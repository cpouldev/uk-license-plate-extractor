package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cpoul/uk-license-plate-extractor/internal/assets"
	"github.com/cpoul/uk-license-plate-extractor/internal/config"
	"github.com/cpoul/uk-license-plate-extractor/internal/httpapi"
	"github.com/cpoul/uk-license-plate-extractor/internal/plate"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(context.Background(), logger); err != nil {
		logger.Error("service stopped", "error", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, logger *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}

	downloadContext, cancelDownloads := context.WithTimeout(ctx, 10*time.Minute)
	defer cancelDownloads()
	downloader := assets.Downloader{}
	for _, model := range assets.Models(cfg.DetectorModelPath, cfg.OCRModelPath) {
		logger.Info("checking model", "name", model.Name, "path", model.Path)
		if err := downloader.Ensure(downloadContext, model); err != nil {
			return err
		}
	}

	// Freeing an ONNX session or unloading the runtime while a cgo Run is still executing
	// segfaults the process, so a shutdown that gives up with requests still in flight
	// deliberately leaks the native handles and lets the process exit instead.
	abandonNative := false
	logClose := func(what string, closer func() error) {
		if abandonNative {
			logger.Warn("leaking native handle; requests still in flight at shutdown", "resource", what)
			return
		}
		if err := closer(); err != nil {
			logger.Error("failed to close "+what, "error", err)
		}
	}

	if err := plate.InitializeONNXRuntime(cfg.RuntimeLibrary); err != nil {
		return err
	}
	defer logClose("ONNX Runtime", plate.DestroyONNXRuntime)

	logger.Info("configuring inference", "intra_op_threads", cfg.IntraOpThreads, "early_exit_confidence", cfg.EarlyExitConfidence)
	sessionSettings := plate.SessionSettings{IntraOpThreads: cfg.IntraOpThreads}
	detector, err := plate.NewONNXDetector(cfg.DetectorModelPath, sessionSettings)
	if err != nil {
		return err
	}
	defer logClose("detector", detector.Close)
	recognizer, err := plate.NewONNXRecognizer(cfg.OCRModelPath, sessionSettings)
	if err != nil {
		return err
	}
	defer logClose("recognizer", recognizer.Close)

	extractor := plate.NewExtractor(detector, recognizer, logger, plate.WithEarlyExitConfidence(cfg.EarlyExitConfidence))
	handler := httpapi.NewHandler(extractor, logger, httpapi.Limits{
		MaxRequestBytes: cfg.MaxRequestBytes,
		MaxImageBytes:   cfg.MaxImageBytes,
		MaxImages:       cfg.MaxImages,
		MaxConcurrent:   cfg.MaxConcurrent,
	})
	server := &http.Server{
		Addr:              cfg.Address,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       1 * time.Minute,
	}

	signalContext, stopSignals := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer stopSignals()
	serverErrors := make(chan error, 1)
	go func() {
		logger.Info("service listening", "address", cfg.Address)
		serverErrors <- server.ListenAndServe()
	}()

	select {
	case err := <-serverErrors:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("serve HTTP: %w", err)
	case <-signalContext.Done():
		// Restore the default disposition so a second signal terminates the process
		// instead of being swallowed by an already-cancelled context.
		stopSignals()
		shutdownContext, cancelShutdown := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancelShutdown()
		if err := server.Shutdown(shutdownContext); err != nil {
			abandonNative = true
			return fmt.Errorf("shut down HTTP server: %w", err)
		}
		return nil
	}
}
