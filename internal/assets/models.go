package assets

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	detectorURL    = "https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx"
	detectorSHA256 = "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a"
	ocrURL         = "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/european_mobile_vit_v2_ocr.onnx"
	ocrSHA256      = "5f388f57ddec318d38d17e420d292f5a049595bec93f111838903f6617f6943f"
)

type Model struct {
	Name   string
	Path   string
	URL    string
	SHA256 string
}

func Models(detectorPath, ocrPath string) []Model {
	return []Model{
		{Name: "detector", Path: detectorPath, URL: detectorURL, SHA256: detectorSHA256},
		{Name: "ocr", Path: ocrPath, URL: ocrURL, SHA256: ocrSHA256},
	}
}

type Downloader struct {
	Client *http.Client
}

func (d Downloader) Ensure(ctx context.Context, model Model) error {
	want, err := hex.DecodeString(model.SHA256)
	if err != nil || len(want) != sha256.Size {
		return fmt.Errorf("%s model has invalid SHA-256 configuration", model.Name)
	}

	if err := verifyFile(model.Path, want); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("verify existing %s model: %w", model.Name, err)
	}

	if err := os.MkdirAll(filepath.Dir(model.Path), 0o755); err != nil {
		return fmt.Errorf("create model directory: %w", err)
	}

	temporary, err := os.CreateTemp(filepath.Dir(model.Path), "."+filepath.Base(model.Path)+".download-*")
	if err != nil {
		return fmt.Errorf("create temporary model file: %w", err)
	}
	temporaryPath := temporary.Name()
	keepTemporary := false
	defer func() {
		_ = temporary.Close()
		if !keepTemporary {
			_ = os.Remove(temporaryPath)
		}
	}()

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, model.URL, nil)
	if err != nil {
		return fmt.Errorf("create %s model request: %w", model.Name, err)
	}
	client := d.Client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	response, err := client.Do(request)
	if err != nil {
		return fmt.Errorf("download %s model: %w", model.Name, err)
	}
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4<<10))
		_ = response.Body.Close()
		return fmt.Errorf("download %s model: unexpected HTTP status %s", model.Name, response.Status)
	}

	hash := sha256.New()
	_, copyErr := io.Copy(io.MultiWriter(temporary, hash), response.Body)
	closeErr := response.Body.Close()
	if copyErr != nil {
		return fmt.Errorf("write %s model: %w", model.Name, copyErr)
	}
	if closeErr != nil {
		return fmt.Errorf("close %s model response: %w", model.Name, closeErr)
	}
	if subtle.ConstantTimeCompare(hash.Sum(nil), want) != 1 {
		return fmt.Errorf("download %s model: SHA-256 mismatch", model.Name)
	}
	if err := temporary.Sync(); err != nil {
		return fmt.Errorf("sync %s model: %w", model.Name, err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close %s model: %w", model.Name, err)
	}
	if err := os.Chmod(temporaryPath, 0o644); err != nil {
		return fmt.Errorf("set %s model permissions: %w", model.Name, err)
	}
	if err := os.Rename(temporaryPath, model.Path); err != nil {
		return fmt.Errorf("install %s model: %w", model.Name, err)
	}
	keepTemporary = true
	return nil
}

func verifyFile(path string, want []byte) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, file)
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if subtle.ConstantTimeCompare(hash.Sum(nil), want) != 1 {
		return fmt.Errorf("SHA-256 mismatch")
	}
	return nil
}
