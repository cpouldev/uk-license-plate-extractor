package config

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	defaultAddress         = ":8080"
	defaultMaxRequestBytes = int64(50 << 20)
	defaultMaxImageBytes   = int64(15 << 20)
	defaultMaxImages       = 20
	// Each in-flight request can hold a fully decoded image, which reaches hundreds of
	// megabytes at MaximumDecodedPixels. Four keeps the worst case within a typical
	// container while still using several cores; raise it only alongside the memory limit.
	defaultMaxConcurrentRequests = 4
	// Zero leaves both inference knobs at their conservative defaults: ONNX Runtime
	// sizes its thread pools itself, and every image in a request is evaluated.
	defaultIntraOpThreads      = 0
	defaultEarlyExitConfidence = 0.0
)

type Config struct {
	Address           string
	RuntimeLibrary    string
	DetectorModelPath string
	OCRModelPath      string
	MaxRequestBytes   int64
	MaxImageBytes     int64
	MaxImages         int
	MaxConcurrent     int
	// IntraOpThreads bounds each ONNX session's intra-op thread pool; zero keeps the
	// runtime default.
	IntraOpThreads int
	// EarlyExitConfidence stops a request's scan at the first plate scoring at least
	// this much; zero evaluates every image.
	EarlyExitConfidence float64
}

func Load() (Config, error) {
	modelDir := envOrDefault("MODEL_DIR", "models")
	address := envOrDefault("ADDR", "")
	if address == "" {
		if port := envOrDefault("PORT", ""); port != "" {
			address = ":" + port
		} else {
			address = defaultAddress
		}
	}

	cfg := Config{
		Address:           address,
		RuntimeLibrary:    envOrDefault("ONNXRUNTIME_SHARED_LIBRARY_PATH", ""),
		DetectorModelPath: envOrDefault("DETECTOR_MODEL_PATH", filepath.Join(modelDir, "yolo-v9-s-608-license-plates-end2end.onnx")),
		OCRModelPath:      envOrDefault("OCR_MODEL_PATH", filepath.Join(modelDir, "european_mobile_vit_v2_ocr.onnx")),
	}

	var err error
	if cfg.MaxRequestBytes, err = positiveInt64Env("MAX_REQUEST_BYTES", defaultMaxRequestBytes); err != nil {
		return Config{}, err
	}
	if cfg.MaxImageBytes, err = positiveInt64Env("MAX_IMAGE_BYTES", defaultMaxImageBytes); err != nil {
		return Config{}, err
	}
	if cfg.MaxImages, err = positiveIntEnv("MAX_IMAGES", defaultMaxImages); err != nil {
		return Config{}, err
	}
	if cfg.MaxConcurrent, err = positiveIntEnv("MAX_CONCURRENT_REQUESTS", defaultMaxConcurrentRequests); err != nil {
		return Config{}, err
	}
	if cfg.IntraOpThreads, err = nonNegativeIntEnv("ONNX_INTRA_OP_THREADS", defaultIntraOpThreads); err != nil {
		return Config{}, err
	}
	if cfg.EarlyExitConfidence, err = unitIntervalEnv("EARLY_EXIT_CONFIDENCE", defaultEarlyExitConfidence); err != nil {
		return Config{}, err
	}
	if cfg.RuntimeLibrary == "" {
		return Config{}, fmt.Errorf("ONNXRUNTIME_SHARED_LIBRARY_PATH is required")
	}
	if cfg.MaxImageBytes > cfg.MaxRequestBytes {
		return Config{}, fmt.Errorf("MAX_IMAGE_BYTES must not exceed MAX_REQUEST_BYTES")
	}

	return cfg, nil
}

func envOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func positiveInt64Env(name string, fallback int64) (int64, error) {
	return int64EnvAtLeast(name, fallback, 1, "a positive integer")
}

func positiveIntEnv(name string, fallback int) (int, error) {
	return intEnvAtLeast(name, fallback, 1, "a positive integer")
}

func nonNegativeIntEnv(name string, fallback int) (int, error) {
	return intEnvAtLeast(name, fallback, 0, "a non-negative integer")
}

// int64EnvAtLeast parses name as a base-10 integer no smaller than minimum. want names
// the accepted values in the error, for example "a positive integer".
func int64EnvAtLeast(name string, fallback, minimum int64, want string) (int64, error) {
	raw := envOrDefault(name, "")
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || value < minimum {
		return 0, fmt.Errorf("%s must be %s", name, want)
	}
	return value, nil
}

func intEnvAtLeast(name string, fallback int, minimum int64, want string) (int, error) {
	value, err := int64EnvAtLeast(name, int64(fallback), minimum, want)
	if err != nil {
		return 0, err
	}
	if value > math.MaxInt {
		return 0, fmt.Errorf("%s is too large", name)
	}
	return int(value), nil
}

// unitIntervalEnv parses name as a decimal fraction between 0 and 1 inclusive.
func unitIntervalEnv(name string, fallback float64) (float64, error) {
	raw := envOrDefault(name, "")
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(value) || value < 0 || value > 1 {
		return 0, fmt.Errorf("%s must be a number between 0 and 1", name)
	}
	return value, nil
}
