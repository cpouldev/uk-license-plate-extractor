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
	raw := envOrDefault(name, "")
	if raw == "" {
		return fallback, nil
	}
	value, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", name)
	}
	return value, nil
}

func positiveIntEnv(name string, fallback int) (int, error) {
	value, err := positiveInt64Env(name, int64(fallback))
	if err != nil {
		return 0, err
	}
	if value > math.MaxInt {
		return 0, fmt.Errorf("%s is too large", name)
	}
	return int(value), nil
}
