package config

import (
	"path/filepath"
	"testing"
)

func TestLoadDefaults(t *testing.T) {
	t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/tmp/libonnxruntime.dylib")
	t.Setenv("MODEL_DIR", "/tmp/plate-models")

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if cfg.Address != ":8080" {
		t.Fatalf("Address = %q, want :8080", cfg.Address)
	}
	if cfg.DetectorModelPath != filepath.Join("/tmp/plate-models", "yolo-v9-s-608-license-plates-end2end.onnx") {
		t.Fatalf("unexpected detector path: %q", cfg.DetectorModelPath)
	}
	if cfg.MaxImages != 20 || cfg.MaxImageBytes != 15<<20 || cfg.MaxRequestBytes != 50<<20 {
		t.Fatalf("unexpected request limits: %+v", cfg)
	}
}

func TestLoadRejectsMissingRuntime(t *testing.T) {
	t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "")

	if _, err := Load(); err == nil {
		t.Fatal("Load() error = nil, want missing runtime error")
	}
}

func TestLoadRejectsImageLimitAboveRequestLimit(t *testing.T) {
	t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/tmp/libonnxruntime.dylib")
	t.Setenv("MAX_REQUEST_BYTES", "100")
	t.Setenv("MAX_IMAGE_BYTES", "101")

	if _, err := Load(); err == nil {
		t.Fatal("Load() error = nil, want invalid limit error")
	}
}
