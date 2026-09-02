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
	if cfg.IntraOpThreads != 0 || cfg.EarlyExitConfidence != 0 {
		t.Fatalf("inference tuning should default to disabled: %+v", cfg)
	}
}

func TestLoadParsesInferenceTuning(t *testing.T) {
	for _, tc := range []struct {
		name           string
		threads        string
		confidence     string
		wantThreads    int
		wantConfidence float64
	}{
		{"production values", "2", "0.85", 2, 0.85},
		{"explicit zeros keep the defaults", "0", "0", 0, 0},
		{"upper bound of confidence", "1", "1", 1, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/tmp/libonnxruntime.dylib")
			t.Setenv("ONNX_INTRA_OP_THREADS", tc.threads)
			t.Setenv("EARLY_EXIT_CONFIDENCE", tc.confidence)

			cfg, err := Load()
			if err != nil {
				t.Fatalf("Load() error = %v", err)
			}
			if cfg.IntraOpThreads != tc.wantThreads {
				t.Fatalf("IntraOpThreads = %d, want %d", cfg.IntraOpThreads, tc.wantThreads)
			}
			if cfg.EarlyExitConfidence != tc.wantConfidence {
				t.Fatalf("EarlyExitConfidence = %v, want %v", cfg.EarlyExitConfidence, tc.wantConfidence)
			}
		})
	}
}

func TestLoadRejectsInvalidInferenceTuning(t *testing.T) {
	for _, tc := range []struct {
		name, variable, value string
	}{
		{"negative threads", "ONNX_INTRA_OP_THREADS", "-1"},
		{"fractional threads", "ONNX_INTRA_OP_THREADS", "1.5"},
		{"non-numeric threads", "ONNX_INTRA_OP_THREADS", "all"},
		{"negative confidence", "EARLY_EXIT_CONFIDENCE", "-0.1"},
		{"confidence above one", "EARLY_EXIT_CONFIDENCE", "1.01"},
		{"non-numeric confidence", "EARLY_EXIT_CONFIDENCE", "high"},
		{"NaN confidence", "EARLY_EXIT_CONFIDENCE", "NaN"},
		{"infinite confidence", "EARLY_EXIT_CONFIDENCE", "Inf"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("ONNXRUNTIME_SHARED_LIBRARY_PATH", "/tmp/libonnxruntime.dylib")
			t.Setenv(tc.variable, tc.value)

			if _, err := Load(); err == nil {
				t.Fatalf("Load() error = nil, want %s=%q rejected", tc.variable, tc.value)
			}
		})
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
