package plate

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
)

func TestONNXIntegration(t *testing.T) {
	runtimePath := os.Getenv("PLATE_INTEGRATION_RUNTIME")
	detectorPath := os.Getenv("PLATE_INTEGRATION_DETECTOR")
	ocrPath := os.Getenv("PLATE_INTEGRATION_OCR")
	imagePath := os.Getenv("PLATE_INTEGRATION_IMAGE")
	if runtimePath == "" || detectorPath == "" || ocrPath == "" || imagePath == "" {
		t.Skip("set PLATE_INTEGRATION_RUNTIME, PLATE_INTEGRATION_DETECTOR, PLATE_INTEGRATION_OCR, and PLATE_INTEGRATION_IMAGE")
	}

	if err := InitializeONNXRuntime(runtimePath); err != nil {
		t.Fatalf("InitializeONNXRuntime() error = %v", err)
	}
	t.Cleanup(func() {
		if err := DestroyONNXRuntime(); err != nil {
			t.Errorf("DestroyONNXRuntime() error = %v", err)
		}
	})
	detector, err := NewONNXDetector(detectorPath)
	if err != nil {
		t.Fatalf("NewONNXDetector() error = %v", err)
	}
	t.Cleanup(func() {
		if err := detector.Close(); err != nil {
			t.Errorf("detector.Close() error = %v", err)
		}
	})
	recognizer, err := NewONNXRecognizer(ocrPath)
	if err != nil {
		t.Fatalf("NewONNXRecognizer() error = %v", err)
	}
	t.Cleanup(func() {
		if err := recognizer.Close(); err != nil {
			t.Errorf("recognizer.Close() error = %v", err)
		}
	})

	data, err := os.ReadFile(imagePath)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	extractor := NewExtractor(detector, recognizer, discardLogger())
	result, err := extractor.ExtractBest(context.Background(), [][]byte{data})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result == nil {
		t.Fatal("ExtractBest() result = nil")
	}

	if expected := os.Getenv("PLATE_INTEGRATION_EXPECTED_TEXT"); expected != "" && result.Text != expected {
		t.Fatalf("text = %q, want %q", result.Text, expected)
	}
	if rawExpected := os.Getenv("PLATE_INTEGRATION_EXPECTED_CONFIDENCE"); rawExpected != "" {
		expected, err := strconv.ParseFloat(rawExpected, 64)
		if err != nil {
			t.Fatalf("parse expected confidence: %v", err)
		}
		if delta := math.Abs(result.Confidence - expected); delta > 1e-4 {
			t.Fatalf("confidence = %.10f, want %.10f (delta %.10f)", result.Confidence, expected, delta)
		}
	}
	if expected := os.Getenv("PLATE_INTEGRATION_EXPECTED_CROP"); expected != "" {
		var width, height int
		if _, err := fmt.Sscanf(expected, "%dx%d", &width, &height); err != nil {
			t.Fatalf("PLATE_INTEGRATION_EXPECTED_CROP must have WIDTHxHEIGHT form: %v", err)
		}
		configuration, err := jpegConfig(result.Crop)
		if err != nil {
			t.Fatalf("decode crop: %v", err)
		}
		if configuration.Width != width || configuration.Height != height {
			t.Fatalf("crop = %dx%d, want %dx%d", configuration.Width, configuration.Height, width, height)
		}
	}
}
