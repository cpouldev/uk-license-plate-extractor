package plate

import (
	"bytes"
	"context"
	"errors"
	"image"
	"image/color"
	"image/png"
	"io"
	"log/slog"
	"math"
	"testing"
)

type detectorResponse struct {
	detections []Detection
	err        error
}

type fakeDetector struct {
	responses []detectorResponse
	calls     int
}

func (f *fakeDetector) Detect(_ context.Context, _ RGBImage) ([]Detection, error) {
	response := f.responses[f.calls]
	f.calls++
	return response.detections, response.err
}

type recognizerResponse struct {
	prediction OCRPrediction
	err        error
}

type fakeRecognizer struct {
	responses []recognizerResponse
	calls     int
}

func (f *fakeRecognizer) Recognize(_ context.Context, _ GrayImage) (OCRPrediction, error) {
	response := f.responses[f.calls]
	f.calls++
	return response.prediction, response.err
}

func TestExtractorSelectsBestPlateAcrossImages(t *testing.T) {
	detector := &fakeDetector{responses: []detectorResponse{
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 10, Y1: 10, X2: 20, Y2: 20}, Confidence: 0.70}}},
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 10, Y1: 10, X2: 20, Y2: 20}, Confidence: 0.91}}},
	}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{
		{prediction: OCRPrediction{Text: "AA11_", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9}}},
		{prediction: OCRPrediction{Text: "BB22_", CharConfidences: []float64{0.8, 0.8, 0.8, 0.8}}},
	}}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 30, 30), testPNG(t, 30, 30)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result == nil {
		t.Fatal("ExtractBest() result = nil")
	}
	if result.Text != "BB22" || result.Confidence != 0.91 {
		t.Fatalf("result = %+v, want BB22 at 0.91", result)
	}
	configuration, err := jpegConfig(result.Crop)
	if err != nil {
		t.Fatalf("decode crop configuration: %v", err)
	}
	if configuration.Width != 20 || configuration.Height != 20 {
		t.Fatalf("crop dimensions = %dx%d, want 20x20", configuration.Width, configuration.Height)
	}
}

func TestExtractorStopsAtFirstPlateReachingEarlyExitConfidence(t *testing.T) {
	// The second image would win a full ranking, so returning the first proves the scan
	// stopped. Scoring exactly the threshold counts as reaching it.
	detector := &fakeDetector{responses: []detectorResponse{
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 10, Y1: 10, X2: 20, Y2: 20}, Confidence: 0.85}}},
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 10, Y1: 10, X2: 20, Y2: 20}, Confidence: 0.99}}},
	}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{
		{prediction: OCRPrediction{Text: "AA11", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9}}},
		{prediction: OCRPrediction{Text: "BB22", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9}}},
	}}
	extractor := NewExtractor(detector, recognizer, discardLogger(), WithEarlyExitConfidence(0.85))

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 30, 30), testPNG(t, 30, 30)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result == nil || result.Text != "AA11" || result.Confidence != 0.85 {
		t.Fatalf("result = %+v, want the first confident plate AA11 at 0.85", result)
	}
	if detector.calls != 1 {
		t.Fatalf("detector calls = %d, want 1: the scan should stop at the first image", detector.calls)
	}
}

func TestExtractorKeepsScanningBelowEarlyExitConfidence(t *testing.T) {
	detector := &fakeDetector{responses: []detectorResponse{
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 10, Y1: 10, X2: 20, Y2: 20}, Confidence: 0.80}}},
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 10, Y1: 10, X2: 20, Y2: 20}, Confidence: 0.84}}},
	}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{
		{prediction: OCRPrediction{Text: "AA11", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9}}},
		{prediction: OCRPrediction{Text: "BB22", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9}}},
	}}
	extractor := NewExtractor(detector, recognizer, discardLogger(), WithEarlyExitConfidence(0.85))

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 30, 30), testPNG(t, 30, 30)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result == nil || result.Text != "BB22" || result.Confidence != 0.84 {
		t.Fatalf("result = %+v, want the best of both images BB22 at 0.84", result)
	}
	if detector.calls != 2 {
		t.Fatalf("detector calls = %d, want 2: nothing reached the threshold", detector.calls)
	}
}

func TestExtractorSkipsInvalidImageAndInferenceError(t *testing.T) {
	detector := &fakeDetector{responses: []detectorResponse{
		{err: errors.New("temporary inference failure")},
		{detections: []Detection{{BoundingBox: BoundingBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Confidence: 0.8}}},
	}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{{
		prediction: OCRPrediction{Text: "AB_12", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9, 0.9}},
	}}}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{
		[]byte("not an image"),
		testPNG(t, 20, 20),
		testPNG(t, 20, 20),
	})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result == nil || result.Text != "AB12" {
		t.Fatalf("result = %+v, want sanitized AB12", result)
	}
}

func TestExtractorRejectsLowDetectionWithoutOCR(t *testing.T) {
	detector := &fakeDetector{responses: []detectorResponse{{detections: []Detection{{
		BoundingBox: BoundingBox{X1: 1, Y1: 1, X2: 10, Y2: 10}, Confidence: 0.649,
	}}}}}
	recognizer := &fakeRecognizer{}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 20, 20)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if recognizer.calls != 0 {
		t.Fatalf("recognizer calls = %d, want 0", recognizer.calls)
	}
}

func TestExtractorUsesOnlyHighestConfidenceDetection(t *testing.T) {
	detector := &fakeDetector{responses: []detectorResponse{{detections: []Detection{
		{BoundingBox: BoundingBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Confidence: 0.9},
		{BoundingBox: BoundingBox{X1: 4, Y1: 4, X2: 14, Y2: 14}, Confidence: 0.8},
	}}}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{{prediction: OCRPrediction{
		Text: "AB12", CharConfidences: []float64{0.4, 0.4, 0.4, 0.4},
	}}}}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 20, 20)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if recognizer.calls != 1 {
		t.Fatalf("recognizer calls = %d, want exactly 1", recognizer.calls)
	}
}

func TestExtractorScoresConfidencesAgainstTheCharactersItKeeps(t *testing.T) {
	// decodeOCR trims only trailing padding, so the leading underscore is still present and
	// index i of Text addresses slot i of CharConfidences. Averaging the first four
	// confidences scores the padding slot and ignores the last real character.
	detector := &fakeDetector{responses: []detectorResponse{{detections: []Detection{
		{BoundingBox: BoundingBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Confidence: 0.9},
	}}}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{{prediction: OCRPrediction{
		Text:            "_AB12",
		CharConfidences: []float64{0.99, 0.60, 0.60, 0.60, 0.01},
	}}}}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 20, 20)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	// A,B,1,2 carry 0.60, 0.60, 0.60, 0.01 -> 0.4525, below MinimumOCRConfidence.
	if result != nil {
		t.Fatalf("result = %+v, want nil: retained characters average 0.4525", result)
	}
}

func TestExtractorRejectsPredictionItCannotFullyScore(t *testing.T) {
	detector := &fakeDetector{responses: []detectorResponse{{detections: []Detection{
		{BoundingBox: BoundingBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Confidence: 0.9},
	}}}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{{
		prediction: OCRPrediction{Text: "AB12"},
	}}}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 20, 20)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil when no confidences back the text", result)
	}
}

func TestExtractorIgnoresNonFiniteDetectionConfidence(t *testing.T) {
	for name, confidence := range map[string]float64{
		"NaN":       math.NaN(),
		"+Inf":      math.Inf(1),
		"-Inf":      math.Inf(-1),
		"very high": math.MaxFloat64,
	} {
		t.Run(name, func(t *testing.T) {
			detector := &fakeDetector{responses: []detectorResponse{{detections: []Detection{
				{BoundingBox: BoundingBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Confidence: confidence},
				{BoundingBox: BoundingBox{X1: 3, Y1: 3, X2: 13, Y2: 13}, Confidence: 0.9},
			}}}}
			recognizer := &fakeRecognizer{responses: []recognizerResponse{{prediction: OCRPrediction{
				Text: "AB12", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9},
			}}}}
			extractor := NewExtractor(detector, recognizer, discardLogger())

			result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 20, 20)})
			if err != nil {
				t.Fatalf("ExtractBest() error = %v", err)
			}
			if result == nil {
				t.Fatal("ExtractBest() result = nil, want the finite detection to win")
			}
			if math.IsNaN(result.Confidence) || math.IsInf(result.Confidence, 0) {
				t.Fatalf("confidence = %v, want a finite score that survives JSON encoding", result.Confidence)
			}
			if name != "very high" && result.Confidence != 0.9 {
				t.Fatalf("confidence = %v, want 0.9", result.Confidence)
			}
		})
	}
}

func TestExtractorSkipsDetectionOutsideTheFrame(t *testing.T) {
	// A detection landing in the letterbox padding inverts to coordinates off the frame.
	// It must not win the ranking and abort the image; the next-best plate should be used.
	detector := &fakeDetector{responses: []detectorResponse{{detections: []Detection{
		{BoundingBox: BoundingBox{X1: 25, Y1: 5, X2: 40, Y2: 15}, Confidence: 0.95},
		{BoundingBox: BoundingBox{X1: 2, Y1: 2, X2: 12, Y2: 12}, Confidence: 0.80},
	}}}}
	recognizer := &fakeRecognizer{responses: []recognizerResponse{{prediction: OCRPrediction{
		Text: "AB12", CharConfidences: []float64{0.9, 0.9, 0.9, 0.9},
	}}}}
	extractor := NewExtractor(detector, recognizer, discardLogger())

	result, err := extractor.ExtractBest(context.Background(), [][]byte{testPNG(t, 20, 20)})
	if err != nil {
		t.Fatalf("ExtractBest() error = %v", err)
	}
	if result == nil || result.Text != "AB12" || result.Confidence != 0.80 {
		t.Fatalf("result = %+v, want the in-frame 0.80 detection", result)
	}
	if recognizer.calls != 1 {
		t.Fatalf("recognizer calls = %d, want 1", recognizer.calls)
	}
}

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func testPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := range height {
		for x := range width {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 100, A: 255})
		}
	}
	var output bytes.Buffer
	if err := png.Encode(&output, img); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}
	return output.Bytes()
}

func jpegConfig(data []byte) (image.Config, error) {
	configuration, _, err := image.DecodeConfig(bytes.NewReader(data))
	return configuration, err
}
