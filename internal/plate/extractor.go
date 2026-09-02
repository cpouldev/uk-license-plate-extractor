package plate

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"unicode"
)

type Extractor struct {
	detector   Detector
	recognizer Recognizer
	logger     *slog.Logger
	// earlyExitConfidence, when positive, ends a request's scan at the first accepted
	// plate the detector scored at least this high instead of ranking every image.
	earlyExitConfidence float64
}

// Option adjusts an Extractor beyond the fixed extraction contract.
type Option func(*Extractor)

// WithEarlyExitConfidence stops scanning a request's images as soon as an accepted plate
// reaches confidence, trading the best-of-batch guarantee for latency on multi-image
// requests. Zero keeps every image in the ranking.
func WithEarlyExitConfidence(confidence float64) Option {
	return func(e *Extractor) { e.earlyExitConfidence = confidence }
}

func NewExtractor(detector Detector, recognizer Recognizer, logger *slog.Logger, options ...Option) *Extractor {
	if logger == nil {
		logger = slog.Default()
	}
	extractor := &Extractor{
		detector:   detector,
		recognizer: recognizer,
		logger:     logger,
	}
	for _, option := range options {
		option(extractor)
	}
	return extractor
}

func (e *Extractor) ExtractBest(ctx context.Context, images [][]byte) (*Result, error) {
	var best *Result
	bestIndex := -1
	for index, data := range images {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		candidate, err := e.extractOne(ctx, data)
		if err != nil {
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, err
			}
			e.logger.Warn("failed to process image", "index", index, "bytes", len(data), "error", err)
			continue
		}
		if candidate != nil && (best == nil || candidate.Confidence > best.Confidence) {
			best = candidate
			bestIndex = index
			if e.earlyExitConfidence > 0 && best.Confidence >= e.earlyExitConfidence {
				e.logger.Info("early exit on confident plate", "index", index, "text", best.Text, "confidence", best.Confidence)
				break
			}
		}
	}
	if best != nil {
		e.logger.Info("selected best plate", "index", bestIndex, "text", best.Text, "confidence", best.Confidence)
	}
	return best, nil
}

func (e *Extractor) extractOne(ctx context.Context, data []byte) (*Result, error) {
	image, err := decodeRGB(data, MaximumDecodedPixels)
	if err != nil {
		return nil, err
	}
	detections, err := e.detector.Detect(ctx, image)
	if err != nil {
		return nil, fmt.Errorf("detect plate: %w", err)
	}

	var best *Detection
	for i := range detections {
		detection := &detections[i]
		if detection.BoundingBox.Width() <= 0 || detection.BoundingBox.Height() <= 0 {
			continue
		}
		if !detection.BoundingBox.overlaps(image.Width, image.Height) {
			continue
		}
		// NaN loses every comparison, so it would slip past the threshold below and then
		// never be displaced as the running best.
		if math.IsNaN(detection.Confidence) || math.IsInf(detection.Confidence, 0) {
			continue
		}
		if detection.Confidence < MinimumDetectionConfidence {
			continue
		}
		if best == nil || detection.Confidence > best.Confidence {
			best = detection
		}
	}
	if best == nil {
		return nil, nil
	}

	crop, err := image.crop(best.BoundingBox, CropPaddingPixels)
	if err != nil {
		return nil, err
	}
	prediction, err := e.recognizer.Recognize(ctx, crop.grayscale())
	if err != nil {
		return nil, fmt.Errorf("recognize plate: %w", err)
	}
	characters, confidences := scoredCharacters(prediction)
	// A recognizer that cannot score every retained character leaves the gate unenforced,
	// so treat that as a rejection rather than passing the plate through unvalidated.
	if len(confidences) != len(characters) {
		return nil, nil
	}
	if len(characters) > 0 {
		total := 0.0
		for _, confidence := range confidences {
			total += confidence
		}
		if total/float64(len(characters)) < MinimumOCRConfidence {
			return nil, nil
		}
	}
	if len(characters) < MinimumPlateCharacters || len(characters) > MaximumPlateCharacters {
		return nil, nil
	}

	encodedCrop, err := encodeJPEG(crop, CropJPEGQuality)
	if err != nil {
		return nil, err
	}
	return &Result{Crop: encodedCrop, Text: string(characters), Confidence: best.Confidence}, nil
}

// scoredCharacters drops padding and punctuation while keeping each retained character
// paired with the confidence of the slot it came from. decodeOCR trims only trailing
// padding, so index i of Text still addresses slot i of CharConfidences; filtering the two
// independently would shift every score after the first interior underscore.
func scoredCharacters(prediction OCRPrediction) ([]rune, []float64) {
	characters := make([]rune, 0, len(prediction.CharConfidences))
	confidences := make([]float64, 0, len(prediction.CharConfidences))
	for index, character := range []rune(prediction.Text) {
		if !unicode.IsLetter(character) && !unicode.IsDigit(character) {
			continue
		}
		characters = append(characters, character)
		if index < len(prediction.CharConfidences) {
			confidences = append(confidences, prediction.CharConfidences[index])
		}
	}
	return characters, confidences
}
