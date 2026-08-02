package plate

import "context"

const (
	MinimumDetectionConfidence = 0.65
	MinimumOCRConfidence       = 0.50
	MinimumPlateCharacters     = 2
	MaximumPlateCharacters     = 8
	CropPaddingPixels          = 5
	CropJPEGQuality            = 90
	MaximumDecodedPixels       = 40_000_000
)

type BoundingBox struct {
	X1 int
	Y1 int
	X2 int
	Y2 int
}

func (b BoundingBox) Width() int  { return b.X2 - b.X1 }
func (b BoundingBox) Height() int { return b.Y2 - b.Y1 }

// overlaps reports whether the box intersects a frame of the given size. Detections that
// land entirely in the letterbox padding invert to coordinates outside the source image.
func (b BoundingBox) overlaps(width, height int) bool {
	return b.X1 < width && b.Y1 < height && b.X2 > 0 && b.Y2 > 0
}

type Detection struct {
	BoundingBox BoundingBox
	Confidence  float64
}

type OCRPrediction struct {
	Text            string
	CharConfidences []float64
}

type Result struct {
	Crop       []byte  `json:"crop"`
	Text       string  `json:"text"`
	Confidence float64 `json:"confidence"`
}

type Detector interface {
	Detect(context.Context, RGBImage) ([]Detection, error)
}

type Recognizer interface {
	Recognize(context.Context, GrayImage) (OCRPrediction, error)
}
