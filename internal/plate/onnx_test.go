package plate

import (
	"math"
	"testing"
)

func TestDecodeDetectionsRestoresOriginalCoordinates(t *testing.T) {
	transform := letterboxTransform{Ratio: 2, PaddingX: 10, PaddingY: 20}
	output := []float32{0, 30, 40, 110, 100, 0, 0.875}

	detections := decodeDetections(output, transform)
	if len(detections) != 1 {
		t.Fatalf("len(detections) = %d, want 1", len(detections))
	}
	want := BoundingBox{X1: 10, Y1: 10, X2: 50, Y2: 40}
	if detections[0].BoundingBox != want {
		t.Fatalf("box = %+v, want %+v", detections[0].BoundingBox, want)
	}
	if math.Abs(detections[0].Confidence-0.875) > 1e-9 {
		t.Fatalf("confidence = %f, want 0.875", detections[0].Confidence)
	}
}

func TestDecodeOCRUsesArgmaxAndTrimsTrailingPadding(t *testing.T) {
	output := make([]float32, 9*len(ocrAlphabet))
	indices := []int{10, 11, 1, 2, 36, 36, 36, 36, 36}
	for slot, index := range indices {
		output[slot*len(ocrAlphabet)+index] = float32(0.91 - float64(slot)*0.01)
	}

	prediction := decodeOCR(output)
	if prediction.Text != "AB12" {
		t.Fatalf("Text = %q, want AB12", prediction.Text)
	}
	if len(prediction.CharConfidences) != 9 {
		t.Fatalf("len(CharConfidences) = %d, want 9", len(prediction.CharConfidences))
	}
	if math.Abs(prediction.CharConfidences[0]-float64(float32(0.91))) > 1e-9 {
		t.Fatalf("first confidence = %f, want float32(0.91)", prediction.CharConfidences[0])
	}
}

func TestDetectorInputKeepsExtremeAspectRatios(t *testing.T) {
	// Past roughly 1216:1 the minor axis rounds to zero, which would leave the tensor as
	// pure padding and silently hide the image from the detector.
	for _, size := range [][2]int{{2000, 1}, {3000, 2}, {1, 2000}, {40000, 1}} {
		image := RGBImage{Width: size[0], Height: size[1], Pix: make([]uint8, size[0]*size[1]*3)}
		for i := range image.Pix {
			image.Pix[i] = 255
		}

		input, _ := detectorInput(image)
		padding := float32(114.0 / 255.0)
		content := 0
		for _, value := range input {
			if value != padding {
				content++
			}
		}
		if content == 0 {
			t.Fatalf("%dx%d produced a tensor of pure padding; the image never reached the detector", size[0], size[1])
		}
	}
}

func TestDetectorInputLetterboxesWithRGBPlanes(t *testing.T) {
	image := RGBImage{Width: 2, Height: 1, Pix: []uint8{255, 0, 0, 0, 255, 0}}
	input, transform := detectorInput(image)
	if len(input) != 3*detectorImageSize*detectorImageSize {
		t.Fatalf("input length = %d", len(input))
	}
	if transform.Ratio != 304 || transform.PaddingX != 0 || transform.PaddingY != 152 {
		t.Fatalf("transform = %+v", transform)
	}
	firstImagePixel := 152 * detectorImageSize
	if input[firstImagePixel] != 1 || input[detectorImageSize*detectorImageSize+firstImagePixel] != 0 {
		t.Fatalf("first resized pixel is not RGB red: R=%f G=%f", input[firstImagePixel], input[detectorImageSize*detectorImageSize+firstImagePixel])
	}
}
