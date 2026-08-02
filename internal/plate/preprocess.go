package plate

import (
	"math"
	"slices"
)

const (
	detectorImageSize = 608
	ocrImageWidth     = 140
	ocrImageHeight    = 70
)

type letterboxTransform struct {
	Ratio    float64
	PaddingX float64
	PaddingY float64
}

func detectorInput(img RGBImage) ([]float32, letterboxTransform) {
	ratio := math.Min(float64(detectorImageSize)/float64(img.Height), float64(detectorImageSize)/float64(img.Width))
	// Past roughly 1216:1 the minor axis rounds to zero, which would discard the image
	// entirely and hand the detector a frame of pure padding. Clamping is a no-op for
	// every aspect ratio that already produced a non-empty resize.
	resizedWidth := max(1, int(math.RoundToEven(float64(img.Width)*ratio)))
	resizedHeight := max(1, int(math.RoundToEven(float64(img.Height)*ratio)))
	resized := resizeRGB(img, resizedWidth, resizedHeight)
	paddingX := float64(detectorImageSize-resizedWidth) / 2
	paddingY := float64(detectorImageSize-resizedHeight) / 2
	left := int(math.RoundToEven(paddingX - 0.1))
	top := int(math.RoundToEven(paddingY - 0.1))

	planeSize := detectorImageSize * detectorImageSize
	tensor := make([]float32, planeSize*3)
	paddingValue := float32(114.0 / 255.0)
	for i := range tensor {
		tensor[i] = paddingValue
	}
	for y := range resized.Height {
		for x := range resized.Width {
			source := (y*resized.Width + x) * 3
			destination := (y+top)*detectorImageSize + x + left
			tensor[destination] = float32(resized.Pix[source]) / 255
			tensor[planeSize+destination] = float32(resized.Pix[source+1]) / 255
			tensor[2*planeSize+destination] = float32(resized.Pix[source+2]) / 255
		}
	}
	return tensor, letterboxTransform{Ratio: ratio, PaddingX: paddingX, PaddingY: paddingY}
}

func ocrInput(img GrayImage) []uint8 {
	return resizeGray(img, ocrImageWidth, ocrImageHeight).Pix
}

func resizeRGB(source RGBImage, width, height int) RGBImage {
	pixels := resizePlanar(source.Pix, source.Width, source.Height, width, height, 3)
	return RGBImage{Width: width, Height: height, Pix: pixels}
}

func resizeGray(source GrayImage, width, height int) GrayImage {
	pixels := resizePlanar(source.Pix, source.Width, source.Height, width, height, 1)
	return GrayImage{Width: width, Height: height, Pix: pixels}
}

// resizePlanar is the bilinear resampler the detector and OCR models were trained against:
// fixed-point weights at a 1<<11 scale and a two-truncation vertical accumulate. Its output
// is pinned by resize_golden_test.go.
//
// The weights and the rounding are load-bearing. Replacing this with golang.org/x/image/draw,
// or "simplifying" the rounding, leaves every other test green while detection confidence
// silently drifts; the golden vectors are the only hermetic guard.
func resizePlanar(source []uint8, sourceWidth, sourceHeight, width, height, channels int) []uint8 {
	if sourceWidth == width && sourceHeight == height {
		return slices.Clone(source)
	}
	xCoefficients := linearCoefficients(sourceWidth, width, horizontalAxis)
	yCoefficients := linearCoefficients(sourceHeight, height, verticalAxis)
	pixels := make([]uint8, width*height*channels)
	for y := range height {
		yCoefficient := yCoefficients[y]
		for x := range width {
			xCoefficient := xCoefficients[x]
			for channel := range channels {
				topLeft := int32(source[(yCoefficient.first*sourceWidth+xCoefficient.first)*channels+channel])
				topRight := int32(source[(yCoefficient.first*sourceWidth+xCoefficient.second)*channels+channel])
				bottomLeft := int32(source[(yCoefficient.second*sourceWidth+xCoefficient.first)*channels+channel])
				bottomRight := int32(source[(yCoefficient.second*sourceWidth+xCoefficient.second)*channels+channel])
				top := topLeft*xCoefficient.weight0 + topRight*xCoefficient.weight1
				bottom := bottomLeft*xCoefficient.weight0 + bottomRight*xCoefficient.weight1
				pixels[(y*width+x)*channels+channel] = verticalLinear(top, bottom, yCoefficient)
			}
		}
	}
	return pixels
}

type linearCoefficient struct {
	first   int
	second  int
	weight0 int32
	weight1 int32
}

type resizeAxis int

const (
	horizontalAxis resizeAxis = iota
	verticalAxis
)

// linearCoefficients builds one axis of the bilinear weight table. The two axes use
// different edge rules, and the asymmetry is deliberate: the horizontal axis clamps the
// interpolation fraction to zero when a tap falls outside the source, while the vertical
// axis keeps the fraction and clamps only the row index. It is visible in the output
// because verticalLinear truncates each row's contribution separately, so a split weight
// pair rounds differently from a single collapsed weight — 1 LSB, on exactly those output
// rows whose source tap falls outside [0, sourceSize-1]. That is a leading and trailing
// band whose width grows with the upscale factor; pure downscale is unaffected. Do not
// collapse the two branches back together.
func linearCoefficients(sourceSize, destinationSize int, axis resizeAxis) []linearCoefficient {
	const coefficientScale = 1 << 11
	scale := float64(sourceSize) / float64(destinationSize)
	coefficients := make([]linearCoefficient, destinationSize)
	for destination := range coefficients {
		position := float32((float64(destination)+0.5)*scale - 0.5)
		first := int(math.Floor(float64(position)))
		fraction := position - float32(first)
		second := first + 1
		if axis == horizontalAxis {
			if first < 0 {
				first, second, fraction = 0, 0, 0
			} else if first >= sourceSize-1 {
				first, second, fraction = sourceSize-1, sourceSize-1, 0
			}
		} else {
			first = min(max(first, 0), sourceSize-1)
			second = min(max(second, 0), sourceSize-1)
		}
		coefficients[destination] = linearCoefficient{
			first:   first,
			second:  second,
			weight0: int32(math.RoundToEven(float64(float32(1-fraction) * coefficientScale))),
			weight1: int32(math.RoundToEven(float64(fraction * coefficientScale))),
		}
	}
	return coefficients
}

func verticalLinear(top, bottom int32, coefficient linearCoefficient) uint8 {
	value := (((coefficient.weight0 * (top >> 4)) >> 16) +
		((coefficient.weight1 * (bottom >> 4)) >> 16) + 2) >> 2
	if value < 0 {
		return 0
	}
	if value > 255 {
		return 255
	}
	return uint8(value)
}
