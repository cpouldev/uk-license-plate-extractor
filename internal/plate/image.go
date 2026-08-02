package plate

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	"image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"
)

type RGBImage struct {
	Width  int
	Height int
	Pix    []uint8
}

type GrayImage struct {
	Width  int
	Height int
	Pix    []uint8
}

func decodeRGB(data []byte, maxPixels int64) (RGBImage, error) {
	if len(data) == 0 {
		return RGBImage{}, fmt.Errorf("empty image")
	}
	configuration, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return RGBImage{}, fmt.Errorf("decode image configuration: %w", err)
	}
	if configuration.Width <= 0 || configuration.Height <= 0 {
		return RGBImage{}, fmt.Errorf("invalid image dimensions %dx%d", configuration.Width, configuration.Height)
	}
	if int64(configuration.Width) > maxPixels/int64(configuration.Height) {
		return RGBImage{}, fmt.Errorf("decoded image exceeds %d pixels", maxPixels)
	}

	decoded, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return RGBImage{}, fmt.Errorf("decode image: %w", err)
	}
	bounds := decoded.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	// DecodeConfig reports the logical dimensions, but a decoder may hand back a smaller
	// or empty frame (an animated GIF's first frame, for example). An empty one would
	// reach detectorInput and turn into int(NaN) there.
	if width <= 0 || height <= 0 {
		return RGBImage{}, fmt.Errorf("decoded image is empty (%dx%d)", width, height)
	}

	source := nonPremultipliedReader(decoded)
	pixels := make([]uint8, width*height*3)
	for y := range height {
		for x := range width {
			pixel := source.RGBA64At(bounds.Min.X+x, bounds.Min.Y+y)
			offset := (y*width + x) * 3
			pixels[offset] = uint8(pixel.R >> 8)
			pixels[offset+1] = uint8(pixel.G >> 8)
			pixels[offset+2] = uint8(pixel.B >> 8)
		}
	}
	return RGBImage{Width: width, Height: height, Pix: pixels}, nil
}

// nonPremultipliedReader picks the cheapest reader that yields straight, non-premultiplied
// colour. RGBA64At is alpha-premultiplied, which darkens translucent pixels toward black and
// discards the stored colour of fully transparent ones outright. The models expect alpha to
// be ignored and the stored colour kept. Formats that cannot carry alpha use RGBA64At directly,
// which avoids boxing every pixel into a color.Color interface — that boxing allocates for the
// YCbCr values the JPEG and WebP decoders return.
func nonPremultipliedReader(img image.Image) image.RGBA64Image {
	switch source := img.(type) {
	case *image.NRGBA:
		return straightNRGBA{source}
	case *image.NRGBA64:
		return straightNRGBA64{source}
	case *image.YCbCr, *image.Gray, *image.Gray16, *image.CMYK:
		return source.(image.RGBA64Image)
	default:
		return straightAlpha{img}
	}
}

type straightNRGBA struct{ *image.NRGBA }

func (s straightNRGBA) RGBA64At(x, y int) color.RGBA64 {
	offset := s.PixOffset(x, y)
	pixel := s.Pix[offset : offset+4 : offset+4]
	return color.RGBA64{
		R: uint16(pixel[0]) * 0x101,
		G: uint16(pixel[1]) * 0x101,
		B: uint16(pixel[2]) * 0x101,
		A: uint16(pixel[3]) * 0x101,
	}
}

type straightNRGBA64 struct{ *image.NRGBA64 }

func (s straightNRGBA64) RGBA64At(x, y int) color.RGBA64 {
	offset := s.PixOffset(x, y)
	pixel := s.Pix[offset : offset+8 : offset+8]
	return color.RGBA64{
		R: uint16(pixel[0])<<8 | uint16(pixel[1]),
		G: uint16(pixel[2])<<8 | uint16(pixel[3]),
		B: uint16(pixel[4])<<8 | uint16(pixel[5]),
		A: uint16(pixel[6])<<8 | uint16(pixel[7]),
	}
}

// straightAlpha covers decoders outside the stdlib set above. Convert cannot recover the
// stored colour of a fully transparent pixel, so those still read as black.
type straightAlpha struct{ image.Image }

func (s straightAlpha) RGBA64At(x, y int) color.RGBA64 {
	pixel := color.NRGBA64Model.Convert(s.Image.At(x, y)).(color.NRGBA64)
	return color.RGBA64{R: pixel.R, G: pixel.G, B: pixel.B, A: pixel.A}
}

func (img RGBImage) crop(box BoundingBox, padding int) (RGBImage, error) {
	x1 := max(0, box.X1-padding)
	y1 := max(0, box.Y1-padding)
	x2 := min(img.Width, box.X2+padding)
	y2 := min(img.Height, box.Y2+padding)
	if x1 >= x2 || y1 >= y2 {
		return RGBImage{}, fmt.Errorf("empty crop after clamping box %+v to %dx%d", box, img.Width, img.Height)
	}

	width, height := x2-x1, y2-y1
	pixels := make([]uint8, width*height*3)
	for y := range height {
		sourceStart := ((y1+y)*img.Width + x1) * 3
		destinationStart := y * width * 3
		copy(pixels[destinationStart:destinationStart+width*3], img.Pix[sourceStart:sourceStart+width*3])
	}
	return RGBImage{Width: width, Height: height, Pix: pixels}, nil
}

func (img RGBImage) grayscale() GrayImage {
	pixels := make([]uint8, img.Width*img.Height)
	for i := range pixels {
		offset := i * 3
		r := uint32(img.Pix[offset])
		g := uint32(img.Pix[offset+1])
		b := uint32(img.Pix[offset+2])
		pixels[i] = uint8((4899*r + 9617*g + 1868*b + 8192) >> 14)
	}
	return GrayImage{Width: img.Width, Height: img.Height, Pix: pixels}
}

func encodeJPEG(img RGBImage, quality int) ([]byte, error) {
	rgba := image.NewRGBA(image.Rect(0, 0, img.Width, img.Height))
	for y := range img.Height {
		for x := range img.Width {
			source := (y*img.Width + x) * 3
			destination := y*rgba.Stride + x*4
			rgba.Pix[destination] = img.Pix[source]
			rgba.Pix[destination+1] = img.Pix[source+1]
			rgba.Pix[destination+2] = img.Pix[source+2]
			rgba.Pix[destination+3] = 0xff
		}
	}

	var output bytes.Buffer
	if err := jpeg.Encode(&output, rgba, &jpeg.Options{Quality: quality}); err != nil {
		return nil, fmt.Errorf("encode crop as JPEG: %w", err)
	}
	return output.Bytes(), nil
}
