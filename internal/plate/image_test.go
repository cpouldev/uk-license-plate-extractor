package plate

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"testing"
)

// emptyFrameGIF declares a 4x4 logical screen but a 0x0 image descriptor, so DecodeConfig
// reports usable dimensions while Decode hands back an empty frame.
var emptyFrameGIF = []byte{
	'G', 'I', 'F', '8', '7', 'a',
	0x04, 0x00, 0x04, 0x00, 0x80, 0x00, 0x00,
	0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF,
	0x2C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x02, 0x00,
	0x3B,
}

func TestDecodeRGBRejectsEmptyDecodedFrame(t *testing.T) {
	configuration, _, err := image.DecodeConfig(bytes.NewReader(emptyFrameGIF))
	if err != nil {
		t.Fatalf("DecodeConfig() error = %v", err)
	}
	if configuration.Width != 4 || configuration.Height != 4 {
		t.Fatalf("fixture no longer reports 4x4 from DecodeConfig, got %dx%d", configuration.Width, configuration.Height)
	}

	if _, err := decodeRGB(emptyFrameGIF, MaximumDecodedPixels); err == nil {
		t.Fatal("decodeRGB() error = nil, want an error for an empty decoded frame")
	}
}

func TestDecodeRGBReadsTransparentPixelsWithoutPremultiplying(t *testing.T) {
	source := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	source.SetNRGBA(0, 0, color.NRGBA{R: 200, G: 150, B: 100, A: 0})
	source.SetNRGBA(1, 0, color.NRGBA{R: 200, G: 150, B: 100, A: 128})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, source); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}

	decoded, err := decodeRGB(encoded.Bytes(), MaximumDecodedPixels)
	if err != nil {
		t.Fatalf("decodeRGB() error = %v", err)
	}
	want := []uint8{200, 150, 100, 200, 150, 100}
	if !bytes.Equal(decoded.Pix, want) {
		t.Fatalf("pixels = %v, want %v (premultiplying darkens translucent pixels toward black)", decoded.Pix, want)
	}
}

func TestDecodeRGBKeepsOpaqueImagesUnchanged(t *testing.T) {
	source := image.NewRGBA(image.Rect(0, 0, 3, 2))
	for y := range 2 {
		for x := range 3 {
			source.Set(x, y, color.RGBA{R: uint8(10 * x), G: uint8(20 * y), B: 30, A: 255})
		}
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, source); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}

	decoded, err := decodeRGB(encoded.Bytes(), MaximumDecodedPixels)
	if err != nil {
		t.Fatalf("decodeRGB() error = %v", err)
	}
	for y := range 2 {
		for x := range 3 {
			offset := (y*3 + x) * 3
			wantR, wantG := uint8(10*x), uint8(20*y)
			if decoded.Pix[offset] != wantR || decoded.Pix[offset+1] != wantG || decoded.Pix[offset+2] != 30 {
				t.Fatalf("pixel (%d,%d) = %v, want %v", x, y,
					decoded.Pix[offset:offset+3], []uint8{wantR, wantG, 30})
			}
		}
	}
}
