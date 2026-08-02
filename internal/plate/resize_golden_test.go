package plate

import (
	"testing"
)

// Golden vectors for resizePlanar. A change to resizePlanar, linearCoefficients or
// verticalLinear leaves the rest of the suite green while detection confidence silently
// drifts, and the opt-in integration test needs ONNX Runtime plus two model downloads —
// these are the only hermetic guard.
//
// Sources are filled with src[i] = (i*37 + 11) % 251, so the fixtures are reproducible
// anywhere. Every expectation was recorded from the reference resampler on x86-64 and
// verified elementwise, with zero differing elements.
//
// Regenerate on x86-64 only. The reference dispatches to architecture-specific kernels on
// aarch64 whose output differs by up to 4/255 — one of these cases differed on 3071 of 9800
// elements — so vectors recorded there are invalid. Which shapes and channel counts deviate
// varies by build and by size, so treat any aarch64 result as an invalid oracle rather than
// working around it.
//
// The cases deliberately include vertical upscales, which are the only place the two axes'
// edge rules differ.
var resizeGoldenCases = []struct {
	name                      string
	sourceWidth, sourceHeight int
	width, height, channels   int
	want                      []uint8
}{
	{
		name: "gray vertical upscale", sourceWidth: 8, sourceHeight: 3, width: 6, height: 7, channels: 1,
		want: []uint8{
			17, 66, 116, 165, 214, 55, 23, 73, 122, 172, 203, 55, 43, 92, 141, 191, 168, 56,
			62, 112, 161, 210, 134, 58, 81, 131, 180, 211, 99, 77, 101, 150, 199, 213, 65, 96,
			107, 156, 206, 213, 53, 103,
		},
	},
	{
		name: "gray horizontal upscale vertical downscale", sourceWidth: 5, sourceHeight: 9, width: 11, height: 4, channels: 1,
		want: []uint8{
			127, 133, 150, 153, 98, 44, 60, 77, 94, 111, 118, 72, 79, 96, 113, 129, 146, 163,
			180, 185, 188, 189, 175, 181, 198, 195, 112, 29, 46, 63, 68, 71, 72, 120, 110, 84,
			67, 83, 100, 117, 134, 151, 167, 174,
		},
	},
	{
		name: "rgb upscale both axes", sourceWidth: 4, sourceHeight: 4, width: 7, height: 7, channels: 3,
		want: []uint8{
			11, 48, 85, 51, 88, 124, 114, 151, 188, 177, 89, 126, 223, 27, 64, 143, 90, 127,
			93, 130, 167, 80, 117, 64, 87, 124, 104, 100, 137, 167, 157, 113, 150, 202, 89,
			126, 122, 102, 138, 72, 109, 146, 190, 227, 31, 147, 183, 71, 77, 114, 134, 123,
			152, 188, 169, 189, 226, 89, 120, 157, 39, 76, 113, 175, 212, 124, 125, 162, 118,
			45, 82, 110, 91, 128, 165, 145, 173, 210, 136, 93, 130, 132, 43, 80, 142, 179, 216,
			98, 135, 166, 28, 65, 86, 66, 103, 131, 121, 141, 178, 184, 71, 108, 224, 28, 65,
			109, 146, 183, 116, 153, 133, 128, 165, 53, 105, 142, 98, 88, 118, 155, 151, 130,
			167, 191, 138, 175, 88, 125, 162, 128, 165, 112, 191, 228, 32, 129, 166, 77, 67,
			104, 141, 130, 167, 204, 170, 207, 244,
		},
	},
	{
		name: "rgb downscale both axes", sourceWidth: 9, sourceHeight: 6, width: 3, height: 2, channels: 3,
		want: []uint8{
			117, 154, 191, 199, 236, 22, 30, 67, 104, 102, 139, 176, 184, 221, 7, 15, 52, 89,
		},
	},
	{
		name: "gray single source row", sourceWidth: 3, sourceHeight: 1, width: 5, height: 5, channels: 1,
		want: []uint8{
			11, 26, 48, 70, 85, 11, 26, 48, 70, 85, 11, 26, 48, 70, 85, 11, 26, 48, 70, 85,
			11, 26, 48, 70, 85,
		},
	},
	{
		name: "rgb narrow source", sourceWidth: 2, sourceHeight: 5, width: 6, height: 2, channels: 3,
		want: []uint8{
			178, 26, 63, 178, 26, 63, 152, 63, 100, 126, 100, 137, 100, 137, 174, 100, 137, 174,
			168, 205, 242, 168, 205, 242, 121, 158, 195, 74, 111, 148, 28, 65, 102, 28, 65, 102,
		},
	},
}

func goldenSource(width, height, channels int) []uint8 {
	source := make([]uint8, width*height*channels)
	for i := range source {
		source[i] = uint8((i*37 + 11) % 251)
	}
	return source
}

func TestResizePlanarMatchesGoldenVectors(t *testing.T) {
	for _, testCase := range resizeGoldenCases {
		t.Run(testCase.name, func(t *testing.T) {
			source := goldenSource(testCase.sourceWidth, testCase.sourceHeight, testCase.channels)
			got := resizePlanar(source, testCase.sourceWidth, testCase.sourceHeight,
				testCase.width, testCase.height, testCase.channels)

			if len(got) != len(testCase.want) {
				t.Fatalf("length = %d, want %d", len(got), len(testCase.want))
			}
			for i := range testCase.want {
				if got[i] != testCase.want[i] {
					row := (i / testCase.channels) / testCase.width
					column := (i / testCase.channels) % testCase.width
					t.Fatalf("element %d (row %d, column %d, channel %d) = %d, want %d",
						i, row, column, i%testCase.channels, got[i], testCase.want[i])
				}
			}
		})
	}
}

// TestResizePlanarPipelineShapesAreStable pins the two shapes the service actually produces.
// Full expectations would be 9800 and 1108992 bytes, so these are checksummed instead.
func TestResizePlanarPipelineShapesAreStable(t *testing.T) {
	for _, testCase := range []struct {
		name                      string
		sourceWidth, sourceHeight int
		width, height, channels   int
		wantSum                   int
	}{
		{"ocr crop upscaled to 140x70", 20, 10, 140, 70, 1, 1219415},
		{"detector letterbox to 608x608", 50, 90, 608, 608, 3, 138490366},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			source := goldenSource(testCase.sourceWidth, testCase.sourceHeight, testCase.channels)
			got := resizePlanar(source, testCase.sourceWidth, testCase.sourceHeight,
				testCase.width, testCase.height, testCase.channels)

			if want := testCase.width * testCase.height * testCase.channels; len(got) != want {
				t.Fatalf("length = %d, want %d", len(got), want)
			}
			sum := 0
			for _, value := range got {
				sum += int(value)
			}
			if sum != testCase.wantSum {
				t.Fatalf("checksum = %d, want %d", sum, testCase.wantSum)
			}
		})
	}
}
