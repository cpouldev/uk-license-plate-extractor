package plate

import (
	"context"
	"fmt"
	"slices"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	ocrAlphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
	ocrSlots    = 9
)

func InitializeONNXRuntime(sharedLibraryPath string) error {
	ort.SetSharedLibraryPath(sharedLibraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("initialize ONNX Runtime: %w", err)
	}
	return nil
}

func DestroyONNXRuntime() error {
	return ort.DestroyEnvironment()
}

type ONNXDetector struct {
	session *ort.DynamicAdvancedSession
}

func NewONNXDetector(modelPath string) (*ONNXDetector, error) {
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"images"}, []string{"output0"}, nil)
	if err != nil {
		return nil, fmt.Errorf("load detector model: %w", err)
	}
	return &ONNXDetector{session: session}, nil
}

func (d *ONNXDetector) Close() error {
	return d.session.Destroy()
}

func (d *ONNXDetector) Detect(ctx context.Context, image RGBImage) ([]Detection, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	inputData, transform := detectorInput(image)
	input, err := ort.NewTensor(ort.NewShape(1, 3, detectorImageSize, detectorImageSize), inputData)
	if err != nil {
		return nil, fmt.Errorf("create detector input tensor: %w", err)
	}
	defer func() { _ = input.Destroy() }()

	data, shape, err := runFloat32Session(d.session, "detector", input)
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[1] != 7 || int64(len(data)) != shape[0]*shape[1] {
		return nil, fmt.Errorf("detector output has unexpected shape %v", shape)
	}
	return decodeDetections(data, transform), nil
}

// runFloat32Session owns the lifetime of the native output tensor, so the returned data
// is cloned out of the memory Destroy releases.
func runFloat32Session(session *ort.DynamicAdvancedSession, model string, input ort.Value) ([]float32, ort.Shape, error) {
	outputs := []ort.Value{nil}
	if err := session.Run([]ort.Value{input}, outputs); err != nil {
		return nil, nil, fmt.Errorf("run %s model: %w", model, err)
	}
	if outputs[0] == nil {
		return nil, nil, fmt.Errorf("%s returned no output", model)
	}
	defer func() { _ = outputs[0].Destroy() }()
	output, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("%s output has unexpected type %T", model, outputs[0])
	}
	return slices.Clone(output.GetData()), output.GetShape(), nil
}

func decodeDetections(output []float32, transform letterboxTransform) []Detection {
	detections := make([]Detection, 0, len(output)/7)
	for offset := 0; offset+6 < len(output); offset += 7 {
		x1 := int((float64(output[offset+1]) - transform.PaddingX) / transform.Ratio)
		y1 := int((float64(output[offset+2]) - transform.PaddingY) / transform.Ratio)
		x2 := int((float64(output[offset+3]) - transform.PaddingX) / transform.Ratio)
		y2 := int((float64(output[offset+4]) - transform.PaddingY) / transform.Ratio)
		detections = append(detections, Detection{
			BoundingBox: BoundingBox{X1: x1, Y1: y1, X2: x2, Y2: y2},
			Confidence:  float64(output[offset+6]),
		})
	}
	return detections
}

type ONNXRecognizer struct {
	session *ort.DynamicAdvancedSession
}

func NewONNXRecognizer(modelPath string) (*ONNXRecognizer, error) {
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"input"}, []string{"concatenate"}, nil)
	if err != nil {
		return nil, fmt.Errorf("load OCR model: %w", err)
	}
	return &ONNXRecognizer{session: session}, nil
}

func (r *ONNXRecognizer) Close() error {
	return r.session.Destroy()
}

func (r *ONNXRecognizer) Recognize(ctx context.Context, image GrayImage) (OCRPrediction, error) {
	if err := ctx.Err(); err != nil {
		return OCRPrediction{}, err
	}
	input, err := ort.NewTensor(ort.NewShape(1, ocrImageHeight, ocrImageWidth, 1), ocrInput(image))
	if err != nil {
		return OCRPrediction{}, fmt.Errorf("create OCR input tensor: %w", err)
	}
	defer func() { _ = input.Destroy() }()

	data, shape, err := runFloat32Session(r.session, "OCR", input)
	if err != nil {
		return OCRPrediction{}, err
	}
	if len(data) != ocrSlots*len(ocrAlphabet) {
		return OCRPrediction{}, fmt.Errorf("OCR output has unexpected shape %v", shape)
	}
	return decodeOCR(data), nil
}

func decodeOCR(output []float32) OCRPrediction {
	characters := make([]byte, ocrSlots)
	confidences := make([]float64, ocrSlots)
	for slot := range ocrSlots {
		start := slot * len(ocrAlphabet)
		bestIndex := 0
		bestConfidence := output[start]
		for index := 1; index < len(ocrAlphabet); index++ {
			if output[start+index] > bestConfidence {
				bestIndex = index
				bestConfidence = output[start+index]
			}
		}
		characters[slot] = ocrAlphabet[bestIndex]
		confidences[slot] = float64(bestConfidence)
	}
	return OCRPrediction{
		Text:            strings.TrimRight(string(characters), "_"),
		CharConfidences: confidences,
	}
}
