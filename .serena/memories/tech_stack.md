# Toolchain
- Go module: `github.com/cpoul/uk-license-plate-extractor`; Go 1.26.
- Inference: `github.com/yalue/onnxruntime_go` v1.27.0 with ONNX Runtime 1.24.1-compatible C API.
- Models: `yolo-v9-s-608-license-plate-end2end` detector and `european-plates-mobile-vit-v2-model` OCR; CPU execution.
- HTTP uses Go standard library.
- Deployment target: OCI/Docker Linux image; local development on Darwin requires an ONNX Runtime dylib path.