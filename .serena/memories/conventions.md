# Conventions
- Standard-library-first Go; apply `gofmt`.
- Keep ONNX session concerns behind detector/recognizer interfaces so behavioral logic is unit-testable without native runtime/model files.
- Preserve extraction contract constants unless the API contract is intentionally versioned: detector threshold 0.65, OCR average threshold 0.50, plate length 2–8, crop padding 5px, JPEG quality 90.
- Per-image decode/inference failures are isolated; one bad upload must not discard valid results from other images.
- Model artifacts are not committed. Pin download URLs and SHA-256 digests; verify existing and downloaded files.
- Use structured `log/slog`; never log raw image contents.
