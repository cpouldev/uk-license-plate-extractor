# Completion gate
- Run `gofmt` on all touched Go files.
- Run `go test ./...`, `go test -race ./...`, `go vet ./...`, and `go build ./...`.
- Run real-model integration verification with the pinned detector/OCR assets: expected cleaned plate text and crop geometry; detector confidence delta from the recorded fixture baseline <= 1e-4. The baseline holds for the PNG reference fixture only — Go's JPEG decoder diverges from the reference decoder (chroma upsampling, IDCT), so a JPEG fixture lands 3e-4 to 3.9e-3 out and would fail this check. See the preprocessing invariants in AGENTS.md.
- Build the Docker image and smoke-test `GET /health` plus `POST /crop-plates`.
- Confirm model download checksums and inspect the target directory for unintended binary artifacts.
- Report any unavailable native/container checks explicitly.
