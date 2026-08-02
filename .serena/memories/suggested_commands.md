# Commands
- Run unit suite: `go test ./...`.
- Run race suite: `go test -race ./...`.
- Static analysis: `go vet ./...`.
- Format: `gofmt -w ./cmd ./internal` (or the Makefile wrapper once present).
- Build: `go build ./...`.
- Run locally: set `ONNXRUNTIME_SHARED_LIBRARY_PATH`, then `go run ./cmd/server`; default model assets are downloaded and checksum-verified if absent.
- Container: `docker build -t uk-license-plate-extractor .`, then publish the configured port.