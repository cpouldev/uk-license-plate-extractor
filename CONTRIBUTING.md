# Contributing

Contributions are welcome. Read [AGENTS.md](AGENTS.md) before changing inference, preprocessing, request limits, or startup wiring; it records the contracts that ordinary unit tests cannot fully protect.

## Prerequisites

- Go at the version declared in `go.mod`;
- Docker with Buildx for container changes;
- ONNX Runtime 1.24.x only when running the service or opt-in integration test locally.

The unit suite is hermetic. It needs neither ONNX Runtime nor model files.

## Development workflow

Use Conventional Commits for commits and squash-merge PR titles. Release Please derives semantic versions from them:

- `fix(plate): reject empty OCR results` produces a patch release;
- `feat(api): add batch extraction` produces a minor release;
- `feat(api)!: change the response schema` produces a major release. A `BREAKING CHANGE: <description>` footer does the same.

Keep commits focused. Use an optional scope when it clarifies the affected area.

Common commands:

```bash
make check     # completion gate: fmt + test + race + vet + build
make test      # go test ./...
make race      # go test -race ./...
make vet       # go vet ./...
make build     # go build ./...
make fmt       # gofmt -w over cmd/ and internal/
make run       # go run ./cmd/server
```

Run one test:

```bash
go test ./internal/plate -run TestExtractorSelectsBestPlateAcrossImages -v
```

`make check` is the definition of done. CI runs the same target and then `git diff --exit-code`, so commit any formatting changes it produces.

## Run from source

Go loads ONNX Runtime dynamically. Point the service at a native ONNX Runtime 1.24.x library:

```bash
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/absolute/path/to/libonnxruntime.dylib
go run ./cmd/server
```

The first startup downloads both models into `./models` and verifies their SHA-256 digests. Every later startup re-verifies them; a mismatch fails startup rather than overwriting the file.

## Integration test

The opt-in integration test exercises both real models against a recorded baseline:

```bash
curl --fail --location \
    --output /tmp/plate-reference.png \
    https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png

PLATE_INTEGRATION_RUNTIME=/absolute/path/to/libonnxruntime.dylib \
PLATE_INTEGRATION_DETECTOR=/absolute/path/to/yolo-v9-s-608-license-plates-end2end.onnx \
PLATE_INTEGRATION_OCR=/absolute/path/to/european_mobile_vit_v2_ocr.onnx \
PLATE_INTEGRATION_IMAGE=/tmp/plate-reference.png \
PLATE_INTEGRATION_EXPECTED_TEXT=5AU5341 \
PLATE_INTEGRATION_EXPECTED_CONFIDENCE=0.9028370380401611 \
PLATE_INTEGRATION_EXPECTED_CROP=91x32 \
go test ./internal/plate -run TestONNXIntegration -v
```

The baseline applies to lossless input. Go's JPEG decoder differs from the reference decoder in chroma upsampling and IDCT, so the same image encoded as 4:2:0 JPEG lands 3e-4 to 3.9e-3 outside the `1e-4` confidence tolerance. See the preprocessing invariants in [AGENTS.md](AGENTS.md).

## Container builds

Build and run the image locally:

```bash
docker build -t uk-license-plate-extractor .
docker run --rm -p 8080:8080 uk-license-plate-extractor
```

The Dockerfile targets Linux `amd64` and `arm64`.

CI uses this policy:

- pushes to `main`, pull requests, and manual runs execute `make check`;
- ordinary pull requests and manual runs also validate both container architectures without pushing;
- generated Release Please PRs and pushes to `main` skip the redundant image build;
- a semantic release tag builds and publishes the release image once.

## Releases

[Release Please](https://github.com/googleapis/release-please) maintains the changelog and release PR. Merging that PR creates a tag and GitHub Release such as `v1.2.3`; the tag triggers the `Publish image` workflow.

The workflow publishes these Docker Hub tags:

- `latest` and the exact version, such as `1.2.3`;
- the moving minor version, such as `1.2`;
- the moving major version, such as `1`, from version 1 onward;
- `sha-<full-commit-sha>` as an immutable revision reference.

Maintainers must configure these repository Actions secrets:

- `DOCKERHUB_TOKEN`: a Docker Hub access token for `cpoul` with write access to `cpoul/uk-license-plate-extractor`;
- `RELEASE_PLEASE_TOKEN`: a fine-grained GitHub personal access token for this repository, with Contents, Issues, and Pull requests read/write access.

Retry an interrupted publication with the existing release tag:

```bash
gh workflow run publish.yml \
    --repo cpouldev/uk-license-plate-extractor \
    -f tag=v1.2.3
```

Do not edit release versions or create semantic release tags manually.

## Pinned inference assets

| Asset | Upstream | SHA-256 |
| --- | --- | --- |
| YOLOv9 608 plate detector | `open-image-models` release `assets` | `2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a` |
| European MobileViT-v2 OCR | `cnn-ocr-lp` release `arg-plates` | `5f388f57ddec318d38d17e420d292f5a049595bec93f111838903f6617f6943f` |

Never commit model binaries. See [NOTICE](NOTICE) for third-party attribution, model licensing, and the x86-64-only procedure for regenerating resize golden vectors.
