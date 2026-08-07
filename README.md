# UK License Plate Extractor

[![CI/CD](https://github.com/cpouldev/uk-license-plate-extractor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cpouldev/uk-license-plate-extractor/actions/workflows/ci.yml)
[![Go version](https://img.shields.io/github/go-mod/go-version/cpouldev/uk-license-plate-extractor)](https://go.dev/)

A small Go HTTP service that detects and reads UK/European licence plates locally with ONNX Runtime. It pins the detection and OCR models, and documents every threshold, ranking rule, and response field below.

## Install

The published image bundles CPU ONNX Runtime 1.24.1 and both checksum-pinned models, so it starts without network access:

```bash
docker run --rm -p 8080:8080 cpoul/uk-license-plate-extractor:latest
```

`latest` tracks the newest semantic release. `X.Y.Z` pins an exact release; `sha-<full-commit-sha>` pins its source revision. All published tags cover `linux/amd64` and `linux/arm64`.

With Compose:

```yaml
services:
  plate-extractor:
    image: cpoul/uk-license-plate-extractor:latest
    ports:
      - "8080:8080"
    environment:
      MAX_CONCURRENT_REQUESTS: "2"
    mem_limit: 2g
    restart: unless-stopped
```

Size `mem_limit` and `MAX_CONCURRENT_REQUESTS` together — the pairing above is measured-safe, and [Configuration](#configuration) gives the formula.

The container listens on `:8080` and runs as UID 65532. It carries no HTTP client, so probe `GET /health` from outside rather than through a Compose `healthcheck`.

## API

### `GET /health`

Returns `{"status":"ok"}` after both ONNX models have loaded.

### `POST /crop-plates`

Send one or more files as repeated multipart fields named `images`:

```bash
curl --fail-with-body \
    -F 'images=@front.jpg' \
    -F 'images=@rear.jpg' \
    http://localhost:8080/crop-plates
```

On an accepted plate, the service responds:

```json
{
    "crop": "<base64 JPEG>",
    "text": "AB12CDE",
    "confidence": 0.91
}
```

`text` is the locally recognised VRM (vehicle registration mark) candidate; `crop` holds the detected plate region alone. When that candidate fails downstream validation or needs review, a client can send the small JPEG, rather than the full source image, to a vision-capable LLM for a second reading. The smaller input cuts bandwidth and image tokens, which keeps the optional second pass fast and cheap. This service itself never contacts an LLM.

When no image yields an accepted plate, the service returns HTTP 200 with JSON `null`. It skips a bad image and processes the rest of the request.

Extraction rules:

- choose only the highest-confidence detection within each image;
- require detector confidence of at least `0.65`;
- crop with five pixels of padding and encode as JPEG quality 90;
- remove non-alphanumeric OCR characters;
- require average OCR character confidence of at least `0.50`;
- accept cleaned text lengths from two through eight characters;
- rank accepted results across images by detector confidence.

## Run locally

Go loads ONNX Runtime dynamically. Point the service at a native ONNX Runtime 1.24.x library:

```bash
export ONNXRUNTIME_SHARED_LIBRARY_PATH=/absolute/path/to/libonnxruntime.dylib
go run ./cmd/server
```

First startup downloads the two pinned models into `./models` and verifies their SHA-256 digests. Every later startup re-verifies the existing files; a mismatch fails startup rather than silently overwriting the file.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ADDR` | `:8080` | Listen address; `PORT` is also supported when `ADDR` is unset |
| `ONNXRUNTIME_SHARED_LIBRARY_PATH` | required | Absolute native ONNX Runtime library path |
| `MODEL_DIR` | `models` | Directory for pinned model assets |
| `DETECTOR_MODEL_PATH` | `$MODEL_DIR/yolo-v9-s-608-license-plates-end2end.onnx` | Detector location |
| `OCR_MODEL_PATH` | `$MODEL_DIR/european_mobile_vit_v2_ocr.onnx` | OCR location |
| `MAX_REQUEST_BYTES` | `52428800` | Maximum multipart request size |
| `MAX_IMAGE_BYTES` | `15728640` | Maximum compressed bytes per image |
| `MAX_IMAGES` | `20` | Maximum images per request |
| `MAX_CONCURRENT_REQUESTS` | `4` | Requests admitted to decode/inference at once; excess queues up to 5s, then gets `503` with `Retry-After` |

**Size `MAX_CONCURRENT_REQUESTS` against your memory limit.** The byte limits above cap compressed uploads; one image at the 40 MP decode cap expands to ~640 MB in memory. Budget roughly `0.15 GiB + 0.64 GiB × MAX_CONCURRENT_REQUESTS` — the default of `4` needs ~3 GiB, and a 1 GiB container should set `MAX_CONCURRENT_REQUESTS=1`. Ordinary traffic costs far less (~85 MB per slot for a 12 MP photo), but the limit must cover the adversarial case.

The service decodes JPEG, PNG, GIF, and WebP uploads up to 40 megapixels.

## Build the image

To build from source instead of pulling the published image:

```bash
docker build -t uk-license-plate-extractor .
docker run --rm -p 8080:8080 uk-license-plate-extractor
```

The Dockerfile targets Linux `amd64` and `arm64`. On ordinary pull requests and manual CI runs, CI builds both architectures after `make check`; generated Release Please PRs run the Go checks only. A release tag performs the single release build and publishes it to Docker Hub.

## Releases

[Release Please](https://github.com/googleapis/release-please) maintains a release PR from Conventional Commits. Merging that PR creates a Git tag and GitHub Release such as `v1.2.3`, then publishes the multi-platform Docker image with these tags:

- `latest` and `1.2.3`;
- moving `1.2`, and `1` from version 1 onward;
- `sha-<full-commit-sha>`, an immutable revision reference.

To retry an interrupted publication without creating another release, manually run the `Publish image` workflow with the existing `vX.Y.Z` tag.

Version bumps follow SemVer: `fix:` releases a patch, `feat:` a minor, and a `!` after the commit type or a `BREAKING CHANGE:` footer a major. Other commit types do not trigger a release by default. Prefer squash-merging pull requests with a Conventional Commit title.

Configure these repository Actions secrets:

- `DOCKERHUB_TOKEN`: a Docker Hub access token for `cpoul` with write access to `cpoul/uk-license-plate-extractor`;
- `RELEASE_PLEASE_TOKEN`: a fine-grained GitHub personal access token for this repository, with read/write access to Contents, Issues, and Pull requests. A dedicated token lets release PRs trigger the normal CI workflow.

## Development

```bash
make check
```

`make check` formats the Go sources, runs the unit tests with and without the race detector, runs `go vet`, and builds every package. The unit suite runs without ONNX Runtime or model files. CI runs the same target and then `git diff --exit-code`, so commit whatever the formatting step rewrites.

An opt-in test exercises both real ONNX models against a recorded baseline for the pinned reference image:

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

The baseline is specific to lossless input. Go's JPEG decoder differs from the reference decoder in chroma upsampling and IDCT, so the same image re-encoded as a 4:2:0 JPEG lands 3e-4 to 3.9e-3 away — outside the `1e-4` check. See the preprocessing invariants in `AGENTS.md`.

## Pinned inference assets

| Asset | Upstream | SHA-256 |
| --- | --- | --- |
| YOLOv9 608 plate detector | `open-image-models` release `assets` | `2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a` |
| European MobileViT-v2 OCR | `cnn-ocr-lp` release `arg-plates` | `5f388f57ddec318d38d17e420d292f5a049595bec93f111838903f6617f6943f` |

The repository excludes model binaries by design. See `NOTICE` for third-party attribution, model licensing, and the procedure for regenerating the resize golden vectors.
