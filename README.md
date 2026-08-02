# UK License Plate Extractor

[![CI](https://github.com/cpouldev/uk-license-plate-extractor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cpouldev/uk-license-plate-extractor/actions/workflows/ci.yml)
[![Go version](https://img.shields.io/github/go-mod/go-version/cpouldev/uk-license-plate-extractor)](https://go.dev/)

A small Go HTTP service that detects and reads UK/European licence plates locally with ONNX Runtime. It uses pinned detection and OCR models with explicit validation thresholds, ranking behavior, and response shape.

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

When a plate is accepted, the response is:

```json
{
    "crop": "<base64 JPEG>",
    "text": "AB12CDE",
    "confidence": 0.91
}
```

The `text` field is the locally recognised VRM (vehicle registration mark) candidate. The `crop` contains only the detected plate region. If the candidate fails downstream validation or needs review, a client can send this small JPEG, rather than the full source image, to a vision-capable LLM for a second visual reading. The smaller input reduces bandwidth and image-token usage, making this optional second pass fast and inexpensive. This service does not contact an LLM.

If no image contains an accepted plate, the service returns HTTP 200 with JSON `null`. A bad image is skipped without preventing later images in the same request from being processed.

The extraction rules are:

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

On first startup, the two pinned models are downloaded into `./models` and SHA-256 verified. Existing files are verified every time; a mismatched file causes startup to fail instead of being overwritten silently.

Useful configuration:

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
| `MAX_CONCURRENT_REQUESTS` | `4` | Requests admitted to decode/inference at once; excess queues up to 5s then gets `503` with `Retry-After` |

**Size `MAX_CONCURRENT_REQUESTS` against your memory limit.** The byte limits above cap compressed uploads; a single image at the 40 MP decode cap expands to ~640 MB once decoded. Budget roughly `0.15 GiB + 0.64 GiB × MAX_CONCURRENT_REQUESTS` for the worst case — so the default of `4` wants ~3 GiB, and a 1 GiB container should set `MAX_CONCURRENT_REQUESTS=1`. Ordinary traffic costs far less (~85 MB per slot for a 12 MP photo), but the limit has to cover the adversarial case.

JPEG, PNG, GIF, and WebP uploads are decoded. The maximum decoded image size is 40 megapixels.

## Docker

The image contains CPU ONNX Runtime 1.24.1 and both checksum-pinned models, so it does not need network access at startup:

```bash
docker build -t uk-license-plate-extractor .
docker run --rm -p 8080:8080 uk-license-plate-extractor
```

The Dockerfile supports Linux `amd64` and `arm64` builds.

## Development

```bash
make check
```

`make check` formats the Go sources, runs unit tests and race detection, runs `go vet`, and builds every package. Unit tests do not require ONNX Runtime or model files.

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

Model binaries are intentionally not committed to the repository. See `NOTICE` for third-party attribution, model licensing, and the procedure for regenerating the resize golden vectors.
