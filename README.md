# UK License Plate Extractor

[![CI/CD](https://github.com/cpouldev/uk-license-plate-extractor/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cpouldev/uk-license-plate-extractor/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/cpouldev/uk-license-plate-extractor)](https://github.com/cpouldev/uk-license-plate-extractor/releases/latest)
[![Go version](https://img.shields.io/github/go-mod/go-version/cpouldev/uk-license-plate-extractor)](https://go.dev/)
[![Docker pulls](https://img.shields.io/docker/pulls/cpoul/uk-license-plate-extractor)](https://hub.docker.com/r/cpoul/uk-license-plate-extractor)
[![Docker image size](https://img.shields.io/docker/image-size/cpoul/uk-license-plate-extractor/latest)](https://hub.docker.com/r/cpoul/uk-license-plate-extractor/tags)
[![License](https://img.shields.io/github/license/cpouldev/uk-license-plate-extractor)](LICENSE.md)

Send it photos of vehicles, get back the licence plate. It reads UK and European plates entirely on your own machine — no API key, no cloud service, no per-request cost.

Everything ships inside one container: the Go binary, CPU ONNX Runtime, and both AI models. After the image is pulled, the service runs with no network access at all.

## Quick start

**1. Start the service.**

```bash
docker run --rm -p 8080:8080 cpoul/uk-license-plate-extractor:latest
```

**2. Check that it is up.** Both models load during startup, so give it a moment on the first run.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

**3. Read a plate.**

```bash
curl --fail-with-body -F 'images=@car.jpg' http://localhost:8080/crop-plates
```

```json
{
    "crop": "<base64 JPEG>",
    "text": "AB12CDE",
    "confidence": 0.91
}
```

You get the plate text, a cropped JPEG of the plate itself, and how confident the detector was. Send several photos of the same vehicle in one request and the service returns the single best reading:

```bash
curl --fail-with-body \
    -F 'images=@front.jpg' \
    -F 'images=@rear.jpg' \
    http://localhost:8080/crop-plates
```

## Docker Compose

For anything beyond a quick try, use Compose. Save this as `compose.yaml`:

```yaml
services:
  plate-extractor:
    image: cpoul/uk-license-plate-extractor:latest
    ports:
      - "8080:8080"
    environment:
      MAX_CONCURRENT_REQUESTS: "2"
      MAX_IMAGES: "20"
      MAX_IMAGE_BYTES: "15728640"
    mem_limit: 2g
    restart: unless-stopped
```

```bash
docker compose up -d
curl http://localhost:8080/health
```

Here is what each key does:

| Key | Why it is there |
| --- | --- |
| `image` | Use `latest` to try things out. Pin `X.Y.Z` in production so a new release never surprises you. |
| `ports` | `host:container`. The service always listens on `8080` inside the container; change the left number to move it on your host. |
| `environment` | Every setting is an environment variable. See [Settings](#settings). |
| `mem_limit` | **Set this.** Decoding a photo costs far more memory than uploading it, and this cap must match `MAX_CONCURRENT_REQUESTS`. See [How much memory?](#how-much-memory) |
| `restart` | Restarts the container after a crash or a host reboot. |

A few container facts worth knowing:

- It listens on `:8080` and runs as UID 65532, never as root.
- It needs no volumes. The models live inside the image, already checksum-verified.
- It carries no `curl` or `wget`, so a Compose `healthcheck` has nothing to run. Probe `GET /health` from outside the container instead.

## Settings

Every setting is an environment variable. These are the ones you are likely to touch:

| Variable | Default | What it does |
| --- | --- | --- |
| `MAX_CONCURRENT_REQUESTS` | `4` | How many requests decode and run inference at once. **This drives memory use.** Extra requests wait up to 5 seconds, then get `503`. |
| `ONNX_INTRA_OP_THREADS` | `0` | Threads each inference session may use for a single operator. `0` lets ONNX Runtime size the pool to every host core, which makes concurrent requests fight each other on small hosts; there, set it to roughly cores ÷ `MAX_CONCURRENT_REQUESTS`. |
| `EARLY_EXIT_CONFIDENCE` | `0` | Stop scanning a request's images as soon as a plate is found at or above this detector confidence. `0` evaluates every image and returns the best; `0.85` is a good production value. |
| `MAX_IMAGES` | `20` | Images allowed per request. |
| `MAX_IMAGE_BYTES` | `15728640` | Largest single upload, 15 MiB by default. Must not exceed `MAX_REQUEST_BYTES`, or the service refuses to start. |
| `MAX_REQUEST_BYTES` | `52428800` | Largest whole request, 50 MiB by default. |
| `ADDR` | `:8080` | Listen address. `PORT` also works when `ADDR` is unset, which suits platforms that inject it. |

The image presets the rest, so you only need them when running from source:

| Variable | Default | What it does |
| --- | --- | --- |
| `ONNXRUNTIME_SHARED_LIBRARY_PATH` | required | Absolute path to the native ONNX Runtime library. |
| `MODEL_DIR` | `models` | Where the model files live. |
| `DETECTOR_MODEL_PATH` | `$MODEL_DIR/yolo-v9-s-608-license-plates-end2end.onnx` | Detector model. |
| `OCR_MODEL_PATH` | `$MODEL_DIR/european_mobile_vit_v2_ocr.onnx` | OCR model. |

The service decodes JPEG, PNG, GIF, and WebP, up to 40 megapixels per image.

### How much memory?

The byte limits above cap **uploads**, and an upload is compressed. Decoding expands it enormously: a 128 KB PNG crafted to hit the 40 megapixel cap becomes roughly 640 MB of raw pixels. `MAX_CONCURRENT_REQUESTS` is what bounds that blowup, so size the two together:

| Container memory | `MAX_CONCURRENT_REQUESTS` | Peak measured | Survives |
| --- | --- | --- | --- |
| 1 GiB | 1 | 788 MiB | yes |
| 1 GiB | 2 | — | **killed** |
| 2 GiB | 2 | 1.31 GiB | yes |
| 4 GiB | 4 (default) | 2.50 GiB | yes |

Roughly: `0.15 GiB + 0.64 GiB × MAX_CONCURRENT_REQUESTS`. The default of `4` therefore wants about 3 GiB, and a 1 GiB container should set `MAX_CONCURRENT_REQUESTS=1`.

Everyday traffic is far cheaper — a 12 megapixel phone photo costs about 85 MB per slot. These numbers are the worst case, which is the one that has to fit.

## API

### `GET /health`

Returns `{"status":"ok"}` once both models have loaded.

### `POST /crop-plates`

Send one or more files as repeated multipart fields named `images`. On success you get HTTP 200 and a single result:

| Field | Meaning |
| --- | --- |
| `text` | The recognised VRM (vehicle registration mark), stripped to letters and digits. |
| `crop` | Base64 JPEG of the plate region alone. |
| `confidence` | The **detector's** score for that plate, between 0 and 1. |

Across several images, the service returns whichever accepted plate the detector was most confident about — not the longest or clearest-looking text. Set `EARLY_EXIT_CONFIDENCE` to stop at the first accepted plate that clears that score instead, which skips the remaining images.

**When no plate is found**, you still get HTTP 200, with the body `null`. That is a normal answer, not an error: no plate in the batch cleared the thresholds below.

A `crop` is a handy second-pass input. If the text fails your own validation, send that small JPEG — rather than the full photo — to a vision-capable model for another opinion; it costs a fraction of the bandwidth and image tokens. This service never contacts an LLM itself.

**A plate is accepted only when all of these hold:**

- it is the highest-confidence detection in its image;
- the detector scored it at least `0.65`;
- the average OCR character confidence is at least `0.50`;
- the text is 2 to 8 characters after non-alphanumerics are stripped.

Accepted crops are padded by 5 pixels and encoded as JPEG quality 90.

### Errors

Failures return the matching status and a JSON body of the form `{"error": "..."}`.

| Status | When | What to do |
| --- | --- | --- |
| `413` | The request or one image exceeded its byte cap, or you sent more than `MAX_IMAGES`. | Send fewer or smaller images, or raise the limits. |
| `400` | The multipart body could not be parsed, or it carried no `images` field at all. | Check that every file field is named `images`. |
| `503` | Every concurrency slot was busy for 5 seconds. Includes `Retry-After: 1`. | Back off and retry, or raise `MAX_CONCURRENT_REQUESTS` along with the memory limit. |
| `500` | Extraction failed outright. | Check the container logs. |

One bad image never sinks the request: the service logs it, skips it, and carries on with the rest.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for source setup, development commands, tests, container builds, and releases.
