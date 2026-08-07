# AGENTS.md

## Working principles

- Clarify requirements that would materially change architecture or product behavior. For small gaps, choose the simplest safe interpretation and record it.
- Preserve unrelated work in this fresh worktree. Do not reset, overwrite, or broadly reformat files outside the task.
- Prefer small, typed, testable changes. Keep external side effects behind durable local state and idempotent/retry-safe boundaries.
- Treat eligibility, consent, billing, authentication, and account deletion as correctness-critical. Fail closed when source evidence or state is ambiguous.
- Never commit plaintext credentials, local certificates, database dumps, or production environment files.

## Overview

Single-binary Go HTTP service that detects and OCRs UK/European licence plates locally via ONNX Runtime. Two pinned ONNX models (YOLOv9 detector, MobileViT-v2 OCR) are downloaded and SHA-256 verified at startup. See `README.md` for the public API contract and env-var table.

## Commands

```bash
make check     # completion gate: fmt + test + race + vet + build
make test      # go test ./...
make race      # go test -race ./...
make vet       # go vet ./...
make build     # go build ./...
make fmt       # gofmt -w over cmd/ and internal/
make run       # go run ./cmd/server (needs ONNXRUNTIME_SHARED_LIBRARY_PATH)
```

Run a single test:

```bash
go test ./internal/plate -run TestExtractorSelectsBestPlateAcrossImages -v
```

`make check` is the definition of done. The unit suite is hermetic — it requires neither ONNX Runtime nor model files, so it must stay that way (see the interface seam below).

The real-model integration test is opt-in and skips unless all four `PLATE_INTEGRATION_*` path vars are set; the expected-value vars additionally assert against the recorded baseline. Full invocation is in `README.md` under Development. Detector confidence must match the baseline within `1e-4`.

Local runs need a native ONNX Runtime 1.24.x library; the Docker image bundles both the runtime and the models, so the container needs no network at startup.

CI (`.github/workflows/ci.yml`) runs `make check`, then `git diff --exit-code` — an uncommitted formatting rewrite fails the build. It also builds the image for `linux/amd64` and `linux/arm64` without pushing, and on `main` runs Release Please. Publishing lives in `.github/workflows/publish.yml`, which fires on a `v*` tag or a manual dispatch naming an existing tag; it is the only workflow that authenticates to Docker Hub. Three constraints govern edits to either file: every action is pinned to a full commit SHA with a trailing `# vX.Y.Z` comment; the Docker Hub login is scoped (`cpoul/uk-license-plate-extractor@push`), which writes credentials to the Buildx config alone, so a plain `docker push` step would be unauthenticated and the scoped login needs Buildx 0.31.0 or newer; and Release Please must keep running under `RELEASE_PLEASE_TOKEN`, because a tag pushed with the default `GITHUB_TOKEN` would not trigger `publish.yml`.

## Architecture

### Wiring and startup order

`cmd/server/main.go` `run()` is the only composition root, and the order is load-bearing:

1. `config.Load()` — env parsing plus validation (`ONNXRUNTIME_SHARED_LIBRARY_PATH` required; `MAX_IMAGE_BYTES` must not exceed `MAX_REQUEST_BYTES`; `ADDR` falls back to `:$PORT`).
2. `assets.Downloader.Ensure()` per model, under a 10-minute context — must complete before any ONNX call.
3. `plate.InitializeONNXRuntime(cfg.RuntimeLibrary)` — process-global; sets the shared library path and initializes the env. Torn down by deferred `DestroyONNXRuntime()`.
4. Detector then recognizer sessions, each with a deferred `Close()` (reverse order).
5. `plate.NewExtractor` → `httpapi.NewHandler` → `http.Server` with explicit timeouts, plus SIGINT/SIGTERM graceful shutdown.

Because the runtime and sessions are process-global and created once, there is no per-request model loading. Sessions are shared across concurrent requests.

### Package layering

- `internal/config` — env → `Config`. No dependencies on other internal packages.
- `internal/assets` — pinned URLs + SHA-256 digests, download/verify/install.
- `internal/plate` — the whole domain: image decode, preprocessing, ONNX sessions, extraction policy.
- `internal/httpapi` — transport only: multipart parsing, limits, JSON.

`httpapi` declares its own one-method `Extractor` interface rather than depending on `*plate.Extractor`, so handler tests inject a fake and never touch inference.

### The testability seam

`plate.Detector` and `plate.Recognizer` (`internal/plate/types.go`) are the boundary between behavioral logic and native inference. `ONNXDetector`/`ONNXRecognizer` in `onnx.go` are the *only* types that touch `onnxruntime_go`. Everything the extraction policy does — thresholds, ranking, cropping, validation — is exercised against `fakeDetector`/`fakeRecognizer`.

When adding inference behavior, put it behind these interfaces. Any change that makes `internal/plate` unit tests require a native library or a model file is a regression.

### Request flow

`POST /crop-plates` → `Handler.readImages` streams multipart parts named `images`, enforcing `maxRequestBytes` via `http.MaxBytesReader`, `maxImages` by count, and `maxImageBytes` via a `LimitReader` set one byte past the cap → `Extractor.ExtractBest` over all images → single JSON result. Size/count violations map to 413; other parse failures to 400.

`Extractor.extractOne` per image: `decodeRGB` (40 MP cap) → `Detector.Detect` → pick the single highest-confidence box passing `MinimumDetectionConfidence` and having positive area → padded `crop` → `grayscale` → `Recognizer.Recognize` → strip non-alphanumerics → average char confidence gate → length gate → JPEG encode.

`ExtractBest` then ranks accepted per-image results by **detector** confidence, not OCR confidence. `Result.Confidence` is the detector score.

### Tensor contracts

The two models disagree on layout, and neither is negotiable:

- **Detector**: NCHW `[1,3,608,608]` `float32`, channel-planar, `/255` normalized, letterboxed with fill `114.0/255`. Output is `[N,7]`; columns 1–4 are the box corners and column 6 is confidence. `decodeDetections` maps coordinates back to original image space by inverting the `letterboxTransform` (subtract padding, divide by ratio).
- **OCR**: NHWC `[1,70,140,1]` **`uint8`, unnormalized** raw grayscale. Output is 9 slots × the 37-character `ocrAlphabet` (`0-9A-Z_`); `decodeOCR` takes a per-slot argmax and trims trailing `_` padding.

## Invariants

**Preprocessing must match what the models were trained on.** `preprocess.go` implements a specific bilinear resampler — `linearCoefficients` uses fixed-point weights at a `1<<11` scale, and `detectorInput` uses `math.RoundToEven` with a `-0.1` padding bias. Do not replace it with `golang.org/x/image/draw` or "simplify" the rounding: the rest of the suite will still pass while detection confidence silently drifts. `internal/plate/resize_golden_test.go` pins the resampler hermetically; the opt-in integration test's `1e-4` check is the second line of defence.

**The two resize axes are not symmetric.** The horizontal axis clamps the interpolation fraction to zero when a tap falls outside the source; the vertical axis keeps the fraction and clamps only the row index. This is visible in the output because `verticalLinear` truncates each row's contribution separately, so a split weight pair rounds differently from a single collapsed weight — 1 LSB, on exactly those output rows whose source tap falls outside `[0, sourceHeight-1]`. That is a leading and trailing band whose width grows with the upscale factor, not just the first and last row. Pure vertical downscale is unaffected. `linearCoefficients` therefore takes a `resizeAxis`; do not collapse the two branches back together.

**Regenerate golden vectors on x86-64 only.** `NOTICE` records the fixture formula, the oracle, and the exact procedure. The recorded vectors were verified elementwise on x86-64 with zero differing elements. The reference resampler dispatches to architecture-specific kernels on aarch64 that differ by up to 4/255, and which shapes and channel counts deviate varies by build and by size — two independent sweeps disagreed on whether the `cn=3` detector letterbox is affected. Treat any aarch64 measurement as an invalid oracle rather than trying to enumerate the exceptions; no pure-Go implementation can match those kernels. The same rule applies to the integration baseline.

**Resize parity does not extend to JPEG decoding.** Decoded JPEG pixels differ from the reference decoder, so do not go looking for the cause in `preprocess.go`. Two measured, independent causes:

- *Chroma upsampling.* Go's `image/jpeg` returns a subsampled `*image.YCbCr` and `image.YCbCr.COffset` resolves 4:2:0 by integer division — nearest-neighbour block replication — where the reference decoder applies a triangle filter. Measured RGB deltas reach 139/255 across 38–72% of pixels on chromatic content (worst on yellow rear plates and EU blue bands), survive the letterbox as 0.17–0.42 in the detector tensor, and move detector confidence by 3e-4 to 3.9e-3 — 3× to 39× the `1e-4` tolerance. Fixable by implementing the triangle filter over Go's Cb/Cr planes (~100 lines), which buys roughly 40× on RGB error but still does not reliably reach `1e-4`.
- *IDCT.* Go's scaled-integer IDCT differs by ±1 on the luma plane (0.3–2.1% of pixels), which is why even 4:4:4 JPEGs exceed `1e-4` in most cases. Irreducible without a cgo decoder binding, which would break the hermetic-unit-test invariant.

Parity does hold for PNG input, to within 1.8e-7 of the recorded baseline. That is also the limit of what the integration test proves, because the reference fixture is a PNG and never exercises the `*image.YCbCr` path. A 4:2:0 JPEG re-encode of the same image would fail against the recorded confidence today — measured at 3.0e-3 off.

**Extraction contract constants** (`internal/plate/types.go`) are part of the public API, not tuning knobs: detector threshold `0.65`, OCR average threshold `0.50`, plate length `2`–`8`, crop padding `5`px, JPEG quality `90`, decode cap 40 MP. Treat a change as an intentional API version bump, and update `README.md` alongside.

**Concurrency is the memory bound.** `MAX_REQUEST_BYTES`/`MAX_IMAGE_BYTES` cap *compressed* uploads, but decoding expands them enormously — a 128 KB PNG at the 40 MP cap decodes to ~640 MB resident. `Handler.admit` bounds requests in decode/inference to `MAX_CONCURRENT_REQUESTS` and is acquired *before* the body is read, so queued requests hold only a connection; excess sheds with `503` and `Retry-After` after `AdmissionWait`. Never move the acquire below `readImages`.

The limit must be sized against the container's memory, because it only converts an unbounded blowup into a bounded one. Measured against 128 KB decode bombs at the cap:

| memory limit | `MAX_CONCURRENT_REQUESTS` | peak RSS | survives |
|---|---|---|---|
| 1 GiB | 1 | 788 MiB | yes |
| 1 GiB | 2 | — | **OOM-killed** |
| 2 GiB | 2 | 1.31 GiB | yes |
| 4 GiB | 4 (default) | 2.50 GiB | yes |

Roughly `0.15 GiB + 0.64 GiB × MAX_CONCURRENT_REQUESTS`. The default of `4` therefore needs ~3 GiB; a 1 GiB container must set `MAX_CONCURRENT_REQUESTS=1`. Typical traffic is far cheaper (a 12 MP photo is ~85 MB per slot) — these figures are the adversarial worst case, which is the one that has to fit.

**Per-image failure isolation.** A bad decode or inference error on one image is logged and skipped so later images still produce results; only context cancellation/deadline propagates as an error. Preserve this when touching `ExtractBest`.

**JSON `null` is a success case.** No accepted plate returns HTTP 200 with a `null` body (a nil `*plate.Result` marshalled directly), not 404 and not an error envelope.

**Model integrity.** `Ensure` verifies pre-existing files on every startup, downloads to a temp file, verifies the digest, then atomically renames. A checksum mismatch fails startup rather than overwriting. Model binaries are never committed; changing a model means updating the URL and digest in both `internal/assets/models.go` and the `Dockerfile` models stage.

## Conventions

### Conventional commits and releases

Release Please derives semantic versions from commits on `main`. Use Conventional Commits for every commit and for any PR title that may become a squash-merge commit:

- `fix(plate): reject empty OCR results` triggers a patch release.
- `feat(api): add batch extraction` triggers a minor release.
- `feat(api)!: change the response schema` triggers a major release. A `BREAKING CHANGE: <description>` footer also marks a breaking change.

Write the description in the imperative mood, lowercase its first word, and omit the trailing period. Use an optional scope when it adds useful context. Choose other Conventional Commit types only when they accurately describe the change; never disguise a user-visible fix or feature as `chore` to avoid a release.

Do not edit release versions or create release tags manually. Merge the Release Please PR when the accumulated changes are ready to publish.

- Standard-library-first Go; `gofmt` all touched files.
- Structured logging with `log/slog`; never log raw image bytes.
- Repo-specific notes also live as Serena memories in `.serena/memories/` (`core` is the graph root). If you establish a durable, non-obvious convention, update those alongside this file; `.serena/memories/memory_maintenance.md` documents the style and the add/update threshold.

### Use Serena MCP for Semantic Code Analysis instead of regular code search and editing

Serena MCP is available for advanced code retrieval and editing capabilities.

**When to use Serena:**
- Symbol-based code navigation (find definitions, references, implementations)
- Precise code manipulation in structured codebases
- Prefer symbol-based operations over file-based grep/sed when available

**Key tools:**
- `find_symbol` - Find symbol by name across the codebase
- `find_referencing_symbols` - Find all symbols that reference a given symbol
- `get_symbols_overview` - Get overview of top-level symbols in a file
- `read_file` - Read file content within the project directory

**Usage notes:**
- Memory files can be manually reviewed/edited in `.serena/memories/`
