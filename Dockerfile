# syntax=docker/dockerfile:1.7

FROM debian:bookworm-slim AS onnxruntime

ARG TARGETARCH
ARG ONNXRUNTIME_VERSION=1.24.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) archive_arch="x64"; archive_sha="9142552248b735920f9390027e4512a2cacf8946a1ffcbe9071a5c210531026f" ;; \
        arm64) archive_arch="aarch64"; archive_sha="0f56edd68f7602df790b68b874a46b115add037e88385c6c842bb763b39b9f89" ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    archive="onnxruntime-linux-${archive_arch}-${ONNXRUNTIME_VERSION}.tgz"; \
    curl --fail --location --retry 3 --output "/tmp/${archive}" \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${archive}"; \
    echo "${archive_sha}  /tmp/${archive}" | sha256sum --check -; \
    tar --extract --gzip --file "/tmp/${archive}" --directory /tmp; \
    install -D -m 0755 \
        "/tmp/onnxruntime-linux-${archive_arch}-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so.${ONNXRUNTIME_VERSION}" \
        "/opt/onnxruntime/libonnxruntime.so.${ONNXRUNTIME_VERSION}"

FROM debian:bookworm-slim AS models

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /models; \
    curl --fail --location --retry 3 --output /models/yolo-v9-s-608-license-plates-end2end.onnx \
        https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-s-608-license-plates-end2end.onnx; \
    echo "2b878b38d9aa07b6ddc3ea75c4ffcb39869bc5c218e0a14002f60ab2f7b0be9a  /models/yolo-v9-s-608-license-plates-end2end.onnx" | sha256sum --check -; \
    curl --fail --location --retry 3 --output /models/european_mobile_vit_v2_ocr.onnx \
        https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/european_mobile_vit_v2_ocr.onnx; \
    echo "5f388f57ddec318d38d17e420d292f5a049595bec93f111838903f6617f6943f  /models/european_mobile_vit_v2_ocr.onnx" | sha256sum --check -

FROM golang:1.26-bookworm AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/plate-extractor ./cmd/server

FROM debian:bookworm-slim

ARG ONNXRUNTIME_VERSION=1.24.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=onnxruntime /opt/onnxruntime/libonnxruntime.so.1.24.1 /usr/local/lib/libonnxruntime.so.1.24.1
COPY --from=models /models /models
COPY --from=build /out/plate-extractor /usr/local/bin/plate-extractor
COPY NOTICE /usr/local/share/doc/uk-license-plate-extractor/NOTICE

ENV ADDR=:8080 \
    MODEL_DIR=/models \
    ONNXRUNTIME_SHARED_LIBRARY_PATH=/usr/local/lib/libonnxruntime.so.1.24.1

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/plate-extractor"]
