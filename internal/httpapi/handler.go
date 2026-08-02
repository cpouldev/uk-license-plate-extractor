package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"time"

	"github.com/cpoul/uk-license-plate-extractor/internal/plate"
)

type Extractor interface {
	ExtractBest(context.Context, [][]byte) (*plate.Result, error)
}

// defaultAdmissionWait is how long a request queues for a concurrency slot before being
// shed. Long enough to smooth a burst, short enough that clients are not held past a load
// balancer's own patience.
const defaultAdmissionWait = 5 * time.Second

// Limits are the request bounds this package enforces. config.Load guarantees
// MaxImageBytes <= MaxRequestBytes for the values it produces.
type Limits struct {
	MaxRequestBytes int64
	MaxImageBytes   int64
	MaxImages       int
	// MaxConcurrent bounds requests that are decoding or running inference. It is the
	// service's memory bound: the byte limits above cap compressed uploads, but a single
	// image at MaximumDecodedPixels expands to hundreds of megabytes once decoded, so
	// without this a few small uploads can exhaust the container. Zero disables the limit.
	MaxConcurrent int
	// AdmissionWait is how long a request queues for a slot before being shed with a 503.
	// Zero uses defaultAdmissionWait.
	AdmissionWait time.Duration
}

type Handler struct {
	extractor Extractor
	logger    *slog.Logger
	limits    Limits
	slots     chan struct{}
}

func NewHandler(extractor Extractor, logger *slog.Logger, limits Limits) http.Handler {
	if logger == nil {
		logger = slog.Default()
	}
	handler := &Handler{
		extractor: extractor,
		logger:    logger,
		limits:    limits,
	}
	if handler.limits.AdmissionWait <= 0 {
		handler.limits.AdmissionWait = defaultAdmissionWait
	}
	if limits.MaxConcurrent > 0 {
		handler.slots = make(chan struct{}, limits.MaxConcurrent)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", handler.health)
	mux.HandleFunc("POST /crop-plates", handler.cropPlates)
	return mux
}

var healthBody = []byte("{\"status\":\"ok\"}\n")

func (h *Handler) health(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(healthBody)
}

// admit reserves a concurrency slot. It is called before the request body is touched, so
// queued requests hold only a connection rather than buffered or decoded image data.
func (h *Handler) admit(ctx context.Context) (func(), bool) {
	if h.slots == nil {
		return func() {}, true
	}
	select {
	case h.slots <- struct{}{}:
		return func() { <-h.slots }, true
	default:
	}

	timer := time.NewTimer(h.limits.AdmissionWait)
	defer timer.Stop()
	select {
	case h.slots <- struct{}{}:
		return func() { <-h.slots }, true
	case <-timer.C:
		return nil, false
	case <-ctx.Done():
		return nil, false
	}
}

func (h *Handler) cropPlates(w http.ResponseWriter, request *http.Request) {
	release, admitted := h.admit(request.Context())
	if !admitted {
		h.logger.Warn("shedding request at capacity", "limit", h.limits.MaxConcurrent)
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, "server is at capacity, retry shortly")
		return
	}
	defer release()

	request.Body = http.MaxBytesReader(w, request.Body, h.limits.MaxRequestBytes)
	images, err := h.readImages(request)
	if err != nil {
		status := http.StatusBadRequest
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) || errors.Is(err, errImageTooLarge) || errors.Is(err, errTooManyImages) {
			status = http.StatusRequestEntityTooLarge
		}
		writeError(w, status, err.Error())
		return
	}

	result, err := h.extractor.ExtractBest(request.Context(), images)
	if err != nil {
		h.logger.Error("plate extraction failed", "error", err)
		writeError(w, http.StatusInternalServerError, "failed to crop plates")
		return
	}
	writeJSON(w, http.StatusOK, result)
}

var (
	errImageTooLarge = errors.New("image exceeds configured byte limit")
	errTooManyImages = errors.New("too many images")
)

func (h *Handler) readImages(request *http.Request) ([][]byte, error) {
	reader, err := request.MultipartReader()
	if err != nil {
		return nil, fmt.Errorf("request must be multipart/form-data: %w", err)
	}

	// Read one byte past the cap so an oversized part is detectable, saturating first so
	// the increment cannot wrap when MaxImageBytes is configured at math.MaxInt64.
	readLimit := min(h.limits.MaxImageBytes, math.MaxInt64-1) + 1

	images := make([][]byte, 0)
	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read multipart request: %w", err)
		}
		if part.FormName() != "images" {
			_ = part.Close()
			continue
		}
		if len(images) >= h.limits.MaxImages {
			_ = part.Close()
			return nil, fmt.Errorf("%w: maximum is %d", errTooManyImages, h.limits.MaxImages)
		}
		// bytes.Buffer grows by doubling; io.ReadAll reallocates far more often at image sizes.
		var buffer bytes.Buffer
		_, readErr := buffer.ReadFrom(io.LimitReader(part, readLimit))
		closeErr := part.Close()
		if readErr != nil {
			return nil, fmt.Errorf("read image %d: %w", len(images), readErr)
		}
		if closeErr != nil {
			return nil, fmt.Errorf("close image %d: %w", len(images), closeErr)
		}
		if int64(buffer.Len()) > h.limits.MaxImageBytes {
			return nil, fmt.Errorf("%w: image %d exceeds %d bytes", errImageTooLarge, len(images), h.limits.MaxImageBytes)
		}
		images = append(images, buffer.Bytes())
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("at least one multipart field named images is required")
	}
	return images, nil
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
