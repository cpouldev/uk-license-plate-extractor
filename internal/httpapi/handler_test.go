package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/cpoul/uk-license-plate-extractor/internal/plate"
)

type fakeExtractor struct {
	result *plate.Result
	err    error
	images [][]byte
}

func (f *fakeExtractor) ExtractBest(_ context.Context, images [][]byte) (*plate.Result, error) {
	f.images = images
	return f.result, f.err
}

func TestHealth(t *testing.T) {
	handler := testHandler(&fakeExtractor{})
	request := httptest.NewRequest(http.MethodGet, "/health", nil)
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK || response.Body.String() != "{\"status\":\"ok\"}\n" {
		t.Fatalf("response = %d %q", response.Code, response.Body.String())
	}
}

func TestCropPlatesAcceptsRepeatedMultipartImages(t *testing.T) {
	extractor := &fakeExtractor{result: &plate.Result{Crop: []byte("jpeg"), Text: "AB12CDE", Confidence: 0.875}}
	handler := testHandler(extractor)
	request := multipartRequest(t, "images", [][]byte{[]byte("first"), []byte("second")})
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if len(extractor.images) != 2 || string(extractor.images[0]) != "first" || string(extractor.images[1]) != "second" {
		t.Fatalf("extractor images = %q", extractor.images)
	}
	var result plate.Result
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if result.Text != "AB12CDE" || string(result.Crop) != "jpeg" || result.Confidence != 0.875 {
		t.Fatalf("result = %+v", result)
	}
}

func TestCropPlatesReturnsJSONNullWhenNoPlateFound(t *testing.T) {
	handler := testHandler(&fakeExtractor{})
	request := multipartRequest(t, "images", [][]byte{[]byte("image")})
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK || response.Body.String() != "null\n" {
		t.Fatalf("response = %d %q, want 200 null", response.Code, response.Body.String())
	}
}

func TestCropPlatesRejectsOversizedImage(t *testing.T) {
	handler := NewHandler(&fakeExtractor{}, discardLogger(), Limits{
		MaxRequestBytes: 1024,
		MaxImageBytes:   4,
		MaxImages:       2,
	})
	request := multipartRequest(t, "images", [][]byte{[]byte("12345")})
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
}

func TestCropPlatesRequiresImagesField(t *testing.T) {
	request := multipartRequest(t, "other", [][]byte{[]byte("value")})
	response := httptest.NewRecorder()

	testHandler(&fakeExtractor{}).ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest || !strings.Contains(response.Body.String(), "images") {
		t.Fatalf("response = %d %q", response.Code, response.Body.String())
	}
}

func TestCropPlatesAcceptsMaximumByteLimits(t *testing.T) {
	// MaxImageBytes+1 wraps negative at math.MaxInt64, which made Grow panic and turned the
	// per-image LimitReader into an immediate EOF.
	extractor := &fakeExtractor{result: &plate.Result{Crop: []byte("jpeg"), Text: "AB12CDE"}}
	handler := NewHandler(extractor, discardLogger(), Limits{
		MaxRequestBytes: math.MaxInt64,
		MaxImageBytes:   math.MaxInt64,
		MaxImages:       2,
	})
	request := multipartRequest(t, "images", [][]byte{[]byte("payload")})
	response := httptest.NewRecorder()

	handler.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if len(extractor.images) != 1 || string(extractor.images[0]) != "payload" {
		t.Fatalf("extractor images = %q, want the full part contents", extractor.images)
	}
}

// blockingExtractor holds each call until release is closed, so a test can pin a
// concurrency slot open.
type blockingExtractor struct {
	entered chan struct{}
	release chan struct{}
}

func (b *blockingExtractor) ExtractBest(_ context.Context, _ [][]byte) (*plate.Result, error) {
	b.entered <- struct{}{}
	<-b.release
	return &plate.Result{Crop: []byte("jpeg"), Text: "AB12CDE"}, nil
}

func TestCropPlatesShedsRequestsBeyondMaxConcurrent(t *testing.T) {
	extractor := &blockingExtractor{entered: make(chan struct{}, 1), release: make(chan struct{})}
	handler := NewHandler(extractor, discardLogger(), Limits{
		MaxRequestBytes: 1 << 20, MaxImageBytes: 1 << 19, MaxImages: 10, MaxConcurrent: 1,
		AdmissionWait: 20 * time.Millisecond,
	})

	first := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		handler.ServeHTTP(first, multipartRequest(t, "images", [][]byte{[]byte("one")}))
	}()
	<-extractor.entered // the only slot is now held

	second := httptest.NewRecorder()
	handler.ServeHTTP(second, multipartRequest(t, "images", [][]byte{[]byte("two")}))

	if second.Code != http.StatusServiceUnavailable {
		t.Fatalf("second status = %d, want 503 while the single slot is held", second.Code)
	}
	if retry := second.Header().Get("Retry-After"); retry == "" {
		t.Error("503 response is missing Retry-After")
	}

	close(extractor.release)
	<-done
	if first.Code != http.StatusOK {
		t.Fatalf("first status = %d, want 200", first.Code)
	}
}

func TestCropPlatesReusesSlotsAfterCompletion(t *testing.T) {
	handler := NewHandler(&fakeExtractor{result: &plate.Result{Text: "AB12CDE"}}, discardLogger(), Limits{
		MaxRequestBytes: 1 << 20, MaxImageBytes: 1 << 19, MaxImages: 10, MaxConcurrent: 1,
	})

	// Sequential requests must all succeed: a released slot has to be reusable.
	for i := range 5 {
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, multipartRequest(t, "images", [][]byte{[]byte("image")}))
		if response.Code != http.StatusOK {
			t.Fatalf("request %d status = %d, want 200 (slot not released?)", i, response.Code)
		}
	}
}

func TestCropPlatesUnlimitedWhenMaxConcurrentUnset(t *testing.T) {
	extractor := &blockingExtractor{entered: make(chan struct{}, 4), release: make(chan struct{})}
	handler := NewHandler(extractor, discardLogger(), Limits{
		MaxRequestBytes: 1 << 20, MaxImageBytes: 1 << 19, MaxImages: 10,
	})

	for range 4 {
		go handler.ServeHTTP(httptest.NewRecorder(), multipartRequest(t, "images", [][]byte{[]byte("x")}))
	}
	for range 4 {
		<-extractor.entered // all four run concurrently; none is shed
	}
	close(extractor.release)
}

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func testHandler(extractor Extractor) http.Handler {
	return NewHandler(extractor, discardLogger(), Limits{
		MaxRequestBytes: 1 << 20,
		MaxImageBytes:   1 << 19,
		MaxImages:       10,
	})
}

func multipartRequest(t *testing.T, field string, images [][]byte) *http.Request {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for index, data := range images {
		part, err := writer.CreateFormFile(field, fmt.Sprintf("image-%d.jpg", index))
		if err != nil {
			t.Fatalf("CreateFormFile() error = %v", err)
		}
		if _, err := part.Write(data); err != nil {
			t.Fatalf("part.Write() error = %v", err)
		}
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("multipart.Close() error = %v", err)
	}
	request := httptest.NewRequest(http.MethodPost, "/crop-plates", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	return request
}
