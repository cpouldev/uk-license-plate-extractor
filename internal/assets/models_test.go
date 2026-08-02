package assets

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDownloaderDownloadsAndReusesVerifiedModel(t *testing.T) {
	content := []byte("model-content")
	digest := sha256.Sum256(content)
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests++
		_, _ = w.Write(content)
	}))
	defer server.Close()

	path := filepath.Join(t.TempDir(), "nested", "model.onnx")
	model := Model{Name: "test", Path: path, URL: server.URL, SHA256: hex.EncodeToString(digest[:])}
	downloader := Downloader{Client: server.Client()}

	if err := downloader.Ensure(context.Background(), model); err != nil {
		t.Fatalf("first Ensure() error = %v", err)
	}
	if err := downloader.Ensure(context.Background(), model); err != nil {
		t.Fatalf("second Ensure() error = %v", err)
	}
	if requests != 1 {
		t.Fatalf("requests = %d, want 1", requests)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("downloaded content = %q, want %q", got, content)
	}
}

func TestDownloaderRejectsChecksumMismatch(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("wrong"))
	}))
	defer server.Close()

	path := filepath.Join(t.TempDir(), "model.onnx")
	model := Model{Name: "test", Path: path, URL: server.URL, SHA256: strings.Repeat("0", 64)}
	if err := (Downloader{Client: server.Client()}).Ensure(context.Background(), model); err == nil {
		t.Fatal("Ensure() error = nil, want checksum error")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("model file should not be installed, Stat() error = %v", err)
	}
}

func TestDownloaderRejectsCorruptExistingModel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.onnx")
	if err := os.WriteFile(path, []byte("corrupt"), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	digest := sha256.Sum256([]byte("expected"))
	model := Model{Name: "test", Path: path, URL: "https://unused.invalid", SHA256: hex.EncodeToString(digest[:])}

	if err := (Downloader{}).Ensure(context.Background(), model); err == nil {
		t.Fatal("Ensure() error = nil, want checksum error")
	}
}
