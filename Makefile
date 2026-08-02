.PHONY: build check fmt race run test vet

build:
	go build ./...

fmt:
	gofmt -w $$(find cmd internal -name '*.go' -type f)

test:
	go test ./...

race:
	go test -race ./...

vet:
	go vet ./...

check: fmt test race vet build

run:
	go run ./cmd/server
