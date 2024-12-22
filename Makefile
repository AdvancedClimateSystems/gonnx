.PHONY: build test

VERSION=$(shell git describe --always --tags --dirty)
LDFLAGS=-ldflags "-s -w -X main.Version=${VERSION}"
TEST=$(shell go list ./... | grep -v /onnx/)

BUILD_PARAMS=CGO_ENABLED=0 


define echotask
	@tput setaf 6
	@echo -n "  $1"
	@tput setaf 3
	@echo -n " - "
	@tput sgr0
	@echo $2
endef

help:
	$(call echotask,"help","Shows this page.")
	$(call echotask,"lint","Runs the GOLANGCI linter.")
	$(call echotask,"test","Runs the Go tests.")
	$(call echotask,"test_data","Downloads data for the ONNX test suite.")
	$(call echotask,"install","Install project dependencies.")
	$(call echotask,"install_lint","Install the Go linter.")
	$(call echotask,"install_gotestsum","Install the Go test runner.")
	$(call echotask,"build_all","Builds the project for both amd64 and arm64")
	$(call echotask,"build_amd64","Go amd64 build of the project.")
	$(call echotask,"build_arm64","Go arm64 build of the project.")

lint: ## Run various linters.
	@golangci-lint run --timeout=1m --config .golangci.yml

test: ## Run tests using gotestsum.
	@ ${BUILD_PARAMS} gotestsum \
	    --format=dots-v2 -- \
	    -timeout=30000ms \
	    -covermode=set \
	    -coverprofile=.coverage.out ${TEST}

test_ci: ## Run tests using normal test runner for ci output.
	@ ${BUILD_PARAMS} go test  \
	    -coverprofile .coverage.out ${TEST} && go tool cover -func=.coverage.out

test_data:  ## Creates test data from the ONNX test module.
	rm -R ./test_data; mkdir ./test_data; touch ./test_data/
	git clone --depth 1 --branch v1.17.0 https://github.com/onnx/onnx.git temp_onnx
	cp -r temp_onnx/onnx/backend/test/data/node/* ./test_data
	rm -Rf temp_onnx

test_html: ## Run tests showing coverage in the browser.
	@$(MAKE) test
	@ go tool cover -html=.coverage.out

install: ## Install project with its depedencies.
	go get ./...
	@go mod download

install_lint: ## Install the linter.
	curl -sfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh \
		| sh -s -- -b $(shell go env GOPATH)/bin v1.61.0

install_gotestsum: ## Install a tool for prettier test output.
	curl -sfL https://github.com/gotestyourself/gotestsum/releases/download/v1.9.0/gotestsum_1.9.0_linux_amd64.tar.gz \
	    | tar -C $(shell go env GOPATH)/bin -zxf - gotestsum

build_all: build_amd64 build_arm64

build_amd64:
	@GOARCH=amd64 GOOS=linux ${BUILD_PARAMS} go build ${LDFLAGs} ./...

build_arm64:
	@GOARCH=arm64 GOOS=linux ${BUILD_PARAMS} go build ${LDFLAGS} ./...
