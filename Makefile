.ONESHELL:
.EXPORT_ALL_VARIABLES:
SHELL := /bin/bash

DIR:=$(strip $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))

.DEFAULT_GOAL := help
PYENV_VERSION = 3.12.0

.PHONY: help
help: ## provides cli help for this makefile (default) 📖
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: install-project-dev
install-project-dev: ## Install the project dev dependencies
	poetry install --with dev

##@ Development targets ✨
.PHONY: run-lint
run-lint: ## Run the linting
	poetry run ruff check autoembed

.PHONY: run-lint-fix
run-lint-fix: ## Run the linting and fix the issues
	poetry run ruff check autoembed --fix

.PHONY: run-tests
run-tests: ## Run the tests
	poetry run pytest

.PHONY: run-tests-coverage
run-tests-coverage: ## Run the tests and generate the coverage report
	poetry run pytest --cov=autoembed
