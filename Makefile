.ONESHELL:
.EXPORT_ALL_VARIABLES:
SHELL := /bin/bash

DIR:=$(strip $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))

.DEFAULT_GOAL := help


.PHONY: help
help: ## provides cli help for this makefile (default) 📖
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Poetry and project setup targets ⚙️
.PHONY: install-project
install-project: ## Install the project core dependencies
	poetry install

.PHONY: install-project-dev
install-project-dev: ## Install the project dev dependencies
	poetry install --with dev

.PHONY: install-project-exploration
install-project-exploration: ## Install the project exploration dependencies
	poetry install --with exploration

##@ Development targets ✨
.PHONY: run-lint
run-lint: ## Run the linting
	poetry run ruff check .

.PHONY: run-lint-fix
run-lint-fix: ## Run the linting and fix the issues
	poetry run ruff check . --fix

.PHONY: run-tests
run-tests: ## Run the tests
	poetry run pytest

.PHONY: run-tests-coverage
run-tests-coverage: ## Run the tests and generate the coverage report
	poetry run pytest --cov=recommendations

##@ Recommendations cli targets ✨🧠
.PHONY: run-train-classified-embedding-model-cli 
run-train-classified-embedding-model-cli: ## Run the train classified embedding model cli 🧠
	train_recommendation_model --online_date 2025-06-24 --bottle_neck_size 64 --hidden_layer_sizes [512,256,128] --epochs 3 --batch_size 64

.PHONY: run-predict-model-cli
run-predict-model-cli: ## Run the predict model cli
	predict_recommendation_model

.PHONY: run-what-is-my-classified-recommendation-cli
run-what-is-my-classified-recommendation-cli: ## Run the what is my classified recommendation cli params: classified_ref
	what_is_my_recommendation --classified_ref $(classified_ref)

##@ Recommendations api targets ✨🧠
.PHONY: run-classified-recommendations-api
run-classified-recommendations-api: ## Run the classified recommendations api
	uvicorn recommendations.src.api.api:app --reload