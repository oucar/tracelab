.DEFAULT_GOAL := help

BACKEND      := backend
FRONTEND     := frontend
VENV         := $(BACKEND)/.venv
PY           := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
UVICORN      := $(VENV)/bin/uvicorn
RUFF         := $(VENV)/bin/ruff
PYTEST       := $(VENV)/bin/pytest

.PHONY: help install install-backend install-frontend dev dev-backend dev-frontend test lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: install-backend install-frontend ## Set up backend venv + frontend deps

install-backend: ## Create the backend virtualenv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e "$(BACKEND)[dev]"

install-frontend: ## Install frontend npm dependencies
	cd $(FRONTEND) && npm install

dev: ## Run backend (:8000) and frontend (:5173) together
	@trap 'kill 0' INT TERM; \
	$(MAKE) dev-backend & \
	$(MAKE) dev-frontend & \
	wait

dev-backend: ## Run the FastAPI backend with reload on :8000
	cd $(BACKEND) && .venv/bin/uvicorn app.main:app --reload --port 8000

dev-frontend: ## Run the Vite dev server on :5173
	cd $(FRONTEND) && npm run dev

test: ## Run backend unit tests (no API key needed)
	cd $(BACKEND) && .venv/bin/pytest

lint: ## Lint backend (ruff) and typecheck frontend (tsc)
	cd $(BACKEND) && .venv/bin/ruff check .
	cd $(FRONTEND) && npm run typecheck

clean: ## Remove the venv and frontend node_modules
	rm -rf $(VENV) $(FRONTEND)/node_modules
