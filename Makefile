.DEFAULT_GOAL := help
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create venv + install dev deps (run this first)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip -q
	$(PIP) install -r requirements-dev.txt -q
	@echo "Done. Activate with: source $(VENV)/bin/activate"

optional: ## Install heavy/optional deps (real providers, vector DBs, frameworks)
	$(PIP) install -r requirements-optional.txt

test: ## Run the full offline suite (no API keys needed)
	$(PY) -m pytest -m "not live"

test-live: ## Run tests that hit real providers (costs money, needs .env)
	$(PY) -m pytest -m live

cov: ## Test with coverage report
	$(PY) -m pytest -m "not live" --cov --cov-report=term-missing

lint: ## Lint + format check
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

fix: ## Auto-fix lint + format
	$(VENV)/bin/ruff check --fix .
	$(VENV)/bin/ruff format .

types: ## Static type check
	$(VENV)/bin/mypy 03-llm-integration 04-rag 05-agents 06-production-service

check: lint types test ## Everything CI runs

serve: ## Run the production FastAPI service locally
	$(VENV)/bin/uvicorn app.main:app --reload --port 8000 --app-dir 06-production-service

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov

.PHONY: help setup optional test test-live cov lint fix types check serve clean
