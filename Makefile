# MCI World Model v4.6.0 — Developer Makefile
# ===========================================
PYTHON ?= python
PYTEST  = PYTHONPATH=src $(PYTHON) -m pytest
RUFF    = ruff

.PHONY: help install test lint format bench clean check all

help: ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install in dev mode with full extras
	pip install -e ".[full]"

install-min: ## Install minimal (no torch, no viz)
	pip install -e "."

test: ## Run all tests (excl slow, realtime)
	$(PYTEST) tests/ -q --tb=short --ignore=tests/test_realtime.py

test-all: ## Run all tests including slow
	$(PYTEST) tests/ -q --tb=short

test-fast: ## Run fast tests only (<5s each)
	$(PYTEST) tests/ -q --tb=line -m "not slow" --ignore=tests/test_realtime.py

bench: ## Run all benchmarks
	$(PYTEST) benchmarks/ -q --tb=short -x

bench-bnlearn: ## Run BNLearn structural discovery benchmark
	$(PYTEST) benchmarks/bnlearn/ -v --tb=short

bench-sota: ## Run SOTA comparison
	$(PYTEST) benchmarks/test_sota_comparison.py -v -s --tb=short

bench-perf: ## Run performance benchmark
	$(PYTEST) benchmarks/test_performance_bench.py -v --tb=short

bench-tuebingen: ## Run Tübingen direction benchmark
	$(PYTEST) benchmarks/real_world/tuebingen_pairs.py -v --tb=short

lint: ## Lint with ruff
	$(RUFF) check src/ tests/ benchmarks/

lint-fix: ## Auto-fix lint issues
	$(RUFF) check src/ tests/ benchmarks/ --fix

format: ## Format with ruff
	$(RUFF) format src/ tests/ benchmarks/

format-check: ## Check formatting without changing
	$(RUFF) format --check src/ tests/ benchmarks/

types: ## Type check with mypy
	mypy src/mci_world_model --strict --ignore-missing-imports

clean: ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name dist -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name build -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

check: lint test bench-bnlearn bench-perf ## Full CI check (lint + test + bench)

cov: ## Test with coverage
	$(PYTEST) tests/ -q --tb=line --cov=mci_world_model --cov-report=term \
		--ignore=tests/test_realtime.py

build: ## Build distribution
	$(PYTHON) -m build

pre-commit-install: ## Install pre-commit hooks
	pre-commit install
