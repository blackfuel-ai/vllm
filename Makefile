# Blackfuel-additive Makefile for bf-vllm.
#
# Upstream vllm-project/vllm ships no Makefile at the repo root, so this
# file is purely BF-additive (ADR-0003) and never conflicts with sync.
# Extend with new targets as bf-vllm grows; keep them thin (delegate to
# real tools, no business logic inline).

# Python interpreter for the dev venv. Defaults to 3.14 (matches
# pyproject.toml's upper bound and the ai-platform CLAUDE.md "Python
# 3.14+ only" rule). Override per-invocation:
#   make install PYTHON=python3.13
PYTHON ?= python3.14
VENV   ?= .venv

# `python` resolved inside the venv, used as the build target so make
# treats the venv as a real file dependency (idempotent).
VENV_PY := $(VENV)/bin/python

.PHONY: help install lint clean

help: ## Show available targets.
	@awk 'BEGIN {FS = ":.*##"; printf "Available targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# Create the venv on demand. uv will auto-download $(PYTHON) if it's not
# already on the system (uv's managed-Python feature).
$(VENV_PY):
	uv venv --python $(PYTHON) $(VENV)

install: $(VENV_PY) ## First-clone bootstrap: create venv, install pre-commit, register git hooks.
	uv pip install --python $(VENV_PY) -r requirements/lint.txt
	$(VENV)/bin/pre-commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
	@echo ""
	@echo "✓ bf-vllm dev env ready."
	@echo "  Activate it for an interactive session:"
	@echo "      source $(VENV)/bin/activate"
	@echo "  Or stay outside the venv and prefix commands with: $(VENV)/bin/<cmd>"
	@echo "  Hooks: pre-commit + commit-msg + pre-push registered, all fire $(VENV)/bin/pre-commit."

lint: ## Run the full pre-commit suite locally, same as the bf-precommit CI workflow.
	$(VENV)/bin/pre-commit run --all-files --hook-stage manual

clean: ## Remove the dev venv.
	rm -rf $(VENV)
