# shellcheck shell=bash disable=SC1091  # shellcheck hint; make ignores this line
SHELL := /bin/bash

.PHONY: run setup test lint

# Serve the dashboard (synthetic frame source when no RealSense is attached).
run:
	@source .venv/bin/activate && python -m posture_guard.dashboard.serve

setup:
	./setup.sh

test:
	@source .venv/bin/activate && pytest

lint:
	@source .venv/bin/activate && ruff check .
