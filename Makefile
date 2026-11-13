.PHONY: run setup test lint

# Serve the dashboard (synthetic frame source when no RealSense is attached).
run:
	@. .venv/bin/activate && python -m posture_guard.dashboard.serve

setup:
	./setup.sh

test:
	@. .venv/bin/activate && pytest

lint:
	@. .venv/bin/activate && ruff check .
