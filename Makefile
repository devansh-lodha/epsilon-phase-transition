.PHONY: setup format lint check clean

setup:
	uv sync

format:
	uv run ruff check --fix .
	uv run ruff format .

lint:
	uv run ruff check .

check: format lint
	uv run ty check .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".ruff_cache" -exec rm -r {} +
