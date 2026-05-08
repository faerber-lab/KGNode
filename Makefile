.PHONY: tests lint format doctest integration_tests_fast evals

lint:
	uv run ruff check .
	uv run mypy .

format:
	ruff check --select I --fix
	uv run ruff format .
	uv run ruff check . --fix

build:
	uv build

publish:
	uv publish --dry-run
