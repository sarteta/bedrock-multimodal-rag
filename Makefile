.PHONY: install test lint clean

install:
	pip install -e .[dev]

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache -o -name "*.egg-info" \) -exec rm -rf {} + 2>/dev/null || true
