.PHONY: help install install-dev test test-cov lint format type-check clean docs build docker-build

help:
	@echo "EVOCOMP Development Commands"
	@echo ""
	@echo "install      Install core dependencies"
	@echo "install-dev  Install development dependencies"
	@echo "test         Run unit tests"
	@echo "test-cov     Run tests with coverage"
	@echo "lint         Run code linters"
	@echo "format       Format code with black and isort"
	@echo "type-check   Run type checking with mypy"
	@echo "clean        Clean build artifacts"
	@echo "docs         Build documentation"
	@echo "build        Build package"
	@echo "docker-build Build Docker image"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-cov:
	pytest tests/ --cov=evocomp --cov-report=html --cov-report=term

lint:
	flake8 src/evocomp tests/
	black --check src/evocomp tests/
	isort --check-only src/evocomp tests/

format:
	black src/evocomp tests/
	isort src/evocomp tests/

type-check:
	mypy src/evocomp

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

build:
	python -m build

docker-build:
	docker build -t evocomp:latest .
