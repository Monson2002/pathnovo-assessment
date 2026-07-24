.PHONY: help install run chat eval test lint clean

PYTHON ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest

PID_A ?= data/samples/pair_01/Export Gas Compressor-P&ID (1).pdf
PID_B ?= data/samples/pair_01/Lift Gas compressor-P&ID.pdf

help:
	@echo "Available commands:"
	@echo "  make install  Install dependencies using uv"
	@echo "  make run      Run delta pipeline on default sample pair"
	@echo "  make chat     Run delta pipeline + interactive grounded chat"
	@echo "  make eval     Run evaluation harness and print scorecard"
	@echo "  make test     Run unit test suite"
	@echo "  make lint     Run pre-commit / linter"
	@echo "  make clean    Remove generated traces, output, and vector DB cache"

install:
	uv sync

run:
	$(PYTHON) -m src.main --pid-a "$(PID_A)" --pid-b "$(PID_B)"

chat:
	$(PYTHON) -m src.main --pid-a "$(PID_A)" --pid-b "$(PID_B)" --chat

eval:
	$(PYTHON) -m eval.run_eval

test:
	$(PYTEST) -s tests/

lint:
	uv run pre-commit run --all-files

clean:
	rm -rf output/ traces/ .chroma/ .pytest_cache/ __pycache__ src/**/__pycache__ tests/__pycache__ eval/__pycache__
