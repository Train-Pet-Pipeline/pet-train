-include ../pet-infra/shared/Makefile.include

.PHONY: setup test lint clean

setup:
	python -m pip install -e ".[dev]"
	cd vendor/LLaMA-Factory && python -m pip install -e ".[metrics]"

test:
	pytest tests/ -v
