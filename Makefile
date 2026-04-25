-include ../pet-infra/shared/Makefile.include

.PHONY: setup test lint clean

setup:
	python -m pip install -e ".[dev]"
	# F010 fix: vendor/LLaMA-Factory is a git submodule (per .gitmodules), not a
	# vendored copy. Fresh clone without --recursive leaves it empty → install
	# fails. Init/update first.
	git submodule update --init --recursive vendor/LLaMA-Factory
	cd vendor/LLaMA-Factory && python -m pip install -e ".[metrics]"

test:
	pytest tests/ -v
