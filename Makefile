# Makefile for PaConvert
#
# 	GitHb: https://github.com/PaddlePaddle/PaConvert
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test
check_dirs := paconvert tests scripts
# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	pre-commit run black

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Install Block # # # # # # # # # # # # # # # 

.PHONY: install
install:
	pip install -r requirements.txt
	pip install -r tests/requirements.txt
	pre-commit install

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

.PHONY: lint
lint:
	bash scripts/code_style_check.sh

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	PYTHONPATH=. pytest tests

# # # # # # # # # # # # # # # Coverage Block # # # # # # # # # # # # # # # 

.PHONY: coverage
coverage:
	bash scripts/coverage_check.sh

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
