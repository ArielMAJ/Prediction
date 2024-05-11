ifneq ("$(wildcard .env)","")
	include .env
	export
endif

.PHONY: install
install: ## Install Python requirements.
	python -m pip install --upgrade pip setuptools wheel poetry
	poetry add numpy
	poetry add pandas
	poetry add matplotlib
	pip install scikit-learn
	pip install tensorflow 
	poetry lock
	poetry install --no-root
	poetry run pre-commit install

.PHONY: run
run: ## Run the project.
	poetry run python .\main.py


	