# Convenience targets. Windows users: run the commands directly, or use the
# bundled Git Bash (`make` ships with it on many setups).
PY ?= python
VENV ?= .venv
V ?= $(VENV)/Scripts/python.exe

.DEFAULT_GOAL := help
.PHONY: help venv data-v1 data-v2 train-baseline train-improved train test lint fmt docker serve clean

help:  ## show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

venv:  ## create the training virtualenv and install dev requirements
	$(PY) -m venv $(VENV)
	$(V) -m pip install --upgrade pip
	$(V) -m pip install -r requirements-dev.txt

data-v1:  ## build + promote dataset v1, then dvc add it
	$(V) ml/data_pipeline.py download-raw
	$(V) ml/data_pipeline.py build-v1
	$(V) ml/data_pipeline.py promote --version v1
	dvc add data/processed

data-v2:  ## build + promote dataset v2, then dvc add it
	$(V) ml/data_pipeline.py build-v2
	$(V) ml/data_pipeline.py promote --version v2
	dvc add data/processed

train-baseline:  ## experiment 1 (v1 data, linear probe)
	$(V) -m ml.train --config ml/configs/baseline.yaml

train-improved:  ## experiment 2 (v2 data, fine-tuned) + promote to artifacts/
	$(V) -m ml.train --config ml/configs/improved.yaml --promote

train: train-baseline train-improved  ## both experiments

test:  ## run the test suite
	$(V) -m pytest -q

lint:  ## black + ruff + flake8, exactly like CI
	$(V) -m black --check .
	$(V) -m ruff check .
	$(V) -m flake8 app ml tests

fmt:  ## autoformat
	$(V) -m black .
	$(V) -m ruff check --fix .

docker:  ## build the slim inference image
	docker build -t tb-cxr:local .

serve:  ## run the API locally with reload
	$(V) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

mlflow-ui:  ## open the experiment UI on :5000
	mlflow ui --backend-store-uri sqlite:///$(CURDIR)/mlflow.db --port 5000

clean:  ## remove build/test caches (never touches data/ or artifacts/)
	rm -rf .pytest_cache .ruff_cache runs/**/__pycache__ **/__pycache__ reports
