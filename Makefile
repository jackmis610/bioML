# bioML developer entry points. Every README number is reproducible from `make report`.
UV ?= uv run
DATA ?= data/synthetic/v1
MODEL ?= models/v2
REPORT ?= reports/v2
N_SUBJECTS ?= 100
SEED ?= 42
ROUNDS ?= 400

.PHONY: install data train report serve demo test lint typecheck check clean

install:
	uv sync --extra ml --extra serve --extra dev

data:
	$(UV) python -m bioml generate --n-subjects $(N_SUBJECTS) --seed $(SEED) --out $(DATA)

train:
	$(UV) python -m bioml train --data $(DATA) --k 5 --n-estimators $(ROUNDS) --save-model $(MODEL)

report:
	$(UV) python -m bioml evaluate --data $(DATA) --k 5 --seed $(SEED) --n-estimators $(ROUNDS) --out $(REPORT)

serve:
	$(UV) python -m bioml serve --model $(MODEL) --port 8000

demo:
	BIOML_MODEL_PATH=$(MODEL) BIOML_DATA_PATH=$(DATA) $(UV) streamlit run src/bioml/demo/app.py

test:
	$(UV) pytest -q

lint:
	$(UV) ruff check src tests
	$(UV) ruff format --check src tests

typecheck:
	$(UV) mypy src

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
