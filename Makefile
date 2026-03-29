PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: venv install serve compare wheel clean

venv:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: venv
	$(BIN)/pip install -e .

serve:
	$(BIN)/python -m coach.cli serve --host 0.0.0.0 --port $${PORT:-8080}

compare:
	$(BIN)/python -m coach.cli compare --target good_lap --reference fast_laps

wheel:
	$(BIN)/python -m coach.cli wheel-to-wheel --reference fast_laps

clean:
	rm -rf .pytest_cache coach/**/__pycache__
