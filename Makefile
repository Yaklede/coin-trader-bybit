.PHONY: setup lint format type test cov run

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

lint:
	ruff check .

format:
	black . && isort .

type:
	mypy src

test:
	pytest -q

cov:
	pytest --cov=src --cov-report=term-missing

run:
	python -m coin_trader_bybit.app --mode=paper --symbol=BTCUSDT --config=configs/params.yaml

