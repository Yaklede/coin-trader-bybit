# Repository Guidelines

## Project Structure & Module Organization
- `src/coin_trader_bybit/` – core package (exchange, strategy, risk, exec, utils).
- `src/coin_trader_bybit/exchange/bybit.py` – Bybit REST/WebSocket adapter.
- `src/coin_trader_bybit/strategy/scalper.py` – BTC scalping logic.
- `src/coin_trader_bybit/app.py` – CLI entry (`--mode=paper|live`).
- `configs/params.yaml` – strategy + risk config; `.env.example` for secrets.
- `tests/` – unit/integration tests; `data/` (gitignored) for cached OHLCV.

## Build, Test, and Development Commands
- Setup: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Lint/Format: `ruff check .` (lint), `black .` (format), `isort .` (imports)
- Types: `mypy src`
- Tests: `pytest -q` and `pytest --cov=src --cov-report=term-missing`
- Run (paper): `python -m coin_trader_bybit.app --mode=paper --symbol=BTCUSDT --config=configs/params.yaml`

## Coding Style & Naming Conventions
- Python 3.11, 4‑space indent, 100‑char lines, type hints for public APIs.
- Names: `snake_case` (func/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (consts), modules lower‑case.
- Keep modules import‑safe (no side effects at import time).

## Testing Guidelines
- Framework: `pytest` with fixtures; network/e2e marked `@pytest.mark.e2e`.
- Layout: `tests/test_*.py`, mirror package paths (e.g., `tests/exchange/test_bybit.py`).
- Coverage: aim ≥ 85% on changed lines; include fee/slippage scenarios in strategy tests.

## Commit & Pull Request Guidelines
- Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `ci:`.
- Messages: subject ≤ 72 chars; body describes intent, trade‑offs; reference issues (`Closes #12`).
- PRs: description, linked issues, screenshots/logs, tests updated, lint+format clean, docs updated.
- Prefer small, focused PRs; avoid mixing refactors with behavior changes.

## Security & Configuration Tips
- Never commit secrets; use `.env` and keep `.env.example` updated (`BYBIT_API_KEY`, `BYBIT_API_SECRET`).
- Default to Bybit testnet; guard live mode behind explicit flag and minimal size.
- Store config in `configs/params.yaml`; log without leaking secrets.

## Agent‑Specific Instructions
- Review and follow `ai_trading_plan.yaml` step‑by‑step.
- Stop on safety gate failures (lint/types/tests/coverage or testnet checks).

