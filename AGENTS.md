# Repository Guidelines

## Project Structure & Module Organization
- Core Python package lives in `src/coin_trader_bybit/`; key modules include `exchange/bybit.py`, `strategy/scalper.py`, and `app.py` for CLI entry.
- Shared configs reside in `configs/params.yaml`; secrets belong in a local `.env` seeded from `.env.example`.
- Tests mirror the package under `tests/`, and transient OHLCV/cache files belong in `data/` (gitignored).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated Python 3.11 env.
- `pip install -r requirements.txt` installs runtime and dev dependencies.
- `python -m coin_trader_bybit.app --mode=paper --symbol=BTCUSDT --config=configs/params.yaml` boots the paper-trading CLI.
- `ruff check .`, `black .`, and `isort .` keep the code formatted and linted; run before committing.

## Coding Style & Naming Conventions
- Use 4-space indentation, ≤100-character lines, and type hints on public APIs.
- Follow `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, and keep modules import-safe.
- Prefer descriptive docstrings where logic is non-obvious; avoid side effects at import time.

## Testing Guidelines
- Write pytest tests in files named `tests/test_*.py`, mirroring module paths.
- Run `pytest -q` for fast feedback and `pytest --cov=src --cov-report=term-missing` to confirm ≥85% coverage on touched code.
- Include scenarios covering fees, slippage, and risk controls to align with the scalper mandate.

## Commit & Pull Request Guidelines
- Use Conventional Commits (e.g., `feat: add bar generator`, `fix: tighten risk limit`) with ≤72-char subjects.
- PRs should describe intent, trade-offs, linked issues, updated tests, and lint/type status; attach logs or screenshots when they clarify outcomes.

## Security & Configuration Tips
- Never commit API keys; populate `.env` locally and update `.env.example` placeholders when fields change.
- Default to Bybit testnet; require explicit opt-in for live trading with minimal size and `reduceOnly` exits.
- Log operational data without leaking secrets; rotate keys promptly if suspicious activity occurs.

## Agent-Specific Instructions
- Consult `ai_trading_plan.yaml` before major changes; adhere to risk (≤0.5% per trade, daily stop at -2R) and execution safety gates.
- Prioritize completing pending plan steps (data ingestion, walk-forward, risk controls, alerts) before expanding scope.
- Halt work if lint, type checks, tests, or testnet validations fail until resolved.
