# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/coin_trader_bybit/`: `exchange/bybit.py` handles API adapters, `strategy/scalper.py` drives signals, and `app.py` exposes the CLI.
- Runtime parameters sit in `configs/params.yaml`; seed `.env` from `.env.example` and never commit live keys.
- Tests mirror the package under `tests/`, transient OHLCV exports belong in git-ignored `data/`, and docs stay in `docs/`.
- Read `ai_trading_plan.yaml` before adjusting automation, risk, or alerting flows.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` — create the Python 3.11 environment.
- `pip install -r requirements.txt` — install runtime and dev dependencies.
- `python -m coin_trader_bybit.app --mode=paper --symbol=BTCUSDT --config=configs/params.yaml` — run the paper-trading CLI.
- `ruff check .`, `black .`, `isort .` — lint, format, and sort imports before commits.
- `pytest --cov=src --cov-report=term-missing` — run the suite and confirm ≥85% coverage on touched modules.

## Coding Style & Naming Conventions
- Use 4-space indentation, keep lines ≤100 characters, and type-hint public functions.
- Stick to `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Keep docstrings concise and avoid side effects at import time.

## Testing Guidelines
- Place new tests in `tests/test_<module>.py` to mirror source layout (e.g., strategy updates → `tests/test_strategy/`).
- Exercise fee, slippage, and risk controls whenever trade logic changes; rerun paper trading after major edits.
- Investigate local test failures before pushing and only commit deterministic fixtures or data.

## Commit & Pull Request Guidelines
- Use Conventional Commit subjects ≤72 characters (e.g., `fix: tighten risk limits`) with bodies focused on rationale and impact.
- Reference linked issues, note trade-offs, and attach logs or screenshots for behavior changes.
- Ensure CI (lint, type checks, coverage) passes before requesting review and call out intentional skips.

## Security & Configuration Tips
- Default to Bybit testnet; require explicit opt-in for live trading and mark live orders `reduceOnly`.
- Rotate API keys on anomalies, redact secrets from logs/screenshots, and keep environment variables in `.env`.

## Agent-Specific Instructions
- Prioritize tasks on data ingestion, walk-forward validation, risk safeguards, and alerting before new features.
- Halt automation if daily loss hits -2R or any trade exceeds 0.5% risk, restoring controls before resuming.

## Current Findings & TODOs (2024-01–2025-09)
- Baseline backtests on BTCUSDT 1m show negative expectancy: PF ≈0.1–0.45, 평균 R < 0, -1R 손절 빈번. 필터를 완화하면 노이즈가 늘고, 강화하면 거래가 과소로 변함.
- 원인
  - 추세 감지 부정확: RSI/EMA/ATR만으로는 레짐 전환(랠리 종료·급락 직후)과 강제청산성 거래량을 거르지 못함.
  - 출구 구조 취약: 작은 이익(≤0.5R) vs. -1R 손실로 기대값이 음수. 브리크이븐 이동·트레일링이 일관되지 않음.
  - 백테스터에 일일 -2% 손실 제한 미반영(라이브 RiskManager에는 존재).

- 해결 과제(우선순위)
  1) 레짐 분류 고도화: 상위 타임프레임(5m/15m) EMA 기울기·ATR 밴딩·거래량 지속성으로 trend/range/fast-market을 게이트.  
  2) 진입 재설계: 추세 풀백은 EMA 재접근+RSI+구조 확인 필수, 레인지 플레이는 저변동 밴드에서만 허용, 저품질 돌파는 비활성.  
  3) 청산 재설계: 이유별 멀티 타깃(예: 20% at 0.8R, runner ≥1.5R), 구조·ATR 트레일, 브리크이븐은 ≥0.8R 이후.  
  4) 백테스트 리스크: 백테스터에 일일 -2% equity stop과 일일 진입 횟수 제한 반영.  
  5) 리포팅: 레짐·시간대별 성능, R 분포, 몬테카를로 등 자동 리포트 추가.

- 수용 기준(전체 기간 기준)
  - PF ≥ 1.2, 평균 R ≥ 0.2, 일일 최대 손실 ≤ 2%(시뮬레이션 반영), 일 평균 1–2 트레이드, MDD 합리적(<15%).

- 재현 방법
  - 데이터: `PYTHONPATH=src python3 scripts/download_bybit_ohlcv.py --symbol BTCUSDT --timeframe 1m --since 2024-01-01T00:00:00Z --until 2025-09-18T23:59:00Z --outfile data/btcusdt_1m_20240101_20250918.csv`
  - 백테스트: `PYTHONPATH=src python3 scripts/run_backtest.py --data=data/btcusdt_1m_20240101_20250918.csv`
