# 시스템 동작 개요

## 진입 경로
- CLI 엔트리포인트 `python -m coin_trader_bybit.app`는 `--mode=paper|live|backtest`, `--symbol`, `--config` 인자를 받아 `.yaml` 설정을 `AppConfig` 모델로 적재한다.
- `.env`와 CLI 인자 조합으로 테스트넷 여부를 결정하고, 로그 포맷(JsonFormatter), Prometheus 레지스트리를 초기화한다.
- `mode=backtest`일 경우 OHLCV CSV를 읽어 `Backtester`를 실행하고, 나머지 모드에서는 `Trader.run()` 루프를 구동한다.

## 실거래 루프(Trader)
- 트레이더는 Bybit REST 캔들 피드(`BybitDataFeed`)로부터 `execution.lookback_candles` 길이의 시계열을 가져온다.
- 전략(`Scalper`)은 1분 엔트리 캔들을 5분 앵커로 리샘플링해 EMA(50/200) 추세, ATR, 마이크로 고점, 거래량 이동평균을 산출한다.
- 최근 신호 이후 캔들 수, 리스크 쿨다운, 일일 손실 한도(`risk.daily_stop_r_multiple`)를 체크한 뒤 조건이 충족되면 시장가 주문을 발행한다. 주문 체결 시 `ActiveTrade` 상태가 생성되며 초기 손절 가격(ATR 기반), 리스크 단위(R), 부분 청산 여부가 저장된다.

## 포지션 관리
- 루프마다 계좌 에쿼티·포지션 스냅샷을 Bybit API로 동기화하고, Prometheus 게이지(포지션 수량, 마크 가격, 미실현 PnL 등)를 갱신한다.
- `ActiveTrade`가 존재하면 신규 진입 대신 출구 로직을 수행한다.
  - **손절**: 저가가 초기 손절 이하일 때 `close_position_market()`로 전체 수량을 reduce-only 시장가 청산한다.
  - **부분익절**: 고가가 1R 목표를 넘으면 절반 수량을 청산하고, 손절 가격을 엔트리 이하로 끌어올린 뒤 트레일링을 활성화한다.
  - **ATR 트레일**: 부분익절 이후 고가-ATR×멀티플 만큼 손절선을 따라올리고, 가격이 트레일을 이탈하면 잔여 수량을 청산한다.
  - **시간 종료**: 설정된 `strategy.time_stop_minutes` 이상 보유하면 시장가로 잔여 수량을 정리한다.
- 각 청산은 실현 PnL을 누적하여 R 배수로 환산하고, `RiskManager`가 일일 손실/이익 합계를 관리한다. 일일 한도를 초과하면 당일 추가 진입이 차단된다.

## 백테스트
- 동일한 전략 지표와 포지션 규칙을 `Backtester`가 재현한다. 부분청산·ATR 트레일·시간 종료·슬리피지·수수료를 반영하여 `BacktestReport`(거래 리스트, 에쿼티 커브, PnL 지표)를 반환한다.

## 위험 관리
- 포지션 사이징은 `RiskManager.position_size()`가 `(에쿼티 × 위험%) ÷ (손절 거리)` 공식을 적용한다.
- 실거래 시 노셔널 상한(`risk.max_live_order_notional_krw`)을 환산하여 주문 수량을 제한한다.
- 쿨다운 카운터로 연속 진입을 제한하며, 일일 실현 R 합계가 한도에 도달하면 자동으로 신규 진입을 차단한다.

## 모니터링
- `MetricsCollector`는 포지션, 에쿼티, 승률, 최근 거래(PnL, 방향, R 등) 정보를 Prometheus 메트릭으로 노출한다.
- 로그는 JSON 포맷으로 구조화되어 Loki/Grafana와 연동하기 쉽다. `docker-compose.monitoring.yml`로 트레이더·Prometheus·Grafana를 동시 구동할 수 있다.

## 테스트 흐름
- Pytest 스위트는 구성요소별 단위 테스트를 제공한다.
  - 거래 루프는 브레이크아웃 진입, 노셔널 상한, 일일 손실 차단, 손절 청산 시나리오를 검증한다.
  - 위험 관리 테스트는 포지션 사이즈 산출과 일일 손실 리셋을 확인한다.
  - Bybit 어댑터 테스트는 실거래 제한과 reduce-only 청산 래퍼 동작을 확인한다.
- `pytest -q` 또는 `pytest --cov=src --cov-report=term-missing`으로 실행한다.

## 실행 요약
1. 설정 적재 → 로그/모니터링 초기화.
2. 캔들 수집 → 전략 특성치 계산.
3. 리스크 게이트(쿨다운, 일일 R 등) 통과 시 시장가 진입.
4. `ActiveTrade` 상태 기반으로 손절·부분익절·트레일·시간 종료를 관리.
5. Prometheus 메트릭과 JSON 로그로 상태를 노출하고, 위험 지표 및 일일 한도를 실시간 반영.
