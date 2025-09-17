# 모니터링 & 로그 수집 가이드

## 1. Prometheus 메트릭 개요
컨테이너가 실행되면 `cfg.monitoring` 설정에 따라 기본적으로 `0.0.0.0:9000`에서 Prometheus 포맷 메트릭이 노출됩니다. 주요 지표는 다음과 같습니다.

| Metric | 설명 |
| --- | --- |
| `coin_trader_account_equity` | 최신 계정 에쿼티 (USDT) |
| `coin_trader_total_return_pct` | 시작 자본 대비 누적 수익률(%) |
| `coin_trader_current_return_pct` | 미실현 손익을 포함한 현재 수익률(%) |
| `coin_trader_realized_pnl_total` | 누적 실현 PnL |
| `coin_trader_open_pnl` | 현재 포지션 미실현 PnL |
| `coin_trader_position_qty` | 현재 보유 수량 (양수=롱, 음수=숏) |
| `coin_trader_position_entry_price`, `coin_trader_position_mark_price`, `coin_trader_position_notional` | 포지션 상세 |
| `coin_trader_trades_total`, `coin_trader_wins_total`, `coin_trader_losses_total` | 거래/승/패 카운트 |
| `coin_trader_win_rate` | 승률 (0~1) |
| `coin_trader_recent_trade_*` | 최근 거래 5건의 PnL, 수량, R배수, 방향(1=Buy/-1=Sell), 타임스탬프 |
| `coin_trader_last_loop_ts` | 가장 최근 전략 루프 실행 시각 (unix) |
| `coin_trader_last_candle_ts` | 최근 처리한 캔들 시각 (unix) |
| `coin_trader_last_signal_ts`, `coin_trader_last_signal_side` | 최근 신호 시각과 방향(1=Buy/-1=Sell) |
| `coin_trader_errors_total` | 루프 내 recoverable 에러 누적 횟수 |
| `coin_trader_trade_pnl_bucket`, `coin_trader_trade_r_multiple_bucket` | PnL/R 분포 히스토그램 |

Grafana에서 위 메트릭을 조합해 현재 포지션, 총/현재 수익률, 승률, 최근 거래 탭을 구성할 수 있습니다.

## 2. Docker Compose로 Prometheus & Grafana 실행
`docker-compose.monitoring.yml`에는 트레이더, Prometheus, Grafana가 정의돼 있습니다.

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (`admin` / `admin`)
- 트레이더 메트릭: `http://localhost:9000`

Grafana에서 **Data Sources → Prometheus**를 추가하고 URL을 `http://prometheus:9090`으로 설정하면 즉시 대시보드를 만들 수 있습니다. 예시 패널:
- *현재 포지션*: `coin_trader_position_qty`
- *계정 잔고*: `coin_trader_account_equity`
- *총 수익률*: `coin_trader_total_return_pct`
- *현재 수익률*: `coin_trader_current_return_pct`
- *승률*: `coin_trader_win_rate`
- *최근 5건 거래 테이블*: `coin_trader_recent_trade_pnl{slot=~"[0-4]"}` 등을 사용해 테이블 변환

종료 시에는 `docker compose -f docker-compose.monitoring.yml down`을 실행하세요.

## 3. 로그 시각화 (Grafana Loki/PromeTail 권장)
애플리케이션 로그는 JSON 포맷으로 STDOUT에 기록됩니다. Grafana에서 로그 패널을 보려면 Loki + Promtail 조합을 추가하는 것이 가장 간단합니다.

1. `docker-compose.monitoring.yml`에 다음 서비스를 추가하세요.
   ```yaml
   loki:
     image: grafana/loki:3.0.0
     ports:
       - "3100:3100"
     command: ["-config.file=/etc/loki/local-config.yaml"]

   promtail:
     image: grafana/promtail:3.0.0
     volumes:
       - /var/run/docker.sock:/var/run/docker.sock
     command: ["-config.file=/etc/promtail/promtail-docker-config.yaml"]
   ```
   (공식 loki/promtail 예제 config를 사용하거나 별도 작성)
2. Grafana에서 **Data Sources → Loki**를 추가하고 `http://loki:3100`을 입력합니다.
3. Explore 탭에서 `{container="coin-trader"}`와 같이 쿼리하면 JSON 로그를 바로 확인할 수 있습니다.

Promtail은 Docker 소켓을 읽어 컨테이너 STDOUT을 자동 수집합니다. JSON 필드는 Grafana에서 파싱 가능하며, `message`, `level`, `logger` 등을 기준으로 패널을 만들 수 있습니다.

## 4. 샘플 대시보드 불러오기
- Grafana → Dashboards → Import → `Upload JSON file`
- `configs/monitoring/grafana-dashboard.json`을 선택합니다. 이 파일에는 다음 패널이 포함되어 있습니다.
  - 총/현재 수익률과 승률, 현재 포지션 수량을 표시하는 Stat 패널
  - 계정 에쿼티 및 누적 실현 손익 시계열
  - 최근 5건 거래를 보여주는 테이블 (PnL, 방향, 수량, R배수, 타임스탬프)
- 데이터 소스 매핑 요청이 나오면 Prometheus 데이터 소스를 지정하세요 (`PROM_DS`).

## 5. 운영 시 권장 워크플로
1. 거래 루프에서 주문 체결/포지션 변경 시 `MetricsCollector`의 `update_position`, `set_equity`, `record_trade`를 호출합니다.
2. 실거래 계정 잔고를 주기적으로 Bybit API에서 조회해 `metrics.set_equity(balance)`로 갱신하면 누적/현재 수익률이 자동으로 계산됩니다.
3. 승률·PnL 대시보드를 Grafana에서 모니터링하고, 최근 5건 거래 패널을 통해 이상 체결 여부를 확인합니다.
4. 로그 탐색은 Grafana Loki Explore를 사용하거나, 필요에 따라 Slack/Telegram 알림과 연동하세요.

이 구성을 통해 “현재 포지션, 잔고, 총/현재 수익률, 승률, 최근 거래 히스토리, 로그”를 하나의 Grafana 대시보드에서 모니터링할 수 있습니다.
## 5. 대시보드 패널 설명
- **총 수익률 (%)**: `coin_trader_total_return_pct`
- **에쿼티 & 누적 실현 손익**: 계정 에쿼티(`coin_trader_account_equity`)와 누적 손익(`coin_trader_realized_pnl_total`) 추세
- **승률**: `coin_trader_win_rate`
- **현재 포지션 수량**: `coin_trader_position_qty`
- **현재 수익률 (%)**: `coin_trader_current_return_pct`
- **최근 거래 5건**: `coin_trader_recent_trade_*` 지표에서 PnL/수량/R/방향을 표시하되, 체결이 없으면 공백
- **Loop Delay (s)**: `time() - coin_trader_last_loop_ts` 로 계산해 마지막 루프 이후 경과 시간
- **마지막 캔들 시각**: `coin_trader_last_candle_ts * 1000` (값이 없으면 "No candles yet" 표시)
- **마지막 신호 시각/방향**: `coin_trader_last_signal_ts * 1000`, `coin_trader_last_signal_side` (값이 없으면 "No signal yet")
- **최근 5분 에러 발생**: `increase(coin_trader_errors_total[5m])` 로 타입별 루프 오류 건수를 표시

