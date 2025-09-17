# Docker 사용 가이드

## 이미지 빌드
```bash
docker build -t coin-trader:latest .
```

## .env 전달
실거래 혹은 테스트넷 키를 사용할 때는 별도의 `.env` 파일을 준비한 뒤 컨테이너에 마운트하거나 `--env-file` 옵션을 사용하세요.

```bash
docker run --rm \
  --env-file ./.env \
  coin-trader:latest --mode=paper --symbol=BTCUSDT --config=configs/params.yaml
```

## 백테스트 실행
CSV 파일(열: `timestamp,open,high,low,close`)을 `/data`로 마운트하고 `--mode=backtest`를 지정하면 됩니다.

```bash
docker run --rm \
  -v $(pwd)/data:/data \
  coin-trader:latest \
  --mode=backtest \
  --config=configs/params.yaml \
  --data=/data/btcusdt_1m.csv
```

## 동작 확인 체크리스트
- **로그 출력**: 컨테이너 로그에서 모드/심볼/testnet 여부가 올바르게 출력되는지 확인합니다.
- **메트릭 노출**: 컨테이너 실행 후 `http://localhost:9000/`에서 Prometheus 포맷 메트릭이 반환되는지 확인합니다.
- **백테스트 결과**: `Backtest complete` 문구와 함께 PnL, Profit Factor, Max DD 등이 표시되는지 확인합니다.
- **실행 종료 코드**: 명령이 성공적으로 끝났다면 종료 코드 `0`이어야 합니다. (`docker run --rm`는 정상 종료 시 자동으로 컨테이너를 제거합니다.)
- **paper/live 모드**: `--mode=paper`로 실행하면 “scaffold run” 로그만 출력되고 주문은 전송되지 않으며, `--mode=live`의 경우 Bybit API 키를 찾았는지 여부를 먼저 확인하세요.

## 추가 팁
- 이미지 업데이트 후에는 `docker image prune`으로 이전 빌드를 정리하세요.
- 실거래 전에는 항상 최신 CSV로 `--mode=backtest`를 돌려 MDD/Win rate 등을 재확인하는 것을 권장합니다.

## 모니터링 스택 (Prometheus + Grafana)
`docker-compose.monitoring.yml`을 사용하면 트레이더, Prometheus, Grafana를 한 번에 띄울 수 있습니다.

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

- `http://localhost:9090` → Prometheus UI에서 타깃 상태 확인
- `http://localhost:3000` → Grafana (기본 계정 `admin`/`admin`)
  - Grafana에서 Prometheus 데이터 소스를 추가하고 `coin_trader_*` 메트릭으로 대시보드를 구성하세요.

정지 시에는 `docker compose -f docker-compose.monitoring.yml down`을 실행하면 됩니다.
