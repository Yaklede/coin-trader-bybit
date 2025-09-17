FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY ai_trading_plan.yaml ./ai_trading_plan.yaml

EXPOSE 9000

ENTRYPOINT ["python", "-m", "coin_trader_bybit.app"]
CMD ["--mode=paper", "--symbol=BTCUSDT", "--config=configs/params.yaml"]
