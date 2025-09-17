from pathlib import Path

import pandas as pd
import pytest

from coin_trader_bybit.app import load_ohlcv


def test_load_ohlcv_parses_timestamp(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z"],
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100.5, 101.5],
        }
    )
    df.to_csv(csv_path, index=False)

    result = load_ohlcv(csv_path)

    assert list(result.columns) == ["open", "high", "low", "close"]
    assert result.index.name == "timestamp"
    assert result.shape == (2, 4)


def test_load_ohlcv_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({"timestamp": ["2024-01-01T00:00:00Z"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        load_ohlcv(csv_path)
