import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import types

import pytest

try:
    import polars  # noqa: F401 - we only check availability for optional dependency
except ModuleNotFoundError:  # pragma: no cover - optional dependency during testing
    POLARS_AVAILABLE = False
    sys.modules.setdefault("polars", types.SimpleNamespace())
else:
    POLARS_AVAILABLE = True

try:
    import ccxt  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency during testing
    sys.modules.setdefault("ccxt", types.SimpleNamespace(exchanges=[]))

# Ensure the backtrader package is importable when running tests from the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEPENDENCIES_PATH = PROJECT_ROOT / "dependencies"
if str(DEPENDENCIES_PATH) not in sys.path:
    sys.path.insert(0, str(DEPENDENCIES_PATH))

MODULE_PATH = DEPENDENCIES_PATH / "backtrader" / "utils" / "ccxt_data.py"
spec = importlib.util.spec_from_file_location("ccxt_data", MODULE_PATH)
ccxt_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(ccxt_data)

get_crypto_data = ccxt_data.get_crypto_data
unix_time_millis = ccxt_data.unix_time_millis


@pytest.mark.parametrize(
    "date_str,time_resolution,expected",
    [
        ("2024-01-01", "1d", int(datetime(2024, 1, 1).timestamp() * 1000)),
        ("2024-01-01 15:30:00", "1h", int(datetime(2024, 1, 1, 15, 30).timestamp() * 1000)),
    ],
)
def test_unix_time_millis_converts_dates(date_str, time_resolution, expected):
    assert unix_time_millis(date_str, time_resolution) == expected


def test_unix_time_millis_adds_midnight_for_intraday():
    # Intraday resolutions expect a timestamp with time information. When only a date is provided
    # the helper should assume midnight to keep the behaviour deterministic.
    expected_midnight = int(datetime(2024, 1, 1).timestamp() * 1000)
    assert unix_time_millis("2024-01-01", "1h") == expected_midnight


class MockExchange:
    last_instance = None

    def __init__(self, *_args, **_kwargs):
        self.calls = 0
        MockExchange.last_instance = self

    def fetch_ohlcv(self, asset, timeframe, since=None):
        self.calls += 1

        if self.calls == 1:
            return []

        if self.calls == 2:
            return [
                [int(datetime(2023, 1, 2).timestamp() * 1000), 1, 2, 3, 4, 5],
                [int(datetime(2023, 1, 3).timestamp() * 1000), 2, 3, 4, 5, 6],
                [int(datetime(2023, 1, 4).timestamp() * 1000), 3, 4, 5, 6, 7],
            ]

        return []


@pytest.fixture
def mock_ccxt(monkeypatch):
    import ccxt

    monkeypatch.setattr(ccxt, "exchanges", ["mockexchange"], raising=False)
    monkeypatch.setattr(ccxt, "mockexchange", MockExchange, raising=False)

    return ccxt


@pytest.mark.skipif(not POLARS_AVAILABLE, reason="polars is required for DataFrame processing")
def test_get_crypto_data_skips_empty_batches_and_trims_end_date(mock_ccxt):
    import polars as pl_local

    start_date = "2023-01-01"
    end_date = "2023-01-03"

    df = get_crypto_data("BTC/USDT", start_date, end_date, "1d", "mockexchange")

    assert isinstance(df, pl_local.DataFrame)
    assert df.height == 2

    expected_dates = [
        datetime(2023, 1, 2),
        datetime(2023, 1, 3),
    ]
    assert df.select(pl_local.col("dt")).to_series().to_list() == expected_dates

    assert df["start_date"].to_list() == [start_date, start_date]
    assert df["end_date"].to_list() == [end_date, end_date]
    assert df["symbol"].to_list() == ["BTC/USDT", "BTC/USDT"]

    # Ensure the mocked exchange was called until data was available and then stopped
    assert MockExchange.last_instance is not None
    assert MockExchange.last_instance.calls == 2

