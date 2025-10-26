import importlib.util
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class _AutoMockModule(ModuleType):
    """Module stub that lazily returns :class:`MagicMock` attributes."""

    def __getattr__(self, item):  # pragma: no cover - simple passthrough
        mock = MagicMock(name=f"{self.__name__}.{item}")
        setattr(self, item, mock)
        return mock


def _load_base_module():
    """Load the BaseStrategy module with a lightweight ``backtrader`` stub."""

    if "backtrader" not in sys.modules:
        bt_stub = ModuleType("backtrader")
        bt_stub.Strategy = object
        bt_stub.Analyzer = object
        bt_stub.observers = SimpleNamespace(BuySell=type("BuySell", (object,), {}))
        bt_stub.analyzers = SimpleNamespace(TradeAnalyzer=object)
        bt_stub.indicators = MagicMock()
        bt_stub.feeds = MagicMock()
        bt_stub.sizers = MagicMock()
        bt_stub.brokers = MagicMock()
        bt_stub.stores = MagicMock()
        bt_stub.WriterBase = object
        sys.modules["backtrader"] = bt_stub

    if "numpy" not in sys.modules:
        numpy_stub = _AutoMockModule("numpy")
        numpy_stub.bool_ = bool
        sys.modules["numpy"] = numpy_stub

    module_name = "btquant_base_strategy_for_tests"
    if module_name in sys.modules:
        return sys.modules[module_name]

    module_path = Path(__file__).resolve().parents[1] / "dependencies/backtrader/strategies/base.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


OrderTracker = _load_base_module().OrderTracker


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path, monkeypatch):
    original_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def test_close_order_updates_state_for_buy():
    tracker = OrderTracker(entry_price=100.0, size=2.0, take_profit_pct=5, symbol="BTCUSDT", backtest=True)

    tracker.close_order(exit_price=110.0)

    assert tracker.closed is True
    assert tracker.exit_price == pytest.approx(110.0)
    assert tracker.profit_pct == pytest.approx(10.0)
    assert tracker.exit_timestamp is not None


def test_close_order_updates_state_for_sell():
    tracker = OrderTracker(entry_price=100.0, size=1.0, take_profit_pct=2, symbol="BTCUSDT", order_type="SELL", backtest=True)

    tracker.close_order(exit_price=90.0)

    assert tracker.closed is True
    # For SELL orders profits are calculated inversely
    assert tracker.profit_pct == pytest.approx(((100.0 / 90.0) - 1) * 100)


def test_load_active_orders_from_csv_filters_and_parses(tmp_path):
    csv_path = tmp_path / "order_tracker.csv"
    rows = [
        {
            "order_id": "1",
            "symbol": "BTCUSDT",
            "order_type": "BUY",
            "entry_price": "30000",
            "size": "0.01",
            "take_profit_price": "31500",
            "timestamp": datetime.now().isoformat(),
            "closed": "False",
            "exit_price": "",
            "exit_timestamp": "",
            "profit_pct": ""
        },
        {
            "order_id": "2",
            "symbol": "ETHUSDT",
            "order_type": "BUY",
            "entry_price": "2000",
            "size": "0.5",
            "take_profit_price": "2100",
            "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
            "closed": "True",
            "exit_price": "2100",
            "exit_timestamp": datetime.now().isoformat(),
            "profit_pct": "5.0"
        },
    ]

    header = [
        "order_id",
        "symbol",
        "order_type",
        "entry_price",
        "size",
        "take_profit_price",
        "timestamp",
        "closed",
        "exit_price",
        "exit_timestamp",
        "profit_pct",
    ]

    with csv_path.open("w", encoding="utf-8") as csv_file:
        csv_file.write(",".join(header) + "\n")
        for row in rows:
            csv_file.write(",".join(str(row[col]) for col in header) + "\n")

    loaded_orders = OrderTracker.load_active_orders_from_csv(symbol="BTCUSDT")

    assert len(loaded_orders) == 1
    order = loaded_orders[0]
    assert order.symbol == "BTCUSDT"
    assert order.size == pytest.approx(0.01)
    assert order.entry_price == pytest.approx(30000.0)
    assert order.closed is False
    assert order.profit_pct is None


def test_load_active_orders_from_csv_without_file():
    assert OrderTracker.load_active_orders_from_csv(symbol="BTCUSDT") == []
