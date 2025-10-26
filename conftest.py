import importlib.util
import pytest


RICH_AVAILABLE = importlib.util.find_spec("rich") is not None


def pytest_collection_modifyitems(config, items):
    if RICH_AVAILABLE:
        return

    skip_rich = pytest.mark.skip(reason="rich dependency not available in test environment")
    for item in items:
        if "dependencies/backtrader" in str(item.fspath):
            item.add_marker(skip_rich)
