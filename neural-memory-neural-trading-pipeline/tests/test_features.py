import numpy as np

from neural_trading_system.data.feature_selector import FeatureSelector


def test_feature_selector_picks_signal_features():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 20))
    y = 2.0 * x[:, 1] - 1.0 * x[:, 5] + rng.normal(scale=0.1, size=1000)
    s = FeatureSelector(importance_threshold=0.0, target_features=5)
    res = s.select(x[:700], y[:700], x[700:], y[700:], [f"f{i}" for i in range(20)])
    assert "f1" in res.selected_features
