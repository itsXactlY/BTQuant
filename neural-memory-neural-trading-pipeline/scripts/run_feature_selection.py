from __future__ import annotations

import numpy as np

from neural_trading_system.data.feature_selector import FeatureSelector


def main() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(2000, 90))
    signal = 0.7 * x[:, 0] - 0.4 * x[:, 3] + 0.25 * x[:, 7]
    y = signal + rng.normal(scale=0.1, size=2000)

    split = 1500
    selector = FeatureSelector(target_features=15)
    result = selector.select(x[:split], y[:split], x[split:], y[split:], [f"f{i}" for i in range(90)])

    print("Selected features:", result.selected_features)
    print("Dropped correlated:", len(result.dropped_correlated))


if __name__ == "__main__":
    main()
