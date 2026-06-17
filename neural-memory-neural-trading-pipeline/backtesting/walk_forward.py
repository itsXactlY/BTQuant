from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.data_loader import WalkForwardConfig, create_walk_forward_splits, slice_split
from .metrics import compute_metrics
from .vectorized_engine import BacktestConfig, VectorizedBacktest


@dataclass
class WalkForwardResult:
    window: int
    split: dict[str, tuple[int, int]]
    metrics: dict[str, float]


class WalkForwardValidator:
    def __init__(self, split_cfg: WalkForwardConfig | None = None, bt_cfg: BacktestConfig | None = None):
        self.split_cfg = split_cfg or WalkForwardConfig()
        self.bt_cfg = bt_cfg or BacktestConfig()

    def run_walk_forward(self, data: dict, model_factory, trainer_factory) -> list[WalkForwardResult]:
        splits = create_walk_forward_splits(len(data["prices"]), self.split_cfg)
        results: list[WalkForwardResult] = []

        for i, split in enumerate(splits):
            train = slice_split(data["features"], data["prices"], split["train"])
            val = slice_split(data["features"], data["prices"], split["val"])
            test = slice_split(data["features"], data["prices"], split["test"])

            model = model_factory()
            trainer = trainer_factory(model)
            trainer.fit(train, val)

            backtest = VectorizedBacktest(model, config=self.bt_cfg)
            pnl, trades = backtest.run(test["features"], test["prices"])
            metrics = compute_metrics(pnl, trades)
            results.append(WalkForwardResult(i, split, metrics))

        return results

    @staticmethod
    def aggregate(results: list[WalkForwardResult]) -> dict[str, float]:
        if not results:
            return {}
        keys = list(results[0].metrics.keys())
        arr = {k: np.array([r.metrics[k] for r in results], dtype=float) for k in keys}
        return {f"{k}_mean": float(v.mean()) for k, v in arr.items()} | {
            f"{k}_std": float(v.std()) for k, v in arr.items()
        }
