from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def wrapper(func):
            return func

        return wrapper


@njit

def _compute_mfe_mae_core(prices: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(prices)
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)
    for i in range(n - horizon):
        entry = prices[i]
        best = -1e9
        worst = 1e9
        for j in range(1, horizon + 1):
            ret = (prices[i + j] - entry) / entry
            if ret > best:
                best = ret
            if ret < worst:
                worst = ret
        mfe[i] = best
        mae[i] = worst
    return mfe, mae


@njit

def _compute_exit_events_core(
    prices: np.ndarray,
    horizon: int,
    tp_thresh: float,
    sl_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(prices)
    tp_hit = np.zeros(n, dtype=np.float64)
    sl_hit = np.zeros(n, dtype=np.float64)
    bars_to_exit = np.full(n, horizon, dtype=np.float64)

    for i in range(n - horizon):
        entry = prices[i]
        for j in range(1, horizon + 1):
            ret = (prices[i + j] - entry) / entry
            if ret >= tp_thresh:
                tp_hit[i] = 1.0
                bars_to_exit[i] = j
                break
            if ret <= sl_thresh:
                sl_hit[i] = 1.0
                bars_to_exit[i] = j
                break
    return tp_hit, sl_hit, bars_to_exit


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = float(np.mean(arr[start : i + 1]))
    return out


@dataclass
class LabelConfig:
    horizon: int = 50
    tp_threshold: float = 0.02
    sl_threshold: float = -0.01
    vol_window: int = 20
    smooth_window: int = 30


class LabelGenerator:
    def __init__(self, config: LabelConfig | None = None) -> None:
        self.config = config or LabelConfig()

    def generate(self, prices: np.ndarray) -> dict[str, np.ndarray]:
        c = self.config
        mfe, mae = _compute_mfe_mae_core(prices.astype(np.float64), c.horizon)
        tp, sl, bars = _compute_exit_events_core(
            prices.astype(np.float64), c.horizon, c.tp_threshold, c.sl_threshold
        )

        future_ret = np.zeros_like(prices, dtype=np.float64)
        for i in range(len(prices) - c.horizon):
            future_ret[i] = (prices[i + c.horizon] - prices[i]) / prices[i]

        raw_returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices, 1e-12)
        volatility = _rolling_mean(raw_returns**2, c.vol_window) ** 0.5
        tp_prob = _rolling_mean(tp, c.smooth_window)
        sl_prob = _rolling_mean(sl, c.smooth_window)

        let_run_score = ((mfe > 0.01) & (future_ret < 0.7 * np.maximum(mfe, 1e-6))).astype(np.float64)

        return {
            "expected_return": future_ret,
            "tp_label": tp,
            "sl_label": sl,
            "tp_prob": tp_prob,
            "sl_prob": sl_prob,
            "optimal_exit_bars": bars,
            "volatility": volatility,
            "let_run_score": let_run_score,
            "mfe": mfe,
            "mae": mae,
        }
