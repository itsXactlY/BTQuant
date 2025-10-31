
import numpy as np
import polars as pl
from numba import jit
from rich.console import Console

console = Console()

@jit(nopython=True)
def compute_excursions(prices: np.ndarray, horizon: int = 50, tp_thresh: float = 0.02, sl_thresh: float = -0.01):

    if len(prices) < horizon + 1:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    entry = prices[0]
    future = prices[1:horizon+1]
    rets = (future - entry) / entry
    mfe = np.max(rets)
    mae = np.min(rets)
    mfe_i = np.argmax(rets)
    mae_i = np.argmin(rets)

    tp = float(int(np.any(rets >= tp_thresh)))
    sl = float(int(np.any(rets <= sl_thresh)))

    opt_bars = float(min(mfe_i, mae_i) + 1)
    vol = float(np.std(rets))

    let_run = float(int(mfe > abs(mae) + 0.01))

    regime = 0.0
    future_r = float(np.mean(rets))
    entry_label = float(int(future_r > 0.01))

    return (entry_label, tp, sl, let_run, regime, future_r, vol, mfe, mae)

def generate_mfe_mae_labels(df: pl.DataFrame) -> pl.DataFrame:
    closes = df['close'].to_numpy()
    labels = []

    for i in range(100, len(closes) - 50):
        window = closes[i-100:i+51]
        labels.append(compute_excursions(window))

    entry_l, tp_l, sl_l, hold_l, regime_l, future_r, vol, mfe, mae = map(np.array, zip(*labels))

    df_labels = pl.DataFrame({
        'entry_label': entry_l,
        'tp_label': tp_l,
        'sl_label': sl_l,
        'hold_label': hold_l,
        'regime_label': regime_l,
        'future_return': future_r,
        'volatility': vol,
        'mfe': mfe,
        'mae': mae
    })

    return df_labels
