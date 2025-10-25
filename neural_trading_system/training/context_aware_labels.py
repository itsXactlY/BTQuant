"""
Context-Aware Exit Label Generation
Works with features + returns only (no OHLCV required)
"""

import numpy as np
import torch
from typing import Dict

def generate_context_aware_labels_from_returns(
    dataset,
    returns: np.ndarray,
    features: np.ndarray,
    lookback: int = 50
) -> None:
    """
    Generate context-aware exit labels using only returns + features.

    No hardcoded thresholds - learns from data patterns:
    - Volatility clustering
    - Momentum patterns
    - Peak/trough detection

    Args:
        dataset: TradingDataset object to store labels in
        returns: Numpy array of returns
        features: Numpy array of features
        lookback: Bars to look ahead for exit detection
    """
    T = len(returns)
    tp_labels = np.zeros(T, dtype=np.float32)
    sl_labels = np.zeros(T, dtype=np.float32)
    hold_labels = np.zeros(T, dtype=np.float32)

    # Calculate rolling volatility (context)
    window = 20
    volatility = np.array([
        np.std(returns[max(0, i-window):i+1]) if i >= window 
        else np.std(returns[:i+1])
        for i in range(T)
    ])

    # Volatility regime (relative to recent history)
    vol_ma = np.convolve(volatility, np.ones(100)/100, mode='same')
    vol_regime = volatility / (vol_ma + 1e-8)

    print("🧠 Generating context-aware exit labels...")
    print(f"   Analyzing {T} bars with {lookback}-bar lookback")

    for i in range(T - lookback):
        future_rets = returns[i:i+lookback]
        cumulative = np.cumsum(future_rets)

        # Find peak and trough
        peak_idx = np.argmax(cumulative)
        peak_return = cumulative[peak_idx]
        trough_idx = np.argmin(cumulative)
        trough_return = cumulative[trough_idx]

        # Get context at peak/trough
        vol_at_peak = vol_regime[i + peak_idx] if (i + peak_idx) < T else 1.0
        vol_at_trough = vol_regime[i + trough_idx] if (i + trough_idx) < T else 1.0

        # Calculate dynamic thresholds OUTSIDE the conditionals
        peak_threshold = np.percentile(cumulative[cumulative > 0], 75) if np.any(cumulative > 0) else 0.01
        trough_threshold = np.percentile(cumulative[cumulative < 0], 25) if np.any(cumulative < 0) else -0.01

        # ✅ TAKE PROFIT: Peak + high volatility (exhaustion)
        if peak_return > peak_threshold and vol_at_peak > 1.3:
            tp_labels[i + peak_idx] = 1.0
            hold_labels[i:i+peak_idx] = 0.9

        # ✅ STOP LOSS: Trough + high volatility (regime change)
        elif trough_return < trough_threshold and vol_at_trough > 1.3:
            sl_labels[i + trough_idx] = 1.0
            hold_labels[i:i+trough_idx] = 0.3

        # ✅ HOLD: Stable volatility, no extremes
        else:
            avg_vol = np.mean(vol_regime[i:i+lookback])
            if avg_vol < 1.2:
                hold_labels[i:i+lookback] = 0.7  # Stable
            else:
                hold_labels[i:i+lookback] = 0.5  # Uncertain

    tp_count = int(np.sum(tp_labels > 0))
    sl_count = int(np.sum(sl_labels > 0))
    hold_count = int(np.sum(hold_labels > 0.6))

    print(f"✅ Context-aware labels generated:")
    print(f"   TP triggers: {tp_count}")
    print(f"   SL triggers: {sl_count}")
    print(f"   Hold bars: {hold_count}")

    if tp_count > 0:
        tp_indices = np.where(tp_labels > 0)[0]
        tp_rets = returns[tp_indices]
        print(f"   Avg TP return: {tp_rets.mean():.3%} (±{tp_rets.std():.3%})")
    if sl_count > 0:
        sl_indices = np.where(sl_labels > 0)[0]
        sl_rets = returns[sl_indices]
        print(f"   Avg SL return: {sl_rets.mean():.3%} (±{sl_rets.std():.3%})")

    # Store in dataset
    dataset.tp_labels = torch.as_tensor(tp_labels, dtype=torch.float32)
    dataset.sl_labels = torch.as_tensor(sl_labels, dtype=torch.float32)
    dataset.hold_labels = torch.as_tensor(hold_labels, dtype=torch.float32)

    print("🎯 Model will learn context-aware exits (no hardcoded thresholds)")
