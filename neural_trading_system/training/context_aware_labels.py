import numpy as np

def context_aware_labels_from_returns(
    future_prices: np.ndarray,
    current_price: float,
    rolling_vol: float,
    max_lookforward: int = 50,
    tp_pctile: float = 0.60,
    sl_pctile: float = 0.40,
    tp_z: float = 0.75,
    sl_z: float = 0.75,
    rc_vol_spike: float = 1.8
):
\
\
\
\
\

    horizon = min(max_lookforward, len(future_prices))
    if horizon < 5:
        return dict(
            take_profit_label=0.0,
            stop_loss_label=0.0,
            let_winner_run_label=0.0,
            regime_change_label=0.0
        )

    fp = future_prices[:horizon]
    future_rets = (fp - current_price) / max(current_price, 1e-12)

    vol = max(float(rolling_vol), 1e-6)
    mu = float(np.mean(future_rets))
    sd = float(np.std(future_rets) + 1e-9)

    tp_thresh = max(np.quantile(future_rets, tp_pctile), mu + tp_z * vol)
    sl_thresh = min(np.quantile(future_rets, sl_pctile), mu - sl_z * vol)

    peak_ret = float(np.max(future_rets))
    trough_ret = float(np.min(future_rets))
    peak_idx = int(np.argmax(future_rets))
    trough_idx = int(np.argmin(future_rets))

    tp = 1.0 if (peak_ret >= tp_thresh and peak_idx <= 10) else (0.7 if peak_idx <= 20 and peak_ret >= tp_thresh else 0.0)

    sl = 1.0 if (trough_ret <= sl_thresh and trough_idx <= 7) else (0.7 if trough_idx <= 14 and trough_ret <= sl_thresh else 0.0)

    if horizon >= 20:
        near = float(np.mean(future_rets[:10]))
        far  = float(np.mean(future_rets[10:20]))
        lr = 1.0 if (far > near and far > tp_thresh * 0.5) else (0.5 if far > near else 0.0)
    else:
        lr = 1.0 if peak_ret >= tp_thresh else 0.0

    if horizon >= 10:
        recent_vol = float(np.std(future_rets[:5]) + 1e-9)
        future_vol = float(np.std(future_rets[5:10]) + 1e-9)
        rc = 1.0 if (future_vol > rc_vol_spike * max(recent_vol, 1e-9)) else 0.0
    else:
        rc = 0.0

    return dict(
        take_profit_label=float(tp),
        stop_loss_label=float(sl),
        let_winner_run_label=float(lr),
        regime_change_label=float(rc),
    )
