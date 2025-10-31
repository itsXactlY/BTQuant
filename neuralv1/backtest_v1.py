
import numpy as np
import torch
import polars as pl
from rich.console import Console
from architecture_v1 import V1FixedTradingModel

console = Console()

def vectorized_walk_forward_backtest(df: pl.DataFrame, model_path: str, selected: list, scaler, config: dict) -> dict:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = V1FixedTradingModel(len(selected), config['seq_len']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    feat_2d = scaler.transform(df.select(selected).to_numpy())
    prices = df['close'].to_numpy()
    seq_len = config['seq_len']

    min_window_size = seq_len + 100
    if len(prices) < min_window_size:
        raise ValueError(f"Insufficient data for walk-forward: {len(prices)} < {min_window_size}")

    all_sharpes, all_dds, exit_rates = [], [], []
    skipped_windows = 0

    for test_start in range(0, len(prices) - 4*(len(prices)//20), len(prices)//20):
        test_end = min(test_start + 4*(len(prices)//20), len(prices))

        if test_start < seq_len:
            console.print(f"⚠️ Skipping window: test_start {test_start} < seq_len {seq_len}")
            skipped_windows += 1
            continue

        if test_end - test_start < 50:
            console.print(f"⚠️ Skipping window: insufficient length {test_end - test_start}")
            skipped_windows += 1
            continue

        if test_end > len(prices):
            console.print(f"⚠️ Skipping window: test_end {test_end} exceeds data length {len(prices)}")
            skipped_windows += 1
            continue

        feat_start = test_start - seq_len
        if feat_start < 0 or test_end > len(feat_2d):
            console.print(f"⚠️ Skipping window: invalid feature bounds [{feat_start}:{test_end}]")
            skipped_windows += 1
            continue

        try:
            test_feat = feat_2d[feat_start:test_end]
            test_prices = prices[test_start:test_end]

            if len(test_feat) < seq_len or len(test_prices) == 0:
                raise ValueError(f"Invalid slice: feat={len(test_feat)}, prices={len(test_prices)}")

            if len(test_feat) < seq_len:
                raise ValueError(f"Cannot create windows: {len(test_feat)} < {seq_len}")

            windows = np.lib.stride_tricks.sliding_window_view(test_feat, (seq_len, len(selected)))
            windows = windows.squeeze(1)

            if len(windows) == 0:
                raise ValueError("Sliding window produced 0 windows")

            positions = np.zeros(len(test_prices))
            model_exits = 0
            total_trades = 0

            with torch.no_grad():
                for i in range(seq_len, len(test_prices)):
                    window_idx = i - seq_len
                    if window_idx >= len(windows):
                        break

                    seq = torch.tensor(windows[window_idx]).float().unsqueeze(0).to(device)
                    context = torch.tensor([[0.0, 5.0]]).to(device)
                    preds = model(seq, context)

                    entry_p = torch.sigmoid(preds['entry']).item()
                    tp_p = torch.sigmoid(preds['tp']).item()
                    sl_p = torch.sigmoid(preds['sl']).item()
                    exp_r = preds['others'][0,0].item()

                    if entry_p > 0.5 and positions[i-1] == 0:
                        kelly = (entry_p * exp_r - (1 - entry_p)) / max(abs(exp_r), 1e-6)
                        positions[i] = np.clip(kelly, 0.1, 0.5)
                        total_trades += 1

                    if positions[i-1] != 0:
                        if tp_p > 0.5 or sl_p > 0.5:
                            positions[i] = 0
                            model_exits += 1
                        else:
                            positions[i] = positions[i-1]

            if len(test_prices) < 2:
                raise ValueError("Insufficient prices for returns calculation")

            rets = np.diff(test_prices) / test_prices[:-1]
            pos_rets = positions[1:] * rets
            pos_rets -= np.abs(np.diff(positions)) * (0.001 + 0.0005)

            if len(pos_rets) == 0 or np.all(pos_rets == 0):
                console.print(f"⚠️ Window produced no returns, skipping metrics")
                skipped_windows += 1
                continue

            cum_rets = np.cumprod(1 + pos_rets)
            sharpe = np.mean(pos_rets) / max(np.std(pos_rets), 1e-6) * np.sqrt(252 * 6)
            peak = np.maximum.accumulate(cum_rets)
            dd = np.max((peak - cum_rets) / peak) if len(peak) > 0 else 0
            exit_rate = model_exits / max(total_trades, 1)

            all_sharpes.append(sharpe)
            all_dds.append(dd)
            exit_rates.append(exit_rate)

        except Exception as e:
            console.print(f"⚠️ Error in window [{test_start}:{test_end}]: {e}")
            skipped_windows += 1
            continue

    if len(all_sharpes) == 0:
        raise ValueError(f"All {skipped_windows} walk-forward windows failed. Check data and parameters.")

    metrics = {
        'mean_sharpe': np.mean(all_sharpes),
        'mean_dd': np.mean(all_dds),
        'mean_exit_rate': np.mean(exit_rates),
        'win_rate': np.mean(pos_rets > 0) if len(pos_rets) > 0 else 0,
        'valid_windows': len(all_sharpes),
        'skipped_windows': skipped_windows
    }

    console.print(f"📈 Walk-Forward Metrics: Sharpe {metrics['mean_sharpe']:.2f}, "
                  f"DD {metrics['mean_dd']:.2%}, Exits {metrics['mean_exit_rate']:.1%}, "
                  f"Valid: {metrics['valid_windows']}, Skipped: {metrics['skipped_windows']}")

    return metrics
