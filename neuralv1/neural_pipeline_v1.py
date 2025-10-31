
import os
import json
import polars as pl
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any
import wandb
from rich.console import Console

console = Console()

from data_utils import load_and_cache_data
from neuralv1.indicators_v1 import compute_core_indicators
from neuralv1.feature_extractor_v1 import extract_and_select_features
from neuralv1.labels_v1 import generate_mfe_mae_labels
from neuralv1.architecture_v1 import V1FixedTradingModel
from neuralv1.trainer_v1 import V1FixedTrainer
from neuralv1.backtest_v1 import vectorized_walk_forward_backtest

CONFIG = {
    'symbol': 'BTCUSDT',
    'interval': '1h',
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'seq_len': 100,
    'n_features': 15,
    'batch_size': 32,
    'epochs': 150,
    'lr': 5e-5,
    'wandb_project': 'neural-trading-v1-1h',
    'cache_dir': Path('neural_cache'),
    'model_dir': Path('neural_models'),
    'data_dir': Path('neural_data'),
    'force_reload': False
}

def main():

    for d in [CONFIG['cache_dir'], CONFIG['model_dir'], CONFIG['data_dir']]:
        d.mkdir(parents=True, exist_ok=True)

    run_name = f"{CONFIG['symbol']}_{CONFIG['interval']}_v1_fixed"
    wandb.init(project=CONFIG['wandb_project'], name=run_name, config=CONFIG)
    console.print(f"🚀 Starting V1 Fixed Workflow: {run_name}")

    data_file = CONFIG['data_dir'] / f"{CONFIG['symbol']}_{CONFIG['interval']}.parquet"
    if data_file.exists() and not CONFIG['force_reload']:
        df = pl.read_parquet(data_file)
        console.print(f"✅ Loaded cached data: {len(df):,} bars")
    else:
        df = load_and_cache_data(CONFIG)
        df.write_parquet(data_file)

    ind_cache = CONFIG['cache_dir'] / f"indicators_{CONFIG['symbol']}_{CONFIG['interval']}.parquet"
    if ind_cache.exists() and not CONFIG['force_reload']:
        df_ind = pl.read_parquet(ind_cache)
    else:
        df_ind = compute_core_indicators(df)
        df_ind.write_parquet(ind_cache)
    console.print(f"📊 Indicators cached: {len(df_ind)} bars")

    feat_cache = CONFIG['cache_dir'] / f"features_3d_{CONFIG['n_features']}.npy"
    label_cache = CONFIG['cache_dir'] / f"labels_mfe_mae.npy"
    scaler_cache = CONFIG['cache_dir'] / f"scaler_{CONFIG['n_features']}.pkl"
    selected_cache = CONFIG['cache_dir'] / f"selected_features_{CONFIG['n_features']}.json"

    if feat_cache.exists() and label_cache.exists() and scaler_cache.exists() and not CONFIG['force_reload']:
        features_3d = np.load(feat_cache)
        labels = np.load(label_cache)
        with open(scaler_cache, 'rb') as f:
            scaler = pickle.load(f)
        with open(selected_cache, 'r') as f:
            selected_features = json.load(f)
        console.print(f"✅ Loaded cached features/labels: {features_3d.shape}")
    else:

        selected_features, scaler = extract_and_select_features(df_ind, CONFIG['n_features'])

        if selected_features is None or scaler is None:
            console.print("❌ Feature extraction failed. Cannot proceed.")
            wandb.finish(exit_code=1)
            return

        try:
            features_3d, labels = generate_sequences_and_labels(
                df_ind, selected_features, CONFIG['seq_len'], scaler
            )
        except Exception as e:
            console.print(f"❌ Sequence generation failed: {e}")
            wandb.finish(exit_code=1)
            return

        try:
            np.save(feat_cache, features_3d)
            np.save(label_cache, labels)
            with open(scaler_cache, 'wb') as f:
                pickle.dump(scaler, f)
            with open(selected_cache, 'w') as f:
                json.dump(selected_features, f)
            console.print(f"💾 Cached features: {features_3d.shape}, labels: {labels.shape}")
        except Exception as e:
            console.print(f"⚠️ Caching failed: {e}. Continuing with in-memory data.")

    actual_n_features = features_3d.shape[2]
    if actual_n_features != CONFIG['n_features']:
        console.print(f"⚠️ Adjusting n_features: {CONFIG['n_features']} → {actual_n_features}")
        CONFIG['n_features'] = actual_n_features

    splits = create_walk_forward_splits(features_3d, labels, CONFIG)
    console.print(f"📈 Created {len(splits['train'])} walk-forward windows")

    model = V1FixedTradingModel(n_features=CONFIG['n_features'], seq_len=CONFIG['seq_len'])
    trainer = V1FixedTrainer(model, splits, CONFIG)
    model_path = Path(trainer.train())

    config_path = model_path.parent / f"{model_path.stem}_config.json"
    with open(config_path, 'w') as f:

        config_serializable = {k: str(v) if isinstance(v, Path) else v
                            for k, v in CONFIG.items()}
        json.dump(config_serializable, f, indent=2)
    console.print(f"💾 Saved config: {config_path}")

    metrics = vectorized_walk_forward_backtest(df_ind, model_path, selected_features, scaler, CONFIG)
    wandb.log(metrics)
    console.print("🎯 Final Metrics:", json.dumps(metrics, indent=2))

    wandb.finish()
    console.print("✅ V1 Fixed Workflow Complete!")

def create_walk_forward_splits(features: np.ndarray, labels: np.ndarray, config: Dict) -> Dict[str, Any]:

    total = len(features)
    window_size = total // 20

    if total < 1000:
        raise ValueError(f"Insufficient total samples: {total} < 1000. Need more data.")

    if len(features) != len(labels):
        raise ValueError(f"Feature/label mismatch: {len(features)} != {len(labels)}")

    train_windows, val_windows, test_windows = [], [], []
    skipped = 0

    console.print(f"📊 Creating splits from {total:,} samples, window_size={window_size}")

    for i in range(20):
        start = i * window_size
        train_end = start + int(12 * window_size / 16)
        val_end = train_end + int(2 * window_size / 16)
        test_end = min(val_end + int(2 * window_size / 16), total)

        try:

            if test_end > total:
                raise ValueError(f"test_end {test_end} exceeds total {total}")

            if train_end >= val_end or val_end >= test_end:
                raise ValueError(f"Invalid window progression: {start}->{train_end}->{val_end}->{test_end}")

            if train_end - start < 100:
                raise ValueError(f"Train window too small: {train_end - start} < 100")

            if val_end - train_end < 20:
                raise ValueError(f"Val window too small: {val_end - train_end} < 20")

            if test_end - val_end < 20:
                raise ValueError(f"Test window too small: {test_end - val_end} < 20")

            train_feat = features[start:train_end]
            train_lab = labels[start:train_end]
            val_feat = features[train_end:val_end]
            val_lab = labels[train_end:val_end]
            test_feat = features[val_end:test_end]
            test_lab = labels[val_end:test_end]

            for name, arr in [('train_feat', train_feat), ('train_lab', train_lab),
                              ('val_feat', val_feat), ('val_lab', val_lab),
                              ('test_feat', test_feat), ('test_lab', test_lab)]:
                if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                    raise ValueError(f"{name} contains NaN/Inf")

            train_windows.append((train_feat, train_lab))
            val_windows.append((val_feat, val_lab))
            test_windows.append((test_feat, test_lab))

        except Exception as e:
            console.print(f"⚠️ Skipping window {i}: {e}")
            skipped += 1
            continue

    if len(train_windows) == 0:
        raise ValueError(f"All {skipped} windows failed validation. Check data quality and parameters.")

    if len(train_windows) < 5:
        console.print(f"⚠️ Warning: Only {len(train_windows)} valid windows (skipped {skipped})")

    console.print(f"✅ Created {len(train_windows)} valid windows (skipped {skipped})")

    return {'train': train_windows, 'val': val_windows, 'test': test_windows}

def generate_sequences_and_labels(df: pl.DataFrame, selected: list, seq_len: int, scaler) -> tuple:

    console.print(f"🔧 Generating sequences from {len(df)} bars with {len(selected)} features...")

    if len(df) < seq_len:
        raise ValueError(f"Insufficient data: {len(df)} rows < seq_len {seq_len}. Need at least {seq_len} bars.")

    feat_2d = scaler.transform(df.select(selected).to_numpy())

    if feat_2d.shape[0] < seq_len:
        raise ValueError(f"Feature array too short: {feat_2d.shape[0]} < {seq_len}")
    if feat_2d.shape[1] != len(selected):
        raise ValueError(f"Feature mismatch: got {feat_2d.shape[1]}, expected {len(selected)}")

    if np.any(np.isnan(feat_2d)) or np.any(np.isinf(feat_2d)):
        nan_count = np.sum(np.isnan(feat_2d))
        inf_count = np.sum(np.isinf(feat_2d))
        raise ValueError(f"Invalid values detected: {nan_count} NaNs, {inf_count} Infs. Clean data first.")

    try:
        features_3d = np.lib.stride_tricks.sliding_window_view(feat_2d, (seq_len, len(selected)))
        features_3d = features_3d.squeeze(1)
    except Exception as e:
        raise RuntimeError(f"Sliding window failed: {e}. Shape: {feat_2d.shape}, seq_len: {seq_len}") from e

    console.print(f"📐 Features shape after sliding window: {features_3d.shape}")

    try:
        labels_df = generate_mfe_mae_labels(df)
    except Exception as e:
        raise RuntimeError(f"Label generation failed: {e}") from e

    console.print(f"🏷️ Generated {len(labels_df)} labels")

    if len(labels_df) == 0:
        raise ValueError("Label generation produced 0 rows. Check generate_mfe_mae_labels implementation.")

    required_cols = ['entry_label', 'tp_label', 'sl_label', 'hold_label', 'regime_label',
                     'future_return', 'volatility', 'mfe', 'mae']
    for col in required_cols:
        if col not in labels_df.columns:
            raise ValueError(f"Missing required label column: {col}")

    labels = np.column_stack([
        labels_df['entry_label'].to_numpy(),
        labels_df['tp_label'].to_numpy(),
        labels_df['sl_label'].to_numpy(),
        labels_df['hold_label'].to_numpy(),
        labels_df['regime_label'].to_numpy(),
        labels_df['future_return'].to_numpy(),
        labels_df['volatility'].to_numpy(),
        labels_df['mfe'].to_numpy(),
        labels_df['mae'].to_numpy(),
        np.zeros(len(labels_df))
    ])

    offset = max(0, 100 - (seq_len - 1))

    if offset > 0:
        feat_end = min(offset + len(labels), len(features_3d))
        features_aligned = features_3d[offset:feat_end]
        labels_aligned = labels[:feat_end - offset]
    else:
        label_offset = abs(offset)
        feat_end = min(len(labels) - label_offset, len(features_3d))
        features_aligned = features_3d[:feat_end]
        labels_aligned = labels[label_offset:label_offset + feat_end]

    min_len = min(len(features_aligned), len(labels_aligned))
    features_aligned = features_aligned[:min_len]
    labels_aligned = labels_aligned[:min_len]

    if min_len < 100:
        raise ValueError(f"Insufficient aligned data: {min_len} samples. Need at least 100.")

    console.print(f"✅ Final aligned shapes - Features: {features_aligned.shape}, Labels: {labels_aligned.shape}")

    return features_aligned, labels_aligned

if __name__ == '__main__':
    main()
