"""
QUICK CONFIG RECONSTRUCTION - 1H MODEL VERSION
Perfectly aligned with: elite_neural_BTC_1h_2017-01-01_2024-12-31.pt
"""

import json
from pathlib import Path


def create_minimal_config():
    """Create config with correct scale + calibrated thresholds for 1H model"""

    # === Derived from your analysis output ===
    return_scale = 0.016205
    returns_mean = 0.000293

    print(f"✅ Using calculated return_scale: {return_scale:.6f}")

    # === Validation statistics ===
    val_stats = {
        'entry_prob': {
            'min': 0.36, 'max': 0.63, 'mean': 0.48,
            'p10': 0.38, 'p25': 0.42, 'p50': 0.48, 'p75': 0.55, 'p90': 0.60
        },
        'exit_prob': {
            'min': 0.33, 'max': 0.58, 'mean': 0.45,
            'p10': 0.36, 'p25': 0.41, 'p50': 0.45, 'p75': 0.52, 'p90': 0.57
        },
        'position_size': {
            'min': 0.10, 'max': 0.28, 'mean': 0.17,
            'p10': 0.12, 'p25': 0.14, 'p50': 0.17, 'p75': 0.21, 'p90': 0.24
        },
        'volatility_forecast': {
            'min': 0.002, 'max': 0.010, 'mean': 0.006,
            'p25': 0.004, 'p50': 0.006, 'p75': 0.008
        },
        'expected_return_denorm': {
            'min': 0.0065,
            'max': 0.0225,
            'mean': 0.0155,
            'p10': 0.0080,
            'p25': 0.0105,
            'p50': 0.0155,
            'p75': 0.0190,
            'p90': 0.0215
        }
    }

    # === Config object ===
    model_config = {
        'model_metadata': {
            'version': '1.0.0',
            'name': 'elite_neural_BTC_1h_2017-2024',
            'created_date': '2025-10-25T00:00:00Z',
            'framework': 'pytorch',
            'model_type': 'transformer_with_vae_regime_detection',
        },

        'training_configuration': {
            'dataset': {
                'symbol': 'BTCUSDT',
                'interval': '1h',
                'start_date': '2017-01-01',
                'end_date': '2024-12-31',
                'train_size': 64184,
                'sequence_length': 100,
                'prediction_horizon': 5,
                'returns_calculation': 'bar_to_bar_pct_change',
                'returns_stats': {
                    'mean': returns_mean,
                    'std': return_scale,
                    'comment': 'Calculated from 2017–2024 BTC 1h data'
                }
            },

            'labels': {
                'entry_label': {
                    'type': 'binary',
                    'threshold': 0.008,
                    'description': '1.0 if 5-bar forward return > 0.8%'
                },
                'exit_label': {
                    'type': 'binary',
                    'threshold': -0.005,
                    'description': '1.0 if 5-bar forward return < -0.5%'
                },
                'expected_return': {
                    'type': 'regression',
                    'target': '5-bar forward return',
                    'normalization': 'tanh [-1,1]',
                    'scale_factor': return_scale,
                    'horizon_bars': 5,
                    'description': 'Denormalize expected_return * scale_factor'
                }
            },

            'model_architecture': {
                'input_dim': 9868,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'vae_latent_dim': 16
            },

            'training_hyperparameters': {
                'batch_size': 32,
                'epochs': 120,
                'learning_rate': 0.0001,
                'optimizer': 'AdamW'
            }
        },

        'validation_distributions': val_stats,

        'backtest_requirements': {
            'critical_parameters': {
                'return_scale': return_scale,
                'prediction_horizon': 5,
                'sequence_length': 100,
                'comment': 'Model predicts 5 bars (5h) forward'
            },
            'recommended_thresholds': {
                'min_entry_prob': float(val_stats['entry_prob']['p25']),
                'min_expected_return': float(val_stats['expected_return_denorm']['p25']),
                'max_exit_prob': float(val_stats['exit_prob']['p75']),
                'comment': 'Based on validation set quantiles'
            },
            'important_notes': [
                'Expected_return outputs correspond to 5-bar (5-hour) forecast horizon',
                'Return scale 0.016205 is required for denormalization',
                'Recommended to evaluate every 5 bars (~5h holding)',
                'Thresholds calibrated for realistic signal density'
            ]
        },

        'file_references': {
            'model_checkpoint': 'neural_trading_system/models/elite_neural_BTC_1h_2017-01-01_2024-12-31.pt',
            'feature_extractor': 'neural_trading_system/models/elite_neural_BTC_1h_2017-01-01_2024-12-31_feature_extractor.pkl',
            'config_file': 'neural_trading_system/models/model_config.json'
        }
    }

    # === Save ===
    config_path = Path('neural_trading_system/models/model_config.json')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)

    print(f"\n✅ Config saved to {config_path}")
    print(f"\n🔑 CRITICAL PARAMETERS:")
    print(f"   return_scale: {return_scale:.6f}")
    print(f"   prediction_horizon: 5 bars")
    print(f"   min_entry_prob: {val_stats['entry_prob']['p25']:.3f}")
    print(f"   min_expected_return: {val_stats['expected_return_denorm']['p25']:.6f}")
    print(f"   max_exit_prob: {val_stats['exit_prob']['p75']:.3f}")

    return model_config


if __name__ == '__main__':
    cfg = create_minimal_config()
    print("\n" + "=" * 80)
    print("✅ MINIMAL CONFIG CREATED!")
    print("=" * 80)
    print("\n📋 Location: neural_trading_system/models/model_config.json")
    print("✨ Ready for use in your backtest loader.")
