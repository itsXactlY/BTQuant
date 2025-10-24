#!/usr/bin/env python3
"""
Quick start script for training elite neural model
"""

import sys
sys.path.append('neural_trading_system')

from neural_pipeline import train_neural_system
import torch

# ✅ ELITE CONFIGURATION - MAXIMUM PERFORMANCE
elite_config = {
    # === ARCHITECTURE - Deep & Expressive ===
    'seq_len': 100,
    'prediction_horizon': 5,
    'lookback_windows': [5, 10, 20, 50, 100, 200],
    'd_model': 512,
    'num_heads': 16,
    'num_layers': 8,
    'd_ff': 2048,
    'dropout': 0.15,
    'latent_dim': 16,
    
    # === TRAINING - Aggressive Learning ===
    'batch_size': 128,
    'num_epochs': 200,
    'lr': 0.0003,
    'min_lr': 1e-7,
    'weight_decay': 1e-4,
    'grad_accum_steps': 2,
    
    # === SCHEDULER ===
    'T_0': 20,
    'patience': 25,
    'save_every': 5,
    
    # === TRACKING ===
    'use_wandb': True,
    'run_name': 'elite_btc_1h_full_history',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'best_model_path': 'models/best_model.pt',
}

if __name__ == '__main__':
    print("🚀 Starting Elite Neural Training System")
    print("=" * 80)
    
    train_neural_system(
        coin='BTC',
        interval='1h',
        start_date='2017-01-01',
        end_date='2024-12-31',
        collateral='USDT',
        config=elite_config
    )