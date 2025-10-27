# Neural Trading System - Configuration Guide

## 🎯 Configuration Tiers

### Tier 1: Fast Prototyping (Debug Mode)
**Use case**: Testing pipeline, quick iterations
**Training time**: ~10-20 minutes
**Expected performance**: Poor (for testing only)
```python
debug_config = {
    'seq_len': 50,
    'prediction_horizon': 1,
    'd_model': 128,
    'num_heads': 4,
    'num_layers': 2,
    'd_ff': 512,
    'dropout': 0.1,
    'latent_dim': 8,
    'batch_size': 32,
    'num_epochs': 20,
    'lr': 0.0001,
}
```

### Tier 2: Baseline Model
**Use case**: Initial experiments, baseline performance
**Training time**: ~2-4 hours
**Expected performance**: Moderate
```python
baseline_config = {
    'seq_len': 100,
    'prediction_horizon': 5,
    'd_model': 256,
    'num_heads': 8,
    'num_layers': 4,
    'd_ff': 1024,
    'dropout': 0.1,
    'latent_dim': 8,
    'batch_size': 64,
    'num_epochs': 100,
    'lr': 0.0001,
}
```

### Tier 3: Elite Model (RECOMMENDED)
**Use case**: Maximum performance, production deployment
**Training time**: ~8-12 hours
**Expected performance**: Best
```python
elite_config = {
    'seq_len': 100,
    'prediction_horizon': 5,
    'lookback_windows': [5, 10, 20, 50, 100, 200],
    'd_model': 512,
    'num_heads': 16,
    'num_layers': 8,
    'd_ff': 2048,
    'dropout': 0.15,
    'latent_dim': 16,
    'batch_size': 128,
    'num_epochs': 200,
    'lr': 0.0003,
    'min_lr': 1e-7,
    'weight_decay': 1e-4,
    'grad_accum_steps': 2,
    'T_0': 20,
    'patience': 25,
}
```

### Tier 4: Experimental (Advanced Users)
**Use case**: Research, pushing limits
**Training time**: 24+ hours
**Expected performance**: Diminishing returns
```python
experimental_config = {
    'seq_len': 150,
    'prediction_horizon': 10,
    'lookback_windows': [5, 10, 20, 50, 100, 200, 500],
    'd_model': 768,
    'num_heads': 24,
    'num_layers': 12,
    'd_ff': 3072,
    'dropout': 0.2,
    'latent_dim': 32,
    'batch_size': 256,
    'num_epochs': 300,
    'lr': 0.0005,
}
```

## 🔧 Key Parameter Explanations

### Architecture Parameters

**d_model**: Model dimension (width)
- 256: Fast, light
- 512: Recommended balance
- 768+: Maximum capacity, slower

**num_heads**: Attention heads
- 8: Standard
- 16: Enhanced (recommended)
- 24+: Experimental

**num_layers**: Transformer depth
- 4: Shallow, fast
- 8: Deep understanding (recommended)
- 12+: Very deep, diminishing returns

**latent_dim**: VAE regime space
- 8: Basic regime detection
- 16: Rich regime space (recommended)
- 32+: Potentially overfitting

### Training Parameters

**batch_size**: 
- 32: Small GPU (<8GB)
- 128: Recommended (16GB+)
- 256+: Large GPU (24GB+)

**lr (learning rate)**:
- 0.0001: Conservative
- 0.0003: Aggressive (recommended)
- 0.0005+: Risk of instability

**num_epochs**:
- 100: Baseline
- 200: Recommended
- 300+: Only if validation loss still improving

**patience**: Early stopping
- 15: Quick stop
- 25: Recommended
- 40+: Very patient

## 📊 Expected Results by Configuration

| Config | Sharpe Ratio | Win Rate | Training Time |
|--------|--------------|----------|---------------|
| Debug | N/A | ~50% | 10 min |
| Baseline | 0.5-1.0 | 52-55% | 2-4 hrs |
| Elite | 1.0-2.0 | 55-60% | 8-12 hrs |
| Experimental | 1.5-2.5 | 58-62% | 24+ hrs |

## 🎯 Optimization Tips

### For Limited GPU Memory
```python
config = {
    'd_model': 256,  # Reduce width
    'num_layers': 4,  # Reduce depth
    'batch_size': 32,  # Smaller batches
    'grad_accum_steps': 4,  # Simulate larger batch
}
```

### For Speed
```python
config = {
    'seq_len': 50,  # Shorter sequences
    'num_layers': 4,  # Fewer layers
    'batch_size': 128,  # Larger batches
    'num_workers': 8,  # More data workers
}
```

### For Accuracy
```python
config = {
    'seq_len': 100,  # Longer context
    'num_layers': 8,  # Deeper model
    'num_epochs': 200,  # More training
    'patience': 30,  # More patience
}
```

## 🚨 Common Issues

### Model predicts ~0.5 for everything
**Symptoms**: All entry_prob around 0.48-0.52
**Solutions**:
1. Increase confidence penalty weight (0.1 → 0.5)
2. Use focal loss (already implemented)
3. Filter low-signal data (already implemented)
4. Increase learning rate (0.0001 → 0.0003)
5. Train longer (100 → 200 epochs)

### Out of memory errors
**Solutions**:
1. Reduce batch_size (128 → 64 → 32)
2. Reduce d_model (512 → 256)
3. Reduce seq_len (100 → 50)
4. Enable gradient checkpointing
5. Use mixed precision (already enabled)

### Loss not decreasing
**Solutions**:
1. Check data quality (NaN, Inf)
2. Reduce learning rate
3. Increase warmup period
4. Check label distribution
5. Verify feature extraction

### Overfitting (low train loss, high val loss)
**Solutions**:
1. Increase dropout (0.1 → 0.2)
2. Increase weight_decay (1e-5 → 1e-4)
3. Reduce model size
4. Add more training data
5. Early stopping (already enabled)

## 📈 Monitoring Training

Watch these metrics:
- **Entry prob std**: Should be >0.15 (wider spread = better)
- **Validation loss**: Should decrease steadily
- **Win rate**: Should exceed 52%
- **Confidence penalty**: Should decrease
- **Temperature values**: Should stabilize around 1.0-3.0

## 🎓 Next Steps

1. **Start with Elite Config**: Use recommended settings
2. **Monitor Training**: Watch wandb dashboard
3. **Analyze Results**: Use analyze_model.py
4. **Backtest**: Test on unseen data
5. **Iterate**: Adjust based on results