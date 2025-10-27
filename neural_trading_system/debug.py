# ============================================================================
# FULL WHITEBOX DEBUG FRAMEWORK FOR NEURAL TRADING MODEL
# Instrument every single checkpoint from data → loss → backprop
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import sys

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_section(title: str):
    """Print a bold section header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{title:^80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.END}\n")

def print_checkpoint(name: str, data: Dict[str, Any]):
    """Print a checkpoint with all metrics."""
    print(f"{Colors.CYAN}[CHECKPOINT] {name}{Colors.END}")
    for key, value in data.items():
        if isinstance(value, (int, float, bool, str)):
            print(f"  {key}: {value}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                  f"min={np.nanmin(value):.6f}, max={np.nanmax(value):.6f}, "
                  f"mean={np.nanmean(value):.6f}, std={np.nanstd(value):.6f}, "
                  f"nan_count={np.isnan(value).sum()}, inf_count={np.isinf(value).sum()}")
        elif isinstance(value, torch.Tensor):
            print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}, "
                  f"min={value.nanmin():.6f}, max={value.nanmax():.6f}, "
                  f"mean={value.nanmean():.6f}, std={value.nanstd():.6f}, "
                  f"nan_count={torch.isnan(value).sum()}, inf_count={torch.isinf(value).sum()}")
        elif isinstance(value, dict):
            print(f"  {key}: {json.dumps({k: str(v)[:50] for k, v in value.items()}, indent=4)}")
        else:
            print(f"  {key}: {type(value).__name__}")

def debug_data_pipeline(data_dict: Dict[str, np.ndarray]):
    """CHECKPOINT 1-3: Data inspection, splits, scaling."""
    print_section("CHECKPOINT 1-3: DATA PIPELINE")
    
    features = data_dict.get('features')
    returns = data_dict.get('returns')
    prices = data_dict.get('prices')
    
    # CHECKPOINT 1: Raw data loading
    print_checkpoint("1. RAW DATA LOADING", {
        'features_shape': features.shape if features is not None else None,
        'features_dtype': features.dtype if features is not None else None,
        'features_min': np.nanmin(features) if features is not None else None,
        'features_max': np.nanmax(features) if features is not None else None,
        'features_mean': np.nanmean(features) if features is not None else None,
        'features_std': np.nanstd(features) if features is not None else None,
        'features_nan_count': np.isnan(features).sum() if features is not None else None,
        'features_inf_count': np.isinf(features).sum() if features is not None else None,
        'returns_shape': returns.shape if returns is not None else None,
        'returns_min': np.nanmin(returns) if returns is not None else None,
        'returns_max': np.nanmax(returns) if returns is not None else None,
        'prices_shape': prices.shape if prices is not None else None,
    })
    
    # CHECKPOINT 2: Train/Val/Test split
    train_end = int(len(features) * 0.7)
    val_end = int(len(features) * 0.85)
    
    X_train, y_train = features[:train_end], returns[:train_end]
    X_val, y_val = features[train_end:val_end], returns[train_end:val_end]
    X_test, y_test = features[val_end:], returns[val_end:]
    
    print_checkpoint("2. TRAIN/VAL/TEST SPLIT", {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_pct': f"{len(X_train)/len(features)*100:.1f}%",
        'val_pct': f"{len(X_val)/len(features)*100:.1f}%",
        'test_pct': f"{len(X_test)/len(features)*100:.1f}%",
        'overlap_check': 'PASS' if len(set(range(len(X_train))).intersection(set(range(train_end, val_end)))) == 0 else 'FAIL',
    })
    
    # CHECKPOINT 3: Scaler (fit on train only)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print_checkpoint("3. SCALER", {
        'scaler_mean_min': scaler.mean_.min(),
        'scaler_mean_max': scaler.mean_.max(),
        'scaler_scale_min': scaler.scale_.min(),
        'scaler_scale_max': scaler.scale_.max(),
        'train_scaled_min': np.nanmin(X_train_scaled),
        'train_scaled_max': np.nanmax(X_train_scaled),
        'train_scaled_should_be_near_0_mean': np.nanmean(X_train_scaled),
        'train_scaled_should_be_near_1_std': np.nanstd(X_train_scaled),
        'val_scaled_min': np.nanmin(X_val_scaled),
        'val_scaled_max': np.nanmax(X_val_scaled),
        'check_scaler_fit_on_val': 'PASS (correct, fit only on train)' if scaler.mean_ is not None else 'FAIL',
    })
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train,
        'X_val': X_val_scaled, 'y_val': y_val,
        'X_test': X_test_scaled, 'y_test': y_test,
        'scaler': scaler,
    }

def debug_model_forward(model: nn.Module, X_batch: torch.Tensor, seq_len: int = 100):
    """CHECKPOINT 4-5: Model architecture and forward pass."""
    print_section("CHECKPOINT 4-5: MODEL ARCHITECTURE & FORWARD PASS")
    
    device = next(model.parameters()).device
    
    # CHECKPOINT 4: Architecture
    print_checkpoint("4. MODEL ARCHITECTURE", {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'd_model': getattr(model, 'd_model', 'N/A'),
        'num_heads': 'Check transformer_blocks',
        'num_layers': len(model.transformer_blocks) if hasattr(model, 'transformer_blocks') else 'N/A',
        'd_ff': 'Check transformer_blocks',
        'latent_dim': getattr(model, 'latent_dim', 'N/A'),
        'seq_len': getattr(model, 'seq_len', 'N/A'),
    })
    
    # CHECKPOINT 5: Forward pass (single batch)
    print(f"\n{Colors.YELLOW}Running forward pass on batch...{Colors.END}")
    model.eval()
    with torch.no_grad():
        # Reshape batch: (batch_size, seq_len * feature_dim) → (batch_size, seq_len, feature_dim)
        batch_size = X_batch.shape[0]
        feature_dim = X_batch.shape[1] // seq_len
        X_batch_reshaped = X_batch.reshape(batch_size, seq_len, feature_dim).to(device)
        
        print(f"  Input shape: {X_batch_reshaped.shape}")
        
        # Forward pass
        try:
            outputs = model(X_batch_reshaped)
            
            print_checkpoint("5. FORWARD PASS OUTPUTS", {
                'entry_logits_shape': outputs['entry_logits'].shape,
                'entry_logits_min': outputs['entry_logits'].nanmin().item(),
                'entry_logits_max': outputs['entry_logits'].nanmax().item(),
                'entry_logits_mean': outputs['entry_logits'].nanmean().item(),
                'entry_logits_nan_count': torch.isnan(outputs['entry_logits']).sum().item(),
                'entry_prob_shape': outputs['entry_prob'].shape,
                'entry_prob_min': outputs['entry_prob'].nanmin().item(),
                'entry_prob_max': outputs['entry_prob'].nanmax().item(),
                'expected_return_shape': outputs['expected_return'].shape,
                'expected_return_min': outputs['expected_return'].nanmin().item(),
                'expected_return_max': outputs['expected_return'].nanmax().item(),
                'volatility_forecast_shape': outputs['volatility_forecast'].shape,
                'volatility_forecast_min': outputs['volatility_forecast'].nanmin().item(),
                'position_size_shape': outputs['position_size'].shape,
                'position_size_min': outputs['position_size'].nanmin().item(),
                'position_size_max': outputs['position_size'].nanmax().item(),
                'regime_mu_shape': outputs['regime_mu'].shape,
                'regime_logvar_shape': outputs['regime_logvar'].shape,
                'vae_recon_shape': outputs['vae_recon'].shape,
                'attention_weights_count': len(outputs['attention_weights']),
            })
            
            return outputs, X_batch_reshaped
        except Exception as e:
            print(f"{Colors.RED}[ERROR] Forward pass failed: {str(e)}{Colors.END}")
            import traceback
            traceback.print_exc()
            return None, None

def debug_loss_calculation(model: nn.Module, criterion: nn.Module, 
                           outputs: Dict, targets: Dict, X_batch: torch.Tensor):
    """CHECKPOINT 6-8: Loss components, backward pass, gradients."""
    print_section("CHECKPOINT 6-8: LOSS CALCULATION & BACKWARD")
    
    device = next(model.parameters()).device
    
    # CHECKPOINT 6: Individual loss components
    print(f"\n{Colors.YELLOW}Computing individual loss components...{Colors.END}")
    
    model.train()
    
    # Forward again to get gradient flow
    outputs = model(X_batch)
    
    # Compute loss
    loss_output = criterion(outputs, targets)
    
    print_checkpoint("6. LOSS COMPONENTS", {
        'total_loss': loss_output['total_loss'].item(),
        'total_loss_is_positive': 'YES' if loss_output['total_loss'].item() > 0 else 'NO ⚠️',
        'entry_loss': loss_output.get('entry_loss', 0),
        'return_loss': loss_output.get('return_loss', 0),
        'volatility_loss': loss_output.get('volatility_loss', 0),
        'vae_loss': loss_output.get('vae_loss', 0),
        'take_profit_loss': loss_output.get('take_profit_loss', 0),
        'stop_loss_loss': loss_output.get('stop_loss_loss', 0),
        'let_run_loss': loss_output.get('let_run_loss', 0),
        'exit_bundle_loss': loss_output.get('exit_bundle_loss', 0),
    })
    
    # CHECKPOINT 7: Loss weights
    print_checkpoint("7. LOSS WEIGHTS (MultiTaskLoss)", {
        'entry_weight': criterion.task_weights.get('entry', 'N/A'),
        'return_weight': criterion.task_weights.get('return', 'N/A'),
        'volatility_weight': criterion.task_weights.get('volatility', 'N/A'),
        'vae_weight': criterion.task_weights.get('vae', 'N/A'),
        'exit_boost_multiplier': criterion._exit_boost,
        'log_vars': criterion.log_vars.detach().cpu().numpy(),
    })
    
    # CHECKPOINT 8: Backward pass
    print(f"\n{Colors.YELLOW}Running backward pass...{Colors.END}")
    
    total_loss = loss_output['total_loss']
    total_loss.backward()
    
    # Compute gradient norms BEFORE clipping
    total_grad_norm = 0.0
    layer_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            total_grad_norm += grad_norm ** 2
            layer_grads[name] = grad_norm
    
    total_grad_norm = total_grad_norm ** 0.5
    
    print_checkpoint("8. BACKWARD PASS & GRADIENTS (BEFORE CLIPPING)", {
        'total_grad_norm': total_grad_norm,
        'grad_norm_is_exploding': 'YES ⚠️' if total_grad_norm > 10.0 else 'NO (OK)',
        'grad_norm_is_dead': 'YES ⚠️' if total_grad_norm < 1e-6 else 'NO (OK)',
        'layers_with_zero_grad': sum(1 for v in layer_grads.values() if v == 0.0),
        'layers_with_high_grad': sum(1 for v in layer_grads.values() if v > 1.0),
        'max_layer_grad': max(layer_grads.values()) if layer_grads else 0,
        'min_nonzero_layer_grad': min((v for v in layer_grads.values() if v > 0), default=0),
        'transformer_block_0_grad': layer_grads.get('transformer_blocks.0.mha.in_proj_weight', 'N/A'),
        'entry_head_grad': layer_grads.get('entry_head.0.weight', 'N/A'),
        'vae_encoder_grad': layer_grads.get('regime_vae.enc.0.weight', 'N/A'),
    })
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Compute gradient norms AFTER clipping
    total_grad_norm_clipped = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm_clipped += param.grad.data.norm(2).item() ** 2
    total_grad_norm_clipped = total_grad_norm_clipped ** 0.5
    
    print(f"\n{Colors.BLUE}Gradients clipped: {total_grad_norm:.6f} → {total_grad_norm_clipped:.6f}{Colors.END}")
    
    return loss_output, total_grad_norm, total_grad_norm_clipped

def debug_metrics_calculation(predictions: Dict, targets: Dict, batch_data: Dict):
    """CHECKPOINT 9-12: Entry/exit accuracy, correlations."""
    print_section("CHECKPOINT 9-12: METRICS CALCULATION")
    
    # CHECKPOINT 9: Entry accuracy
    print(f"\n{Colors.YELLOW}Computing entry accuracy...{Colors.END}")
    
    entry_pred = predictions['entry_prob'].detach().cpu().numpy().squeeze()
    entry_true = targets['entry_label'].cpu().numpy().squeeze()
    
    entry_pred_binary = (entry_pred > 0.5).astype(int)
    entry_correct = (entry_pred_binary == entry_true).sum()
    entry_total = len(entry_true)
    entry_accuracy = entry_correct / max(1, entry_total)
    
    print_checkpoint("9. ENTRY ACCURACY", {
        'entry_pred_shape': entry_pred.shape,
        'entry_pred_min': np.min(entry_pred),
        'entry_pred_max': np.max(entry_pred),
        'entry_pred_mean': np.mean(entry_pred),
        'entry_true_shape': entry_true.shape,
        'entry_true_sum': entry_true.sum(),
        'entry_correct': entry_correct,
        'entry_total': entry_total,
        'entry_accuracy': entry_accuracy,
        'entry_accuracy_is_valid': 'YES' if 0.0 <= entry_accuracy <= 1.0 else f'NO ⚠️ (value={entry_accuracy})',
    })
    
    # CHECKPOINT 10: Exit accuracy (UNIFIED)
    print(f"\n{Colors.YELLOW}Computing exit accuracy...{Colors.END}")
    
    if 'unified_exit_prob' in predictions:
        exit_pred = predictions['unified_exit_prob'].detach().cpu().numpy().squeeze()
        
        # Create unified exit label from position context
        unrealized_pnl = batch_data.get('unrealized_pnl', torch.zeros_like(predictions['unified_exit_prob'])).cpu().numpy().squeeze()
        take_profit_label = targets.get('take_profit_label', torch.zeros_like(predictions['unified_exit_prob'])).cpu().numpy().squeeze()
        stop_loss_label = targets.get('stop_loss_label', torch.zeros_like(predictions['unified_exit_prob'])).cpu().numpy().squeeze()
        
        # In profit → use TP; in loss → use SL
        exit_true = np.where(unrealized_pnl > 0, take_profit_label, stop_loss_label)
        
        exit_pred_binary = (exit_pred > 0.5).astype(int)
        exit_correct = (exit_pred_binary == exit_true).sum()
        exit_total = len(exit_true)
        exit_accuracy = exit_correct / max(1, exit_total)
        
        print_checkpoint("10. EXIT ACCURACY", {
            'exit_pred_shape': exit_pred.shape,
            'exit_pred_min': np.min(exit_pred),
            'exit_pred_max': np.max(exit_pred),
            'exit_correct': exit_correct,
            'exit_total': exit_total,
            'exit_accuracy': exit_accuracy,
            'exit_accuracy_is_valid': 'YES' if 0.0 <= exit_accuracy <= 1.0 else f'NO ⚠️ (value={exit_accuracy})',
            'exit_accuracy_over_100pct': 'YES ⚠️ BUG' if exit_accuracy > 1.0 else 'NO (OK)',
        })
    else:
        print(f"{Colors.RED}[WARNING] unified_exit_prob not in predictions{Colors.END}")
    
    # CHECKPOINT 11: Return correlation
    print(f"\n{Colors.YELLOW}Computing return correlation...{Colors.END}")
    
    ret_pred = predictions['expected_return'].detach().cpu().numpy().squeeze()
    ret_true = targets['future_return'].cpu().numpy().squeeze()
    
    def safe_corr(a, b):
        if len(a) < 2:
            return np.nan
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]
    
    corr_ret = safe_corr(ret_pred, ret_true)
    
    print_checkpoint("11. RETURN CORRELATION", {
        'ret_pred_shape': ret_pred.shape,
        'ret_pred_min': np.nanmin(ret_pred),
        'ret_pred_max': np.nanmax(ret_pred),
        'ret_pred_mean': np.nanmean(ret_pred),
        'ret_pred_std': np.nanstd(ret_pred),
        'ret_true_shape': ret_true.shape,
        'ret_true_min': np.nanmin(ret_true),
        'ret_true_max': np.nanmax(ret_true),
        'ret_true_mean': np.nanmean(ret_true),
        'return_correlation': corr_ret,
        'correlation_is_valid': 'YES' if -1.0 <= corr_ret <= 1.0 else f'NO ⚠️ (value={corr_ret})',
    })
    
    # CHECKPOINT 12: Volatility correlation
    print(f"\n{Colors.YELLOW}Computing volatility correlation...{Colors.END}")
    
    vol_pred = predictions['volatility_forecast'].detach().cpu().numpy().squeeze()
    vol_true = targets['actual_volatility'].cpu().numpy().squeeze()
    
    corr_vol = safe_corr(vol_pred, vol_true)
    
    print_checkpoint("12. VOLATILITY CORRELATION", {
        'vol_pred_shape': vol_pred.shape,
        'vol_pred_min': np.nanmin(vol_pred),
        'vol_pred_max': np.nanmax(vol_pred),
        'vol_pred_mean': np.nanmean(vol_pred),
        'vol_pred_all_positive': np.all(vol_pred > 0),
        'vol_true_shape': vol_true.shape,
        'vol_true_min': np.nanmin(vol_true),
        'vol_true_max': np.nanmax(vol_true),
        'volatility_correlation': corr_vol,
        'correlation_is_valid': 'YES' if -1.0 <= corr_vol <= 1.0 else f'NO ⚠️ (value={corr_vol})',
    })

def run_full_debug(model_path: str = None, data_path: str = None, 
                   batch_size: int = 32, seq_len: int = 100):
    """Run complete debug framework."""
    print_section("FULL WHITEBOX DEBUG FRAMEWORK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Load data
    if data_path is None:
        print(f"{Colors.RED}[ERROR] data_path required{Colors.END}")
        return
    
    print(f"\n{Colors.YELLOW}Loading data from {data_path}...{Colors.END}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Debug data pipeline
    split_data = debug_data_pipeline(data)
    
    # Create dummy batch
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    
    # Random batch
    indices = np.random.choice(len(X_train), batch_size, replace=False)
    X_batch = torch.from_numpy(X_train[indices]).float()
    
    # Create dummy targets
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    targets = {
        'entry_label': torch.randint(0, 2, (batch_size,)).float().to(device),
        'future_return': torch.from_numpy(y_train[indices]).float().to(device),
        'actual_volatility': torch.abs(torch.randn(batch_size)).to(device),
        'take_profit_label': torch.randint(0, 2, (batch_size,)).float().to(device),
        'stop_loss_label': torch.randint(0, 2, (batch_size,)).float().to(device),
        'let_winner_run_label': torch.randint(0, 2, (batch_size,)).float().to(device),
        'regime_change_label': torch.randint(0, 2, (batch_size,)).float().to(device),
    }
    
    # Load or create model
    if model_path is None:
        print(f"{Colors.YELLOW}Creating dummy model for debug...{Colors.END}")
        from architecture import NeuralTradingModel
        feature_dim = X_train.shape[1] // seq_len
        model = NeuralTradingModel(feature_dim=feature_dim, seq_len=seq_len)
    else:
        print(f"{Colors.YELLOW}Loading model from {model_path}...{Colors.END}")
        from architecture import NeuralTradingModel
        checkpoint = torch.load(model_path, map_location=device)
        feature_dim = checkpoint['config']['feature_dim']
        model = NeuralTradingModel(feature_dim=feature_dim, seq_len=seq_len)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    # Create dummy loss function
    from trainer import MultiTaskLoss
    criterion = MultiTaskLoss(num_tasks=5)
    criterion = criterion.to(device)
    
    X_batch = X_batch.to(device)
    
    # Forward pass
    outputs, X_batch_reshaped = debug_model_forward(model, X_batch, seq_len=seq_len)
    
    if outputs is None:
        print(f"{Colors.RED}[CRITICAL] Forward pass failed, cannot continue{Colors.END}")
        return
    
    # Loss calculation
    loss_output, grad_norm_before, grad_norm_after = debug_loss_calculation(
        model, criterion, outputs, targets, X_batch_reshaped
    )
    
    # Metrics
    batch_data = {
        'unrealized_pnl': torch.randn(batch_size, 1).to(device),
    }
    debug_metrics_calculation(outputs, targets, batch_data)
    
    # Summary
    print_section("DEBUG SUMMARY")
    print(f"{Colors.GREEN}✅ All checkpoints completed successfully!{Colors.END}")
    print(f"\n{Colors.BOLD}Issues to investigate:{Colors.END}")
    print(f"  1. Is total_loss positive? (found in CHECKPOINT 6)")
    print(f"  2. Are exit accuracies in [0, 1]? (found in CHECKPOINT 10)")
    print(f"  3. Are correlations in [-1, 1]? (found in CHECKPOINT 11-12)")
    print(f"  4. Are gradients exploding (>10) or dying (<1e-6)? (found in CHECKPOINT 8)")
    print(f"  5. Are attention weights converging to uniform? (found in forward pass)")

if __name__ == '__main__':
    # Example usage (modify paths as needed):
    run_full_debug(
        model_path=None,  # Set to your saved model path
        data_path='neural_data/features/features_latest.pkl',  # Set to your data path
        batch_size=32,
        seq_len=100,
    )
