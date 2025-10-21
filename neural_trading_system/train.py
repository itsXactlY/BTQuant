import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from data.feature_extractor import IndicatorFeatureExtractor
from data.data_pipeline import DataPipeline
from models.architecture import create_model
from training.trainer import NeuralTrainer, TradingDataset

def main():
    # Configuration
    config = {
        # Data
        'data_path': 'data/btc_4h_indicators.csv',  # Your indicator data
        'seq_len': 100,
        'prediction_horizon': 5,  # Predict 5 bars ahead
        
        # Model architecture
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'latent_dim': 8,
        
        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'lr': 1e-4,
        'min_lr': 1e-6,
        'weight_decay': 1e-5,
        'grad_accum_steps': 4,
        'T_0': 10,  # Cosine annealing restart
        
        # Early stopping
        'patience': 15,
        'save_every': 10,
        
        # Experiment tracking
        'use_wandb': True,
        'run_name': 'neural_trading_v1',
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("🚀 Starting Neural Trading System Training")
    print(f"Device: {config['device']}")
    
    # Initialize feature extractor
    feature_extractor = IndicatorFeatureExtractor(
        lookback_windows=[5, 10, 20, 50, 100]
    )
    
    # Initialize data pipeline
    pipeline = DataPipeline(feature_extractor)
    
    # Check if processed data exists
    processed_path = 'data/processed_features.pkl'
    
    if Path(processed_path).exists():
        print("📁 Loading processed data...")
        data = pipeline.load_processed_data(processed_path)
        features = data['features']
        returns = data['returns']
    else:
        print("📊 Processing raw indicator data...")
        
        # Load indicator data
        indicator_data = pipeline.load_indicator_data(config['data_path'])
        
        # Prepare sequences
        features, returns = pipeline.prepare_sequences(
            indicator_data,
            seq_len=config['seq_len'],
            prediction_horizon=config['prediction_horizon']
        )
        
        # Save processed data
        pipeline.save_processed_data(
            {'features': features, 'returns': returns},
            processed_path
        )
    
    print(f"✅ Feature shape: {features.shape}")
    
    # Update config with feature dimension
    config['feature_dim'] = features.shape[1]
    
    # Fit scaler on training data
    train_end = int(len(features) * 0.7)
    feature_extractor.fit_scaler(features[:train_end])
    
    # Normalize all features
    features_normalized = np.array([
        feature_extractor.transform(f) for f in features
    ])
    
    # Train/val/test split
    (train_features, train_returns), \
    (val_features, val_returns), \
    (test_features, test_returns) = pipeline.train_val_test_split(
        features_normalized, returns
    )
    
    # Create datasets
    train_dataset = TradingDataset(
        train_features, train_returns,
        seq_len=config['seq_len'],
        prediction_horizon=config['prediction_horizon']
    )
    
    val_dataset = TradingDataset(
        val_features, val_returns,
        seq_len=config['seq_len'],
        prediction_horizon=config['prediction_horizon']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # Don't shuffle time series!
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    print("🏗️ Building neural architecture...")
    model = create_model(config['feature_dim'], config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total parameters: {total_params:,}")
    print(f"📊 Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = NeuralTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=config['device']
    )
    
    # Train
    print("\n🎯 Starting training...")
    trainer.train(config['num_epochs'])
    
    # Test set evaluation
    print("\n🧪 Evaluating on test set...")
    test_dataset = TradingDataset(
        test_features, test_returns,
        seq_len=config['seq_len'],
        prediction_horizon=config['prediction_horizon']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Load best model
    trainer.load_checkpoint('best_model.pt')
    
    # Evaluate
    test_loss, test_components, entry_acc, exit_acc = trainer.validate()
    
    print("\n" + "="*50)
    print("📈 FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Entry Accuracy: {entry_acc:.3f}")
    print(f"Exit Accuracy: {exit_acc:.3f}")
    print(f"Loss Components:")
    for name, value in test_components.items():
        print(f"  - {name}: {value:.4f}")
    print("="*50)
    
    # Save feature extractor for inference
    import pickle
    with open('feature_extractor.pkl', 'wb') as f:
        pickle.dump(feature_extractor, f)
    print("\n💾 Saved feature extractor for inference")
    
    print("\n✅ Training pipeline completed!")

if __name__ == '__main__':
    main()