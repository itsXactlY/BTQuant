import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import pickle
from pathlib import Path

class DataPipeline:
    """
    Efficient data pipeline for loading and preprocessing indicator data.
    """
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.feature_cache = {}
    
    def load_indicator_data(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load indicator data from CSV or pickle.
        
        Expected format: CSV with columns like:
        - timestamp
        - close, open, high, low, volume
        - schaff_v1, schaff_v2, schaff_f1, schaff_pf, schaff_f2, schaff_final
        - cyber_cycle, cyber_cycle_smooth
        - rsx, qqe_line, qqe_signal
        - damiani_v, damiani_atr_lag
        - ... etc for all indicator internals
        """
        if filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            df = pd.read_csv(filepath)
            data = {col: df[col].values for col in df.columns}
        
        return data
    
    def prepare_sequences(
        self,
        indicator_data: Dict[str, np.ndarray],
        seq_len: int = 100,
        prediction_horizon: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert indicator data to feature sequences and labels.
        
        Returns:
            features: [num_samples, feature_dim]
            returns: [num_samples] - forward returns for labeling
        """
        # Calculate forward returns
        close_prices = indicator_data['close']
        returns = np.zeros(len(close_prices))
        
        for i in range(len(close_prices) - prediction_horizon):
            returns[i] = (close_prices[i + prediction_horizon] - close_prices[i]) / close_prices[i]
        
        # Extract features for each timestep
        features_list = []
        
        print("Extracting features from indicator data...")
        for i in range(len(close_prices)):
            # Get data up to current point
            current_data = {}
            for key, values in indicator_data.items():
                if key != 'timestamp':
                    current_data[key] = values[:i+1]
            
            # Extract features
            if i >= seq_len:  # Need enough history
                features = self.feature_extractor.extract_all_features(current_data)
                features_list.append(features)
        
        features = np.array(features_list, dtype=np.float32)
        
        # Align returns with features
        returns = returns[seq_len:]
        
        print(f"Extracted {len(features)} feature vectors of dimension {features.shape[1]}")
        
        return features, returns
    
    def train_val_test_split(
        self,
        features: np.ndarray,
        returns: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple:
        """
        Split data chronologically (no shuffling for time series).
        """
        n = len(features)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_features = features[:train_end]
        train_returns = returns[:train_end]
        
        val_features = features[train_end:val_end]
        val_returns = returns[train_end:val_end]
        
        test_features = features[val_end:]
        test_returns = returns[val_end:]
        
        print(f"Train: {len(train_features)} | Val: {len(val_features)} | Test: {len(test_features)}")
        
        return (train_features, train_returns), (val_features, val_returns), (test_features, test_returns)
    
    def save_processed_data(self, data: Dict, filepath: str):
        """Save processed data for faster loading."""
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved processed data to {filepath}")
    
    def load_processed_data(self, filepath: str) -> Dict:
        """Load preprocessed data."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded processed data from {filepath}")
        return data