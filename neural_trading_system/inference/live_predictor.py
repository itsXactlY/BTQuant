import torch
import numpy as np
from typing import Dict, Tuple
from collections import deque

class LivePredictor:
    """
    Real-time prediction engine for live trading.
    Optimized for minimal latency.
    """
    
    def __init__(
        self,
        model_path: str,
        feature_extractor,
        seq_len: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = self._load_model(model_path, device)
        self.feature_extractor = feature_extractor
        self.seq_len = seq_len
        self.device = device
        
        # Circular buffer for features
        self.feature_buffer = deque(maxlen=seq_len)
        
        # Cache for faster inference
        self.last_prediction = None
        self.prediction_cache_size = 0
        
        print(f"🚀 LivePredictor initialized on {device}")
    
    def _load_model(self, model_path: str, device: str):
        """Load and prepare model for inference."""
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        from models.architecture import NeuralTradingModel
        
        model = NeuralTradingModel(
            feature_dim=config.get('feature_dim', 500),
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 1024),
            dropout=0.0,
            latent_dim=config.get('latent_dim', 8),
            seq_len=config.get('seq_len', 100)
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Optional: Convert to TorchScript for faster inference
        # model = torch.jit.script(model)
        
        return model
    
    def update(self, indicator_data: Dict[str, np.ndarray]):
        """
        Update with new bar data.
        
        Args:
            indicator_data: Dict with all indicator values and internals
        """
        # Extract features from current state
        features = self.feature_extractor.extract_all_features(indicator_data)
        features = self.feature_extractor.transform(features)
        
        # Add to buffer
        self.feature_buffer.append(features)
    
    def predict(self) -> Dict[str, float]:
        """
        Get current predictions.
        
        Returns:
            Dict with:
                - entry_prob: float
                - exit_prob: float
                - expected_return: float
                - volatility_forecast: float
                - position_size: float
                - regime_embedding: np.ndarray
                - confidence_score: float (meta-metric)
        """
        # Need full sequence
        if len(self.feature_buffer) < self.seq_len:
            return None
        
        # Prepare input
        feature_sequence = np.array(list(self.feature_buffer))  # [seq_len, feature_dim]
        feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(feature_tensor)
        
        # Extract values
        result = {
            'entry_prob': predictions['entry_prob'].cpu().item(),
            'exit_prob': predictions['exit_prob'].cpu().item(),
            'expected_return': predictions['expected_return'].cpu().item(),
            'volatility_forecast': predictions['volatility_forecast'].cpu().item(),
            'position_size': predictions['position_size'].cpu().item(),
            'regime_embedding': predictions['regime_z'].cpu().numpy()[0],
        }
        
        # Compute confidence score (meta-metric combining multiple signals)
        confidence = self._compute_confidence(predictions)
        result['confidence_score'] = confidence
        
        self.last_prediction = result
        return result
    
    def _compute_confidence(self, predictions: Dict) -> float:
        """
        Compute overall confidence in the prediction.
        
        Combines multiple factors:
        - Entry probability certainty (far from 0.5)
        - Low predicted volatility
        - Expected return magnitude
        - Regime stability (low variance in latent space)
        """
        entry_prob = predictions['entry_prob'].item()
        vol_forecast = predictions['volatility_forecast'].item()
        exp_return = abs(predictions['expected_return'].item())
        
        # Entry certainty (0.5 = uncertain, 0 or 1 = certain)
        entry_certainty = abs(entry_prob - 0.5) * 2  # Scale to [0, 1]
        
        # Volatility confidence (lower vol = higher confidence)
        vol_confidence = 1.0 / (1.0 + vol_forecast)
        
        # Return magnitude
        return_confidence = min(exp_return * 10, 1.0)  # Cap at 1.0
        
        # Weighted combination
        confidence = (
            0.4 * entry_certainty +
            0.3 * vol_confidence +
            0.3 * return_confidence
        )
        
        return confidence
    
    def get_trade_signal(
        self,
        min_entry_prob: float = 0.6,
        min_confidence: float = 0.5,
        min_expected_return: float = 0.01
    ) -> Tuple[str, Dict]:
        """
        Get actionable trade signal with filters.
        
        Returns:
            signal: 'ENTRY', 'EXIT', or 'HOLD'
            details: Dict with prediction details
        """
        if self.last_prediction is None:
            return 'HOLD', {}
        
        pred = self.last_prediction
        
        # Entry conditions
        if (pred['entry_prob'] > min_entry_prob and
            pred['confidence_score'] > min_confidence and
            pred['expected_return'] > min_expected_return):
            return 'ENTRY', pred
        
        # Exit conditions
        if pred['exit_prob'] > 0.7:
            return 'EXIT', pred
        
        return 'HOLD', pred
    
    def reset(self):
        """Reset the predictor state."""
        self.feature_buffer.clear()
        self.last_prediction = None