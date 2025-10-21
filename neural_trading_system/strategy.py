import backtrader as bt
import torch
import numpy as np
from typing import Dict
from models.architecture import NeuralTradingModel
from data.feature_extractor import IndicatorFeatureExtractor

class NeuralTradingStrategy(bt.Strategy):
    """
    Backtrader strategy using the neural network for decisions.
    NO RULES - pure neural inference.
    """
    
    params = dict(
        model_path='best_model.pt',
        seq_len=100,
        feature_dim=None,  # Will be set automatically
        position_size_mode='neural',  # 'neural' or 'fixed'
        fixed_position_size=0.3,
        confidence_threshold=0.6,  # Only enter if entry_prob > this
        stop_loss_atr=2.0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        debug=False
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Indicator setup (same as before, but we'll extract internals)
        self.atr = bt.indicators.ATR(self.data, period=14)
        
        # Initialize all indicators with internal line access
        from backtrader.indicators.CyberCycle import CyberCycle
        from backtrader.indicators.SchaffTrendCycle import SchaffTrendCycle
        from backtrader.indicators.RSX import RSX
        from backtrader.indicators.qqe import QQE
        from backtrader.indicators.HurstExponent import HurstExponent
        from backtrader.indicators.DamianiVolatmeter import DamianiVolatmeter
        # ... import all your indicators
        
        self.cyber_cycle = CyberCycle(self.data)
        self.schaff = SchaffTrendCycle(self.data)
        self.rsx = RSX(self.data)
        self.qqe = QQE(self.data)
        self.hurst = HurstExponent(self.data)
        self.damiani = DamianiVolatmeter(self.data)
        # ... initialize all indicators
        
        # Feature extractor
        self.feature_extractor = IndicatorFeatureExtractor()
        
        # Load trained model
        self.model = self._load_model()
        self.model.eval()  # Inference mode
        
        # History buffer for sequences
        self.feature_history = []
        
        # Trade tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.highest_price = None
        
        self._warmup = 200  # Need history for sequence
    
    def _load_model(self):
        """Load the trained neural model."""
        checkpoint = torch.load(self.p.model_path, map_location=self.p.device)
        
        config = checkpoint.get('config', {})
        feature_dim = config.get('feature_dim', 500)  # Default
        
        model = NeuralTradingModel(
            feature_dim=feature_dim,
            d_model=config.get('d_model', 256),
            num_heads=config.get('num_heads', 8),
            num_layers=config.get('num_layers', 6),
            d_ff=config.get('d_ff', 1024),
            dropout=0.0,  # No dropout in inference
            latent_dim=config.get('latent_dim', 8),
            seq_len=self.p.seq_len
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.p.device)
        
        print(f"✅ Loaded neural model from {self.p.model_path}")
        return model
    
    def _extract_current_features(self) -> np.ndarray:
        """Extract features from current indicator state."""
        # Build indicator data dictionary with all internal lines
        indicator_data = {}
        
        # Schaff internals
        indicator_data['schaff_macd'] = np.array(self.schaff.l.macd.get(size=len(self)))
        indicator_data['schaff_f1'] = np.array(self.schaff.l.f1.get(size=len(self)))
        indicator_data['schaff_pf'] = np.array(self.schaff.l.pf.get(size=len(self)))
        indicator_data['schaff_f2'] = np.array(self.schaff.l.f2.get(size=len(self)))
        indicator_data['schaff_final'] = np.array(self.schaff.l.schaff.get(size=len(self)))
        
        # Cyber Cycle
        indicator_data['cyber_cycle'] = np.array(self.cyber_cycle.get(size=len(self)))
        
        # RSX
        indicator_data['rsx'] = np.array(self.rsx.get(size=len(self)))
        
        # QQE
        indicator_data['qqe_line'] = np.array(self.qqe.qqe_line.get(size=len(self)))
        indicator_data['qqe_signal'] = np.array(self.qqe.qqe_signal.get(size=len(self)))
        
        # Hurst
        indicator_data['hurst'] = np.array(self.hurst.get(size=len(self)))
        
        # Damiani
        indicator_data['damiani_v'] = np.array(self.damiani.v.get(size=len(self)))
        indicator_data['damiani_atr_lag'] = np.array(self.damiani.atr_lag.get(size=len(self)))
        
        # Price data
        indicator_data['close'] = np.array(self.data.close.get(size=len(self)))
        indicator_data['open'] = np.array(self.data.open.get(size=len(self)))
        indicator_data['high'] = np.array(self.data.high.get(size=len(self)))
        indicator_data['low'] = np.array(self.data.low.get(size=len(self)))
        indicator_data['volume'] = np.array(self.data.volume.get(size=len(self)))
        
        # ... add all other indicators and their internal lines
        
        # Extract features
        features = self.feature_extractor.extract_all_features(indicator_data)
        
        return features
    
    def next(self):
        """Main strategy logic - pure neural inference."""
        
        # Warmup period
        if len(self) < self._warmup:
            return
        
        # Extract current features
        current_features = self._extract_current_features()
        self.feature_history.append(current_features)
        
        # Keep only seq_len history
        if len(self.feature_history) > self.p.seq_len:
            self.feature_history = self.feature_history[-self.p.seq_len:]
        
        # Need full sequence
        if len(self.feature_history) < self.p.seq_len:
            return
        
        # Prepare input tensor
        feature_sequence = np.array(self.feature_history)  # [seq_len, feature_dim]
        feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0)  # [1, seq_len, feature_dim]
        feature_tensor = feature_tensor.to(self.p.device)
        
        # Neural inference
        with torch.no_grad():
            predictions = self.model(feature_tensor)
        
        # Extract predictions
        entry_prob = predictions['entry_prob'].cpu().item()
        exit_prob = predictions['exit_prob'].cpu().item()
        expected_return = predictions['expected_return'].cpu().item()
        volatility_forecast = predictions['volatility_forecast'].cpu().item()
        neural_position_size = predictions['position_size'].cpu().item()
        regime_z = predictions['regime_z'].cpu().numpy()[0]
        
        if self.p.debug:
            self.log(f"Neural Output - Entry: {entry_prob:.3f}, Exit: {exit_prob:.3f}, "
                    f"ExpRet: {expected_return:.4f}, Vol: {volatility_forecast:.4f}")
        
        # Trading logic
        if not self.position:
            # ENTRY DECISION - purely based on neural network
            if entry_prob > self.p.confidence_threshold and expected_return > 0:
                # Calculate position size
                if self.p.position_size_mode == 'neural':
                    size_pct = neural_position_size
                else:
                    size_pct = self.p.fixed_position_size
                
                available_cash = self.broker.getcash()
                size = (available_cash * size_pct) / self.data.close[0]
                
                # Set stop loss based on volatility forecast
                volatility_adjusted_stop = max(
                    self.p.stop_loss_atr * volatility_forecast,
                    self.p.stop_loss_atr * self.atr[0]
                )
                self.stop_price = self.data.close[0] - volatility_adjusted_stop
                
                # Calculate target based on expected return
                self.target_price = self.data.close[0] * (1 + abs(expected_return) * 2)
                
                self.buy(size=size)
                self.entry_bar = len(self)
                self.entry_price = self.data.close[0]
                self.highest_price = self.data.close[0]
                
                self.log(f"🚀 NEURAL ENTRY - Confidence: {entry_prob:.2f}, "
                        f"Exp Return: {expected_return:.2%}, "
                        f"Position Size: {size_pct:.2%}, "
                        f"Regime: {regime_z}")
        
        else:
            # EXIT DECISION - neural network decides
            self.highest_price = max(self.highest_price, self.data.high[0])
            
            # Update trailing stop based on volatility
            trailing_stop = self.highest_price - (volatility_forecast * 1.5)
            self.stop_price = max(self.stop_price, trailing_stop)
            
            # Exit conditions
            stop_hit = self.data.close[0] <= self.stop_price
            target_hit = self.data.close[0] >= self.target_price
            neural_exit = exit_prob > 0.7  # Neural network says exit
            
            if stop_hit:
                self.close()
                self.log(f"🛑 EXIT - Stop Loss")
            elif target_hit:
                self.close()
                self.log(f"🎯 EXIT - Target Reached")
            elif neural_exit:
                self.close()
                self.log(f"🧠 EXIT - Neural Signal (confidence: {exit_prob:.2f})")
    
    def notify_order(self, order):
        """Track order execution."""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED @ {order.executed.price:.2f}")
            elif order.issell():
                pnl = order.executed.price - self.entry_price
                pnl_pct = (pnl / self.entry_price) * 100
                self.log(f"SELL EXECUTED @ {order.executed.price:.2f} | P&L: {pnl_pct:.2f}%")
    
    def log(self, txt, dt=None):
        """Logging utility."""
        if self.p.debug:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()} {txt}')