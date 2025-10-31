import torch
import numpy as np
from typing import Dict, Tuple
from collections import deque

class LivePredictor:
\
\

    def __init__(self, model_path: str, feature_extractor, seq_len: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = self._load_model(model_path, device)
        self.feature_extractor = feature_extractor
        self.seq_len = seq_len
        self.device = device

        self.feature_buffer = deque(maxlen=seq_len)

        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.current_bars_in_position = 0

        print(f"🚀 LivePredictor initialized with exit management on {device}")

    def _load_model(self, model_path: str, device: str):

        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})

        from models.architecture_v2 import NeuralTradingModel

        model = NeuralTradingModel(
            feature_dim=config.get('feature_dim', 500),
            d_model=config.get('d_model', 512),
            num_heads=config.get('num_heads', 16),
            num_layers=config.get('num_layers', 8),
            d_ff=config.get('d_ff', 2048),
            dropout=0.0,
            latent_dim=config.get('latent_dim', 16),
            seq_len=config.get('seq_len', 100)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model

    def update(self, indicator_data: Dict[str, np.ndarray], current_price: float):

        features = self.feature_extractor.extract_all_features(indicator_data)
        features = self.feature_extractor.transform(features)
        self.feature_buffer.append(features)

        if self.in_position:
            self.current_bars_in_position += 1

    def predict(self, current_price: float) -> Dict[str, float]:
\
\

        if len(self.feature_buffer) < self.seq_len:
            return None

        feature_sequence = np.array(list(self.feature_buffer))
        feature_tensor = torch.FloatTensor(feature_sequence).unsqueeze(0).to(self.device)

        position_context = None
        if self.in_position and self.entry_price is not None:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            position_context = {
                'unrealized_pnl': torch.tensor([[unrealized_pnl]], dtype=torch.float32, device=self.device),
                'time_in_position': torch.tensor([[float(self.current_bars_in_position)]], dtype=torch.float32, device=self.device),
            }

        with torch.no_grad():
            predictions = self.model(feature_tensor, position_context=position_context)

        result = {
            'entry_prob': predictions['entry_prob'].cpu().item(),
            'expected_return': predictions['expected_return'].cpu().item(),
            'volatility_forecast': predictions['volatility_forecast'].cpu().item(),
            'position_size': predictions['position_size'].cpu().item(),
            'regime_embedding': predictions['regime_z'].cpu().numpy()[0],
        }

        regime_change = predictions['regime_change']
        result.update({
            'regime_change_score': regime_change['regime_change_score'].cpu().item(),
            'regime_stability': regime_change['stability'].cpu().item(),
            'vol_spike': regime_change['vol_change'].cpu().item(),
            'volume_anomaly': regime_change['volume_anomaly'].cpu().item(),
        })

        exit_signals = predictions['exit_signals']
        if self.in_position:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price

            if unrealized_pnl > 0:

                if exit_signals['profit_taking'] is not None:
                    result.update({
                        'take_profit_prob': exit_signals['profit_taking']['take_profit_prob'].cpu().item(),
                        'momentum_fade': exit_signals['profit_taking']['momentum_fade'].cpu().item(),
                        'resistance_near': exit_signals['profit_taking']['resistance_near'].cpu().item(),
                        'profit_optimal': exit_signals['profit_taking']['profit_optimal'].cpu().item(),
                    })

                if exit_signals['let_winner_run'] is not None:
                    result.update({
                        'hold_score': exit_signals['let_winner_run']['hold_score'].cpu().item(),
                        'trend_strength': exit_signals['let_winner_run']['trend_strength'].cpu().item(),
                        'momentum_accel': exit_signals['let_winner_run']['momentum_accel'].cpu().item(),
                    })

                result['should_exit'] = exit_signals['should_exit_profit'].cpu().item()
                result['exit_reason'] = 'TAKE_PROFIT' if result['should_exit'] > 0.7 else 'HOLD_WINNER'

            else:

                if exit_signals['stop_loss'] is not None:
                    result.update({
                        'stop_loss_prob': exit_signals['stop_loss']['stop_loss_prob'].cpu().item(),
                        'pattern_failure': exit_signals['stop_loss']['pattern_failure'].cpu().item(),
                        'acceleration_down': exit_signals['stop_loss']['acceleration_down'].cpu().item(),
                        'support_break': exit_signals['stop_loss']['support_break'].cpu().item(),
                    })

                result['should_exit'] = exit_signals['should_exit_loss'].cpu().item()
                result['exit_reason'] = 'STOP_LOSS' if result['should_exit'] > 0.7 else 'HOLD_LOSS'

            result['unrealized_pnl'] = unrealized_pnl
            result['bars_in_position'] = self.current_bars_in_position
        else:
            result['should_exit'] = 0.0
            result['exit_reason'] = 'NOT_IN_POSITION'

        result['unified_exit_prob'] = predictions['unified_exit_prob'].cpu().item()

        return result

    def enter_position(self, entry_price: float):

        self.in_position = True
        self.entry_price = entry_price
        self.entry_time = 0
        self.current_bars_in_position = 0
        print(f"📈 Entered position at {entry_price}")

    def exit_position(self, exit_price: float, reason: str):

        if self.in_position and self.entry_price is not None:
            pnl = (exit_price - self.entry_price) / self.entry_price
            print(f"📉 Exited position at {exit_price} | Reason: {reason}")
            print(f"    P&L: {pnl*100:.2f}% | Held for {self.current_bars_in_position} bars")

        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.current_bars_in_position = 0

    def get_trade_signal(
        self,
        current_price: float,
        min_entry_prob: float = 0.65,
        min_confidence: float = 0.5,
        min_expected_return: float = 0.015,
        exit_threshold: float = 0.7
    ) -> Tuple[str, Dict]:
\
\
\
\
\
\

        pred = self.predict(current_price)

        if pred is None:
            return 'HOLD', {}

        if self.in_position:

            if pred['should_exit'] > exit_threshold:
                return 'EXIT', pred

            if pred.get('unrealized_pnl', 0) < 0 and pred['regime_change_score'] > 0.8:
                pred['exit_reason'] = 'REGIME_CHANGE_CUTLOSS'
                return 'EXIT', pred

            if pred.get('hold_score', 0) > 0.75 and pred.get('unrealized_pnl', 0) > 0.02:
                pred['action_detail'] = 'Letting winner run - strong continuation signals'
                return 'HOLD', pred

            return 'HOLD', pred

        if (pred['entry_prob'] > min_entry_prob and
            pred['expected_return'] > min_expected_return and
            pred['regime_stability'] > 0.4):

            return 'ENTRY', pred

        return 'HOLD', pred

    def reset(self):

        self.feature_buffer.clear()
        self.in_position = False
        self.entry_price = None
        self.entry_time = None
        self.current_bars_in_position = 0
