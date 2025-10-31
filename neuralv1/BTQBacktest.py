from backtrader.strategies.base import BaseStrategy
import torch
import numpy as np
import pickle
import json
from architecture_v1_fixed import V1FixedTradingModel

class NeuralV1Strategy(BaseStrategy):
    params = (
        ("model_path", "neural_models/v1_fixed_best.pt"),
        ("scaler_path", "neural_cache/scaler_15.pkl"),
        ("features_path", "neural_cache/selected_features_15.json"),
        ("seq_len", 100),
        ("n_features", 15),
        ("entry_threshold", 0.5),
        ("tp_threshold", 0.5),
        ("sl_threshold", 0.5),
        ("percent_sizer", 0.025),
        ("min_size", 0.1),
        ("maker_fee", 0.0002),
        ("taker_fee", 0.0005),
        ("slippage", 0.0005),
        ('debug', False),
        ("backtest", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(self.p.model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                actual_n_features = checkpoint['config']['n_features']
            else:
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                actual_n_features = state_dict['embed.weight'].shape[1]
        else:
            actual_n_features = checkpoint['embed.weight'].shape[1]

        self.neural_model = V1FixedTradingModel(actual_n_features, self.p.seq_len)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.neural_model.load_state_dict(checkpoint)

        self.neural_model = self.neural_model.to(self.device)
        self.neural_model.eval()

        with open(self.p.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(self.p.features_path, 'r') as f:
            self.selected_features = json.load(f)

        self.features_scaled = None
        self.features_ready = False
        self.bars_held = 0

        print(f"✅ Model loaded: {actual_n_features} features")

    def next(self):

        if not self.features_ready and len(self.data) >= self.p.seq_len + 50:
            self._prepare_features()
            if self.features_scaled is not None:
                self.features_ready = True

        if self.features_ready:
            super().next()

    def _prepare_features(self):

        import polars as pl
        from indicators_v1_fixed import compute_core_indicators

        data_len = len(self.data)

        min_data = self.p.seq_len + 50
        if data_len < min_data:
            return

        df = pl.DataFrame({
            'open': [float(self.data.open[-i]) for i in range(data_len, 0, -1)],
            'high': [float(self.data.high[-i]) for i in range(data_len, 0, -1)],
            'low': [float(self.data.low[-i]) for i in range(data_len, 0, -1)],
            'close': [float(self.data.close[-i]) for i in range(data_len, 0, -1)],
            'volume': [float(self.data.volume[-i]) for i in range(data_len, 0, -1)]
        })

        df_ind = compute_core_indicators(df)

        features_2d = df_ind.select(self.selected_features).to_numpy()
        self.features_scaled = self.scaler.transform(features_2d)

        print(f"✅ Prepared {len(self.features_scaled)} feature vectors")

    def _get_predictions(self):

        if self.features_scaled is None:
            return None

        if len(self.features_scaled) < self.p.seq_len:
            return None

        sequence = self.features_scaled[-self.p.seq_len:]

        with torch.no_grad():
            seq_tensor = torch.tensor(sequence).float().unsqueeze(0).to(self.device)
            position = 1.0 if self.position else 0.0
            context = torch.tensor([[position, self.bars_held]]).float().to(self.device)

            preds = self.neural_model(seq_tensor, context)

            return {
                'entry': torch.sigmoid(preds['entry']).item(),
                'tp': torch.sigmoid(preds['tp']).item(),
                'sl': torch.sigmoid(preds['sl']).item(),
                'exp_return': preds['others'][0, 0].item()
            }

    def buy_or_short_condition(self):

        preds = self._get_predictions()
        if preds is None:
            return False

        if preds['entry'] > self.p.entry_threshold:
            win_prob = preds['entry']
            exp_return = preds['exp_return']
            kelly = (win_prob * exp_return - (1 - win_prob)) / max(abs(exp_return), 1e-6)
            size = np.clip(kelly, self.p.min_size, self.p.percent_sizer)

            self.create_order(action='BUY', size=size)
            self.bars_held = 0

            if self.p.debug:
                print(f"ENTRY @ {self.data.close[0]:.2f} | prob: {preds['entry']:.2%} | size: {size:.2%}")
            return True

        return False

    def dca_or_short_condition(self):
        return False

    def sell_or_cover_condition(self):

        if not self.position:
            return False

        preds = self._get_predictions()
        if preds is None:
            return False

        self.bars_held += 1

        if preds['tp'] > self.p.tp_threshold or preds['sl'] > self.p.sl_threshold:
            for order_tracker in list(self.active_orders):
                self.close_order(order_tracker)

            reason = "TP" if preds['tp'] > self.p.tp_threshold else "SL"
            if self.p.debug:
                print(f"EXIT @ {self.data.close[0]:.2f} | {reason} | held: {self.bars_held}")

            self.bars_held = 0
            return True

        return False

if __name__ == '__main__':
    from backtrader.utils.backtest import backtest
    try:
        backtest(
            NeuralV1Strategy,
            coin='BTC',
            collateral='USDT',
            start_date="2020-01-01",
            end_date="2024-12-31",
            interval="1h",
            init_cash=10000,
            plot=True
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
