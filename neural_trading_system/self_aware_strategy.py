
"""
Pure Self-Aware Strategy - ZERO Hardcoded Rules

The model has COMPLETE autonomy over trading decisions:
- Entry: Model decides when (no threshold)
- Exit: Model decides when (TP/SL/Hold ensemble)
- Position sizing: Model confidence-based
- Hold duration: Model decides (no max bars)

Philosophy: Trust the neural network completely.
"""

import backtrader as bt
import numpy as np
import torch

class PureSelfAwareStrategy(bt.Strategy):
    """
    Pure neural decision-making strategy.

    NO hardcoded rules. Model outputs are directly translated to actions:
    - entry_prob > exit_prob → Enter
    - tp_prob is highest → Take profit
    - sl_prob is highest → Stop loss
    - hold_prob is highest → Hold

    Position sizing scales with model confidence.
    """

    params = (
        ('model', None),
        ('scaler', None),
        ('feature_extractor', None),
        ('max_position_size', 0.95),  # Safety: max 95% of capital
        ('seq_len', 100),
        ('min_confidence', 0.51),  # Only filter: must be >50% (better than random)
    )

    def __init__(self):
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.entry_confidence = None

        # Track decisions for analysis
        self.decisions = []

        print("     🧠 Pure Self-Aware Neural Strategy                          ")
        print("     🎯 ZERO hardcoded rules - model has full autonomy           ")
        print(f"    Min confidence: {self.params.min_confidence} (>random)      ")
        print(f"    Max position size: {self.params.max_position_size}          ")

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                self.entry_bar = len(self)
                self.log(
                    f'BUY @ {order.executed.price:.2f} '
                    f'(confidence: {self.entry_confidence:.2%})'
                )
            elif order.issell():
                pnl = order.executed.price - self.entry_price if self.entry_price else 0
                pnl_pct = (pnl / self.entry_price * 100) if self.entry_price else 0
                bars_held = len(self) - self.entry_bar if self.entry_bar else 0
                self.log(
                    f'SELL @ {order.executed.price:.2f} '
                    f'| PnL: {pnl_pct:+.2f}% | Held: {bars_held} bars'
                )
                self.entry_price = None
                self.entry_bar = None
                self.entry_confidence = None

        self.order = None

    def get_model_predictions(self):
        """Get raw model predictions (no filtering)."""
        if len(self) < self.params.seq_len:
            return None

        # Extract OHLCV sequence
        closes = np.array([self.data.close[-i] for i in range(self.params.seq_len, 0, -1)])
        opens = np.array([self.data.open[-i] for i in range(self.params.seq_len, 0, -1)])
        highs = np.array([self.data.high[-i] for i in range(self.params.seq_len, 0, -1)])
        lows = np.array([self.data.low[-i] for i in range(self.params.seq_len, 0, -1)])
        volumes = np.array([self.data.volume[-i] for i in range(self.params.seq_len, 0, -1)])

        ohlcv = np.column_stack([opens, highs, lows, closes, volumes])

        # Feature extraction & normalization
        features = self.params.feature_extractor.extract_features(ohlcv)
        features_scaled = self.params.scaler.transform(features.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)

        # Get predictions
        with torch.no_grad():
            preds = self.params.model(features_tensor)

        return {
            'entry_prob': preds['entry_prob'].item(),
            'exit_prob': preds['exit_prob'].item(),
            'tp_prob': preds['take_profit_prob'].item(),
            'sl_prob': preds['stop_loss_prob'].item(),
            'hold_prob': preds['hold_prob'].item(),
            'expected_return': preds['expected_return'].item(),
            'volatility_forecast': preds['volatility_forecast'].item()
        }

    def next(self):
        if self.order:
            return

        preds = self.get_model_predictions()
        if preds is None:
            return

        current_price = self.data.close[0]

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # POSITION MANAGEMENT: Let model decide
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if self.position:
            # Calculate actual P&L
            actual_return = (current_price - self.entry_price) / self.entry_price
            bars_held = len(self) - self.entry_bar

            # Model's exit preferences (which signal is strongest?)
            exit_probs = {
                'take_profit': preds['tp_prob'],
                'stop_loss': preds['sl_prob'],
                'hold': preds['hold_prob']
            }

            # Winner-takes-all: highest probability decides action
            max_prob = max(exit_probs.values())
            action = max(exit_probs, key=exit_probs.get)

            # Only enforce min confidence (must be >50% = better than random)
            if max_prob > self.params.min_confidence:

                if action == 'take_profit':
                    self.log(
                        f'🎯 MODEL TP: {preds["tp_prob"]:.2%} '
                        f'(actual: {actual_return:+.2%}, held: {bars_held})'
                    )
                    self.order = self.sell()
                    self.decisions.append({
                        'action': 'tp',
                        'confidence': preds['tp_prob'],
                        'return': actual_return,
                        'bars_held': bars_held
                    })

                elif action == 'stop_loss':
                    self.log(
                        f'🛑 MODEL SL: {preds["sl_prob"]:.2%} '
                        f'(actual: {actual_return:+.2%}, held: {bars_held})'
                    )
                    self.order = self.sell()
                    self.decisions.append({
                        'action': 'sl',
                        'confidence': preds['sl_prob'],
                        'return': actual_return,
                        'bars_held': bars_held
                    })

                # hold: do nothing (model wants to stay)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ENTRY LOGIC: Pure model decision
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        else:
            # Model wants to enter if entry_prob > exit_prob
            entry_signal = preds['entry_prob'] > preds['exit_prob']

            # Only filter: entry confidence must be >50%
            if entry_signal and preds['entry_prob'] > self.params.min_confidence:

                # Position sizing based on model confidence
                # Higher confidence → larger position
                confidence_multiplier = (preds['entry_prob'] - 0.5) / 0.5
                position_size = self.params.max_position_size * confidence_multiplier

                size = (self.broker.getcash() * position_size) / current_price

                self.entry_confidence = preds['entry_prob']
                self.log(
                    f'🚀 MODEL ENTRY: {preds["entry_prob"]:.2%} '
                    f'(exp_ret: {preds["expected_return"]:+.2%}, size: {position_size:.1%})'
                )
                self.order = self.buy(size=size)
                self.decisions.append({
                    'action': 'entry',
                    'confidence': preds['entry_prob'],
                    'expected_return': preds['expected_return'],
                    'position_size': position_size
                })

    def stop(self):
        """Analyze model decisions at end of backtest."""
        print("\n" + "="*80)
        print("🎯 Pure Self-Aware Strategy - Decision Analysis")
        print("="*80)

        if not self.decisions:
            print("No trades executed")
            return

        # Separate by action type
        entries = [d for d in self.decisions if d['action'] == 'entry']
        tps = [d for d in self.decisions if d['action'] == 'tp']
        sls = [d for d in self.decisions if d['action'] == 'sl']

        print(f"\n📊 Trading Activity:")
        print(f"   Entries: {len(entries)}")
        print(f"   Take Profits: {len(tps)}")
        print(f"   Stop Losses: {len(sls)}")

        if entries:
            print(f"\n🚀 Entry Decisions:")
            print(f"   Avg confidence: {np.mean([e['confidence'] for e in entries]):.2%}")
            print(f"   Avg exp return: {np.mean([e['expected_return'] for e in entries]):.2%}")
            print(f"   Avg position size: {np.mean([e['position_size'] for e in entries]):.1%}")

        if tps:
            print(f"\n🎯 Take Profit Decisions:")
            print(f"   Avg confidence: {np.mean([t['confidence'] for t in tps]):.2%}")
            print(f"   Avg return: {np.mean([t['return'] for t in tps]):+.2%}")
            print(f"   Avg hold time: {np.mean([t['bars_held'] for t in tps]):.0f} bars")

        if sls:
            print(f"\n🛑 Stop Loss Decisions:")
            print(f"   Avg confidence: {np.mean([s['confidence'] for s in sls]):.2%}")
            print(f"   Avg loss: {np.mean([s['return'] for s in sls]):+.2%}")
            print(f"   Avg hold time: {np.mean([s['bars_held'] for s in sls]):.0f} bars")

        print("="*80)