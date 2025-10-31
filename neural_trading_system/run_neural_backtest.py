
\
\
\
\
\

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pickle
import json
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import datetime

import backtrader as bt
from backtrader.utils.backtest import PolarsDataLoader, DataSpec
from backtrader.TransparencyPatch import activate_patch, capture_patch, optimized_patch

from neural_pipeline import DataCollectionStrategy

console = Console()

class FastUnifiedNeuralStrategy(DataCollectionStrategy):
\
\

    params = dict(
        model_path='models/best_exit_aware_model.pt',
        feature_extractor_path='models/exit_aware_feature_extractor.pkl',
        config_path='models/exit_aware_config.json',

        min_entry_prob=0.35,
        min_expected_return=0.01,

        min_warmup_bars=500,
        feature_check_interval=100,
        prediction_interval=5,

        debug=True,
    )

    def __init__(self):
        super().__init__()

        console.print("[cyan]📦 Loading training configuration...[/cyan]")
        with open(self.p.config_path, 'r') as f:
            self.config = json.load(f)

        console.print(f"[yellow]📋 Config loaded: {self.config.get('run_name', 'unknown')}[/yellow]")

        self.expected_features = self.config['feature_dim']
        self.seq_len = self.config['seq_len']
        self.d_model = self.config['d_model']
        self.num_heads = self.config['num_heads']
        self.num_layers = self.config['num_layers']
        self.d_ff = self.config['d_ff']
        self.dropout = self.config['dropout']

        console.print(
            f"[yellow]🏗️  Architecture: feature_dim={self.expected_features}, "
            f"d_model={self.d_model}, layers={self.num_layers}, seq_len={self.seq_len}[/yellow]"
        )

        console.print(f"[cyan]📦 Loading model from {self.p.model_path}[/cyan]")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(self.p.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        from models.architecture import create_model
        model_cfg = {
            'seq_len': self.seq_len,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout
        }
        self.model = create_model(self.expected_features, model_cfg).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        console.print(f"[cyan]📦 Loading feature extractor[/cyan]")
        with open(self.p.feature_extractor_path, 'rb') as f:
            self.feature_extractor = pickle.load(f)

        expected_scaler_features = getattr(self.feature_extractor.scaler, 'n_features_in_', None)
        if expected_scaler_features and expected_scaler_features != self.expected_features:
            console.print(
                f"[red]⚠️  Scaler mismatch: {expected_scaler_features} vs {self.expected_features}[/red]"
            )

        self.feature_buffer = []
        self.prediction_counter = 0
        self.last_prediction = None
        self.warmup_complete = False
        self.feature_ready_bar = None

        self.entry_bar = None
        self.entry_price = None

        self.total_predictions = 0
        self.feature_checks = 0
        self.last_feature_count = 0

        console.print("[green]✅ Fast unified strategy initialized[/green]")

    def _fast_feature_check(self):
\
\
\

        try:
            batch = optimized_patch.current_batch
            if not batch:
                return False, 0

            available_keys = sum(1 for key in batch[0].keys() if key not in ('bar', 'datetime'))

            estimated_features = available_keys * 279

            return estimated_features >= self.expected_features, estimated_features

        except:
            return False, 0

    def _extract_features(self):
\
\

        try:
            batch = optimized_patch.current_batch
            if not batch:
                return None

            indicator_arrays = {}
            for key in batch[0].keys():
                if key not in ('bar', 'datetime'):
                    try:
                        values = [b.get(key, 0.0) for b in batch]
                        values = [float(v) if v is not None and not np.isnan(v) else 0.0 for v in values]
                        indicator_arrays[key] = np.array(values, dtype=np.float32)
                    except:
                        continue

            if not indicator_arrays:
                return None

            features = self.feature_extractor.extract_all_features(indicator_arrays)
            features = np.asarray(features, dtype=np.float32).ravel()

            if len(features) != self.expected_features:
                self.last_feature_count = len(features)
                return None

            features_scaled = self.feature_extractor.transform(features.reshape(1, -1)).ravel()

            if not self.warmup_complete:
                self.warmup_complete = True
                self.feature_ready_bar = len(self)
                console.print(
                    f"[green bold]✅ All {self.expected_features} features ready @ bar {len(self)}![/green bold]"
                )

            return features_scaled

        except Exception as e:
            if self.p.debug:
                console.print(f"[red]Feature extraction error @ bar {len(self)}: {e}[/red]")
            return None

    def _predict(self):

        try:
            seq = np.array(self.feature_buffer, dtype=np.float32)
            if seq.shape[0] != self.seq_len or seq.shape[1] != self.expected_features:
                return None

            input_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)

            if self.position:
                bars_in_position = len(self) - self.entry_bar if self.entry_bar else 0
                current_price = float(self.data.close[0])
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price else 0.0
            else:
                bars_in_position = 0
                unrealized_pnl = 0.0

            position_context = {
                'unrealized_pnl': torch.tensor([[unrealized_pnl]], dtype=torch.float32, device=self.device),
                'time_in_position': torch.tensor([[bars_in_position]], dtype=torch.float32, device=self.device)
            }

            with torch.no_grad():
                out = self.model(input_tensor, position_context=position_context)

            self.total_predictions += 1

            return {
                'entry_prob': float(out['entry_prob'].cpu().item()),
                'exit_prob': float(out['unified_exit_prob'].cpu().item()),
                'expected_return': float(out['expected_return'].cpu().item()),
                'position_size': float(out['position_size'].cpu().item()),
            }

        except Exception as e:
            if self.p.debug:
                console.print(f"[red]Prediction error: {e}[/red]")
            return None

    def next(self):
\
\

        capture_patch(self)

        if len(self) < self.p.min_warmup_bars:
            return

        if not self.warmup_complete:

            if (len(self) % self.p.feature_check_interval) == 0:
                is_ready, est_count = self._fast_feature_check()
                self.feature_checks += 1

                if self.p.debug:
                    progress = (est_count / self.expected_features) * 100
                    console.print(
                        f"[yellow]⏳ Check #{self.feature_checks} @ bar {len(self)}: "
                        f"~{est_count}/{self.expected_features} features (~{progress:.0f}%)[/yellow]"
                    )

                if is_ready:
                    f = self._extract_features()
                    if f is not None:

                        self.feature_buffer.append(f)
                        if len(self.feature_buffer) > self.seq_len:
                            self.feature_buffer.pop(0)
            return

        f = self._extract_features()
        if f is None:
            return

        self.feature_buffer.append(f)
        if len(self.feature_buffer) > self.seq_len:
            self.feature_buffer.pop(0)

        if len(self.feature_buffer) < self.seq_len:
            return

        if (self.prediction_counter % self.p.prediction_interval) == 0:
            self.last_prediction = self._predict()
        self.prediction_counter += 1

        if not self.last_prediction:
            return

        if not self.position:
            self._check_entry(self.last_prediction)
        else:
            self._check_exit(self.last_prediction)

    def _check_entry(self, pred):

        if pred['entry_prob'] < self.p.min_entry_prob:
            return
        if pred['expected_return'] < self.p.min_expected_return:
            return

        pos_size = max(0.1, min(0.5, pred['position_size']))
        cash = self.broker.getcash()
        price = float(self.data.close[0])
        size = (cash * pos_size) / price

        if size <= 0:
            return

        self.buy(size=size)
        self.entry_bar = len(self)
        self.entry_price = price

        if self.p.debug:
            console.print(
                f"[green]🚀 ENTRY @ {price:.2f}[/green] | "
                f"prob={pred['entry_prob']:.3f} | "
                f"exp={pred['expected_return']:+.4f} | "
                f"size={pos_size:.1%}"
            )

    def _check_exit(self, pred):

        if not self.position or not self.entry_price:
            return

        current_price = float(self.data.close[0])
        pnl_pct = (current_price - self.entry_price) / self.entry_price
        bars_held = len(self) - self.entry_bar if self.entry_bar else 0

        exit_prob = pred['exit_prob']

        if pnl_pct > 0:
            should_exit = (
                exit_prob > 0.30 or
                (pnl_pct > 0.05 and exit_prob > 0.20) or
                pnl_pct > 0.10 or
                bars_held > 500
            )
        else:
            should_exit = (
                exit_prob > 0.40 or
                pnl_pct < -0.03 or
                bars_held > 300
            )

        if should_exit:
            self.close()

            if self.p.debug:
                console.print(
                    f"[red]🛑 EXIT @ {current_price:.2f}[/red] | "
                    f"exit_prob={exit_prob:.3f} | "
                    f"pnl={pnl_pct:+.2%} | "
                    f"bars={bars_held}"
                )

            self.entry_bar = None
            self.entry_price = None

    def stop(self):

        console.print("\n" + "="*60)
        console.print("[cyan bold]📊 Strategy Statistics[/cyan bold]")
        console.print("="*60)

        if self.feature_ready_bar:
            console.print(f"[green]✅ Features ready at bar: {self.feature_ready_bar}[/green]")
        else:
            console.print(f"[red]⚠️  Features never fully ready![/red]")

        console.print(f"[yellow]Feature checks during warmup: {self.feature_checks}[/yellow]")
        console.print(f"[yellow]Total bars processed: {len(self)}[/yellow]")
        console.print(f"[cyan]Total predictions: {self.total_predictions}[/cyan]")
        console.print("="*60 + "\n")

def run_unified_backtest(
    coin='BTC',
    interval='15m',
    start_date='2023-01-01',
    end_date='2024-01-01',
    collateral='USDT',
    init_cash=10000.0,
):
\
\

    console.print(Panel.fit(
        f"[bold cyan]⚡ FAST UNIFIED BACKTEST[/bold cyan]\n"
        f"[yellow]Config-Driven Architecture[/yellow]\n\n"
        f"{coin}/{collateral} @ {interval}\n"
        f"{start_date} → {end_date}\n"
        f"${init_cash:,.2f}",
        title="🚀 Neural Backtest",
        border_style="cyan"
    ))

    activate_patch(debug=False)

    cerebro = bt.Cerebro(
        oldbuysell=True,
        runonce=False,
        stdstats=False,
        exactbars=False,
    )

    console.print("\n[cyan]📊 Loading data...[/cyan]")
    loader = PolarsDataLoader()
    spec = DataSpec(
        symbol=coin,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        collateral=collateral
    )
    df = loader.load_data(spec, use_cache=True)
    console.print(f"[green]✅ Loaded {len(df)} bars[/green]")

    data_feed = loader.make_backtrader_feed(df, spec)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(FastUnifiedNeuralStrategy)

    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    console.print("\n[bold cyan]🔄 Running backtest...[/bold cyan]\n")
    start_time = datetime.now()

    results = cerebro.run()
    strat = results[0]

    duration = (datetime.now() - start_time).total_seconds()

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - init_cash) / init_cash * 100

    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    table = Table(title="📊 Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Value", justify="right", style="yellow", width=20)

    table.add_row("Initial Cash", f"${init_cash:,.2f}")
    table.add_row("Final Value", f"${final_value:,.2f}")
    table.add_row("Total Return", f"{total_return:+.2f}%")
    table.add_row("Sharpe Ratio", f"{sharpe:.3f}" if sharpe else "N/A")
    table.add_row("Max Drawdown", f"{drawdown.get('max',{}).get('drawdown', 0):.2f}%")

    total_trades = trades.get('total', {}).get('total', 0)
    won = trades.get('won', {}).get('total', 0)

    table.add_row("─" * 25, "─" * 20)
    table.add_row("Total Trades", str(total_trades))

    if total_trades > 0:
        win_rate = (won / total_trades) * 100
        table.add_row("Win Rate", f"{win_rate:.1f}%")

    table.add_row("─" * 25, "─" * 20)
    table.add_row("Execution Time", f"{duration:.1f}s")

    console.print("\n")
    console.print(table)
    console.print("\n")

    return {
        'final_value': final_value,
        'total_return': total_return,
        'sharpe': sharpe,
        'duration': duration
    }

if __name__ == '__main__':
    results = run_unified_backtest(
        coin='BTC',
        interval='15m',
        start_date='2023-01-01',
        end_date='2024-01-01',
        collateral='USDT',
        init_cash=10000.0
    )
