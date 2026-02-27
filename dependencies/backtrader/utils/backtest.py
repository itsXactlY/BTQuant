import backtrader as bt
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import gc
import pandas as pd
import polars as pl
import hashlib
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table


# Hacky workaround for now... meh
from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC, ptu
import optuna

INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"

# Cache directory
CACHE_DIR = Path(os.getenv("BTQ_CACHE_DIR", ".btq_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

class PolarsDataLoader:
    """Polars-based data loader with caching capabilities"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, spec: DataSpec) -> str:
        ranges = spec.ranges or [(spec.start_date or "", spec.end_date or "")]
        ranges_str = ",".join([f"{s}:{e}" for s, e in ranges])
        raw = f"{spec.symbol}|{spec.interval}|{spec.collateral}|{ranges_str}"
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{spec.symbol}_{spec.interval}_{spec.collateral}_{h}.parquet"
    
    def _cache_path(self, spec: DataSpec) -> Path:
        return self.cache_dir / self._cache_key(spec)
    
    def _load_from_cache(self, spec: DataSpec) -> Optional[pl.DataFrame]:
        cache_path = self._cache_path(spec)
        if not cache_path.exists():
            return None
        try:
            return pl.scan_parquet(str(cache_path)).collect()
        except Exception as e:
            console.print(f"[yellow]Cache read failed for {spec.symbol}: {e}[/yellow]")
            return None
    
    def _save_to_cache(self, spec: DataSpec, df: pl.DataFrame) -> None:
        cache_path = self._cache_path(spec)
        try:
            df.write_parquet(str(cache_path), compression="zstd", statistics=True)
            console.print(f"[green]Cached {spec.symbol} -> {cache_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Cache write failed for {spec.symbol}: {e}[/yellow]")
    
    def _fetch_from_database(self, spec: DataSpec) -> pl.DataFrame:
        """Fetch data from database using existing MSSQL function"""
        from backtrader.feeds.mssql_crypto import get_database_data
        
        if spec.ranges:
            parts = []
            for start_date, end_date in spec.ranges:
                part = get_database_data(
                    ticker=spec.symbol, 
                    start_date=start_date, 
                    end_date=end_date,
                    time_resolution=spec.interval, 
                    pair=spec.collateral
                )
                if part is None or part.is_empty():
                    raise ValueError(f"No data for {spec.symbol} {spec.interval} {start_date}->{end_date}")
                parts.append(part)
            df = pl.concat(parts).sort("TimestampStart")
        else:
            df = get_database_data(
                ticker=spec.symbol, 
                start_date=spec.start_date, 
                end_date=spec.end_date,
                time_resolution=spec.interval, 
                pair=spec.collateral
            )
        
        if df is None or df.is_empty():
            raise ValueError(f"No data for {spec.symbol} {spec.interval} {spec.start_date}->{spec.end_date}")
        
        return df.sort("TimestampStart")
    
    def load_data(self, spec: DataSpec, use_cache: bool = True) -> pl.DataFrame:
        """Load data with caching support"""
        # Try cache first
        df = None
        if use_cache:
            df = self._load_from_cache(spec)
            if df is not None:
                console.print(f"[cyan]Loaded from cache: {spec.symbol}[/cyan]")
        
        # Fetch from database if not in cache
        if df is None:
            df = self._fetch_from_database(spec)
            if use_cache:
                self._save_to_cache(spec, df)
        
        # Validate required columns
        required = {"TimestampStart", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{spec.symbol}: missing required columns. Found: {df.columns}")
        
        return df
    
    def make_backtrader_feed(self, df: pl.DataFrame, spec: DataSpec):
        """Convert Polars DataFrame to Backtrader feed"""
        from backtrader.feeds.mssql_crypto import MSSQLData
        
        feed = MSSQLData(
            dataname=df,
            datetime="TimestampStart", 
            open="Open", 
            high="High", 
            low="Low", 
            close="Close", 
            volume="Volume",
            timeframe=bt.TimeFrame.Minutes, 
            compression=1
        )
        
        # Set timeframe attributes safely
        try:
            feed.p.timeframe = bt.TimeFrame.Minutes
            feed.p.compression = 1
        except Exception:
            setattr(feed, "_timeframe", bt.TimeFrame.Minutes)
            setattr(feed, "_compression", 1)
        
        # Set feed identifiers
        try:
            feed._name = f"{spec.symbol}-{spec.interval}"
            feed._dataname = f"{spec.symbol}{spec.collateral}"
        except Exception:
            pass
        
        return feed

def fetch_single_data(coin, start_date, end_date, interval, collateral="USDT"):
    """Legacy compatibility function - now uses PolarsDataLoader"""
    try:
        loader = PolarsDataLoader()
        spec = DataSpec(
            symbol=coin,
            interval=interval, 
            start_date=start_date, 
            end_date=end_date,
            collateral=collateral
        )
        df = loader.load_data(spec)
        feed = loader.make_backtrader_feed(df, spec)
        feed._dataname = f"{coin}{collateral}"
        return feed
    except Exception as e:
        console.print(f"[red]Error fetching data for {coin}: {str(e)}[/red]")
        return None

def backtest(
    strategy,
    data=None,
    coin=None,
    start_date="1970-01-01",
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=COMMISSION_PER_TRANSACTION,
    init_cash=INIT_CASH,
    plot=False,
    quantstats=False,
    asset_name=None,
    bulk=False,
    show_progress=True,
    exchange=None,
    slippage_bps=5,
    min_qty=0.0, 
    qty_step=1.0,
    price_tick=None,
    params=None,
    add_mtf_resamples=False,
    **kwargs,
):
    from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomData
    # ---------- Data ----------
    if data is None:
        if not all([coin, start_date, end_date, interval]):
            raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")
        if show_progress and not bulk:
            console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")
        
        # Use OOP data loader
        loader = PolarsDataLoader()
        spec = DataSpec(
            symbol=coin,
            interval=interval, 
            start_date=start_date, 
            end_date=end_date,
            collateral=collateral
        )
        df = loader.load_data(spec, use_cache=True)
        data_feed = loader.make_backtrader_feed(df, spec)
        
        if show_progress and not bulk:
            console.print(f"✅ [bold green]Data fetched for {coin}[/bold green]")
    else:
        # Handle existing data input
        from backtrader.feed import DataBase
        if isinstance(data, DataBase):
            data_feed = data
        else:
            data_feed = CustomData(dataname=data)

    # ---------- Engine & feeds ----------
    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    cerebro.adddata(data_feed)

    # Add multi-timeframe resamples
    if add_mtf_resamples:
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=5,  name='5m',  boundoff=1)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=15, name='15m', boundoff=1)
        cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name='60m', boundoff=1)

    # ---------- Strategy parameters ----------
    if not bulk:
        strat_kwargs = {
            'backtest': True
        }
    else:
        strat_kwargs = {
            'backtest': True,
            'bulk': bulk,
        }

    def _strategy_param_keys(cls):
        try:
            return set(cls.params._getkeys())
        except Exception:
            try:
                return set(k for k, _ in cls.params)
            except Exception:
                return set()

    valid_keys = _strategy_param_keys(strategy)

    # Exchange-specific settings
    if 'can_short' in valid_keys and exchange is not None:
        strat_kwargs['can_short'] = (str(exchange).lower() == "mexc")
    
    # Order sizing parameters
    if 'min_qty' in valid_keys:
        strat_kwargs['min_qty'] = min_qty
    if 'qty_step' in valid_keys:
        strat_kwargs['qty_step'] = qty_step
    if 'price_tick' in valid_keys:
        strat_kwargs['price_tick'] = price_tick

    # Handle Optuna parameters
    if isinstance(params, dict):
        # For now, assume MTF format - you can add compat conversion later if needed
        converted_params = dict(params)
        
        # Merge safely
        reserved = set(strat_kwargs.keys())
        for k, v in converted_params.items():
            if k in valid_keys and k not in reserved:
                strat_kwargs[k] = v

    # Allow explicit kwargs
    for k, v in kwargs.items():
        if k not in strat_kwargs:
            strat_kwargs[k] = v

    if 'backtest' not in valid_keys and 'backtest' in strat_kwargs:
        del strat_kwargs['backtest']

    cerebro.addstrategy(strategy, **strat_kwargs)

    # ---------- Broker & analyzers ----------
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass

    # Analyzers
    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')
    
    if not add_mtf_resamples:
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    display_name = asset_name or (f"{coin}/{collateral}" if coin else "Asset")

    if show_progress and not bulk:
        console.print(f"🔄 [bold green]Running backtest for {display_name}...")

    strategy_result = cerebro.run()[0]

    # ---------- Metrics ----------
    dd_info = strategy_result.analyzers.drawdown.get_analysis()
    max_drawdown = dd_info.get('max', {}).get('drawdown', 0.0)
    trade_analyzer = strategy_result.analyzers.trade_analyzer.get_analysis()

    # ---------- Returns for QuantStats using Polars ----------
    tr = strategy_result.analyzers.time_return.get_analysis()

    if tr:
        # Create Polars DataFrame from time returns
        dates = list(tr.keys())
        returns_values = list(tr.values())
        
        returns_df = pl.DataFrame({
            'date': dates,
            'returns': returns_values
        })
        
        # Check if dates are already datetime objects or need parsing
        if returns_df.dtypes[0] == pl.String:
            # Parse string dates
            returns_df = returns_df.with_columns([
                pl.col('date').str.strptime(pl.Datetime, format='%Y-%m-%d', strict=False)
            ]).sort('date').drop_nulls()
        elif returns_df.dtypes[0] in [pl.Datetime, pl.Date]:
            # Already datetime, just sort and clean
            returns_df = returns_df.sort('date').drop_nulls()
        else:
            # Try to convert whatever type it is to datetime
            returns_df = returns_df.with_columns([
                pl.col('date').cast(pl.Datetime)
            ]).sort('date').drop_nulls()
        
        # Convert to pandas Series for QuantStats
        if not returns_df.is_empty():
            returns_pd = returns_df.to_pandas().set_index('date')['returns']
        else:
            returns_pd = pd.Series(dtype=float)
    else:
        returns_pd = pd.Series(dtype=float)

    if quantstats and not returns_pd.empty:
        import quantstats_lumi as quantstats
        from datetime import datetime
        import os
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        if asset_name:
            coin_name = asset_name.replace('/', '_')
        elif coin:
            coin_name = f"{coin}_{collateral}"
        elif hasattr(data_feed, '_dataname'):
            coin_name = str(data_feed._dataname)
        else:
            coin_name = "Unknown_Asset"
        folder = os.path.join("QuantStats")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{coin_name}_{current_date}_{current_time}.html")
        try:
            quantstats.reports.html(returns_pd, output=filename, title=f'QuantStats_{coin_name}_{current_date}')
        except Exception as e:
            console.print(f"[yellow]QuantStats report generation failed: {e}[/yellow]")

    if not bulk:
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        pnl_net = trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - init_cash

        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS - {display_name}")
        print(f"{'='*50}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades}")
        print(f"Losing Trades: {lost_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Net P&L: ${pnl_net:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Final Portfolio Value: ${portvalue:.2f}")
        print(f"Total P/L: ${pnl:.2f}")
        print(f"Return: {(pnl / init_cash * 100):.2f}%")
        print(f"{'='*50}")

    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')

    # Store final value before cleanup
    final_value = cerebro.broker.getvalue()

    # Clean up
    del cerebro, data_feed, strategy_result
    gc.collect()

    return final_value

def backtest_with_leverage(
    strategy,
    data=None,
    coin=None,
    start_date=None,
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=0.00075,
    init_cash=1000.0,
    leverage=100,
    max_leverage=100,
    margin_mode='cross',
    maintenance_margin_rate=0.005,  # 0.5% for 100x
    initial_margin_rate=0.01,  # 1% for 100x
    position_pct=0.0025,  # 0.25% per position
    plot=False,
    quantstats=False,
    asset_name=None,
    bulk=False,
    show_progress=True,
    exchange=None,
    slippage_bps=5,
    min_qty=0.0,
    qty_step=1.0,
    price_tick=None,
    params=None,
    params_mode="mtf",
    add_mtf_resamples=False,
    **kwargs,
):
    """
    REAL BITMEX CROSS MARGIN BACKTEST
    
    Formula for Long Position:
    - Position Value = entry_price × size
    - Initial Margin = Position Value × initial_margin_rate
    - Maintenance Margin = Position Value × maintenance_margin_rate
    - Available Balance = Wallet - Total Initial Margin
    - Bankruptcy Price = Entry - (Available Balance / Position Size)
    - Liquidation Price = Bankruptcy Price + (Maintenance Margin / Position Size)
    """
    from backtrader.analyzers import SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomData
    from rich.console import Console

    console = Console()

    # Load data
    if data is None:
        if not all([coin, start_date, end_date, interval]):
            raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")

        from backtrader.feeds.mssql_crypto import get_database_data

        if show_progress and not bulk:
            console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")

        df = get_database_data(
            ticker=coin,
            start_date=start_date,
            end_date=end_date,
            time_resolution=interval,
            pair=collateral
        )

        data_feed = CustomData(dataname=df)

        if show_progress and not bulk:
            console.print(f"✅ [bold green]Data fetched for {coin}[/bold green]")
    else:
        from backtrader.feed import DataBase
        if isinstance(data, DataBase):
            data_feed = data
        else:
            data_feed = CustomData(dataname=data)

    cerebro = bt.Cerebro(oldbuysell=True, runonce=False, stdstats=False)
    cerebro.adddata(data_feed)
    
    cerebro.broker.setcash(init_cash)
    cerebro.broker.setcommission(commission=commission)
    
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass

    # Cross margin tracker
    cross_margin = {
        'wallet_balance': init_cash,
        'initial_wallet': init_cash,
        'positions': [],
        'liquidations': [],
        'total_pnl': 0.0,
        'total_fees': 0.0,
        'wins': 0,
        'losses': 0,
        'liquidated': 0,
    }

    if show_progress and not bulk:
        console.print(f"💪 [bold yellow]Leverage: {leverage}x Cross | "
                     f"Position Size: {position_pct*100}% | "
                     f"Maint. Margin: {maintenance_margin_rate*100}%[/bold yellow]")

    class BitmexCrossMarginWrapper(strategy):
        def __init__(self):
            super().__init__()
            self.cross_positions = []  # Track all open positions
            
        def next(self):
            current_price = self.data.close[0]
            current_dt = self.data.datetime.datetime(0)
            
            # STEP 1: Calculate cross margin metrics
            total_position_value = 0.0
            total_initial_margin = 0.0
            total_maintenance_margin = 0.0
            unrealized_pnl = 0.0
            
            for pos in self.cross_positions:
                pos_value = pos['entry_price'] * pos['size']
                total_position_value += pos_value
                total_initial_margin += pos_value * initial_margin_rate
                total_maintenance_margin += pos_value * maintenance_margin_rate
                
                # Calculate unrealized PnL
                price_change = current_price - pos['entry_price']
                pnl = price_change * pos['size']
                unrealized_pnl += pnl
            
            # Available balance = Wallet + Unrealized PnL - Initial Margin
            available_balance = cross_margin['wallet_balance'] + unrealized_pnl - total_initial_margin
            
            # STEP 2: Check liquidation condition
            # Liquidation when: Available Balance <= Maintenance Margin
            if self.cross_positions and available_balance <= total_maintenance_margin:
                # LIQUIDATION EVENT
                total_loss = cross_margin['wallet_balance'] + unrealized_pnl - total_maintenance_margin
                
                cross_margin['liquidations'].append({
                    'timestamp': current_dt,
                    'price': current_price,
                    'positions': len(self.cross_positions),
                    'total_loss': abs(total_loss),
                    'wallet_before': cross_margin['wallet_balance'],
                })
                
                cross_margin['liquidated'] += len(self.cross_positions)
                cross_margin['losses'] += len(self.cross_positions)
                cross_margin['wallet_balance'] = total_maintenance_margin
                
                if show_progress:
                    console.print(
                        f"💥 [bold red]LIQUIDATED[/bold red] | {current_dt} | "
                        f"Price: ${current_price:,.2f} | "
                        f"{len(self.cross_positions)} positions | "
                        f"Loss: ${abs(total_loss):.2f}"
                    )
                
                # Force close all positions
                for pos in self.cross_positions:
                    try:
                        self.sell(size=pos['size'], exectype=bt.Order.Market)
                    except:
                        pass
                
                self.cross_positions.clear()
                self.reset_position_state()
                
                # Skip strategy logic after liquidation
                return
            
            # STEP 3: Run normal strategy logic
            super().next()
        
        def notify_trade(self, trade):
            super().notify_trade(trade)
            
            if trade.isclosed:
                # Track closed trade
                pnl = trade.pnlcomm
                cross_margin['total_pnl'] += pnl
                cross_margin['total_fees'] += trade.commission
                cross_margin['wallet_balance'] += pnl
                
                if pnl > 0:
                    cross_margin['wins'] += 1
                else:
                    cross_margin['losses'] += 1
                
                # Remove from cross positions
                self.cross_positions = [p for p in self.cross_positions 
                                       if p['entry_price'] != trade.price]
        
        def create_order(self, action='BUY', size=None, price=None):
            """Override to track cross margin positions"""
            if size is None:
                # Calculate size based on position_pct
                margin_to_use = cross_margin['wallet_balance'] * position_pct
                position_value = margin_to_use * leverage
                size = position_value / self.data.close[0]
            
            if price is None:
                price = self.data.close[0]
            
            # Track position for cross margin
            self.cross_positions.append({
                'entry_price': price,
                'size': size,
                'timestamp': self.data.datetime.datetime(0)
            })
            
            # Call parent
            return super().create_order(action=action, size=size, price=price)

    # Strategy setup
    strat_kwargs = {'backtest': True}

    def _strategy_param_keys(cls):
        try:
            return set(cls.params._getkeys())
        except Exception:
            try:
                return set(k for k, _ in cls.params)
            except Exception:
                return set()

    valid_keys = _strategy_param_keys(BitmexCrossMarginWrapper)

    if 'can_short' in valid_keys and exchange is not None:
        strat_kwargs['can_short'] = (str(exchange).lower() == "mexc")
    if 'min_qty' in valid_keys:
        strat_kwargs['min_qty'] = min_qty
    if 'qty_step' in valid_keys:
        strat_kwargs['qty_step'] = qty_step
    if 'price_tick' in valid_keys:
        strat_kwargs['price_tick'] = price_tick

    if isinstance(params, dict):
        for k, v in params.items():
            if k in valid_keys and k not in strat_kwargs:
                strat_kwargs[k] = v

    for k, v in kwargs.items():
        if k not in strat_kwargs:
            strat_kwargs[k] = v

    cerebro.addstrategy(BitmexCrossMarginWrapper, **strat_kwargs)

    # Analyzers
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')

    if not add_mtf_resamples:
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    display_name = asset_name or (f"{coin}/{collateral}" if coin else "Asset")

    if show_progress and not bulk:
        console.print(f"🔄 [bold green]Running backtest for {display_name}...")

    # Run
    strategy_result = cerebro.run()[0]

    # Results
    if not bulk:
        total_trades = cross_margin['wins'] + cross_margin['losses']
        win_rate = (cross_margin['wins'] / total_trades * 100) if total_trades > 0 else 0
        
        final_value = cross_margin['wallet_balance']
        total_return = ((final_value - init_cash) / init_cash) * 100

        print(f"\n{'='*80}")
        print(f"BITMEX CROSS MARGIN BACKTEST - {display_name}")
        print(f"{'='*80}\n")
        
        print(f"💰 ACCOUNT")
        print(f"Initial Balance:        ${init_cash:.2f}")
        print(f"Final Balance:          ${final_value:.2f}")
        print(f"Total P&L:              ${cross_margin['total_pnl']:.2f}")
        print(f"Total Fees:             ${cross_margin['total_fees']:.2f}")
        print(f"Return:                 {total_return:.2f}%")
        print(f"")
        
        print(f"📊 TRADES")
        print(f"Total:                  {total_trades}")
        print(f"Wins:                   {cross_margin['wins']} ({win_rate:.1f}%)")
        print(f"Losses:                 {cross_margin['losses']}")
        print(f"Liquidated:             {cross_margin['liquidated']}")
        print(f"")
        
        print(f"⚙️  SETTINGS")
        print(f"Leverage:               {leverage}x")
        print(f"Margin Mode:            Cross")
        print(f"Position Size:          {position_pct*100}%")
        print(f"Initial Margin Rate:    {initial_margin_rate*100}%")
        print(f"Maintenance Margin:     {maintenance_margin_rate*100}%")
        
        if cross_margin['liquidations']:
            print(f"\n💥 LIQUIDATIONS ({len(cross_margin['liquidations'])})")
            print(f"{'─'*80}")
            for i, liq in enumerate(cross_margin['liquidations'], 1):
                print(f"{i}. {liq['timestamp']} | Price: ${liq['price']:,.2f} | "
                      f"Positions: {liq['positions']} | Loss: ${liq['total_loss']:.2f}")
        
        print(f"\n{'='*80}\n")

    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')

    import gc
    del cerebro, data_feed, strategy_result
    gc.collect()

    return final_value

def backtest_study_params(
    study_name: str,
    strategy_class,
    odbc_connection: Optional[str],
    coin: str = "BTC",
    collateral: str = "USDT",
    interval: str = "1d",
    start_date="1970-01-01",
    end_date="2030-01-01",
    init_cash: float = 1000,
    plot: bool = True,
    quantstats: bool = False,
    debug: bool = False,
):
    """
    Run a backtest using the best trial parameters from a given Optuna study.
    """
    console.rule(f"[bold cyan]🔍 Fetching best params from study: {study_name}[/bold cyan]")

    from .optimize import ensure_storage_or_sqlite, get_param_names
    storage = ensure_storage_or_sqlite(odbc_connection, study_name)
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        console.print(f"[red]Failed to load study {study_name}: {e}[/red]")
        return None

    if not hasattr(study, "best_trial") or study.best_trial is None:
        console.print("[red]❌ No best trial found in this study.[/red]")
        return None

    best_trial = study.best_trial
    raw_params = best_trial.params
    param_names = get_param_names(strategy_class)
    params = {k: v for k, v in raw_params.items() if k in param_names}

    console.print(f"[green]✅ Loaded {len(params)} optimized params from study '{study_name}':[/green]")
    for k, v in params.items():
        console.print(f"  • {k} = {v}")

    console.rule("[bold green]🚀 Launching Backtest[/bold green]")
    results = backtest(
        strategy_class,
        coin=coin,
        collateral=collateral,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        init_cash=init_cash,
        plot=plot,
        quantstats=quantstats,
        debug=debug,
        params=params,
        param_names=param_names,
    )

    console.rule("[bold green]🏁 Backtest Completed[/bold green]")
    return results

def run_multi_isolated_pairs(strategy, pairs_config, start_date, end_date, interval, collateral="USDT"):
    """
    Run multiple isolated leveraged backtests at once — each pair has its own leverage, margin, and isolation.
    """
    from rich.console import Console
    from rich.table import Table
    import concurrent.futures
    import gc

    console = Console()
    results = []

    def run_pair(pair_cfg):
        coin = pair_cfg["symbol"]
        lev = float(pair_cfg.get("leverage", 1))
        cap = float(pair_cfg.get("capital", 100))
        iso_margin = cap / lev

        console.print(f"[cyan]Running {coin} with x{lev} isolated (margin {iso_margin:.2f}$)[/cyan]")

        try:
            loader = PolarsDataLoader()
            spec = DataSpec(
                symbol=coin,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                collateral=collateral
            )
            df = loader.load_data(spec, use_cache=True)
            feed = loader.make_backtrader_feed(df, spec)

            # simulate isolated margin by setting broker cash = margin * leverage
            init_cash = iso_margin * lev

            final_value = backtest(
                strategy,
                data=feed,
                coin=coin,
                init_cash=init_cash,
                collateral=collateral,
                bulk=True,
                show_progress=True,
                quantstats=False,
            )

            pnl = final_value - init_cash
            ret_pct = (pnl / init_cash) * 100

            gc.collect()
            return {
                "coin": coin,
                "leverage": lev,
                "capital": cap,
                "iso_margin": iso_margin,
                "final_value": final_value,
                "pnl": pnl,
                "return_pct": ret_pct,
                "status": "success",
            }

        except Exception as e:
            return {"coin": coin, "error": str(e), "status": "failed"}

    # Parallel run all pairs
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs_config)) as executor: # ProcessPoolExecutor
        futures = [executor.submit(run_pair, p) for p in pairs_config]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    # Print results summary
    table = Table(title="📊 Multi-Isolated Backtest Results")
    table.add_column("Pair", style="cyan")
    table.add_column("Lev", justify="right")
    table.add_column("Margin", justify="right")
    table.add_column("Final Value", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Return %", justify="right")

    for r in results:
        if r["status"] == "success":
            table.add_row(
                r["coin"],
                f"x{r['leverage']}",
                f"${r['iso_margin']:.2f}",
                f"${r['final_value']:.2f}",
                f"${r['pnl']:.2f}",
                f"{r['return_pct']:.2f}%",
            )
        else:
            table.add_row(r["coin"], "-", "-", "-", "-", f"❌ {r['error']}")

    console.print(table)
    return results

import sys
import io

def silent_run(args):
    """Wrapper to silence all output inside the subprocess."""
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        # import traceback for error capture
        import traceback
        result = run_single_backtest_with_data_fetch(args)
        return result
    except Exception as e:
        return {"coin": args[0], "status": "failed", "error": str(e), "traceback": traceback.format_exc()}

def run_backtest_with_data(args):
    """Helper function for parallel execution in bulk_backtest"""
    coin, data, strategy_class, init_cash, backtest_params, collateral = args

    if data is None:
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": "No data available", "status": "skipped"}

    try:
        asset = f'{coin}/{collateral}'
        
        result = backtest(
            strategy_class,
            data=data,
            init_cash=init_cash,
            asset_name=asset,
            bulk=True,
            capture_data=False, # No transparence on indicator buildup chain needed here
            **backtest_params
        )

        del data
        gc.collect()

        return {
            "coin": coin,
            "asset": asset,
            "result": result,
            "final_value": result,
            "pnl": result - init_cash,
            "return_pct": ((result - init_cash) / init_cash) * 100,
            "status": "success"
        }

    except Exception as e:
        console.print(f"[red]Error in {coin}: {str(e)}[/red]")
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": str(e), "status": "failed"}

def bulk_backtest(
        strategy, 
        coins=None, 
        start_date="1970-01-01", 
        end_date="2030-01-01", 
        interval=None, 
        collateral="USDT", 
        init_cash=1000, 
        max_workers=8, 
        save_results=True, 
        output_file='backtest_results.json', 
        params_mode="mtf",
        **backtest_kwargs):

    """Enhanced bulk backtest with unified progress tracking and auto-discovery
    Example Coinlist: coinlist = ['1000CAT','AAVE', 'ACA', 'ACE', 'ACH', 'ACM', 'ACT', 'ACX']
    Or just None to fully fetch whole Database available of Coins
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # Validate required parameters first
    if not all([start_date, end_date, interval]):
        raise ValueError("start_date, end_date, and interval are required")

    # Auto-discover coins from database if not provided
    if coins is None:
        console.print("[bold blue]🔍 Auto-discovering coins from database...[/bold blue]")
        try:
            coins = get_available_coins_from_database(collateral=collateral)
            if not coins:
                console.print("[red]❌ No coins found in database[/red]")
                return []
            
            sample_coins = coins[:10]
            remaining = len(coins) - 10
            coins_display = ', '.join(sample_coins)
            if remaining > 0:
                coins_display += f" (and {remaining} more)"
            
            console.print(f"[green]✅ Found {len(coins)} coins: {coins_display}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to get coins from database: {e}[/red]")
            return []

    default_backtest_params = {
        'plot': False,
        'debug': False,
        'quantstats': False, # Default to False
        'params_mode': params_mode,
        'add_mtf_resamples': False, # TODO :: Fetch timeframe(s) from startup
    }
    default_backtest_params.update(backtest_kwargs)
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    results = []
    successful_count = 0
    failed_count = 0

    with progress:
        backtest_task = progress.add_task("🚀 Running backtests...", total=len(coins))
        
        # arguments for parallel execution
        backtest_args = [
            (coin, start_date, end_date, interval, collateral, strategy, init_cash, default_backtest_params)
            for coin in coins
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_coin = {
                executor.submit(silent_run, args): args[0]
                for args in backtest_args
            }
            
            for future in concurrent.futures.as_completed(future_to_coin):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    coin = result.get('coin', 'Unknown')
                    completed = successful_count + failed_count
                    
                    progress.update(
                        backtest_task, 
                        completed=completed,
                        description=f"🚀 Backtests: {completed}/{len(coins)} (✅{successful_count} ❌{failed_count}) - Last: {coin}"
                    )
                    
                except Exception as exc:
                    coin = future_to_coin[future]
                    failed_count += 1
                    completed = successful_count + failed_count
                    
                    progress.console.print(f"[red]Exception for {coin}: {exc}[/red]")
                    results.append({"coin": coin, "asset": f"{coin}/{collateral}", "error": str(exc), "status": "failed"})
                    
                    progress.update(
                        backtest_task, 
                        completed=completed,
                        description=f"🚀 Backtests: {completed}/{len(coins)} (✅{successful_count} ❌{failed_count}) - Last: {coin} ❌"
                    )
        
        progress.update(backtest_task, completed=len(coins), description="🚀 All backtests complete")
    
    if save_results:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n[green]Results saved to '{output_file}'[/green]")
    
    console.print(f"\n[bold green]🎉 Backtesting completed![/bold green]")
    return print_detailed_results(results)

def run_single_backtest_with_data_fetch(args):
    """Fetch data and run backtest for a single coin - designed for parallel execution"""
    coin, start_date, end_date, interval, collateral, strategy_class, init_cash, backtest_params = args
    
    try:
        # Step 1: Fetch data using existing caching system
        loader = PolarsDataLoader()
        spec = DataSpec(
            symbol=coin,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            collateral=collateral
        )
        
        # This will use cache if available, fetch from DB if not
        df = loader.load_data(spec, use_cache=True)
        
        if df is None or df.is_empty():
            return {"coin": coin, "asset": f"{coin}/{collateral}", "error": "No data available", "status": "skipped"}
        
        # Step 2: Create backtrader feed
        data_feed = loader.make_backtrader_feed(df, spec)
        data_feed._dataname = f"{coin}{collateral}"
        
        # Step 3: Run backtest
        asset = f'{coin}/{collateral}'
        
        result = backtest(
            strategy_class,
            data=data_feed,
            init_cash=init_cash,
            asset_name=asset,
            bulk=True,
            show_progress=False,  # No progress for individual backtests
            **backtest_params
        )
        
        # Clean up
        del df, data_feed
        gc.collect()
        
        return {
            "coin": coin,
            "asset": asset,
            "result": result,
            "final_value": result,
            "pnl": result - init_cash,
            "return_pct": ((result - init_cash) / init_cash) * 100,
            "status": "success"
        }
        
    except Exception as e:
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": str(e), "status": "failed"}

def get_available_coins_from_database(collateral="USDT"):
    """Get all available coins from the database"""
    try:
        from backtrader.feeds.mssql_crypto import MSSQLData
        from backtrader.dontcommit import connection_string
        
        all_pairs = MSSQLData.get_all_pairs(connection_string)
        
        coins = []
        suffix_pattern = f"{collateral}_klines"
        
        for pair in all_pairs:
            if pair.endswith(suffix_pattern):
                # Extract coin name by removing the suffix
                coin = pair.replace(suffix_pattern, "")
                # Remove any trailing underscores
                coin = coin.rstrip("_")
                if coin:
                    coins.append(coin)
        
        # Remove duplicates and sort
        coins = sorted(list(set(coins)))
        
        return coins
        
    except Exception as e:
        console.print(f"[red]❌ Error getting coins from database: {e}[/red]")
        # Fallback to common coins
        return ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOT', 'DOGE', 'AVAX', 'MATIC']

def print_detailed_results(results):
    """Print detailed backtest results in a styled Rich table format"""

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]

    console.rule("[bold yellow]🚀 BTQuant Results[/]")

    # ✅ Successful backtests
    if successful:
        success_table = Table(title=f"🟢 Successful Backtests ({len(successful)})")
        success_table.add_column("Asset", style="cyan")
        success_table.add_column("Final Value", style="green")
        success_table.add_column("P&L", style="magenta")

        successful_sorted = sorted(successful, key=lambda x: x.get("pnl", 0), reverse=True)
        for r in successful_sorted:
            asset = r.get("asset", r.get("coin", "N/A"))
            final_value = r.get("final_value")
            pnl = r.get("pnl")

            final_value_str = f"${final_value:,.2f}" if isinstance(final_value, (int, float)) else str(final_value)
            pnl_str = f"${pnl:,.2f}" if isinstance(pnl, (int, float)) else str(pnl)

            success_table.add_row(asset, final_value_str, pnl_str)

        console.print(success_table)

    # ❌ Failed backtests
    if failed:
        fail_table = Table(title=f"🔴 Failed Backtests ({len(failed)})")
        fail_table.add_column("Asset", style="red")
        fail_table.add_column("Error", style="yellow")

        for r in failed:
            fail_table.add_row(
                r.get("asset", r.get("coin", "N/A")),
                str(r.get("error", "Unknown error"))
            )

        console.print(fail_table)

    # ⚪ Skipped backtests
    if skipped:
        skip_table = Table(title=f"⚪ Skipped Backtests (No Data) ({len(skipped)})")
        skip_table.add_column("Assets", style="white")

        skipped_assets = [r.get("asset", r.get("coin", "N/A")) for r in skipped]
        skip_table.add_row(", ".join(skipped_assets))

        console.print(skip_table)

    # 📊 Summary
    summary = Table(title="📊 Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")

    summary.add_row("Successful", str(len(successful)))
    summary.add_row("Failed", str(len(failed)))
    summary.add_row("Skipped (No Data)", str(len(skipped)))

    console.print(summary)

    return results


def bulk_backtest_with_optuna(
        study_name: str,
        strategy_class,
        odbc_connection: Optional[str],
        coins=None,
        start_date="2016-01-01",
        end_date="2026-01-08",
        interval=None,
        collateral="USDT",
        init_cash=1000,
        max_workers=8,
        save_results=True,
        output_file='backtest_results_optuna.json',
        params_mode="mtf",
        **backtest_kwargs
):
    """
    Enhanced bulk backtest that automatically loads best parameters from an Optuna study
    and applies them to all backtests.
    
    Args:
        study_name: Name of the Optuna study to load best parameters from
        strategy_class: Strategy class to backtest
        odbc_connection: Database connection string for Optuna
        coins: List of coins to backtest (auto-discovers if None)
        start_date: Start date for backtesting
        end_date: End date for backtesting
        interval: Timeframe interval (e.g., '1m', '5m', '1h', '1d')
        collateral: Collateral currency (default: 'USDT')
        init_cash: Initial capital for each backtest
        max_workers: Number of parallel workers
        save_results: Save results to JSON file
        output_file: Output filename for results
        params_mode: Parameter mode ('mtf' or other)
        **backtest_kwargs: Additional backtest parameters
        
    Returns:
        List of backtest results
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    console.rule(f"[bold cyan]🔍 Loading best params from Optuna study: {study_name}[/bold cyan]")
    
    # ==================== Load Optuna Best Parameters ====================
    from .optimize import ensure_storage_or_sqlite, get_param_names
    
    storage = ensure_storage_or_sqlite(odbc_connection, study_name)
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        console.print(f"[red]❌ Failed to load study '{study_name}': {e}[/red]")
        console.print("[yellow]⚠ Falling back to bulk_backtest without Optuna params[/yellow]")
        return bulk_backtest(
            study_name=None,
            strategy_class=strategy_class,
            odbc_connection=odbc_connection,
            coins=coins,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            collateral=collateral,
            init_cash=init_cash,
            max_workers=max_workers,
            save_results=save_results,
            output_file=output_file,
            params_mode=params_mode,
            **backtest_kwargs
        )
    
    if not hasattr(study, "best_trial") or study.best_trial is None:
        console.print("[red]❌ No best trial found in this study.[/red]")
        console.print("[yellow]⚠ Falling back to bulk_backtest without Optuna params[/yellow]")
        return bulk_backtest(
            study_name=None,
            strategy_class=strategy_class,
            odbc_connection=odbc_connection,
            coins=coins,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            collateral=collateral,
            init_cash=init_cash,
            max_workers=max_workers,
            save_results=save_results,
            output_file=output_file,
            params_mode=params_mode,
            **backtest_kwargs
        )
    
    # Extract best parameters
    best_trial = study.best_trial
    raw_params = best_trial.params
    param_names = get_param_names(strategy_class)
    best_params = {k: v for k, v in raw_params.items() if k in param_names}
    
    # Display loaded parameters
    console.print(f"[green]✅ Loaded {len(best_params)} optimized parameters:[/green]")
    
    # Create a nice table for parameters
    params_table = Table(title="Optuna Best Parameters", show_header=True)
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")
    
    for k, v in sorted(best_params.items()):
        params_table.add_row(k, str(v))
    
    console.print(params_table)
    console.print(f"[bold blue]Study Best Value: {best_trial.value}[/bold blue]")
    console.print(f"[dim]Trial Number: {best_trial.number} | Trials Completed: {len(study.trials)}[/dim]\n")
    
    if not all([start_date, end_date, interval]):
        raise ValueError("start_date, end_date, and interval are required")
    
    if coins is None:
        console.print("[bold blue]🔍 Auto-discovering coins from database...[/bold blue]")
        try:
            coins = get_available_coins_from_database(collateral=collateral)
            if not coins:
                console.print("[red]❌ No coins found in database[/red]")
                return []
            
            sample_coins = coins[:10]
            remaining = len(coins) - 10
            coins_display = ', '.join(sample_coins)
            if remaining > 0:
                coins_display += f" (and {remaining} more)"
            
            console.print(f"[green]✅ Found {len(coins)} coins: {coins_display}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to get coins from database: {e}[/red]")
            return []
    
    # ==================== Prepare Backtest Parameters ====================
    default_backtest_params = {
        'plot': False,
        'debug': False,
        'quantstats': False,
        'params_mode': params_mode,
        'add_mtf_resamples': False,
        'params': best_params,
    }
    default_backtest_params.update(backtest_kwargs)
    
    # ==================== Progress Tracking ====================
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )
    
    results = []
    successful_count = 0
    failed_count = 0
    
    # ==================== Run Backtests in Parallel ====================
    with progress:
        backtest_task = progress.add_task(
            f"🚀 Running backtests with Optuna params...", 
            total=len(coins)
        )
        
        # Prepare arguments for parallel execution
        backtest_args = [
            (coin, start_date, end_date, interval, collateral, strategy_class, init_cash, default_backtest_params)
            for coin in coins
        ]
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_coin = {
                executor.submit(silent_run, args): args[0]
                for args in backtest_args
            }
            
            for future in concurrent.futures.as_completed(future_to_coin):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result["status"] == "success":
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    coin = result.get('coin', 'Unknown')
                    completed = successful_count + failed_count
                    
                    progress.update(
                        backtest_task,
                        completed=completed,
                        description=f"🚀 Backtests: {completed}/{len(coins)} (✅{successful_count} ❌{failed_count}) - Last: {coin}"
                    )
                    
                except Exception as exc:
                    coin = future_to_coin[future]
                    failed_count += 1
                    completed = successful_count + failed_count
                    
                    progress.console.print(f"[red]Exception for {coin}: {exc}[/red]")
                    results.append({
                        "coin": coin,
                        "asset": f"{coin}/{collateral}",
                        "error": str(exc),
                        "status": "failed"
                    })
                    
                    progress.update(
                        backtest_task,
                        completed=completed,
                        description=f"🚀 Backtests: {completed}/{len(coins)} (✅{successful_count} ❌{failed_count}) - Last: {coin} ❌"
                    )
        
        progress.update(
            backtest_task,
            completed=len(coins),
            description="🚀 All backtests complete with Optuna params"
        )
    
    # ==================== Save Results ====================
    if save_results:
        import json
        
        # Add metadata about the Optuna study
        results_with_metadata = {
            "metadata": {
                "study_name": study_name,
                "best_trial_number": best_trial.number,
                "best_trial_value": best_trial.value,
                "total_trials": len(study.trials),
                "best_params": best_params,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval,
                "collateral": collateral,
                "init_cash": init_cash,
                "coins_tested": len(coins)
            },
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ Results saved to '{output_file}'[/green]")
    
    console.print(f"\n[bold green]🎉 Bulk backtesting with Optuna params completed![/bold green]")
    
    # ==================== Print Summary ====================
    return print_detailed_results_with_optuna(results, best_params, study_name)


def print_detailed_results_with_optuna(results, best_params, study_name):
    """Enhanced results printer that includes Optuna information"""
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    
    console.rule(f"[bold yellow]🚀 BTQuant Results (Optuna Study: {study_name})[/]")
    
    # ✅ Successful backtests
    if successful:
        success_table = Table(title=f"🟢 Successful Backtests ({len(successful)})")
        success_table.add_column("Asset", style="cyan")
        success_table.add_column("Final Value", style="green", justify="right")
        success_table.add_column("P&L", style="magenta", justify="right")
        success_table.add_column("Return %", style="yellow", justify="right")
        
        successful_sorted = sorted(successful, key=lambda x: x.get("pnl", 0), reverse=True)
        for r in successful_sorted:
            asset = r.get("asset", r.get("coin", "N/A"))
            final_value = r.get("final_value")
            pnl = r.get("pnl")
            return_pct = r.get("return_pct", 0)
            
            final_value_str = f"${final_value:,.2f}" if isinstance(final_value, (int, float)) else str(final_value)
            pnl_str = f"${pnl:,.2f}" if isinstance(pnl, (int, float)) else str(pnl)
            return_pct_str = f"{return_pct:.2f}%" if isinstance(return_pct, (int, float)) else str(return_pct)
            
            success_table.add_row(asset, final_value_str, pnl_str, return_pct_str)
        
        console.print(success_table)
    
    # ❌ Failed backtests
    if failed:
        fail_table = Table(title=f"🔴 Failed Backtests ({len(failed)})")
        fail_table.add_column("Asset", style="red")
        fail_table.add_column("Error", style="yellow")
        
        for r in failed:
            fail_table.add_row(
                r.get("asset", r.get("coin", "N/A")),
                str(r.get("error", "Unknown error"))
            )
        
        console.print(fail_table)
    
    # ⚪ Skipped backtests
    if skipped:
        skip_table = Table(title=f"⚪ Skipped Backtests (No Data) ({len(skipped)})")
        skip_table.add_column("Assets", style="white")
        
        skipped_assets = [r.get("asset", r.get("coin", "N/A")) for r in skipped]
        skip_table.add_row(", ".join(skipped_assets))
        
        console.print(skip_table)
    
    # 📊 Summary
    summary = Table(title="📊 Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    
    summary.add_row("Optuna Study", study_name)
    summary.add_row("Parameters Used", str(len(best_params)))
    summary.add_row("Successful", str(len(successful)))
    summary.add_row("Failed", str(len(failed)))
    summary.add_row("Skipped (No Data)", str(len(skipped)))
    
    # Calculate aggregate statistics
    if successful:
        total_pnl = sum(r.get("pnl", 0) for r in successful)
        avg_return = sum(r.get("return_pct", 0) for r in successful) / len(successful)
        summary.add_row("Total P&L", f"${total_pnl:,.2f}")
        summary.add_row("Average Return %", f"{avg_return:.2f}%")
    
    console.print(summary)
    
    return results

