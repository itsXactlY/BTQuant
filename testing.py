from backtrader.utils.backtest import backtest
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX4 as strategy
# from backtrader.strategies.MACD_ADX import VectorMACD_ADX as strategy
import optuna
from testing_optuna_newmacd import build_optuna_storage
# from backtrader.dontcommit import connection_string as MSSQL_ODBC
from backtrader.dontcommit import optuna_connection_string as MSSQL_ODBC
storage = build_optuna_storage(MSSQL_ODBC)
# study = optuna.load_study(study_name="BullBearMarketBTC-ETH-LTC-XRP-BCH_1m_MACD_ADXV3", storage=storage)
study = optuna.load_study(study_name="Optimized_1m_MTF_VEC_V7", storage=storage)

from rich.console import Console
console = Console()


##### Strategy
import math
import backtrader as bt
class VectorMACD_ADX(bt.Strategy):
    params = (
        # position sizing / risk
        ('risk_per_trade_pct', 0.005),
        ('exposure_pct_cap', 0.95),
        ('reserve_cash_pct', 0.02),
        ('slip_fee_buffer_bps', 10.0),
        ('max_leverage', 1.0),

        # exits / trailing
        ('use_dynamic_exits', True),
        ('tp_r_multiple', 2.0),
        ('use_partial_exits', True),
        ('partial_exit_r', 1.5),
        ('partial_exit_pct', 0.5),
        ('time_limit_bars', 120),
        ('atr_stop_mult', 2.5),
        ('use_trailing_stop', True),
        ('trail_mode', 'chandelier'),
        ('trail_atr_mult', 4.0),
        ('move_to_breakeven_R', 1.0),

        # execution / position management
        ('reentry_cooldown_bars', 5),
        ('min_qty', 0.0),
        ('qty_step', 0.001),
        ('price_tick', 0.1),
        ('round_prices', True),
        ('can_short', False),

        # indicators / signals (base)
        ('ema_fast', 20),
        ('ema_slow', 50),
        ('ema_trend', 200),
        ('atr_period', 14),
        ('rsi_overheat', 75),
        ('rsi_oversold', 25),
        ('adxth', 20),
        ('confirm_bars', 2),
        ('max_stretch_atr_mult', 1.0),
        ('ema_band_mult', 2.0),
        ('donchian_trail_period', 55),
        ('close_based_stop', True),
        ('trail_update_every', 2),
        ('max_bars_in_trade', 360),
        ('use_pyramiding', False),
        ('max_adds', 5),
        ('add_cooldown', 20),
        ('add_atr_mult', 1.0),
        ('add_min_R', 1.0),

        # volume filter
        ('use_volume_filter', False),
        ('volume_filter_mult', 1.2),

        # legacy / flags
        ('backtest', True),
        ('debug', False),

        # regime
        ('regime_mode_long', 'ema'),
        ('regime_mode_short', 'neutral'),

        # enable HTF usage
        ('use_htf', True),

        # fallback TP if dynamic exits disabled
        ('take_profit', 4.0),

        # --- New params that Optuna may pass ---
        ('breakout_period', 40),
        ('htf1_adx_period', 14),
        ('htf1_ema_fast', 40),
        ('htf1_ema_slow', 120),
        ('htf2_ema_fast', 80),
        ('htf2_ema_slow', 200),

        # compatibility names
        ('tf5m_breakout_period', 55),
        ('tf15m_adx_period', 14),
        ('tf15m_ema_fast', 50),
        ('tf15m_ema_slow', 200),
        ('tf60m_ema_fast', 50),
        ('tf60m_ema_slow', 200),
    )

    def __init__(self):
        self.d = self.datas[0]
        self._reset_position_state()
        self.active_orders = []

    def _reset_position_state(self):
        self.entry_bar = None
        self.trail_stop = None
        self.init_stop = None
        self.initial_risk = None
        self.run_high = None
        self.run_low = None
        self.partial_exit_done = False
        self.trail_stop_active = False
        self.last_exit_bar = None

    def _round_qty(self, s):
        step = self.p.qty_step if self.p.qty_step > 0 else 0.001
        q = math.floor(max(0.0, s) / step) * step
        if q == 0.0 and s > 0:
            q = step
        return 0.0 if (self.p.min_qty > 0 and q < self.p.min_qty) else q

    def _round_price(self, p):
        return round(p / self.p.price_tick) * self.p.price_tick if (self.p.round_prices and self.p.price_tick > 0) else p

    def _risk_based_size(self, entry_price, stop_price):
        eq = float(self.broker.getvalue())
        cash = float(self.broker.getcash())

        dist = abs(entry_price - stop_price)
        if dist <= 0 or math.isclose(dist, 0.0):
            return 0.0

        risk_dollars = max(0.0, eq * self.p.risk_per_trade_pct)
        raw_units = risk_dollars / dist

        buffer_mult = 1.0 + (self.p.slip_fee_buffer_bps / 10000.0)
        max_cash_for_position = max(
            0.0,
            (cash * self.p.exposure_pct_cap) - (eq * self.p.reserve_cash_pct)
        )
        cash_cap_units = (max_cash_for_position / (entry_price * buffer_mult)) if entry_price > 0 else 0.0

        size = min(raw_units, cash_cap_units)
        return self._round_qty(size)

    def _get_atr_value(self):
        """Safely get ATR value with fallback options"""
        atr_value = None
        
        # Try atr_brk first
        try:
            atr_attr = getattr(self.d, 'atr_brk', None)
            if atr_attr is not None:
                atr_val = atr_attr[0] if hasattr(atr_attr, '__getitem__') else atr_attr
                if atr_val is not None and not math.isnan(float(atr_val)):
                    atr_value = float(atr_val)
        except (AttributeError, IndexError, ValueError, TypeError):
            pass
        
        # Try atr_base as fallback
        if atr_value is None:
            try:
                atr_attr = getattr(self.d, 'atr_base', None)
                if atr_attr is not None:
                    atr_val = atr_attr[0] if hasattr(atr_attr, '__getitem__') else atr_attr
                    if atr_val is not None and not math.isnan(float(atr_val)):
                        atr_value = float(atr_val)
            except (AttributeError, IndexError, ValueError, TypeError):
                pass
        
        # Final fallback - estimate from price data
        if atr_value is None:
            try:
                if hasattr(self.d, 'high') and hasattr(self.d, 'low'):
                    high_val = self.d.high[0]
                    low_val = self.d.low[0]
                    if high_val is not None and low_val is not None:
                        atr_value = abs(float(high_val) - float(low_val))
            except (IndexError, ValueError, TypeError):
                pass
        
        return atr_value

    def _enter(self, direction: int):
        try:
            entry = self._round_price(float(self.d.close[0]))
        except (ValueError, TypeError):
            return

        atr_value = self._get_atr_value()
        if atr_value is None or atr_value <= 0:
            return

        stop_dist = self.p.atr_stop_mult * atr_value
        init_stop = self._round_price(entry - stop_dist * direction)

        size = self._risk_based_size(entry, init_stop)
        if size <= 0:
            return

        self.initial_risk = abs(entry - init_stop)

        if self.p.use_dynamic_exits:
            self.take_profit_price = self._round_price(entry + self.p.tp_r_multiple * self.initial_risk * direction)
            self.partial_tp_price = self._round_price(entry + self.p.partial_exit_r * self.initial_risk * direction)
        else:
            self.take_profit_price = self._round_price(entry * (1 + (getattr(self.p, "take_profit", 4.0) / 100.0) * direction))
            self.partial_tp_price = None

        if direction > 0:
            o = self.buy(size=size)
            try:
                self.run_high = float(self.d.high[0])
            except (ValueError, TypeError):
                self.run_high = entry
        else:
            o = self.sell(size=size)
            try:
                self.run_low = float(self.d.low[0])
            except (ValueError, TypeError):
                self.run_low = entry

        self.init_stop = init_stop
        self.trail_stop = init_stop
        self.entry_bar = len(self)
        self.active_orders.append(o)

    def notify_order(self, o):
        if o.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return

        if o.status == bt.Order.Completed:
            try:
                self.entry_bar = len(self)
            except Exception:
                self.entry_bar = 0

            if o.isbuy():
                try:
                    high_val = self.d.high[0]
                    self.run_high = float(high_val) if high_val is not None else float(self.d.close[0])
                except (ValueError, TypeError, IndexError):
                    try:
                        self.run_high = float(self.d.close[0])
                    except (ValueError, TypeError):
                        self.run_high = o.executed.price
                        
                if self.p.debug:
                    console.log(f'BUY EXECUTED, Price: {o.executed.price:.2f}, Cost: {o.executed.value:.2f}, Comm: {o.executed.comm:.2f}')
                    
            elif o.issell():
                try:
                    low_val = self.d.low[0]
                    self.run_low = float(low_val) if low_val is not None else float(self.d.close[0])
                except (ValueError, TypeError, IndexError):
                    try:
                        self.run_low = float(self.d.close[0])
                    except (ValueError, TypeError):
                        self.run_low = o.executed.price
                        
                if self.p.debug:
                    console.log(f'SELL EXECUTED, Price: {o.executed.price:.2f}, Cost: {o.executed.value:.2f}, Comm: {o.executed.comm:.2f}')

        elif o.status in [bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected]:
            try:
                status_name = bt.Order.Status[o.status]
            except Exception:
                status_name = str(o.status)
            if self.p.debug:
                console.print(f'Order {status_name}')

        if o in self.active_orders:
            self.active_orders.remove(o)

    def _R(self):
        if not self.position:
            return 0.0
        try:
            current_price = float(self.d.close[0])
            direction = 1 if self.position.size > 0 else -1
            return (current_price - self.init_stop) * direction / (self.initial_risk if self.initial_risk else 1e-9)
        except (ValueError, TypeError):
            return 0.0

    def _update_trailing_stop(self):
        if not self.position:
            self.trail_stop = None
            return

        # Safety checks for data availability
        try:
            current_high = self.d.high[0]
            current_low = self.d.low[0]
            if current_high is None or current_low is None:
                return
            current_high = float(current_high)
            current_low = float(current_low)
        except (ValueError, TypeError, IndexError):
            return

        # Update run_high/run_low safely
        if self.position.size > 0:
            if self.run_high is None:
                self.run_high = current_high
            else:
                self.run_high = max(self.run_high, current_high)
        else:
            if self.run_low is None:
                self.run_low = current_low
            else:
                self.run_low = min(self.run_low, current_low)

        # Get ATR value
        atr_value = self._get_atr_value()
        if atr_value is None or atr_value <= 0:
            return

        cand = None
        if self.p.trail_mode == "chandelier":
            if self.position.size > 0 and self.run_high is not None:
                cand = self.run_high - self.p.trail_atr_mult * atr_value
            elif self.position.size < 0 and self.run_low is not None:
                cand = self.run_low + self.p.trail_atr_mult * atr_value
        elif self.p.trail_mode == "donchian":
            try:
                if self.position.size > 0:
                    dc_attr = getattr(self.d, 'dc_exit_low', None)
                    if dc_attr is not None:
                        cand = dc_attr[0] if hasattr(dc_attr, '__getitem__') else dc_attr
                else:
                    dc_attr = getattr(self.d, 'dc_exit_high', None)
                    if dc_attr is not None:
                        cand = dc_attr[0] if hasattr(dc_attr, '__getitem__') else dc_attr
                
                if cand is not None:
                    cand = float(cand)
            except (AttributeError, IndexError, ValueError, TypeError):
                cand = None

        if cand is not None:
            # Apply safety bounds
            if self.position.size > 0:
                cand = max(cand, self.init_stop if self.init_stop is not None else -1e18)
            else:
                cand = min(cand, self.init_stop if self.init_stop is not None else 1e18)
            
            # Update trailing stop
            if self.trail_stop is None:
                self.trail_stop = cand
            else:
                if self.position.size > 0:
                    self.trail_stop = max(self.trail_stop, cand)
                else:
                    self.trail_stop = min(self.trail_stop, cand)

    def _stop_hit(self):
        if self.trail_stop is None:
            return False
        
        try:
            if self.position.size > 0:
                if self.p.close_based_stop:
                    return float(self.d.close[0]) <= self.trail_stop
                else:
                    return float(self.d.low[0]) <= self.trail_stop
            else:
                if self.p.close_based_stop:
                    return float(self.d.close[0]) >= self.trail_stop
                else:
                    return float(self.d.high[0]) >= self.trail_stop
        except (ValueError, TypeError, IndexError):
            return False

    def next(self):
        if not self.position:
            # Check reentry cooldown
            if hasattr(self, 'last_exit_bar') and self.last_exit_bar is not None:
                if (len(self) - self.last_exit_bar) < self.p.reentry_cooldown_bars:
                    return
            
            # Check for entry signals
            try:
                long_sig = getattr(self.d, 'long_entry_signal', None)
                short_sig = getattr(self.d, 'short_entry_signal', None)
                
                if long_sig is not None and hasattr(long_sig, '__getitem__'):
                    if long_sig[0]:
                        self._enter(1)
                elif self.p.can_short and short_sig is not None and hasattr(short_sig, '__getitem__'):
                    if short_sig[0]:
                        self._enter(-1)
            except (IndexError, AttributeError):
                pass
                
        else:
            try:
                current_price = float(self.d.close[0])
                direction = 1 if self.position.size > 0 else -1

                # Partial exits
                if (self.p.use_partial_exits and not self.partial_exit_done and 
                    hasattr(self, 'partial_tp_price') and self.partial_tp_price is not None):
                    
                    if ((direction > 0 and current_price >= self.partial_tp_price) or 
                        (direction < 0 and current_price <= self.partial_tp_price)):
                        
                        partial_size = self._round_qty(abs(self.position.size) * self.p.partial_exit_pct)
                        if partial_size > 0:
                            if direction > 0:
                                self.sell(size=partial_size)
                            else:
                                self.buy(size=partial_size)
                            self.partial_exit_done = True
                            self.trail_stop_active = True

                # Take profit
                if hasattr(self, 'take_profit_price') and self.take_profit_price is not None:
                    if ((direction > 0 and current_price >= self.take_profit_price) or 
                        (direction < 0 and current_price <= self.take_profit_price)):
                        self.close()
                        return

                # Time limit
                if (self.p.time_limit_bars > 0 and self.entry_bar is not None and 
                    (len(self) - self.entry_bar) > self.p.time_limit_bars and not self.partial_exit_done):
                    self.close()
                    return

                # Trailing stop
                if (self.p.use_trailing_stop and 
                    (not self.p.use_partial_exits or self.partial_exit_done)):
                    self._update_trailing_stop()
                    if self._stop_hit():
                        self.close()
                        return
                        
            except (ValueError, TypeError, IndexError):
                pass

    def notify_trade(self, trade):
        if trade.isclosed:
            self.last_exit_bar = len(self)
            self._reset_position_state()


strategy = VectorMACD_ADX
##### Strategy end


trial_num = None # or None for best
trial = (study.best_trial if trial_num is None
         else next(t for t in study.get_trials(deepcopy=False) if t.number == trial_num))

raw_params = trial.params

def get_param_names(cls) -> set:
    names = set()
    try:
        names = set(cls.params._getkeys())  # type: ignore[attr-defined]
    except Exception:
        try:
            # legacy tuple-of-tuples style
            names = set(k for k, _ in cls.params)  # type: ignore[assignment]
        except Exception:
            # fallback: just trust trial params
            names = set(raw_params.keys())
    return names

param_names = get_param_names(strategy)
params = {k: v for k, v in raw_params.items() if k in param_names}


# --------------- Data spec ---------------
bull_start = "2020-09-28"
bull_end = "2021-05-31"
bear_start = "2022-05-28"
bear_end = "2023-06-23"
# Optional holdout test period
test_bull_start="2023-06-12"
test_bull_end="2025-05-31"
tf = "15m"


###
import backtrader as bt
import gc
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Callable
import optuna
import polars as pl
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData


INIT_CASH = 1000.0
COMMISSION_PER_TRANSACTION = 0.00075
DEFAULT_COLLATERAL = "USDT"


@dataclass(frozen=True)
class DataSpec:
    symbol: str
    interval: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    ranges: Optional[List[Tuple[str, str]]] = None
    collateral: str = DEFAULT_COLLATERAL

def score_sharpe_dd(strat, lam_dd=0.03):
    """
    Score = Sharpe - lam_dd * (MaxDD/100). Maximiert Sharpe, bestraft hohen DD.
    """
    import math
    try:
        sr = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        sharpe = float(sr) if sr is not None and not math.isnan(sr) else 0.0
    except Exception:
        sharpe = 0.0

    try:
        draw = strat.analyzers.drawdown.get_analysis()
        mdd = float(draw.get("max", {}).get("drawdown", 0.0))
    except Exception:
        mdd = 0.0

    try:
        ta = strat.analyzers.trades.get_analysis()
        trades = ta.get("total", {}).get("total", 0) or 0
        won = ta.get("won", {}).get("total", 0) or 0
        lost = ta.get("lost", {}).get("total", 0) or 0
        win_rate = (won / trades) if trades > 0 else 0.0
    except Exception:
        trades, won, lost, win_rate = 0, 0, 0, 0.0

    score = sharpe - lam_dd * (mdd / 100.0)
    metrics = dict(mdd=mdd, sharpe=sharpe, trades=trades, win_rate=win_rate)
    return score, metrics

def compat_params_to_mtf(p: dict) -> dict:
    mtf = dict(
        # MTF/Guards
        tf5m_breakout_period=p.get('breakout_period', 55),
        adxth=p.get('adxth', 22),
        confirm_bars=p.get('confirm_bars', 2),
        max_stretch_atr_mult=p.get('stretch_atr_mult', 1.0),

        # Stops/Trail
        atr_stop_mult=p.get('init_sl_atr_mult', 1.25),
        trail_mode=p.get('trail_mode', 'chandelier'),
        trail_atr_mult=p.get('trail_atr_mult', 4.0),
        ema_band_mult=p.get('ema_band_mult', 2.0),
        donchian_trail_period=p.get('donchian_trail_period', 55),
        move_to_breakeven_R=p.get('trail_arm_R', 1.0),
        trail_update_every=p.get('trail_update_every', 3),

        # Core
        ema_fast=p.get('ema_fast', 20),
        ema_slow=p.get('ema_slow', 50),
        ema_trend=p.get('ema_trend', 200),
        atr_period=p.get('atr_period', 14),

        # TP
        take_profit=p.get('take_profit', 4.0),

        # Volume
        use_volume_filter=(p.get('volume_filter_mult', 1.2) > 1.0),
        volume_filter_mult=p.get('volume_filter_mult', 1.2),

        # Pyramiding (ersetzt DCA)
        use_pyramiding=(p.get('max_adds', 0) > 0 or p.get('use_dca', False)),
        max_adds=p.get('max_adds', 0),
        add_cooldown=p.get('add_cooldown', 20),
        add_atr_mult=p.get('dca_atr_mult', 1.0),
        add_min_R=p.get('add_min_R', 1.0),

        # Defaults
        rsi_overheat=75,
        use_htf=True,
    )
    return mtf

def preload_polars(
    specs: List[DataSpec],
) -> Dict[str, pl.DataFrame]:
    """
    Load each asset once using your get_database_data (Polars), keep in memory.
    Keyed by symbol. Assumes same interval/date per symbol as in specs.
    """
    df_map: Dict[str, pl.DataFrame] = {}
    seen = set()

    for spec in specs:
        if spec.symbol in seen:
            continue

        if spec.ranges:
            dfs = []
            for s, e in spec.ranges:
                part = get_database_data(
                    ticker=spec.symbol,
                    start_date=s,
                    end_date=e,
                    time_resolution=spec.interval,
                    pair=spec.collateral,
                )
                if part is None or part.is_empty():
                    raise ValueError(f"No data for {spec.symbol} {spec.interval} {s}->{e}")
                dfs.append(part)
            df = pl.concat(dfs).sort("TimestampStart")
        else:
            df = get_database_data(
                ticker=spec.symbol,
                start_date=spec.start_date,
                end_date=spec.end_date,
                time_resolution=spec.interval,
                pair=spec.collateral,
            )
        if df is None or df.is_empty():
            raise ValueError(f"No data for {spec.symbol} {spec.interval} {spec.start_date}->{spec.end_date}")

        # Ensure expected columns exist (your loader provides these)
        required = {"TimestampStart", "Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"{spec.symbol}: missing required columns. Found: {df.columns}")

        # Sort by time
        df = df.sort("TimestampStart")

        # Data validation - check for null/invalid values
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                # Replace null values with forward fill or drop rows
                if df[col].null_count() > 0:
                    console.print(f"[yellow]Warning: {spec.symbol} has {df[col].null_count()} null values in {col}[/yellow]")
                    df = df.with_columns(pl.col(col).fill_null(strategy="forward"))
                
                # Check for invalid values (negative prices, etc.)
                invalid_count = df.filter(pl.col(col) <= 0).height
                if invalid_count > 0:
                    console.print(f"[yellow]Warning: {spec.symbol} has {invalid_count} invalid values in {col}[/yellow]")
                    df = df.filter(pl.col(col) > 0)

        # Ensure High >= Low, Close/Open within range
        df = df.filter(
            (pl.col("High") >= pl.col("Low")) &
            (pl.col("Close") <= pl.col("High")) &
            (pl.col("Close") >= pl.col("Low")) &
            (pl.col("Open") <= pl.col("High")) &
            (pl.col("Open") >= pl.col("Low"))
        )

        if df.is_empty():
            raise ValueError(f"No valid data remaining for {spec.symbol} after cleaning")

        df_map[spec.symbol] = df
        seen.add(spec.symbol)

    return df_map


def make_feed_from_df(df: pl.DataFrame, spec: DataSpec) -> MSSQLData:
    """
    Wrap a cloned Polars DataFrame into your PolarsData feed.
    Always pass explicit mapping, no guessing.
    """
    cloned = df.clone()  # safe, cheap
    feed = MSSQLData(
        dataname=cloned,
        datetime="TimestampStart",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
    )
    try:
        feed._name = f"{spec.symbol}-{spec.interval}"
        feed._dataname = f"{spec.symbol}{spec.collateral}"
    except Exception:
        pass
    return feed

def run_single_backtest_eval(
    strategy_class,
    df: pl.DataFrame,
    spec: DataSpec,
    init_cash: float,
    commission: float,
    params: Dict[str, Any],
    exchange: Optional[str] = None,    # NEW
    slippage_bps: float = 5.0,         # NEW
    min_qty: float = 0.0,              # NEW
    qty_step: float = 1.0,             # NEW
    price_tick: Optional[float] = None,# NEW
    params_mode: str = "compat",       # "compat" (map altes Set) oder "mtf"
    score_fn: Callable = score_sharpe_dd,  # NEW
) -> Tuple[float, Dict[str, float], float]:
    
    # Data validation
    # if df.is_empty():
    #     return -999.0, {"error": 1.0}, 0.0
    
    # Check for required columns and valid data
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in df.columns:
            return -999.0, {"error": 1.0}, 0.0
        if df[col].null_count() == len(df):
            return -999.0, {"error": 1.0}, 0.0
    
    cerebro = bt.Cerebro(oldbuysell=True)

    # Feed
    feed = make_feed_from_df(df, spec)
    cerebro.adddata(feed)
    
    # MTF Resamples
    try:
        cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=5)
        cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=15)
        cerebro.resampledata(feed, timeframe=bt.TimeFrame.Minutes, compression=60)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not create resampled data: {e}[/yellow]")

    # Params vorbereiten
    if params_mode == "compat":
        strat_params = compat_params_to_mtf(params)
    else:
        strat_params = dict(params)

    # Short nur auf MEXC
    can_short = (str(exchange).lower() == "mexc") if exchange else False
    strat_params.update(dict(
        can_short=can_short,
        min_qty=min_qty,
        qty_step=qty_step,
        price_tick=price_tick,
        debug=False  # Disable debug in batch runs
    ))

    try:
        cerebro.addstrategy(strategy_class, backtest=True, **strat_params)
    except Exception as e:
        console.print(f"[red]Failed to add strategy: {e}[/red]")
        return -999.0, {"error": 1.0}, 0.0

    # Analyzers (gleich wie vorher, Namen wichtig)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)

    cerebro.broker.setcommission(commission=commission)
    try:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10000.0)
    except Exception:
        pass
    cerebro.broker.setcash(init_cash)

    try:
        strats = cerebro.run()
        if not strats:
            return -999.0, {"error": 1.0}, 0.0
        strat = strats[0]
    except Exception as e:
        console.print(f"[red]Backtest failed for {spec.symbol}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return -999.0, {"error": 1.0}, 0.0

    try:
        score, metrics = score_fn(strat)
        final_value = cerebro.broker.getvalue()
    except Exception as e:
        console.print(f"[red]Failed to calculate metrics for {spec.symbol}: {e}[/red]")
        score, metrics, final_value = -999.0, {"error": 1.0}, init_cash

    # Cleanup
    try:
        del cerebro, feed, strats, strat
        gc.collect()
    except Exception:
        pass

    return score, metrics, final_value

if __name__ == '__main__':
    console.print(f"Using params: {params}")
    console.print(f"Trial number: {trial.number}")
    
    # Create data specs
    specs = [DataSpec("BTC", interval="1m", ranges=[("2020-09-28", "2021-05-31")])]
    
    try:
        # Load data
        console.print("[cyan]Loading data...[/cyan]")
        df_map = preload_polars(specs)
        console.print(f"[green]Data loaded successfully. BTC shape: {df_map['BTC'].shape}[/green]")
        
        # Run backtest
        console.print("[cyan]Running backtest...[/cyan]")
        score, metrics, final_value = run_single_backtest_eval(
            strategy_class=strategy,
            df=df_map["BTC"],
            spec=specs[0],
            init_cash=1000,
            commission=0.00075,
            params=params,
            params_mode="compat",  # Use compatibility mode
        )
        
        # Print results
        console.print(f"[green]Backtest completed![/green]")
        console.print(f"Score: {score:.4f}")
        console.print(f"Final Value: ${final_value:.2f}")
        console.print(f"Metrics: {metrics}")
        
        # Calculate additional metrics
        if final_value > 0:
            total_return = (final_value / 1000.0 - 1.0) * 100
            console.print(f"Total Return: {total_return:.2f}%")

    except Exception as e:
        console.print(f"[red]An error occurred: {e}[/red]")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        console.print("[yellow]Process interrupted by user.[/yellow]")