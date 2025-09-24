
import backtrader as bt
import numpy as np
import numba
from rich.console import Console
import math

console = Console()

# Keep all your existing Numba functions - they're fine
@numba.jit(nopython=True, cache=True)
def fast_rsi(prices, period=14):
    """Ultra-fast RSI calculation using Numba JIT compilation"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, 50.0)

    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_values = np.full(n, 50.0)

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period

        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi_values

@numba.jit(nopython=True, cache=True)
def fast_ema_calc(prices, alpha):
    """Ultra-fast EMA calculation with proper type handling"""
    n = len(prices)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    ema = np.empty(n, dtype=np.float64)
    ema[0] = prices[0]

    for i in range(1, n):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

    return ema

@numba.jit(nopython=True, cache=True)
def fast_macd(prices, fast_period, slow_period, signal_period):
    """Ultra-fast MACD calculation"""
    n = len(prices)
    if n < slow_period:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    fast_alpha = 2.0 / (fast_period + 1)
    slow_alpha = 2.0 / (slow_period + 1)
    signal_alpha = 2.0 / (signal_period + 1)

    fast_ema_vals = fast_ema_calc(prices, fast_alpha)
    slow_ema_vals = fast_ema_calc(prices, slow_alpha)
    macd_line = fast_ema_vals - slow_ema_vals
    signal_line = fast_ema_calc(macd_line, signal_alpha)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram

@numba.jit(nopython=True, cache=True)
def fast_adx(high, low, close, period):
    """Fast ADX calculation for trend strength"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 0.0)

    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))

        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move

    # Smooth TR and DM
    atr = np.zeros(n)
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    adx = np.zeros(n)

    # Initialize first values
    if period < n:
        atr[period] = np.mean(tr[1:period+1])
        sum_dm_plus = np.sum(dm_plus[1:period+1])
        sum_dm_minus = np.sum(dm_minus[1:period+1])

        if atr[period] > 0:
            di_plus[period] = 100 * sum_dm_plus / (period * atr[period])
            di_minus[period] = 100 * sum_dm_minus / (period * atr[period])

    # Calculate smoothed values
    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha

        if atr[i] > 0:
            di_plus[i] = di_plus[i-1] * (1 - alpha) + (100 * dm_plus[i] / atr[i]) * alpha
            di_minus[i] = di_minus[i-1] * (1 - alpha) + (100 * dm_minus[i] / atr[i]) * alpha

            dx = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i] + 1e-10) * 100
            adx[i] = adx[i-1] * (1 - alpha) + dx * alpha

    return adx

@numba.jit(nopython=True, cache=True)
def fast_vwap_rolling(prices, volumes, window):
    """Fast rolling VWAP calculation"""
    n = len(prices)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    vwap = np.empty(n, dtype=np.float64)

    for i in range(n):
        start_idx = max(0, i - window + 1)
        price_vol_sum = 0.0
        vol_sum = 0.0

        for j in range(start_idx, i + 1):
            price_vol_sum += prices[j] * volumes[j]
            vol_sum += volumes[j]

        if vol_sum > 0:
            vwap[i] = price_vol_sum / vol_sum
        else:
            vwap[i] = prices[i] if i == 0 else vwap[i-1]

    return vwap

def calculate_tke_simple(close_prices, high_prices, low_prices, period):
    """Simplified but accurate TKE calculation"""
    if len(close_prices) < period:
        return 'neutral'

    # RSI component
    rsi_vals = fast_rsi(close_prices, period)
    rsi_current = rsi_vals[-1]

    # Momentum component
    if len(close_prices) > period:
        momentum = (close_prices[-1] / close_prices[-period]) * 100
    else:
        momentum = 100

    # Williams %R component
    period_high = np.max(high_prices[-period:])
    period_low = np.min(low_prices[-period:])
    if period_high != period_low:
        willr = (period_high - close_prices[-1]) / (period_high - period_low) * -100
    else:
        willr = 0

    tke_line = (rsi_current + momentum + willr) / 3

    # More conservative thresholds
    if tke_line >= 80:  # Was 85
        return 'overbought'
    elif tke_line <= 20:  # Was -5
        return 'oversold'
    else:
        return 'neutral'

def calculate_hull_trend(close_prices, period):
    """Hull MA trend calculation"""
    if len(close_prices) < period:
        return 'neutral'

    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    # Calculate WMAs manually for better control
    def wma(data, n):
        if len(data) < n:
            return np.array([np.mean(data)])
        weights = np.arange(1, n + 1)
        result = []
        for i in range(len(data)):
            if i < n - 1:
                w = weights[:i+1]
                result.append(np.average(data[:i+1], weights=w))
            else:
                result.append(np.average(data[i-n+1:i+1], weights=weights))
        return np.array(result)

    wma_half = wma(close_prices, half_period)
    wma_full = wma(close_prices, period)

    if len(wma_half) == 0 or len(wma_full) == 0:
        return 'neutral'

    wma_diff = 2 * wma_half - wma_full
    hull_ma = wma(wma_diff, sqrt_period)

    if len(hull_ma) >= 2:
        if hull_ma[-1] > hull_ma[-2]:
            return 'bullish'
        elif hull_ma[-1] < hull_ma[-2]:
            return 'bearish'

    return 'neutral'


class CheatcodeStrategyFixed(bt.Strategy):
    """
    FIXED version of CheatcodeStrategy with proper risk management
    - Reduced over-trading
    - Better signal quality
    - Trend filtering
    - Volatility filtering  
    - Improved risk/reward
    """

    params = (
        # Core RSI/MACD Parameters (more conservative)
        ('rsi_period', 14),
        ('rsi_overbought', 75),  # Was 70 - more conservative
        ('rsi_oversold', 25),    # Was 30 - more conservative

        # MACD params
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),

        # GRaB System
        ('ema_grab_period', 34),

        # TKE Oscillator (more conservative)
        ('tke_period', 14),
        ('tke_overbought', 80),  # Was 85
        ('tke_oversold', 20),    # Was -5

        # Hull MA
        ('hull_period', 55),

        # RSI/MACD Combo (more conservative)
        ('combo_rsi_length', 14),    # Was 10
        ('combo_rsi_oversold', 35),  # Was 49 - much more conservative
        ('combo_rsi_overbought', 65), # Was 51 - much more conservative
        ('combo_macd_fast', 8),
        ('combo_macd_slow', 16),
        ('combo_macd_signal', 11),

        # VWAP-MACD
        ('vwap_window', 390),

        # NEW: Trend and Volatility Filters
        ('trend_ema_period', 200),    # Long-term trend
        ('adx_period', 14),
        ('adx_threshold', 25),        # Minimum trend strength
        ('volatility_threshold', 2.0), # Minimum volatility for trading

        # Trading Parameters (MUCH more conservative)
        ('min_signals_for_trade', 4), # Was 2 - require MORE confirmation
        ('max_trades_per_day', 5),    # NEW: Limit trades per day
        ('position_size', 25),        # Was 95 - MUCH smaller position

        # Risk Management (Better risk/reward)
        ('stop_loss_pct', 1.5),       # Tighter stop loss
        ('take_profit_pct', 3.0),     # Better risk/reward (1:2)

        # NEW: Drawdown Protection
        ('max_daily_loss_pct', 5.0),  # Stop trading after 5% daily loss
        ('max_total_dd_pct', 15.0),   # Stop trading after 15% total drawdown

        # System
        ('debug', False),
        ('backtest', None),
    )

    def __init__(self):
        # Your existing buffer setup
        self.data_length = 2000
        self.current_idx = 0

        self.high_data = np.zeros(self.data_length, dtype=np.float64)
        self.low_data = np.zeros(self.data_length, dtype=np.float64)
        self.close_data = np.zeros(self.data_length, dtype=np.float64)
        self.volume_data = np.ones(self.data_length, dtype=np.float64)

        # Pre-compute values
        self.grab_alpha = 2.0 / (self.p.ema_grab_period + 1)
        self.trend_alpha = 2.0 / (self.p.trend_ema_period + 1)

        # Signal tracking
        self.long_signals = set()
        self.short_signals = set()

        # Position tracking
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

        # NEW: Risk management tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.peak_value = 0.0
        self.current_dd = 0.0
        self.trading_disabled = False
        self.last_trade_date = None

        console.print(f"[green]CheatcodeStrategyFixed initialized - Conservative Settings Enabled[/green]")

    def update_data_arrays(self):
        """Your existing efficient data update"""
        idx = self.current_idx % self.data_length

        self.high_data[idx] = float(self.data.high[0])
        self.low_data[idx] = float(self.data.low[0])
        self.close_data[idx] = float(self.data.close[0])

        if hasattr(self.data, 'volume') and self.data.volume[0] is not None:
            self.volume_data[idx] = float(self.data.volume[0])
        else:
            self.volume_data[idx] = 1.0

        self.current_idx += 1

    def get_recent_data(self, lookback=None):
        """Your existing efficient data retrieval"""
        if lookback is None:
            lookback = min(self.current_idx, self.data_length)

        actual_length = min(self.current_idx, lookback)

        if self.current_idx <= self.data_length:
            return {
                'high': self.high_data[:actual_length].copy(),
                'low': self.low_data[:actual_length].copy(),
                'close': self.close_data[:actual_length].copy(),
                'volume': self.volume_data[:actual_length].copy()
            }
        else:
            start_idx = self.current_idx % self.data_length
            high = np.concatenate([self.high_data[start_idx:], self.high_data[:start_idx]])
            low = np.concatenate([self.low_data[start_idx:], self.low_data[:start_idx]])
            close = np.concatenate([self.close_data[start_idx:], self.close_data[:start_idx]])
            volume = np.concatenate([self.volume_data[start_idx:], self.volume_data[:start_idx]])

            return {
                'high': high[-actual_length:],
                'low': low[-actual_length:],
                'close': close[-actual_length:],
                'volume': volume[-actual_length:]
            }

    def check_trading_conditions(self, close_prices, high_prices, low_prices):
        """NEW: Check if we should trade based on trend and volatility"""
        if self.trading_disabled:
            return False

        # Check daily trade limit
        current_date = self.data.datetime.date(0)
        if self.last_trade_date != current_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date

        if self.daily_trades >= self.p.max_trades_per_day:
            if self.p.debug:
                console.print(f"[yellow]Daily trade limit reached ({self.daily_trades})[/yellow]")
            return False

        # Check daily loss limit
        portfolio_value = self.broker.getvalue()
        if self.daily_pnl / portfolio_value * 100 <= -self.p.max_daily_loss_pct:
            if self.p.debug:
                console.print(f"[red]Daily loss limit hit: {self.daily_pnl/portfolio_value*100:.2f}%[/red]")
            return False

        # Check total drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        self.current_dd = (self.peak_value - portfolio_value) / self.peak_value * 100
        if self.current_dd >= self.p.max_total_dd_pct:
            self.trading_disabled = True
            if self.p.debug:
                console.print(f"[red]Total drawdown limit hit: {self.current_dd:.2f}%[/red]")
            return False

        # Trend filter - only trade with the trend
        if len(close_prices) >= self.p.trend_ema_period:
            trend_ema = fast_ema_calc(close_prices, self.trend_alpha)[-1]
            current_price = close_prices[-1]

            # Only allow longs above trend EMA, shorts below trend EMA
            trend_direction = 'up' if current_price > trend_ema else 'down'
        else:
            trend_direction = 'neutral'

        # Volatility filter - need sufficient volatility
        if len(close_prices) >= self.p.adx_period:
            adx = fast_adx(high_prices, low_prices, close_prices, self.p.adx_period)
            current_adx = adx[-1]

            if current_adx < self.p.adx_threshold:
                if self.p.debug:
                    console.print(f"[yellow]Low volatility - ADX: {current_adx:.1f}[/yellow]")
                return False

        return True

    def calculate_indicators_batch(self):
        """Enhanced indicator calculation with filters"""
        data = self.get_recent_data()

        if len(data['close']) < max(self.p.macd_slow, self.p.rsi_period, self.p.trend_ema_period):
            return {}

        close_prices = data['close']
        high_prices = data['high']
        low_prices = data['low']
        volumes = data['volume']

        indicators = {}

        # Check trading conditions first
        if not self.check_trading_conditions(close_prices, high_prices, low_prices):
            indicators['trading_allowed'] = False
            return indicators

        indicators['trading_allowed'] = True

        # RSI (main indicator)
        rsi_values = fast_rsi(close_prices, self.p.rsi_period)
        indicators['rsi'] = rsi_values[-1]

        # VWAP
        vwap_series = fast_vwap_rolling(close_prices, volumes, self.p.vwap_window)
        indicators['vwap'] = vwap_series[-1]

        # GRaB EMA System
        if len(close_prices) >= self.p.ema_grab_period:
            ema_high = fast_ema_calc(high_prices, self.grab_alpha)
            ema_low = fast_ema_calc(low_prices, self.grab_alpha)

            current_close = close_prices[-1]
            if current_close > ema_high[-1]:
                indicators['grab'] = 'bullish'
            elif current_close < ema_low[-1]:
                indicators['grab'] = 'bearish'
            else:
                indicators['grab'] = 'neutral'

        # RSI/MACD Combo (more conservative)
        if len(close_prices) >= max(self.p.combo_macd_slow, self.p.combo_rsi_length):
            combo_rsi = fast_rsi(close_prices, self.p.combo_rsi_length)[-1]
            macd, signal, _ = fast_macd(close_prices, self.p.combo_macd_fast, 
                                     self.p.combo_macd_slow, self.p.combo_macd_signal)

            if len(macd) > 0 and len(signal) > 0:
                # Much more conservative thresholds
                indicators['combo'] = {
                    'long': (combo_rsi < self.p.combo_rsi_oversold and 
                           signal[-1] < macd[-1] and macd[-1] < 0),  # Only when MACD negative
                    'short': (combo_rsi > self.p.combo_rsi_overbought and 
                            signal[-1] > macd[-1] and macd[-1] > 0)   # Only when MACD positive
                }

        # TKE (more conservative)
        indicators['tke'] = calculate_tke_simple(close_prices, high_prices, low_prices, self.p.tke_period)

        # Hull MA trend
        indicators['hull'] = calculate_hull_trend(close_prices, self.p.hull_period)

        # Trend direction
        if len(close_prices) >= self.p.trend_ema_period:
            trend_ema = fast_ema_calc(close_prices, self.trend_alpha)[-1]
            indicators['trend_direction'] = 'up' if close_prices[-1] > trend_ema else 'down'

        return indicators

    def evaluate_signals_conservative(self, indicators, current_price):
        """Much more conservative signal evaluation"""
        if not indicators.get('trading_allowed', False):
            return

        self.long_signals.clear()
        self.short_signals.clear()

        trend_direction = indicators.get('trend_direction', 'neutral')

        # Only generate signals in trend direction
        if trend_direction == 'up':
            # RSI oversold in uptrend
            rsi_val = indicators.get('rsi', 50)
            if rsi_val <= self.p.rsi_oversold:
                self.long_signals.add('rsi_oversold')

            # GRaB bullish
            if indicators.get('grab') == 'bullish':
                self.long_signals.add('grab_bullish')

            # Combo long
            combo = indicators.get('combo', {})
            if combo.get('long'):
                self.long_signals.add('combo_long')

            # TKE oversold
            if indicators.get('tke') == 'oversold':
                self.long_signals.add('tke_oversold')

            # Hull bullish
            if indicators.get('hull') == 'bullish':
                self.long_signals.add('hull_bullish')

        elif trend_direction == 'down':
            # RSI overbought in downtrend
            rsi_val = indicators.get('rsi', 50)
            if rsi_val >= self.p.rsi_overbought:
                self.short_signals.add('rsi_overbought')

            # GRaB bearish  
            if indicators.get('grab') == 'bearish':
                self.short_signals.add('grab_bearish')

            # Combo short
            combo = indicators.get('combo', {})
            if combo.get('short'):
                self.short_signals.add('combo_short')

            # TKE overbought
            if indicators.get('tke') == 'overbought':
                self.short_signals.add('tke_overbought')

            # Hull bearish
            if indicators.get('hull') == 'bearish':
                self.short_signals.add('hull_bearish')

    def next(self):
        """Main strategy logic with enhanced risk management"""
        self.update_data_arrays()
        current_price = float(self.data.close[0])

        if self.current_idx < max(self.p.macd_slow, self.p.rsi_period, self.p.trend_ema_period):
            return

        indicators = self.calculate_indicators_batch()

        if not indicators:
            return

        self.evaluate_signals_conservative(indicators, current_price)

        if self.position:
            self.manage_position_enhanced(current_price)
        else:
            self.evaluate_entry_conservative(current_price)

    def evaluate_entry_conservative(self, current_price):
        """Much more conservative entry logic"""
        long_count = len(self.long_signals)
        short_count = len(self.short_signals)

        # Require MORE confirmations and NO opposing signals
        if (long_count >= self.p.min_signals_for_trade and 
            short_count == 0 and  # NO opposing signals
            long_count > short_count):
            self.enter_position_conservative(current_price, True)
        elif (short_count >= self.p.min_signals_for_trade and 
              long_count == 0 and  # NO opposing signals  
              short_count > long_count):
            self.enter_position_conservative(current_price, False)

    def enter_position_conservative(self, current_price, is_long):
        """Conservative position entry with smaller size"""
        available_cash = self.broker.get_cash()
        # Much smaller position size
        size = int((available_cash * self.p.position_size / 100) / current_price)

        if size > 0:
            if is_long:
                self.buy(size=size)
                self.stop_loss_price = current_price * (1 - self.p.stop_loss_pct / 100)
                self.take_profit_price = current_price * (1 + self.p.take_profit_pct / 100)

                if self.p.debug:
                    console.print(f'[green]CONSERVATIVE LONG[/green]: ${current_price:.4f} Size:{size}')
                    console.print(f'[blue]Signals[/blue]: {self.long_signals}')
            else:
                self.sell(size=size)
                self.stop_loss_price = current_price * (1 + self.p.stop_loss_pct / 100)
                self.take_profit_price = current_price * (1 - self.p.take_profit_pct / 100)

                if self.p.debug:
                    console.print(f'[red]CONSERVATIVE SHORT[/red]: ${current_price:.4f} Size:{size}')
                    console.print(f'[blue]Signals[/blue]: {self.short_signals}')

            self.entry_price = current_price
            self.daily_trades += 1

    def manage_position_enhanced(self, current_price):
        """Enhanced position management"""
        if not self.position:
            return

        is_long = self.position.size > 0

        # Enhanced exit logic
        if is_long:
            if current_price <= self.stop_loss_price:
                self.close()
                pnl = (current_price - self.entry_price) * abs(self.position.size)
                self.daily_pnl += pnl
                if self.p.debug:
                    console.print(f'[red]LONG STOP LOSS[/red]: ${current_price:.4f} PnL: {pnl:.2f}')
                return
            elif current_price >= self.take_profit_price:
                self.close()
                pnl = (current_price - self.entry_price) * abs(self.position.size)
                self.daily_pnl += pnl
                if self.p.debug:
                    console.print(f'[green]LONG TAKE PROFIT[/green]: ${current_price:.4f} PnL: {pnl:.2f}')
                return
        else:
            if current_price >= self.stop_loss_price:
                self.close()
                pnl = (self.entry_price - current_price) * abs(self.position.size)
                self.daily_pnl += pnl
                if self.p.debug:
                    console.print(f'[red]SHORT STOP LOSS[/red]: ${current_price:.4f} PnL: {pnl:.2f}')
                return
            elif current_price <= self.take_profit_price:
                self.close()
                pnl = (self.entry_price - current_price) * abs(self.position.size)
                self.daily_pnl += pnl
                if self.p.debug:
                    console.print(f'[green]SHORT TAKE PROFIT[/green]: ${current_price:.4f} PnL: {pnl:.2f}')
                return

    def notify_trade(self, trade):
        """Enhanced trade notification with P&L tracking"""
        if trade.isclosed:
            self.daily_pnl += trade.pnl
            if self.p.debug:
                color = "[green]" if trade.pnl > 0 else "[red]"
                console.print(f'{color}TRADE CLOSED: PnL ${trade.pnl:.2f}[/{color.strip("[]")}]')


# Alias for compatibility
CheatcodeStrategy = CheatcodeStrategyFixed
