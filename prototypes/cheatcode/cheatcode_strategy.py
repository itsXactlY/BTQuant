
import backtrader as bt
import numpy as np
import pandas as pd
import math
from collections import deque

class TechnicalIndicators:
    """Custom technical indicators for the Cheatcode strategy"""

    @staticmethod
    def rsi(prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)

        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(period).mean()
        avg_loss = pd.Series(loss).rolling(period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50.0], rsi.fillna(50.0).values])

    @staticmethod
    def ema(prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) == 0:
            return np.array([])
        return pd.Series(prices).ewm(span=period).mean().values

    @staticmethod
    def sma(prices, period):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.mean(prices) if len(prices) > 0 else 0)
        return pd.Series(prices).rolling(period).mean().fillna(method='bfill').values

    @staticmethod
    def wma(prices, period):
        """Calculate Weighted Moving Average"""
        if len(prices) < period:
            return np.full(len(prices), np.mean(prices) if len(prices) > 0 else 0)

        weights = np.arange(1, period + 1)
        wma_values = []

        for i in range(len(prices)):
            if i < period - 1:
                wma_values.append(np.mean(prices[:i+1]))
            else:
                window = prices[i-period+1:i+1]
                wma_values.append(np.dot(window, weights) / weights.sum())

        return np.array(wma_values)

class CheatcodeStrategy(bt.Strategy):
    """
    Complete Cheatcode Strategy combining Cheatcode-ZERO© and Cheatcode-A© by aLca

    This strategy implements:
    - Camarilla Pivot Points for support/resistance
    - RSI/MACD combination signals
    - GRaB EMA system
    - TKE composite oscillator
    - Hull Moving Average trend detection
    - VWAP analysis
    - MACD forecast with divergence detection
    """

    params = (
        # RSI Parameters
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),

        # MACD Parameters
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),

        # GRaB System
        ('ema_grab_period', 34),

        # TKE Oscillator
        ('tke_period', 14),
        ('tke_ema_period', 5),
        ('tke_overbought', 85),
        ('tke_oversold', -5),

        # Hull MA
        ('hull_period', 55),

        # RSI/MACD Combo
        ('combo_rsi_length', 10),
        ('combo_rsi_oversold', 49),
        ('combo_rsi_overbought', 51),
        ('combo_macd_fast', 8),
        ('combo_macd_slow', 16),
        ('combo_macd_signal', 11),

        # Forecast
        ('forecast_periods', 6),
        ('linreg_len', 3),

        # Trading Parameters
        ('min_signals_for_trade', 2),  # Minimum confirmations needed
        ('position_size', 0.95),       # Position size as % of available cash

        # Risk Management
        ('stop_loss_pct', 2.0),        # Stop loss percentage
        ('take_profit_pct', 4.0),      # Take profit percentage

        # BTQuant
        ('backtest', None),
    )

    def __init__(self):
        # Data series for calculations
        self.high_data = deque(maxlen=200)
        self.low_data = deque(maxlen=200)
        self.close_data = deque(maxlen=200)
        self.volume_data = deque(maxlen=200)

        # Indicator storage
        self.indicators = {}

        # Signal tracking
        self.current_signals = {'long': [], 'short': []}
        self.last_pivot_levels = {}

        # Position tracking
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

        # Add Backtrader built-in indicators for comparison
        self.rsi_bt = bt.indicators.RSI(period=self.p.rsi_period)
        self.macd_bt = bt.indicators.MACD(
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )

        print(f"CheatcodeStrategy initialized with {len(self.params._getkwargs())} parameters")

    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

    def calculate_camarilla_pivots(self):
        """Calculate Camarilla pivot points"""
        if len(self.close_data) < 1:
            return {}

        # Use previous day's data
        high_prev = self.high_data[-1]
        low_prev = self.low_data[-1]
        close_prev = self.close_data[-1]

        range_val = high_prev - low_prev

        # Camarilla levels
        H5 = high_prev / low_prev * close_prev if low_prev != 0 else close_prev
        H4 = close_prev + range_val * 1.1 / 2
        H3 = close_prev + range_val * 1.1 / 4
        H2 = close_prev + range_val * 1.1 / 6
        H1 = close_prev + range_val * 1.1 / 12
        H6 = H5 + 1.168 * (H5 - H4)

        L1 = close_prev - range_val * 1.1 / 12
        L2 = close_prev - range_val * 1.1 / 6
        L3 = close_prev - range_val * 1.1 / 4
        L4 = close_prev - range_val * 1.1 / 2
        L5 = close_prev - (H5 - close_prev)
        L6 = close_prev - (H6 - close_prev)

        # Standard pivots
        pivot = (high_prev + low_prev + close_prev) / 3.0
        bc = (high_prev + low_prev) / 2.0
        tc = pivot - bc + pivot

        return {
            'H6': H6, 'H5': H5, 'H4': H4, 'H3': H3, 'H2': H2, 'H1': H1,
            'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4, 'L5': L5, 'L6': L6,
            'P': pivot, 'BC': bc, 'TC': tc
        }

    def calculate_grab_signals(self):
        """Calculate GRaB EMA system signals"""
        if len(self.close_data) < self.p.ema_grab_period:
            return None

        period = self.p.ema_grab_period

        # Calculate EMAs
        ema_high = TechnicalIndicators.ema(list(self.high_data), period)
        ema_low = TechnicalIndicators.ema(list(self.low_data), period)

        current_close = self.close_data[-1]
        current_ema_high = ema_high[-1]
        current_ema_low = ema_low[-1]

        # Signal generation
        if current_close > current_ema_high:
            return 'bullish'
        elif current_close < current_ema_low:
            return 'bearish'
        else:
            return 'neutral'

    def calculate_rsi_macd_combo(self):
        """Calculate RSI/MACD combination signals"""
        if len(self.close_data) < max(self.p.combo_macd_slow, self.p.combo_rsi_length):
            return {'long': False, 'short': False}

        close_array = np.array(list(self.close_data))

        # MACD calculation
        fast_ma = TechnicalIndicators.ema(close_array, self.p.combo_macd_fast)
        slow_ma = TechnicalIndicators.ema(close_array, self.p.combo_macd_slow)
        macd = fast_ma - slow_ma
        signal = TechnicalIndicators.sma(macd, self.p.combo_macd_signal)

        # RSI calculation
        rsi = TechnicalIndicators.rsi(close_array, self.p.combo_rsi_length)

        # Signal conditions
        current_rsi = rsi[-1]
        current_signal = signal[-1]
        current_macd = macd[-1]

        long_condition = (current_rsi > self.p.combo_rsi_overbought and 
                         current_signal < current_macd)
        short_condition = (current_rsi < self.p.combo_rsi_oversold and 
                          current_signal > current_macd)

        return {
            'long': long_condition,
            'short': short_condition,
            'rsi': current_rsi,
            'macd': current_macd,
            'signal': current_signal
        }

    def calculate_tke_signals(self):
        """Calculate TKE oscillator signals"""
        if len(self.close_data) < self.p.tke_period:
            return None

        close_array = np.array(list(self.close_data))
        high_array = np.array(list(self.high_data))
        low_array = np.array(list(self.low_data))

        # Simplified TKE calculation
        rsi = TechnicalIndicators.rsi(close_array, self.p.tke_period)

        # Momentum component
        if len(close_array) > self.p.tke_period:
            momentum = close_array[self.p.tke_period:] / close_array[:-self.p.tke_period] * 100
            momentum = np.concatenate([np.full(self.p.tke_period, 100), momentum])
        else:
            momentum = np.full(len(close_array), 100)

        # Williams %R approximation
        period = self.p.tke_period
        willr = np.zeros_like(close_array)

        for i in range(period, len(close_array)):
            highest_high = np.max(high_array[i-period:i])
            lowest_low = np.min(low_array[i-period:i])
            if highest_high != lowest_low:
                willr[i] = (highest_high - close_array[i]) / (highest_high - lowest_low) * -100

        # Combine components
        tke_line = (rsi + momentum + willr) / 3

        # Current TKE value
        current_tke = tke_line[-1]

        if current_tke >= self.p.tke_overbought:
            return 'overbought'
        elif current_tke <= self.p.tke_oversold:
            return 'oversold'
        else:
            return 'neutral'

    def calculate_hull_trend(self):
        """Calculate Hull MA trend"""
        if len(self.close_data) < self.p.hull_period:
            return None

        close_array = np.array(list(self.close_data))

        # Hull MA calculation
        wma_half = TechnicalIndicators.wma(close_array, self.p.hull_period // 2)
        wma_full = TechnicalIndicators.wma(close_array, self.p.hull_period)

        wma_diff = 2 * wma_half - wma_full
        sqrt_period = int(math.sqrt(self.p.hull_period))
        hull_ma = TechnicalIndicators.wma(wma_diff, sqrt_period)

        # Trend determination
        if len(hull_ma) >= 2 and hull_ma[-1] > hull_ma[-2]:
            return 'bullish'
        elif len(hull_ma) >= 2 and hull_ma[-1] < hull_ma[-2]:
            return 'bearish'
        else:
            return 'neutral'

    def evaluate_pivot_action(self, current_price, pivots):
        """Evaluate action based on current price and pivot levels"""
        actions = []

        # Check for pivot level interactions
        tolerance = current_price * 0.001  # 0.1% tolerance

        # Support level bounces (bullish)
        for level in ['L3', 'L4', 'L5']:
            if level in pivots:
                level_price = pivots[level]
                if abs(current_price - level_price) <= tolerance and current_price >= level_price:
                    actions.append(f'bounce_from_{level}')

        # Resistance level rejections (bearish)  
        for level in ['H3', 'H4', 'H5']:
            if level in pivots:
                level_price = pivots[level]
                if abs(current_price - level_price) <= tolerance and current_price <= level_price:
                    actions.append(f'reject_at_{level}')

        return actions

    def next(self):
        """Main strategy logic executed on each bar"""
        # Update data arrays
        self.high_data.append(self.data.high[0])
        self.low_data.append(self.data.low[0])
        self.close_data.append(self.data.close[0])
        if hasattr(self.data, 'volume'):
            self.volume_data.append(self.data.volume[0])

        current_price = self.data.close[0]

        # Calculate all indicators
        pivots = self.calculate_camarilla_pivots()
        grab_signal = self.calculate_grab_signals()
        rsi_macd_combo = self.calculate_rsi_macd_combo()
        tke_signal = self.calculate_tke_signals()
        hull_trend = self.calculate_hull_trend()

        # Reset current signals
        self.current_signals = {'long': [], 'short': []}

        # Collect long signals
        if rsi_macd_combo['long']:
            self.current_signals['long'].append('rsi_macd')

        if grab_signal == 'bullish':
            self.current_signals['long'].append('grab_bullish')

        if tke_signal == 'oversold':
            self.current_signals['long'].append('tke_oversold')

        if hull_trend == 'bullish':
            self.current_signals['long'].append('hull_bullish')

        # Collect short signals  
        if rsi_macd_combo['short']:
            self.current_signals['short'].append('rsi_macd')

        if grab_signal == 'bearish':
            self.current_signals['short'].append('grab_bearish')

        if tke_signal == 'overbought':
            self.current_signals['short'].append('tke_overbought')

        if hull_trend == 'bearish':
            self.current_signals['short'].append('hull_bearish')

        # Evaluate pivot actions
        pivot_actions = self.evaluate_pivot_action(current_price, pivots)
        for action in pivot_actions:
            if 'bounce' in action:
                self.current_signals['long'].append(f'pivot_{action}')
            elif 'reject' in action:
                self.current_signals['short'].append(f'pivot_{action}')

        # Position management
        if self.position:
            self.manage_position(current_price)
        else:
            self.evaluate_entry_signals(current_price, pivots)

    def evaluate_entry_signals(self, current_price, pivots):
        """Evaluate entry signals"""
        long_signals = len(self.current_signals['long'])
        short_signals = len(self.current_signals['short'])

        # Entry logic - require minimum confirmations
        if long_signals >= self.p.min_signals_for_trade and not self.position:
            self.enter_long(current_price, pivots)
        elif short_signals >= self.p.min_signals_for_trade and not self.position:
            self.enter_short(current_price, pivots)

    def enter_long(self, current_price, pivots):
        """Enter long position"""
        size = int((self.broker.get_cash() * self.p.position_size / 100) / current_price)

        if size > 0:
            self.buy(size=size)
            self.entry_price = current_price

            # Set stop loss and take profit
            self.stop_loss_price = current_price * (1 - self.p.stop_loss_pct / 100)
            self.take_profit_price = current_price * (1 + self.p.take_profit_pct / 100)

            self.log(f'LONG ENTRY: Price {current_price:.2f}, Size {size}, '
                    f'Signals: {self.current_signals["long"]}')

    def enter_short(self, current_price, pivots):
        """Enter short position"""
        size = int((self.broker.get_cash() * self.p.position_size / 100) / current_price)

        if size > 0:
            self.sell(size=size)
            self.entry_price = current_price

            # Set stop loss and take profit  
            self.stop_loss_price = current_price * (1 + self.p.stop_loss_pct / 100)
            self.take_profit_price = current_price * (1 - self.p.take_profit_pct / 100)

            self.log(f'SHORT ENTRY: Price {current_price:.2f}, Size {size}, '
                    f'Signals: {self.current_signals["short"]}')

    def manage_position(self, current_price):
        """Manage existing position"""
        if not self.position:
            return

        # Check stop loss
        if self.position.size > 0:  # Long position
            if current_price <= self.stop_loss_price:
                self.close()
                self.log(f'LONG STOP LOSS: Price {current_price:.2f}')
                return
            elif current_price >= self.take_profit_price:
                self.close()
                self.log(f'LONG TAKE PROFIT: Price {current_price:.2f}')
                return

        else:  # Short position
            if current_price >= self.stop_loss_price:
                self.close()
                self.log(f'SHORT STOP LOSS: Price {current_price:.2f}')
                return
            elif current_price <= self.take_profit_price:
                self.close()
                self.log(f'SHORT TAKE PROFIT: Price {current_price:.2f}')
                return

    def notify_order(self, order):
        """Order notification"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED: Price {order.executed.price:.2f}, '
                        f'Size {order.executed.size}')
            else:
                self.log(f'SELL EXECUTED: Price {order.executed.price:.2f}, '
                        f'Size {order.executed.size}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):
        """Trade notification"""
        if trade.isclosed:
            self.log(f'TRADE CLOSED: PnL {trade.pnl:.2f}')

