import numpy as np
from sklearn.neighbors import NearestNeighbors # pip install scikit-learn
from fastquant.strategys.base import BaseStrategy, bt
'''
It was actually planned as a “keep it simple stupid” proof of concept. but as things happen, it totally escalated once again. 
But anyway, I can pull more rabbits out of my hat - so I've decided to make this available to everyone. 

Sharing is caring.

© by aLca (itsXactlY) // BTQuant
'''

class RationalQuadraticKernel(bt.indicators.PeriodN):
    lines = ('yhat',)
    params = (('h', 8), ('r', 8), ('x', 25))

    def __init__(self):
        super(RationalQuadraticKernel, self).__init__()
        self.addminperiod(self.p.x)

    def next(self):
        x = np.arange(len(self.data.get(size=self.p.x)))
        y = self.data.get(size=self.p.x)
        w = (1 + ((x - self.p.x)**2) / (2 * self.p.r * (self.p.h**2)))**(-self.p.r)
        self.lines.yhat[0] = np.sum(w * y) / np.sum(w)

class NearestNeighbors_RationalQuadraticKernel_DCA_Strategy(BaseStrategy):
    params = (
        # ML Parameters
        ('source', 'close'),
        ('neighbors_count', 8),
        ('max_bars_back', 2000),
        ('use_volatility_filter', True),
        ('regime_threshold', -0.1),
        ('adx_threshold', 20),
        ('trade_with_kernel', True),
        ('kernel_lookback', 8),
        ('kernel_r', 8),
        ('kernel_x', 25),
        ('use_kernel_smoothing', True),
        ('kernel_smoothing_lag', 2),
        # DCA Parameters
        ("dca_deviation", 1.5),
        ("take_profit", 2),
        ('percent_sizer', 0.01),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Primary data series (e.g., close)
        self.source = getattr(self.data, self.p.source)
        # Standard technical indicators
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.rsi = bt.indicators.RSI(self.data, period=14)
        self.wt = bt.indicators.WilliamsR(self.data, period=10)
        self.cci = bt.indicators.CCI(self.data, period=20)
        self.adx = bt.indicators.ADX(self.data, period=14)
        
        # Kernel estimator (and smoothing) for an extra filter
        self.kernel_estimate = RationalQuadraticKernel(self.data.close, 
                                                       h=self.p.kernel_lookback, 
                                                       r=self.p.kernel_r, 
                                                       x=self.p.kernel_x)
        if self.p.use_kernel_smoothing:
            self.kernel_smooth = bt.indicators.SMA(self.kernel_estimate, period=self.p.kernel_smoothing_lag)

        # Features for the ML model – note the duplicate RSI to mimic the original 5-feature setup
        self.features = [self.rsi, 
                         self.wt, 
                         self.cci, 
                         self.adx, 
                         self.rsi]
        self.feature_data = []

        self.DCA = True


    def compute_ml_signal(self):
        """
        Computes the ML signal based on the historical feature data and applies
        additional filters (volatility, regime, ADX, and kernel smoothing).
        """
        # Build a list of current feature values
        
        # Standard indicators: RSI, Williams %R, CCI, ADX, and duplicate RSI (for the original 5-feature setup)
        standard_indicators = [self.rsi, self.wt, self.cci, self.adx, self.rsi]
        
        current_features = []
        
        # Append values for standard indicators
        for indicator in standard_indicators:
            try:
                current_features.append(indicator[0])
            except IndexError:
                current_features.append(np.nan)
        
        # Add the current features to our history
        self.feature_data.append(current_features)
        
        # Maintain a fixed history length
        if len(self.feature_data) > self.p.max_bars_back:
            self.feature_data.pop(0)
            
        # Ensure we have enough data for our nearest-neighbors search
        if len(self.feature_data) < self.p.neighbors_count:
            return 0

        # Convert feature history to a NumPy array and handle any NaNs
        X = np.array(self.feature_data, dtype=float)
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        # Create labels by comparing past prices to the current price
        y = np.array([
            1 if self.source[-i] < self.source[0] 
            else -1 if self.source[-i] > self.source[0] 
            else 0 
            for i in range(1, len(X)+1)
        ])
        
        # Nearest Neighbors calculation (using Manhattan distance)
        nn = NearestNeighbors(n_neighbors=min(self.p.neighbors_count, len(X)), metric='manhattan')
        nn.fit(X)
        distances, indices = nn.kneighbors([X[-1]])
        prediction = np.sum(y[indices[0]])
        signal = 1 if prediction > 0 else -1 if prediction < 0 else 0

        # Apply volatility filter (using ATR)
        if self.p.use_volatility_filter and len(self.atr) > 20:
            atr_value = self.atr[0]
            atr_20_bars_ago = self.atr[-20]
            signal *= 1 if atr_value > atr_20_bars_ago else 0
        else:
            signal = 0

        # Apply regime filter (price change over 20 bars)
        regime = (self.data.close[0] / self.data.close[-20] - 1) * 100
        signal *= 1 if regime > self.p.regime_threshold else 0

        # Apply ADX filter
        signal *= 1 if self.adx[0] > self.p.adx_threshold else 0

        # Apply kernel smoothing filter if enabled
        if self.p.trade_with_kernel:
            if self.p.use_kernel_smoothing:
                kernel_bullish = self.kernel_smooth[0] > self.kernel_estimate[0]
                kernel_bearish = self.kernel_smooth[0] < self.kernel_estimate[0]
            else:
                kernel_bullish = self.kernel_estimate[0] > self.kernel_estimate[-1]
                kernel_bearish = self.kernel_estimate[0] < self.kernel_estimate[-1]
            if (signal > 0 and not kernel_bullish) or (signal < 0 and not kernel_bearish):
                signal = 0

        return signal

    def buy_or_short_condition(self):
        signal = self.compute_ml_signal()
        if not self.buy_executed:
            if signal > 0:
                if self.p.backtest is False:
                    self.entry_prices.append(self.data.close[0])
                    print(f'\n\nBUY EXECUTED AT {self.data.close[0]:.9f}\n')
                    self.sizes.append(self.amount)
                    self.load_trade_data()
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    # Set take profit price based on the take_profit parameter
                    self.take_profit_price = self.data.close[0] * (1 + self.p.take_profit / 100)
                elif self.p.backtest is True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.conditions_checked = True

    def dca_or_short_condition(self):
        signal = self.compute_ml_signal()
        if signal > 0:
            if self.buy_executed and not self.conditions_checked:
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.p.dca_deviation / 100):
                    if self.p.backtest is False:
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.amount)
                        self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                        self.calc_averages()
                        self.buy_executed = True
                        self.conditions_checked = True
                    elif self.p.backtest is True:
                        self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                        self.buy_executed = True
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.stake)
                        self.calc_averages()
                        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            average_entry_price = sum(self.entry_prices) / len(self.entry_prices) if self.entry_prices else 0

            # Avoid selling at a loss or below the take profit price
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                print(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )

            if self.params.backtest == False:
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()

            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True


    def stop(self):
        print('Final Portfolio Value: %.2f' % self.broker.getvalue())