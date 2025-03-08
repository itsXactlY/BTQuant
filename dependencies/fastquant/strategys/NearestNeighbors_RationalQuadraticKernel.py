import backtrader as bt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastquant.strategys.base import BaseStrategy, datetime

'''
It was actually planned as a “keep it simple stupid” proof of concept. but as things happen, it totally escalated once again. 
But anyway, I can pull more rabbits out of my hat - so I've decided to make this available to everyone. 

Sharing is caring.

© by aLca (itsXactlY) // BTQuant
'''
# 2179.040000000000 TakeProfit: 2183.398080000000
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

class NRK(BaseStrategy):
    params = (
        # ML Parameters
        ('source', 'close'),
        ('neighbors_count', 8),
        ('max_bars_back', 100), # Slow, better for higher TF than 5m+
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
        ('dca_deviation', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.1),
        ('debug', True),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.source = getattr(self.data, self.p.source)
        self.atr = bt.indicators.ATR(self.data, period=14, plot=False)
        self.rsi = bt.indicators.RSI(self.data, period=14, plot=False)
        self.wt = bt.indicators.WilliamsR(self.data, period=10, plot=False)
        self.cci = bt.indicators.CCI(self.data, period=20, plot=False)
        self.adx = bt.indicators.ADX(self.data, period=14, plot=False)
        self.kernel_estimate = RationalQuadraticKernel(self.data.close, 
                                                       h=self.p.kernel_lookback, 
                                                       r=self.p.kernel_r, 
                                                       x=self.p.kernel_x)
        if self.p.use_kernel_smoothing:
            self.kernel_smooth = bt.indicators.SMA(self.kernel_estimate, period=self.p.kernel_smoothing_lag, plot=False)
        
        self.features = [self.rsi, 
                         self.wt, 
                         self.cci, 
                         self.adx, 
                         self.rsi]
        self.feature_data = []
        ####
        self.buy_executed = False
        self.conditions_checked = False
        self.DCA = True


    def compute_ml_signal(self):
        """
        Computes the ML signal based on the historical feature data and applies
        additional filters (volatility, regime, ADX, and kernel smoothing).
        """

        current_features = []
        for f in self.features:
            try:
                current_features.append(f[0])
            except IndexError:
                current_features.append(np.nan)
        self.feature_data.append(current_features)
        if len(self.feature_data) > self.p.max_bars_back:
            self.feature_data.pop(0)
        if len(self.feature_data) < self.p.neighbors_count:
            return 0

        X = np.array(self.feature_data, dtype=float)
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        y = np.array([
            1 if self.source[-i] < self.source[0] 
            else -1 if self.source[-i] > self.source[0] 
            else 0 
            for i in range(1, len(X)+1)
        ])
        
        nn = NearestNeighbors(n_neighbors=min(self.p.neighbors_count, len(X)), metric='manhattan')
        nn.fit(X)
        distances, indices = nn.kneighbors([X[-1]])
        prediction = np.sum(y[indices[0]])
        signal = 1 if prediction > 0 else -1 if prediction < 0 else 0

        if self.p.use_volatility_filter and len(self.atr) > 20:
            atr_value = self.atr[0]
            atr_20_bars_ago = self.atr[-20]
            if atr_value <= atr_20_bars_ago:
                if self.p.debug:
                    print(f"Volatility filter zeroed signal. ATR now={atr_value}, ATR 20 bars ago={atr_20_bars_ago}")
                signal = 0
        else:
            print("Volatility filter triggered - not enough bars or something else did go broke along the way")
            signal = 0
        if self.p.debug:
            print(f"Signal after volatility filter: {signal}")


        regime = (self.data.close[0] / self.data.close[-20] - 1) * 100
        if regime <= self.p.regime_threshold:
            if self.p.debug:
                print(f"Regime filter zeroed signal. Regime={regime}, threshold={self.p.regime_threshold}")
            signal = 0

        if self.adx[0] <= self.p.adx_threshold:
            if self.p.debug:
                print(f"ADX filter zeroed signal. ADX={self.adx[0]}, threshold={self.p.adx_threshold}")
            signal = 0
        if self.p.debug:
            print(f"Signal after ADX filter: {signal}")

        if self.p.trade_with_kernel:
            if self.p.use_kernel_smoothing:
                kernel_bullish = self.kernel_smooth[0] > self.kernel_estimate[0]
                kernel_bearish = self.kernel_smooth[0] < self.kernel_estimate[0]
            else:
                kernel_bullish = self.kernel_estimate[0] > self.kernel_estimate[-1]
                kernel_bearish = self.kernel_estimate[0] < self.kernel_estimate[-1]
            if (signal > 0 and not kernel_bullish) or (signal < 0 and not kernel_bearish):
                if self.p.debug:
                    print(f"Kernel filter zeroed signal. signal={signal}, kernel_bullish={kernel_bullish}, kernel_bearish={kernel_bearish}")
                signal = 0
        if self.p.debug:
            print(f"DEBUG: Final signal = {signal}")
        return signal

    def buy_or_short_condition(self):
        signal = self.compute_ml_signal()
        if not self.buy_executed:
            if signal > 0:
                if self.p.backtest is False:
                    self.entry_prices.append(self.data.close[0])
                    print(f'\n\nBUY EXECUTED AT {self.data.close[0]:.9f}\n')
                    self.sizes.append(self.amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    alert_message = f"""\nBuy Alert arrived!
Exchange: {self.exchange}
Action: buy {self.asset}
Entry Price: {self.data.close[0]:.9f}
Take Profit: {self.take_profit_price:.9f}"""

                    self.send_alert(alert_message)
                elif self.p.backtest is True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.conditions_checked = True

    def dca_or_short_condition(self):
        print('dca_or_short_condition')
        if self.live_data and self.buy_executed:
            print(f'|\n| {datetime.now()}\n| Price: {self.data.close[0]:.12f} Entry: {self.average_entry_price:.12f} TakeProfit: {self.take_profit_price:.12f}')
        signal = self.compute_ml_signal()
        print(f'Signal: {signal}')
        
        if signal > 0:
            if self.p.backtest is False:
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    alert_message = f"""\nDCA Alert arrived!
        Exchange: {self.exchange}
        Action: buy {self.asset}
        Entry Price: {self.data.close[0]:.9f}
        Take Profit: {self.take_profit_price:.9f}"""
                    self.send_alert(alert_message)
                elif self.p.backtest is True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                print(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {self.average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )

                if self.params.backtest == False:
                    self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
                    alert_message = f"""Close {self.asset}"""
                    self.send_alert(alert_message)
                    self.reset_position_state()
                    self.buy_executed = False
                elif self.params.backtest == True:
                    self.close()


            self.conditions_checked = True