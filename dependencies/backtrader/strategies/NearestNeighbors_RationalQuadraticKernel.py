from datetime import datetime
import backtrader as bt
from sklearn.neighbors import NearestNeighbors
from backtrader.strategies.base import BaseStrategy, np, OrderTracker
from backtrader import Order

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
        
        # Add check for zero denominator
        sum_w = np.sum(w)
        if sum_w != 0:
            self.lines.yhat[0] = np.sum(w * y) / sum_w
        else:
            # Handle zero case - use the most recent data point as fallback
            self.lines.yhat[0] = y[-1] if len(y) > 0 else 0

class NRK(BaseStrategy):
    params = (
        # ML Parameters
        ('source', 'close'),
        ('neighbors_count', 8),
        ('max_bars_back', 100), # Standard: 2000 what is slow on HFT, better suited for higher TF than 5m+
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
        ('dca_deviation', 2),
        ('take_profit', 4),
        ('percent_sizer', 0.05),
        ('debug', True),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        print("Initializing strategy for", self.p.asset)
        super().__init__(**kwargs)
        self.addminperiod(self.p.max_bars_back)
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

        current_features = [] # TODO dequeue
        for f in self.features:
            try:
                current_features.append(f[0])
            except IndexError:
                current_features.append(np.nan)
        self.feature_data.append(current_features)
        if len(self.feature_data) > self.p.max_bars_back:
            self.feature_data.pop()
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


        # New code with zero check
        if self.data.close[-20] != 0:
            regime = (self.data.close[0] / self.data.close[-20] - 1) * 100
        else:
            # Handle zero case - can use a default value or log the issue
            regime = 0
            if self.p.debug:
                print("Warning: self.data.close[-20] is zero, setting regime to 0")
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
                if self.p.backtest == False:
                    self.calculate_position_size()
                    order_tracker = OrderTracker(
                        entry_price=self.data.close[0],
                        size=self.usdt_amount,
                        take_profit_pct=self.params.take_profit,
                        symbol=self.symbol,
                        order_type="BUY"
                    )

                    order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    self.active_orders.append(order_tracker)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.usdt_amount)
                    self.order = self.buy(size=self.usdt_amount, exectype=Order.Market)
                    print(f"Buy order placed: {self.usdt_amount} at {self.data.close[0]}\n{self.order}")
                    if not self.buy_executed:
                        if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                            self.first_entry_price = self.data.close[0]
                        self.buy_executed = True
                    self.calc_averages()
        
                elif self.p.backtest == True:
                    print(self.stake)
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]

                    self.calc_averages()
                    self.buy_executed = True

        self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
            signal = self.compute_ml_signal()
            if signal > 0:
                if self.p.backtest == False:
                    self.calculate_position_size()
                    order_tracker = OrderTracker(
                        entry_price=self.data.close[0],
                        size=self.usdt_amount,
                        take_profit_pct=self.params.take_profit,
                        symbol=self.symbol,
                        order_type="BUY"
                    )

                    order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    self.active_orders.append(order_tracker)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.usdt_amount)
                    self.order = self.buy(size=self.usdt_amount, exectype=Order.Market)
                    print(f"Buy order placed: {self.usdt_amount} at {self.data.close[0]}\n{self.order}")
                    if not self.buy_executed:
                        if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                            self.first_entry_price = self.data.close[0]
                        self.buy_executed = True
                    self.calc_averages()

                elif self.p.backtest is True:
                    print(self.stake)
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.buy_executed = True
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.active_orders and self.p.backtest == False:
            current_price = self.data.close[0]
            orders_to_remove = []
            for idx, order in enumerate(self.active_orders):
                if current_price >= order.take_profit_price:
                    if self.params.backtest:
                        self.order = self.sell(size=order.size, exectype=Order.Market)
                    else:
                        self.order = self.sell(self.data, size=order.size, exectype=Order.Market)

                    print(f"TP hit: Selling {order.size} at {current_price} (entry: {order.entry_price})")
                    
                    order.close_order(current_price)
                    orders_to_remove.append(idx)
            
            for idx in sorted(orders_to_remove, reverse=True):
                removed_order = self.active_orders.pop(idx)
                profit_pct = ((current_price / removed_order.entry_price) - 1) * 100
                print(f"Order removed: {profit_pct:.2f}% profit")
            
            if orders_to_remove:
                self.entry_prices = [order.entry_price for order in self.active_orders]
                self.sizes = [order.size for order in self.active_orders]
                if not self.active_orders:
                    self.reset_position_state()
                    self.buy_executed = False
                else:
                    self.calc_averages()
        elif self.p.backtest == True:
            if self.buy_executed:
                current_price = round(self.data.close[0], 9)
                avg_price = round(self.average_entry_price, 9)
                tp_price = round(self.take_profit_price, 9)

                if current_price >= tp_price:
                    if self.params.backtest:
                        self.close()
                        self.buy_executed = False
                        if self.p.debug:
                            print(f"Position closed at {current_price:.9f}, profit taken")
                        self.reset_position_state()

        self.conditions_checked = True

    def next(self):
        super().next()
        if self.p.backtest == False:
            self.report_positions()
            dt = self.datas[0].datetime.datetime(0)
            print(f'Realtime: {datetime.now()} processing candle date: {dt}, with {self.data.close[0]}')
