from backtrader.strategies.base import BaseStrategy, bt, OrderTracker, datetime
from backtrader.indicators import BollingerBands, RSI

class MeanReversionVolatileAltcoins(BaseStrategy):
    params = (
        ('take_profit', 5),  # Higher TP for volatile altcoins to capture larger swings
        ('percent_sizer', 0.01),  # Smaller position size to manage volatility risk
        ('dca_deviation', 3.0),  # Wider deviation for DCA in volatile markets
        ('bollinger_period', 20),  # Period for Bollinger Bands
        ('bollinger_dev', 2.5),  # Wider deviation for volatile altcoins
        ('rsi_period', 14),  # RSI period
        ('rsi_low', 30),  # Oversold threshold for buy
        ('rsi_high', 70),  # Overbought threshold for sell
        ('debug', False),
        ('backtest', None),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Indicators for mean reversion
        self.bbands = BollingerBands(period=self.p.bollinger_period, devfactor=self.p.bollinger_dev, subplot=True)
        self.rsi = RSI(period=self.p.rsi_period, subplot=True)
        self.DCA = True  # Enable DCA for handling volatility dips

    def buy_or_short_condition(self):
        if not self.buy_executed:
            # Mean reversion buy condition: Price below lower Bollinger Band and RSI oversold
            if self.data.close[0] < self.bbands.lines.bot[0] and self.rsi[0] < self.p.rsi_low:
                size = self._determine_size()
                order_tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.params.take_profit,
                    symbol=getattr(self, 'symbol', self.p.asset),
                    order_type="BUY",
                    backtest=self.params.backtest
                )
                order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if not hasattr(self, 'active_orders'):
                    self.active_orders = []
                    
                self.active_orders.append(order_tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"Buy order placed: {size} at {self.data.close[0]} (Mean Reversion Entry)")
                if not self.buy_executed:
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.buy_executed = True
                self.calc_averages()
        self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
            # DCA condition with mean reversion confirmation: Still below lower band and RSI oversold
            if self.data.close[0] < self.bbands.lines.bot[0] and self.rsi[0] < self.p.rsi_low:
                size = self._determine_size() * 1.5  # Scale up DCA size slightly for volatility
                order_tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.params.take_profit,
                    symbol=getattr(self, 'symbol', self.p.asset),
                    order_type="BUY",
                    backtest=self.params.backtest
                )
                order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if not hasattr(self, 'active_orders'):
                    self.active_orders = []

                self.active_orders.append(order_tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)

                if self.p.debug:
                    print(f"DCA Buy order placed: {size} at {self.data.close[0]} (Mean Reversion DCA)")

                if not self.buy_executed:
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.buy_executed = True
                self.calc_averages()
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if hasattr(self, 'active_orders') and self.active_orders and self.buy_executed:
            current_price = self.data.close[0]
            orders_to_remove = []

            for idx, order in enumerate(self.active_orders):
                # Mean reversion sell: Price above mid Bollinger Band or RSI overbought, or take profit
                if (current_price > self.bbands.lines.mid[0] or self.rsi[0] > self.p.rsi_high) or current_price >= order.take_profit_price:
                    self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    if self.p.debug:
                        print(f"Sell order placed: {order.size} at {current_price} (entry: {order.entry_price}) - Mean Reversion Exit")
                    order.close_order(current_price)
                    orders_to_remove.append(idx)
            for idx in sorted(orders_to_remove, reverse=True):
                removed_order = self.active_orders.pop(idx)
                profit_pct = ((current_price / removed_order.entry_price) - 1) * 100
                if self.p.debug:
                    print(f"Order removed: {profit_pct:.2f}% profit")
            if orders_to_remove:
                self.entry_prices = [order.entry_price for order in self.active_orders]
                self.sizes = [order.size for order in self.active_orders]
                if not self.active_orders:
                    self.reset_position_state()
                    self.buy_executed = False
                else:
                    self.calc_averages()
        self.conditions_checked = True

from backtrader.indicators import BollingerBands, RSI, MACD, ATR, Stochastic, EMA
from backtrader.indicators import CrossOver, CrossUp, CrossDown

class QuantumMeanReversionMonster(BaseStrategy):
    params = (
        ('take_profit', 8),  # Aggressive TP for volatile altcoins to capture big reversions
        ('percent_sizer', 0.005),  # Conservative sizing to handle extreme volatility
        ('dca_deviation', 4.0),  # Wider DCA trigger for deep dips in volatile markets
        ('bollinger_period', 20),  # BB period
        ('bollinger_dev', 3.0),  # Wider bands for volatility
        ('rsi_period', 14),  # RSI period
        ('rsi_low', 25),  # Deeper oversold for strong signals
        ('rsi_high', 75),  # Higher overbought for exits
        ('macd_fast', 12),  # MACD fast period
        ('macd_slow', 26),  # MACD slow period
        ('macd_signal', 9),  # MACD signal period
        ('atr_period', 14),  # ATR for volatility and stops
        ('stoch_period', 14),  # Stochastic period
        ('stoch_low', 20),  # Stochastic oversold
        ('stoch_high', 80),  # Stochastic overbought
        ('trend_ema_period', 50),  # EMA for trend filter
        ('trailing_stop_atr_mult', 2.0),  # Trailing stop multiplier based on ATR
        ('max_dca_levels', 5),  # Limit DCA to prevent over-exposure
        ('volatility_filter_mult', 1.5),  # Only trade if ATR > EMA(ATR) * mult (high vol filter)
        ('debug', False),
        ('backtest', None),
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Core Mean Reversion Indicators
        self.bbands = BollingerBands(period=self.p.bollinger_period, devfactor=self.p.bollinger_dev, subplot=True)
        self.rsi = RSI(period=self.p.rsi_period, subplot=True)
        
        # Momentum and Confirmation Indicators
        self.macd = MACD(period_me1=self.p.macd_fast, period_me2=self.p.macd_slow, period_signal=self.p.macd_signal, subplot=True)
        self.macd_crossover = CrossOver(self.macd.lines.macd, self.macd.lines.signal)  # +1 for bullish crossover
        
        # Volatility and Risk Management
        self.atr = ATR(period=self.p.atr_period, subplot=True)
        self.atr_ema = EMA(self.atr, period=self.p.atr_period)  # Smoothed ATR for volatility filter
        
        # Oscillator for Additional Confirmation
        self.stoch = Stochastic(period=self.p.stoch_period, subplot=True)
        self.stoch_crossup = CrossUp(self.stoch.lines.percK, self.stoch.lines.percD)  # Bullish cross
        
        # Trend Filter to Avoid Counter-Trend Trades in Strong Downtrends
        self.trend_ema = EMA(period=self.p.trend_ema_period, subplot=False)
        
        self.DCA = True  # Enable DCA with limits
        self.dca_count = 0  # Track DCA levels

    def buy_or_short_condition(self):
        if not self.buy_executed and self.dca_count < self.p.max_dca_levels:
            # High Volatility Filter: Only enter if current ATR > smoothed ATR * multiplier
            high_vol = self.atr[0] > self.atr_ema[0] * self.p.volatility_filter_mult
            
            # Mean Reversion Buy: Price < lower BB, RSI oversold, Stochastic oversold
            mean_reversion_buy = (self.data.close[0] < self.bbands.lines.bot[0] and 
                                  self.rsi[0] < self.p.rsi_low and 
                                  self.stoch.lines.percK[0] < self.p.stoch_low)
            
            # Momentum Confirmation: MACD bullish crossover and price above trend EMA (avoid strong downtrends)
            momentum_confirm = (self.macd_crossover[0] == 1 and 
                                self.data.close[0] > self.trend_ema[0])
            
            # Stochastic bullish cross for timing
            timing_confirm = self.stoch_crossup[0] == 1
            
            if high_vol and mean_reversion_buy and momentum_confirm and timing_confirm:
                size = self._determine_size() * (1 + (self.atr[0] / self.data.close[0]))  # Volatility-adjusted sizing (larger in high vol)
                order_tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.params.take_profit,
                    symbol=getattr(self, 'symbol', self.p.asset),
                    order_type="BUY",
                    backtest=self.params.backtest
                )
                # Dynamic trailing stop based on ATR
                order_tracker.trailing_stop = self.data.close[0] - (self.atr[0] * self.p.trailing_stop_atr_mult)
                order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if not hasattr(self, 'active_orders'):
                    self.active_orders = []
                    
                self.active_orders.append(order_tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"Buy order placed: {size} at {self.data.close[0]} (Quantum Monster Entry - Vol: {self.atr[0]:.2f})")
                if not self.buy_executed:
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.buy_executed = True
                self.calc_averages()
                self.dca_count += 1  # Count as initial entry
        self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.entry_prices and self.dca_count < self.p.max_dca_levels and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
            # Similar conditions but slightly relaxed for DCA
            high_vol = self.atr[0] > self.atr_ema[0] * self.p.volatility_filter_mult
            mean_reversion_dca = (self.data.close[0] < self.bbands.lines.bot[0] and self.rsi[0] < self.p.rsi_low)
            momentum_confirm = self.data.close[0] > self.trend_ema[0]  # Still respect trend filter
            timing_confirm = self.stoch.lines.percK[0] < self.p.stoch_low  # Oversold stochastic
            
            if high_vol and mean_reversion_dca and momentum_confirm and timing_confirm:
                size = self._determine_size() * (1.5 + self.dca_count * 0.5)  # Progressive sizing for DCA (increase with levels)
                order_tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.params.take_profit,
                    symbol=getattr(self, 'symbol', self.p.asset),
                    order_type="BUY",
                    backtest=self.params.backtest
                )
                # Dynamic trailing stop for DCA orders
                order_tracker.trailing_stop = self.data.close[0] - (self.atr[0] * self.p.trailing_stop_atr_mult)
                order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if not hasattr(self, 'active_orders'):
                    self.active_orders = []

                self.active_orders.append(order_tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)

                if self.p.debug:
                    print(f"DCA Buy order placed: {size} at {self.data.close[0]} (Quantum Monster DCA - Level: {self.dca_count + 1}, Vol: {self.atr[0]:.2f})")

                if not self.buy_executed:
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.buy_executed = True
                self.calc_averages()
                self.dca_count += 1
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if hasattr(self, 'active_orders') and self.active_orders and self.buy_executed:
            current_price = self.data.close[0]
            orders_to_remove = []

            # Global Exit Conditions: RSI overbought, price > upper BB, MACD bearish crossover, Stochastic overbought
            global_exit = (self.rsi[0] > self.p.rsi_high or 
                           current_price > self.bbands.lines.top[0] or 
                           self.macd_crossover[0] == -1 or 
                           self.stoch.lines.percK[0] > self.p.stoch_high)

            for idx, order in enumerate(self.active_orders):
                # Update trailing stop: Move up if price increases
                new_trailing_stop = current_price - (self.atr[0] * self.p.trailing_stop_atr_mult)
                if new_trailing_stop > order.trailing_stop:
                    order.trailing_stop = new_trailing_stop
                
                # Exit if global condition, TP hit, or trailing stop hit
                if global_exit or current_price >= order.take_profit_price or current_price <= order.trailing_stop:
                    self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    if self.p.debug:
                        print(f"Sell order placed: {order.size} at {current_price} (entry: {order.entry_price}) - Quantum Monster Exit (Global: {global_exit}, TP: {current_price >= order.take_profit_price}, Trail: {current_price <= order.trailing_stop})")
                    order.close_order(current_price)
                    orders_to_remove.append(idx)
            for idx in sorted(orders_to_remove, reverse=True):
                removed_order = self.active_orders.pop(idx)
                profit_pct = ((current_price / removed_order.entry_price) - 1) * 100
                if self.p.debug:
                    print(f"Order removed: {profit_pct:.2f}% profit")
            if orders_to_remove:
                self.entry_prices = [order.entry_price for order in self.active_orders]
                self.sizes = [order.size for order in self.active_orders]
                if not self.active_orders:
                    self.reset_position_state()
                    self.buy_executed = False
                    self.dca_count = 0  # Reset DCA count
                else:
                    self.calc_averages()
        self.conditions_checked = True

from backtrader.utils.backtest import backtest

if __name__ == '__main__':
    try:
        backtest(
            QuantumMeanReversionMonster, # MeanReversionVolatileAltcoins,
            coin='DOGE',
            collateral='USDT',
            start_date="2023-04-18", 
            end_date="2024-12-12", 
            interval="1m",
            init_cash=1000,
            plot=True, 
            quantstats=True
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
