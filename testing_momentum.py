### From another idea of @Rose, to an full proof of concept with >1000%+ growthrate over the time.

# -*- coding: utf-8 -*-
import backtrader as bt
import math
from datetime import datetime
from backtrader.utils.backtest import backtest, bulk_backtest

class BuySellArrows(bt.observers.BuySell):
    def next(self):
        super().next()
        if self.lines.buy[0]:
            self.lines.buy[0] -= self.data.low[0] * 0.02
        if self.lines.sell[0]:
            self.lines.sell[0] += self.data.high[0] * 0.02
    
    plotlines = dict(
        buy=dict(marker='$\u21E7$', markersize=8.0),
        sell=dict(marker='$\u21E9$', markersize=8.0)
    )

# --- Hull Moving Average ---
class HMA(bt.Indicator):
    lines = ('hma',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        p = self.p.period
        wma_half = bt.ind.WeightedMovingAverage(self.data, period=int(p / 2))
        wma_full = bt.ind.WeightedMovingAverage(self.data, period=p)
        raw_hma = 2 * wma_half - wma_full
        self.lines.hma = bt.ind.WeightedMovingAverage(raw_hma, period=int(math.sqrt(p)))

# --- Zero Lag Indicator (based on HMA) ---
class ZeroLag(bt.Indicator):
    lines = ('zerolag',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        hma = HMA(self.data, period=self.p.period)
        hma_sma = bt.ind.SMA(hma, period=int(self.p.period / 2))
        self.lines.zerolag = 2 * hma - hma_sma

# --- Sine Weighted Moving Average ---
class SineWeightedMA(bt.Indicator):
    lines = ('sine_wma',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        # TODO :: For now: approximated using SMA, but you could replace with a true sine-weight
        self.lines.sine_wma = bt.ind.SMA(self.data, period=self.p.period)

# --- Order Tracker Class ---
class OrderTracker:
    def __init__(self, entry_price, size, take_profit_pct, symbol, order_type="BUY", backtest=True):
        self.entry_price = entry_price
        self.size = size
        self.take_profit_pct = take_profit_pct
        self.take_profit_price = entry_price * (1 + take_profit_pct / 100)
        self.symbol = symbol
        self.order_type = order_type
        self.backtest = backtest
        self.order_id = None
        self.closed = False
        
    def close_order(self, exit_price):
        self.closed = True
        self.exit_price = exit_price

# --- Enhanced Momentum Strategy with disabled DCA for maximum returns on SPOT LONG Only Markets ---
class ZeroLagSineMomentumDCAStrategy(bt.Strategy):
    params = dict(

        zl_period=14,
        sine_period=20,
        sma_filter=200,
        mom_period=14,
        mom_lookback=3,

        dca_deviation=1.5,
        take_profit=20,
        percent_sizer=0.95,
        max_positions=0,  # Maximum number of DCA positions
        # Additional filters
        use_volatility_filter=False,
        adx_threshold=20,
        debug=False,
        backtest=True
    )
    
    def __init__(self):
        self.zero_lag = ZeroLag(self.data, period=self.p.zl_period)
        self.sine_wma = SineWeightedMA(self.zero_lag, period=self.p.sine_period)

        self.sma_200 = bt.ind.SMA(self.data, period=self.p.sma_filter)
        self.momentum = bt.ind.Momentum(self.data, period=self.p.mom_period)
        self.atr = bt.indicators.ATR(self.data, period=14, plot=False)
        self.adx = bt.indicators.ADX(self.data, period=14, plot=False)

        self.crossover = bt.ind.CrossOver(self.zero_lag, self.sine_wma)

        self.active_orders = []
        self.entry_prices = []
        self.sizes = []
        self.buy_executed = False
        self.first_entry_price = None
        
        BuySellArrows(self.data0, barplot=True)
    
    def _determine_size(self):
        available_cash = self.broker.get_cash()
        size = (available_cash * self.p.percent_sizer) / self.data.close[0]
        return max(size, 0.001)  # Minimum size
    
    def reset_position_state(self):
        self.active_orders = []
        self.entry_prices = []
        self.sizes = []
        self.first_entry_price = None
    
    def calc_averages(self):
        if self.entry_prices and self.sizes:
            total_cost = sum(price * size for price, size in zip(self.entry_prices, self.sizes))
            total_size = sum(self.sizes)
            self.avg_entry_price = total_cost / total_size if total_size > 0 else 0
    
    def get_signal(self):
        momentum_trending_up = all(
            self.momentum[0] > self.momentum[-i]
            for i in range(1, self.p.mom_lookback + 1)
            if len(self.momentum) > i
        )
        
        price_above_sma = self.data.close[0] > self.sma_200[0]
        
        signal = 0
        if self.crossover > 0 and momentum_trending_up and price_above_sma:
            signal = 1
        elif self.crossover < 0:
            signal = -1
            
        if self.p.use_volatility_filter and len(self.atr) > 20:
            atr_value = self.atr[0]
            atr_20_bars_ago = self.atr[-20]
            if atr_value <= atr_20_bars_ago:
                if self.p.debug:
                    print(f"Volatility filter zeroed signal. ATR now={atr_value}, ATR 20 bars ago={atr_20_bars_ago}")
                signal = 0
        
        if len(self.adx) > 0 and self.adx[0] <= self.p.adx_threshold:
            if self.p.debug:
                print(f"ADX filter zeroed signal. ADX={self.adx[0]}, threshold={self.p.adx_threshold}")
            signal = 0
            
        return signal
    
    def buy_or_initial_entry(self):
        """Handle initial buy entry"""
        signal = self.get_signal()
        
        if not self.buy_executed and signal > 0:
            size = self._determine_size()
            
            order_tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.p.take_profit,
                symbol=getattr(self, 'symbol', 'ASSET'),
                order_type="BUY",
                backtest=self.p.backtest
            )
            order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            self.active_orders.append(order_tracker)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.order = self.buy(size=size, exectype=bt.Order.Market)
            
            if self.p.debug:
                print(f"Initial buy order placed: {size} at {self.data.close[0]}")
                
            self.first_entry_price = self.data.close[0]
            self.buy_executed = True
            self.calc_averages()
    
    def dca_condition(self):
        """Handle DCA (Dollar Cost Averaging) entries"""
        if (self.entry_prices and 
            len(self.active_orders) < self.p.max_positions and
            self.data.close[0] < self.entry_prices[-1] * (1 - self.p.dca_deviation / 100)):
            
            signal = self.get_signal()
            
            if signal > 0:
                size = self._determine_size()
                
                order_tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.p.take_profit,
                    symbol=getattr(self, 'symbol', 'ASSET'),
                    order_type="BUY",
                    backtest=self.p.backtest
                )
                order_tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                self.active_orders.append(order_tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                
                if self.p.debug:
                    print(f"DCA buy order placed: {size} at {self.data.close[0]}")
                    
                self.calc_averages()
    
    def sell_condition(self):
        """Handle sell conditions (take profit or signal-based exit)"""
        if hasattr(self, 'active_orders') and self.active_orders and self.buy_executed:
            current_price = self.data.close[0]
            orders_to_remove = []
            signal = self.get_signal()
            
            for idx, order in enumerate(self.active_orders):
                if current_price >= order.take_profit_price:
                    self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    if self.p.debug:
                        print(f"TP hit: Selling {order.size} at {current_price} (entry: {order.entry_price})")
                    order.close_order(current_price)
                    orders_to_remove.append(idx)
            
            if signal < 0 and self.data.close[0] < self.sma_200[0]:
                for idx, order in enumerate(self.active_orders):
                    if idx not in orders_to_remove: 
                        self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                        if self.p.debug:
                            print(f"Signal exit: Selling {order.size} at {current_price}")
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
    
    def next(self):
        if not self.buy_executed:
            self.buy_or_initial_entry()
        else:
            self.dca_condition()
        
        self.sell_condition()
        
        if self.p.debug and not self.p.backtest:
            dt = self.datas[0].datetime.datetime(0)
            print(f'Processing candle: {dt}, Close: {self.data.close[0]}, Active orders: {len(self.active_orders)}')

# --- Run Backtest ---
if __name__ == '__main__':
    try:
        # backtest(
        bulk_backtest(
            ZeroLagSineMomentumDCAStrategy,
            # coin='RVN',
            collateral='USDT',
            start_date="2017-01-01",
            end_date="2026-01-08",
            interval="4h",
            init_cash=100,
            plot=False,
            quantstats=True
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()