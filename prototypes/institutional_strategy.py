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

class ZeroLag(bt.Indicator):
    lines = ('zerolag',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        hma = HMA(self.data, period=self.p.period)
        hma_sma = bt.ind.SMA(hma, period=int(self.p.period / 2))
        self.lines.zerolag = 2 * hma - hma_sma

class SineWeightedMA(bt.Indicator):
    lines = ('sine_wma',)
    params = dict(period=20)
    plotinfo = dict(plot=True, subplot=False)
    
    def __init__(self):
        self.lines.sine_wma = bt.ind.SMA(self.data, period=self.p.period)

class OptimizedOrderTracker:
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
        
        self.trailing_stop = None
        self.highest_price = entry_price
        self.profit_levels_hit = []
        
    def update_optimized_trailing_stop(self, current_price, atr_value=None):
        if self.order_type == "BUY":
            self.highest_price = max(self.highest_price, current_price)
            profit_pct = (current_price / self.entry_price - 1) * 100

            if profit_pct >= 100:
                atr_multiplier = 1.2
            elif profit_pct >= 50:
                atr_multiplier = 1.5  
            elif profit_pct >= 25:
                atr_multiplier = 1.8
            elif profit_pct >= 10:
                atr_multiplier = 2.0
            else:
                atr_multiplier = 2.5

            if atr_value and atr_value > 0:
                distance = atr_value * atr_multiplier
                new_stop = self.highest_price - distance
            else:
                if profit_pct >= 50:
                    trail_pct = 0.12  # 12% trailing for big winners
                elif profit_pct >= 25:
                    trail_pct = 0.15  # 15% trailing
                else:
                    trail_pct = 0.18  # 18% trailing for early profits
                new_stop = self.highest_price * (1 - trail_pct)
            
            # Only move stop up, never down
            if self.trailing_stop is None:
                self.trailing_stop = new_stop
            else:
                self.trailing_stop = max(self.trailing_stop, new_stop)
        
    def should_take_partial_profit(self, current_price):
        profit_pct = (current_price / self.entry_price - 1) * 100
        
        profit_levels = [50, 100, 200, 400]  # 50%, 100%, 200%, 400%
        
        for level in profit_levels:
            if profit_pct >= level and level not in self.profit_levels_hit:
                self.profit_levels_hit.append(level)
                return level
        return None
        
    def close_order(self, exit_price):
        self.closed = True
        self.exit_price = exit_price

class OptimizedZeroLagStrategy(bt.Strategy):
    """
        params = dict(


        zl_period=14,
        sine_period=20,
        sma_filter=200,
        mom_period=14,
        mom_lookback=3,


        dca_deviation=1.5,
        take_profit=20,
        percent_sizer=0.95,
        max_positions=0,
        # Additional filters
        use_volatility_filter=False,
        adx_threshold=20,
        debug=True,
        backtest=True
    )
    
    """
    
    params = dict(
        zl_period=14,
        sine_period=20,
        sma_filter=200,
        mom_period=14,
        mom_lookback=3,

        dca_deviation=1.5,
        take_profit=20,
        percent_sizer=0.95,
        max_positions=0,

        use_optimized_trailing=True,
        atr_period=11,

        use_partial_profits=True,
        partial_profit_size=0.3,
        max_drawdown_protection=40,
        position_risk_limit=0.15,

        use_volatility_filter=False,
        adx_threshold=13,
        debug=False,
        backtest=True
    )
    
    def __init__(self):
        self.zero_lag = ZeroLag(self.data, period=self.p.zl_period)
        self.sine_wma = SineWeightedMA(self.zero_lag, period=self.p.sine_period)
        self.sma_200 = bt.ind.SMA(self.data, period=self.p.sma_filter)
        self.momentum = bt.ind.Momentum(self.data, period=self.p.mom_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period, plot=False)
        self.adx = bt.indicators.ADX(self.data, period=14, plot=False)
        self.crossover = bt.ind.CrossOver(self.zero_lag, self.sine_wma)

        self.active_orders = []
        self.entry_prices = []
        self.sizes = []
        self.buy_executed = False
        self.first_entry_price = None

        self.peak_value = 0
        self.current_drawdown = 0
        
        BuySellArrows(self.data0, barplot=True)
    
    def _determine_size(self):
        available_cash = self.broker.get_cash()
        base_size = (available_cash * self.p.percent_sizer) / self.data.close[0]
        if len(self.atr) > 0:
            atr_value = self.atr[0]
            stop_distance = atr_value * 2.0
            risk_per_share = stop_distance
            max_risk_amount = available_cash * self.p.position_risk_limit
            risk_based_size = max_risk_amount / risk_per_share if risk_per_share > 0 else base_size
            size = min(base_size, risk_based_size)
        else:
            size = base_size
        
        return max(size, 0.001)
    
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
            
        if len(self.adx) > 0 and self.adx[0] <= self.p.adx_threshold:
            signal = 0
            
        return signal
    
    def buy_or_initial_entry(self):
        signal = self.get_signal()
        
        if not self.buy_executed and signal > 0:
            size = self._determine_size()
            
            order_tracker = OptimizedOrderTracker(
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
                risk_pct = (size * self.data.close[0] / self.broker.get_value()) * 100
                print(f"Optimized entry: {size:.4f} at {self.data.close[0]:.2f} ({risk_pct:.1f}% of portfolio)")
                
            self.first_entry_price = self.data.close[0]
            self.buy_executed = True
            self.calc_averages()
    
    def dca_condition(self):
        if (self.entry_prices and 
            len(self.active_orders) < self.p.max_positions and
            self.data.close[0] < self.entry_prices[-1] * (1 - self.p.dca_deviation / 100)):
            
            signal = self.get_signal()
            
            if signal > 0:
                size = self._determine_size()
                
                order_tracker = OptimizedOrderTracker(
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
                    print(f"Optimized DCA: {size:.4f} at {self.data.close[0]:.2f}")
                    
                self.calc_averages()
    
    def sell_condition(self):
        if hasattr(self, 'active_orders') and self.active_orders and self.buy_executed:
            current_price = self.data.close[0]
            orders_to_remove = []
            orders_to_reduce = []
            signal = self.get_signal()
            
            for idx, order in enumerate(self.active_orders):
                should_exit = False
                should_partial = False
                exit_reason = ""
                partial_level = None

                if self.p.use_optimized_trailing:
                    atr_value = self.atr[0] if len(self.atr) > 0 else None
                    order.update_optimized_trailing_stop(current_price, atr_value)

                    if self.p.use_partial_profits:
                        partial_level = order.should_take_partial_profit(current_price)
                        if partial_level:
                            should_partial = True

                    if order.trailing_stop and current_price <= order.trailing_stop:
                        should_exit = True
                        exit_reason = "Optimized Trailing Stop"

                if not should_exit and current_price >= order.take_profit_price:
                    should_exit = True
                    exit_reason = "Take Profit (Original)"

                if not should_exit and signal < 0 and self.data.close[0] < self.sma_200[0]:
                    should_exit = True
                    exit_reason = "Signal Exit"
                
                if should_partial and not should_exit:
                    partial_size = order.size * self.p.partial_profit_size
                    remaining_size = order.size - partial_size
                    
                    self.sell(size=partial_size, exectype=bt.Order.Market)
                    
                    order.size = remaining_size
                    
                    if self.p.debug:
                        profit_pct = ((current_price / order.entry_price) - 1) * 100
                        print(f"Partial profit at {partial_level}%: Sold {partial_size:.4f} shares, {profit_pct:.1f}% profit")
                
                elif should_exit:
                    self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                    if self.p.debug:
                        profit_pct = ((current_price / order.entry_price) - 1) * 100
                        print(f"Exit ({exit_reason}): {profit_pct:.2f}% profit")
                    order.close_order(current_price)
                    orders_to_remove.append(idx)
            
            for idx in sorted(orders_to_remove, reverse=True):
                self.active_orders.pop(idx)
            
            if orders_to_remove:
                self.entry_prices = [order.entry_price for order in self.active_orders]
                self.sizes = [order.size for order in self.active_orders]
                
                if not self.active_orders:
                    self.reset_position_state()
                    self.buy_executed = False
                else:
                    self.calc_averages()
    
    def next(self):
        current_value = self.broker.get_value()
        if current_value > self.peak_value:
            self.peak_value = current_value
        self.current_drawdown = (self.peak_value - current_value) / self.peak_value * 100
        if self.current_drawdown > self.p.max_drawdown_protection:
            if self.p.debug:
                print(f"🚨 DRAWDOWN PROTECTION: {self.current_drawdown:.1f}% drawdown - closing all positions")
            for order in self.active_orders:
                if not order.closed:
                    self.sell(size=order.size, exectype=bt.Order.Market)
                    order.close_order(self.data.close[0])
                    self.reset_position_state()
                    self.buy_executed = False
            
            self.reset_position_state()
            self.buy_executed = False
            return
        if not self.buy_executed:
            self.buy_or_initial_entry()
        else:
            self.dca_condition()
        self.sell_condition()

        if self.p.debug and not self.p.backtest:
            dt = self.datas[0].datetime.datetime(0)
            print(f'Processing: {dt}, Close: {self.data.close[0]}, DD: {self.current_drawdown:.1f}%')

if __name__ == '__main__':
    try:
        backtest(
            OptimizedZeroLagStrategy,
            coin='BNB',
            collateral='USDT',
            start_date="2017-01-01",
            end_date="2026-01-08",
            interval="4h",
            init_cash=100,
            plot=True,
            quantstats=True
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()