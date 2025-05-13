import backtrader as bt
from .base import BaseStrategy
from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.indicators.RSX import RSX
from backtrader import Order
from datetime import datetime

class SuperSTrend_Scalp(BaseStrategy):
    params = (
        ("dca_deviation", 0.2),
        ("take_profit", 0.2),
        ("signal_tp", 99),
        ('percent_sizer', 0.05),
        ('trailing_stop_pct', 0.4),
        # Trend Strenght
        ("adx_period", 13),
        ("adx_strength", 31),
        ("di_period", 14),
        ("adxth", 25),

        # Supertrends
        ("st_fast", 2),
        ('st_fast_multiplier', 3),
        ("st_slow", 6),
        ('st_slow_multiplier', 7),

        # RSX
        ('rsxlen', 14),

        ('debug', False),
        ("backtest", None)
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period, plot=False)
        self.plusDI = bt.indicators.PlusDI(self.data, period=self.p.di_period, plot=False)
        self.minusDI = bt.indicators.MinusDI(self.data, period=self.p.di_period, plot=False)
        self.supertrend_fast = SuperTrend(period=self.p.st_fast, multiplier=self.p.st_fast_multiplier, plotname='SuperTrend Fast: ', plot=False)
        self.supertrend_slow = SuperTrend(period=self.p.st_slow, multiplier=self.p.st_slow_multiplier, plotname='SuperTrend Slow: ', plot=False)
        self.supertrend_uptrend_signal = bt.indicators.CrossOver(self.supertrend_fast, self.supertrend_slow, plot=False)
        self.rsx = RSX(self.data, length=self.p.rsxlen, plot=False)
        
        self.DCA = True
        self.peak = 0
        self.buy_executed = False
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):
            if not self.buy_executed:
                if self.p.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.calculate_position_size()
                    print(f'\nBUY EXECUTED AT {self.data.close[0]:.9f}\n')
                    self.sizes.append(self.usdt_amount)
                    self.order = self.buy(size=self.usdt_amount, exectype=Order.Market)
                    print(self.order)
                    # self.enqueue_order('buy', exchange=self.exchange, account=self.p.account, asset=self.asset, amount=self.usdt_amount)
                    
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.buy_executed = True
                    
                    self.calc_averages()
                    alert_message = f"""\nBuy Alert arrived!\nAction: buy {self.asset}\nEntry Price: {self.data.close[0]:.9f}\nTake Profit: {self.take_profit_price:.9f}"""
                    self.send_alert(alert_message)
                elif self.p.backtest == True:
                    self.buy(self.data, size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.calc_averages()
                    self.buy_executed = True
        self.conditions_checked = True

    def dca_or_short_condition(self):
        self.peak = max(self.peak, self.data.close[0])
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):
            if self.buy_executed and not self.conditions_checked:
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):    
                    if self.p.backtest is False:
                        self.calculate_position_size()
                        self.calc_averages()
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.usdt_amount)
                        self.order = self.buy(size=self.usdt_amount, exectype=Order.Market)
                        print(self.order)
                        # self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.usdt_amount)
                        self.buy_executed = True
                        self.conditions_checked = True
                        alert_message = f"""\nDCA Alert arrived!\nAction: buy {self.asset}\nEntry Price: {self.data.close[0]:.9f}\nTake Profit: {self.take_profit_price:.9f}"""
                        self.send_alert(alert_message)
                    elif self.p.backtest is True:
                        self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.stake)
                        self.calc_averages()
                        self.buy_executed = True
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed:
            current_price = round(self.data.close[0], 9)
            # avg_price = round(self.average_entry_price, 9)
            tp_price = round(self.take_profit_price, 9)

            if current_price >= tp_price:
                if self.params.backtest:
                    self.order = self.close()
                    print(self.order)
                    self.buy_executed = False
                    if self.p.debug:
                        print(f"Position closed at {current_price:.9f}, profit taken")
                        self.log_exit("Sell Signal - Take Profit")
                    self.reset_position_state()
                else:
                    _size = self.broker.get_position_info(self.data)['size']
                    print('sellsize: ', _size)
                    self.order = self.close(self.data, size=_size, exectype=Order.Market)
                    print(self.order)
                    # self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
                    alert_message = f"""Close {self.asset}"""
                    self.send_alert(alert_message)
                    self.reset_position_state()
                    self.buy_executed = False
            # else:
                # if self.p.debug == True:
                #     print(
                #         f"| - Awaiting take profit target.\n"
                #         f"| - Current close price: {self.data.close[0]:.12f},\n"
                #         f"| - Average entry price: {self.average_entry_price:.12f},\n"
                #         f"| - Take profit price: {self.take_profit_price:.12f}")
        self.conditions_checked = True

    def next(self):
        # print(f"Close price: {self.data.close[0]:.9f}")
        dt = self.datas[0].datetime.datetime(0)  # Or self.data.datetime.datetime(0)
        print(f'Realtime: {datetime.now()} processing candle date: {dt}, with close price: {self.data.close[0]}')
        super().next()

        # print(datetime.now(), self.data.close[0])