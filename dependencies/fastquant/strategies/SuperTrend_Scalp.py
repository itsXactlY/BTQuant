import backtrader as bt
from fastquant.strategies.base import BaseStrategy, BuySellArrows
from fastquant.strategies.custom_indicators.SuperTrend import SuperTrend

class SuperSTrend_Scalper(BaseStrategy):
    params = (
        ("dca_deviation", 0.2),
        ("take_profit_percent", 0.4),
        ('percent_sizer', 0.01), # 0.01 -> 1%
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

        ('debug', False),
        ("backtest", None)
        )

    def __init__(self, **kwargs):
        BuySellArrows(self.data0, barplot=True)
        super().__init__(**kwargs)
        self.adx = bt.indicators.ADX(self.data, plot=False)
        self.plusDI = bt.indicators.PlusDI(self.data, plot=False)
        self.minusDI = bt.indicators.MinusDI(self.data, plot=False)
        self.supertrend_fast = SuperTrend(period=self.p.st_fast, multiplier=self.p.st_fast_multiplier, plotname='SuperTrend Fast: ', plot=True)
        self.supertrend_slow = SuperTrend(period=self.p.st_slow, multiplier=self.p.st_slow_multiplier, plotname='SuperTrend Slow: ', plot=True)
        self.supertrend_uptrend_signal = bt.indicators.CrossOver(self.supertrend_fast, self.supertrend_slow, plot=False)
        # self.supertrend_downtrend_signal = bt.indicators.CrossDown(self.supertrend_fast, self.supertrend_slow, plot=False) # NOT USED IN THIS EXAMPLE
        self.DCA = True
        self.buy_executed = False
        self.conditions_checked = False

    def buy_or_short_condition(self):
        print('buy_or_short_condition')
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):

            if self.params.backtest == False:
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(self.amount)
                self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                self.calc_averages()
                self.buy_executed = True
                self.conditions_checked = True
            elif self.params.backtest == True:
                self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                self.buy_executed = True
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(self.stake)
                self.calc_averages()

    def dca_or_short_condition(self):
        print('dca_or_short_condition')
        if (self.position and \
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):

            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):    
                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    # self.buy_executed = True
                    # self.conditions_checked = True
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()

    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            average_entry_price = sum(self.entry_prices) / len(self.entry_prices) if self.entry_prices else 0

            # Avoid selling at a loss or below the take profit price
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                self.log(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )
                return

        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            if self.data.close[0] < self.average_entry_price:
                print(f'Nothing Todo here. {self.average_entry_price, self.take_profit_price}')
                return
            
            if self.params.backtest == False:
                self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
            
        self.reset_position_state()
        self.buy_executed = False
        self.conditions_checked = True

    def next(self):
        # print('buy executed :', True or False)
        BaseStrategy.next(self)