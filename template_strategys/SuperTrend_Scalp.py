import backtrader as bt
from live_strategys.live_functions import BaseStrategy
from custom_indicators.SuperTrend import SuperTrend

class SuperSTrend_Scalper(BaseStrategy):
    params = (
        ("dca_threshold", 2.5),
        ("take_profit", 4),
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
        # RevFinder
        ('reversal_lookback', 10),
        ('reversal_malen', 40),
        ('reversal_mult', 2.2),
        ('reversal_rangethreshold', 0.9),
        ('debug', False),
        ("backtest", None)
        )

    def __init__(self):
        super().__init__()
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
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):
            
            if self.params.backtest == False:
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(self.amount)
                self.load_trade_data()
                self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                self.buy_executed = True
                self.conditions_checked = True
            elif self.params.backtest == True:
                self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                self.take_profit_price = self.data.close[-1] * (1 + self.params.take_profit / 100)
                self.buy_executed = True
                self.conditions_checked = True

    def dca_or_short_condition(self):
        if (self.position and \
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
            ):
            
            if self.params.backtest == False:
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(self.amount)
                self.load_trade_data()
                self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                self.buy_executed = True
                self.conditions_checked = True
            elif self.params.backtest == True:
                self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                self.take_profit_price = self.data.close[-1] * (1 + self.params.take_profit / 100)
                self.buy_executed = True
                self.conditions_checked = True

    
    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            if self.params.backtest == False:
                self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    def next(self):
        BaseStrategy.next(self)