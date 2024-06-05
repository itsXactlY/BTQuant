from live_strategys.live_functions import BaseStrategy
import backtrader as bt

class VolumeOscillator(bt.Indicator):
    lines = ('osc',)
    params = (('shortlen', 5),
            ('longlen', 10))
    
    def __init__(self):
        shortlen, longlen = self.params.shortlen, self.params.longlen
        self.lines.short = bt.indicators.ExponentialMovingAverage(self.data.volume, period=shortlen)
        self.lines.long = bt.indicators.ExponentialMovingAverage(self.data.volume, period=longlen)
        self.lines.osc = (self.lines.short - self.lines.long) / self.lines.long * 100

    def next(self):
        self.osc[0] = (self.lines.short[0] - self.lines.long[0]) / self.lines.long[0] * 100

class QQEIndicator(bt.Indicator):
    params = (
        ("period", 6),
        ("fast", 5),
        ("q", 3.0),
    )
    lines = ("qqe_line",)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        self.matr = bt.indicators.EMA(self.atr, period=int((self.p.period * 2) - 1))
        self.dar = bt.indicators.EMA(self.atr - self.p.q, period=int((self.p.period * 2) - 1))
        self.lines.qqe_line = self.rsi + self.dar

class QQE_Example(BaseStrategy):
    params = (
        ("ema_length", 20),
        ('hull_length', 53),
        ("printlog", True),
        ("backtest", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qqe = QQEIndicator(self.data)
        self.hma = bt.indicators.HullMovingAverage(self.data, period=self.p.hull_length)
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_length)
        self.volosc = VolumeOscillator(self.data)
        self.DCA = False
        self.buy_executed = False
        self.conditions_checked = False
        self.stake = self.broker.getcash()

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
            (self.data.close[-1] > self.hma[0]) and \
            (self.volosc.osc[-1] > self.volosc.lines[0]):
                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.load_trade_data()
                    self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.buy_executed = True
                    self.conditions_checked = True
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed and (self.qqe.qqe_line[-1] > 0) and \
        (self.data.close[-1] < self.hma[0]) and \
        (self.volosc.osc[-1] < self.volosc.lines[0]):
            if self.params.backtest == False:
                self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True


    def next(self):
        BaseStrategy.next(self)
        self.conditions_checked = False
