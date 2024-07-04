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

live = True

class QQE_DCA_Example(BaseStrategy):
    params = (
        ('dca_threshold', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.01), # 0.01 -> 1%
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
        self.DCA = True
        self.buy_executed = False
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
                (self.data.close[-1] > self.hma[0]) and \
                (self.volosc.osc[-1] > self.volosc.lines[0]
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
        if self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
                (self.data.close[-1] > self.hma[0]) and \
                (self.volosc.osc[-1] > self.volosc.lines[0]
            ):
            
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):    
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

    def sell_or_cover_condition(self):
        if self.p.debug:
            print(f'| - sell_or_cover_condition {self.data._name} Entry:{self.average_entry_price:.12f} TakeProfit: {self.take_profit_price:.12f}')
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

            if self.params.backtest == False:
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()

            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    def next(self):
        BaseStrategy.next(self)
        self.conditions_checked = False
