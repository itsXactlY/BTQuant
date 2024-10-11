import backtrader as bt
from fastquant.strategies.base import BaseStrategy, BuySellArrows
from numpy import isnan

class VolumeOscillator(bt.Indicator):
    lines = ('short', 'long', 'osc')
    params = (
        ('shortlen', 5),
        ('longlen', 10),
        ('debug', False),
        )

    def __init__(self):
        # self.addminperiod(self.p.longlen)
        shortlen, longlen = self.params.shortlen, self.params.longlen
        self.lines.short = bt.indicators.ExponentialMovingAverage(self.data.volume, period=shortlen)
        self.lines.long = bt.indicators.ExponentialMovingAverage(self.data.volume, period=longlen)

    def next(self):
        try:
            if self.p.debug:
                # Log the volume data and EMAs to check for issues
                print(f"Volume: {self.data.volume[0]}, Short EMA: {self.lines.short[0]}, Long EMA: {self.lines.long[0]}")

            # Check if the volume data or EMAs are None, zero, or NaN to avoid division by zero
            if not (self.lines.long[0] and self.lines.long[0] > 0 and not isnan(self.lines.long[0])):
                print(f"Invalid volume data detected: Long EMA is {self.lines.long[0]}. Setting oscillator to 0.")
                self.lines.osc[0] = 0
            else:
                # Calculate oscillator only when valid values exist
                self.lines.osc[0] = (self.lines.short[0] - self.lines.long[0]) / self.lines.long[0] * 100
        except Exception as e:
            print(f"Error calculating Volume Oscillator: {e}")
            self.lines.osc[0] = 0

class QQEIndicator(bt.Indicator):
    params = (
        ("period", 6),
        ("fast", 5),
        ("q", 3.0),
        ("debug", False)
    )
    lines = ("qqe_line",)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        self.dar = bt.If(self.atr > 0, bt.indicators.EMA(self.atr - self.p.q, period=int((self.p.period * 2) - 1)), 0)
        self.lines.qqe_line = bt.If(self.rsi > 0, self.rsi + self.dar, 0)

    def next(self):
        # check if ATR is not zero to avoid division by zero errors
        if self.atr[0] == 0:
            print("ATR is zero, skipping this iteration to avoid division by zero.")
            return

        # check if RSI and DAR are valid before computing the QQE line
        if self.rsi[0] != 0 and self.dar[0] != 0:
            self.lines.qqe_line[0] = self.rsi[0] + self.dar[0]
        else:
            self.lines.qqe_line[0] = 0
        
        if self.p.debug:
            print(f"RSI: {self.rsi[0]}, DAR: {self.dar[0]}, ATR: {self.atr[0]}, QQE: {self.lines.qqe_line[0]}")

class QQE_Example(BaseStrategy):
    params = (
        ("ema_length", 20),
        ('hull_length', 53),
        ('take_profit_percent', 2),
        ('dca_deviation', 1.5),  # DCA deviation
        ('percent_sizer', 0.1),
    )

    def __init__(self, **kwargs):
        print('Initialized QQE')
        BuySellArrows(self.data0, barplot=True)
        super().__init__(**kwargs)
        self.qqe = QQEIndicator(self.data)
        self.hma = bt.indicators.HullMovingAverage(self.data, period=self.p.hull_length)
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_length)
        self.volosc = VolumeOscillator(self.data)
        self.DCA = True
        self.conditions_checked = False

        if self.strategy_logging:
            print("===Strategy level arguments===")
            print("ema_length :", self.p.ema_length)
            print("hull_length :", self.p.hull_length)
            print("takeprofit :", self.p.take_profit_percent)
            print("dca_deviation :", self.p.dca_deviation)
            print("percent_sizer :", self.p.percent_sizer)

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
               (self.data.close[-1] < self.hma[0]) and \
               (self.volosc.osc[-1] < self.volosc.lines.short[0]):

                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
                    self.sizes.append(self.amount)
                    self.load_trade_data()
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
                    self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
               (self.data.close[-1] < self.hma[0]) and \
               (self.volosc.osc[-1] < self.volosc.lines.short[0]):

                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):    
                    if self.params.backtest == False:
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.amount)
                        self.load_trade_data()
                        print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
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
                        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            average_entry_price = sum(self.entry_prices) / len(self.entry_prices) if self.entry_prices else 0

            # Avoid selling at a loss or below the take profit price
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                print(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )
                self.conditions_checked = True
                return

            if self.params.backtest == False:
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()

            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    # def next(self):
    #     self.conditions_checked = False  # Reset at the start of every next iteration
    #     BaseStrategy.next(self)