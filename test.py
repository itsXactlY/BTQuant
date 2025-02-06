from fastquant import get_database_data, backtest, bt, BaseStrategy
data = get_database_data("BTC", "2024-07-01", "2024-08-08", "1m")

class AccumulativeSwingIndex(bt.Indicator):
    lines = ('asi',)
    params = (
        ('period', 14),  # Period for the ASI calculation
    )

    def __init__(self):
        # Calculate ASI using a custom function
        self.addminperiod(self.params.period)
        self.lines.asi = bt.indicators.WMA(
        self.data.close - self.data.open,
        period=self.params.period
        ) + bt.indicators.SumN(
        self.data.high - self.data.low,
        period=self.params.period
        )

class SuperTrend(bt.Indicator):
    lines = ('super_trend',)
    params = (('period', 7), ('multiplier', 3))

    def __init__(self):
        self.st = [0]
        self.finalupband = [0]
        self.finallowband = [0]
        self.addminperiod(self.p.period)
        atr = bt.ind.ATR(self.data, period=self.p.period)
        self.upperband = (self.data.high + self.data.low) / 2 + self.p.multiplier * atr
        self.lowerband = (self.data.high + self.data.low) / 2 - self.p.multiplier * atr

    def next(self):
        pre_upband = self.finalupband[0]
        pre_lowband = self.finallowband[0]
        if self.upperband[0] < self.finalupband[-1] or self.data.close[-1] > self.finalupband[-1]:
            self.finalupband[0] = self.upperband[0]
        else:
            self.finalupband[0] = self.finalupband[-1]
        if self.lowerband[0] > self.finallowband[-1] or self.data.close[-1] < self.finallowband[-1]:
            self.finallowband[0] = self.lowerband[0]
        else:
            self.finallowband[0] = self.finallowband[-1]
        if self.data.close[0] <= self.finalupband[0] and ((self.st[-1] == pre_upband)):
            self.st[0] = self.finalupband[0]
            self.lines.super_trend[0] = self.finalupband[0]
        elif (self.st[-1] == pre_upband) and (self.data.close[0] > self.finalupband[0]):
            self.st[0] = self.finallowband[0]
            self.lines.super_trend[0] = self.finallowband[0]
        elif (self.st[-1] == pre_lowband) and (self.data.close[0] >= self.finallowband[0]):
            self.st[0] = self.finallowband[0]
            self.lines.super_trend[0] = self.finallowband[0]
        elif (self.st[-1] == pre_lowband) and (self.data.close[0] < self.finallowband[0]):
            self.st[0] = self.finalupband[0]
            self.lines.super_trend[0] = self.st[0]

class aLcas_STrend_AccumulativeSwingIndex(BaseStrategy):
    params = (
        ('stlen', 7),
        ('stmult', 7.0),
        ("dca_deviation", 4.5),
        ("take_profit", 2),
        ('percent_sizer', 0.1),
        ('backtest', None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DCA = True
        
        # Indicators
        self.asi_short = AccumulativeSwingIndex(period=7, plot=True)
        self.asi_long = AccumulativeSwingIndex(period=14, plot=True)
        self.sttrend = SuperTrend(self.data, period=self.p.stlen, multiplier=self.p.stmult, plot=True)

        # Buy/Sell Signals
        self.stLong = bt.ind.CrossOver(self.data.close, self.sttrend, plot=True)
        self.stShort = bt.ind.CrossDown(self.data.close, self.sttrend, plot=True)

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1]:

                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
                    self.sizes.append(self.amount)
                    # self.load_trade_data()
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
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1]:

                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):    
                    if self.params.backtest == False:
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.amount)
                        # self.load_trade_data()
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

            if self.params.backtest == False:
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()

            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True



backtest("bbands", data, init_cash=1000, backtest=True, plot=True, verbose=0)