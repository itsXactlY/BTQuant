from backtrader.indicators.RSX import RSX
from backtrader.indicators.AccumulativeSwingIndex import AccumulativeSwingIndex
from backtrader.indicators.SuperTrend import SuperTrend
from .base import BaseStrategy, bt

class STrend_RSX_AccumulativeSwingIndex(BaseStrategy):
    params = (
        ('stlen', 7),
        ('stmult', 7.0),
        ('rsxlen', 14),
        ("dca_deviation", 2),
        ("take_profit", 8),
        ('percent_sizer', 0.1), # 0.1 - 10%
        ('trailing_stop_pct', 1.5),
        ('backtest', None)
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DCA = True
        self.peak = 0
        
        self.asi_short = AccumulativeSwingIndex(period=7, plot=True)
        self.asi_long = AccumulativeSwingIndex(period=14, plot=True)
        self.rsx = RSX(self.data, length=self.p.rsxlen, plot=False)
        self.sttrend = SuperTrend(self.data, period=self.p.stlen, multiplier=self.p.stmult, plot=True)
        self.stLong = bt.ind.CrossOver(self.data.close, self.sttrend, plot=True)

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1] and self.rsx[0] < 30:

                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
                    self.sizes.append(self.amount)
                    # self.load_trade_data()
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    alert_message = f"""\nBuy Alert arrived!
Strategy: Supertrend, RSX, AccumulativeSwingIndex + Trailing Exit
Action: long {self.asset}
Entry Price: {self.data.close[0]:.9f}
Take Profit: {self.take_profit_price:.9f} - Hard exit at temporary 99% - Algo using an Trailing Stop approach."""

                    self.send_alert(alert_message)
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if self.buy_executed:
                self.peak = max(self.peak, self.data.close[0])
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1] and self.rsx[0] < 30:

                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):    
                    if self.params.backtest == False:
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.amount)
                        # self.load_trade_data()
                        print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
                        self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                        self.calc_averages()
                        self.buy_executed = True
                        self.conditions_checked = True
                        alert_message = f"""\nDCA Buy Alert arrived!
Strategy: Supertrend RSX AccumulativeSwingIndex + Trailing Exit
Action: long {self.asset}
Entry Price: {self.data.close[0]:.9f}
Take Profit: {self.take_profit_price:.9f} - Hard exit at temporary 99% - Algo using an Trailing Stop approach."""

                        self.send_alert(alert_message)
                    elif self.params.backtest == True:
                        self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                        self.buy_executed = True
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.stake)
                        self.calc_averages()
                        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price and self.rsx[0] > 70: # use RSX as extra trailing option
                trailing_trigger = self.peak * (1 - self.p.trailing_stop_pct / 100)
                if self.data.close[0] < trailing_trigger:
                    if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                        self.conditions_checked = True
                        return

                    if self.params.backtest == False:
                        self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
                        alert_message = f"""Close {self.asset} - Trailing Takeprofit reached"""
                        self.send_alert(alert_message)
                    elif self.params.backtest == True:
                        self.close()

                    print(f"Exiting trade at {self.data.close[0]:.12f} after trailing stop triggered.")
                    self.peak = 0
                    self.reset_position_state()
                    self.buy_executed = False
                    self.conditions_checked = True
