from datetime import datetime
from fastquant.strategies.base import BaseStrategy
from fastquant.strategies.custom_indicators.SuperTrend import SuperTrend
from fastquant.strategies.custom_indicators.WilliamsAligator import WilliamsAlligator

class AliG_STrend(BaseStrategy):
    params = (
        ('dca_threshold', 20.5),
        ('take_profit', 5),
        ('percent_sizer', 0.045),
        ('premium', 0.003),
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('INIT AliG_STrend')
        self.supertrend = SuperTrend(plotname='Supertrend: ')
        self.alligator = WilliamsAlligator(plotname='WilliamsAlligator: ')
        self.DCA = True
        self.buy_executed = False
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.supertrend.lines.super_trend[0] > 0 and \
                    self.alligator.lines.jaw[0] > 0 and \
                    self.alligator.lines.teeth[0] > 0 and \
                    self.alligator.lines.lips[0] > 0:

                if self.p.backtest == False:
                    self.calculate_position_size()
                    self.entry_prices.append(self.data.close[0])
                    print(f'\n\nBUY EXECUTED AT {self.data.close[0]:.9f}\n')
                    self.sizes.append(self.usdt_amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.usdt_amount)
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]
                    self.calc_averages()
                    self.buy_executed = True
                    alert_message = f"""\nBuy Alert arrived!\nExchange: {self.exchange}\nAction: buy {self.asset}\nEntry Price: {self.data.close[0]:.9f}\nTake Profit: {self.take_profit_price:.9f}"""
                    self.send_alert(alert_message)
                elif self.p.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                        self.first_entry_price = self.data.close[0]

                    self.calc_averages()
                    self.buy_executed = True
                    if self.p.debug:
                        self.log_entry()
        self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if self.supertrend.lines.super_trend[0] > 0 and \
                    self.alligator.lines.jaw[0] > 0 and \
                    self.alligator.lines.teeth[0] > 0 and \
                    self.alligator.lines.lips[0] > 0:

                if self.p.backtest is False:
                    self.calculate_position_size()
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.usdt_amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.usdt_amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    alert_message = f"""\nDCA Alert arrived!\nExchange: {self.exchange}\nAction: buy {self.asset}\nEntry Price: {self.data.close[0]:.9f}\nTake Profit: {self.take_profit_price:.9f}"""
                    self.send_alert(alert_message)
                elif self.p.backtest is True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.buy_executed = True
                    if self.p.debug:
                        self.log_entry()
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed:
            current_price = round(self.data.close[0], 9)
            avg_price = round(self.average_entry_price, 9)
            tp_price = round(self.take_profit_price, 9)

            if current_price >= tp_price:
                if self.params.backtest:
                    self.close()
                    self.buy_executed = False
                    if self.p.debug:
                        print(f"Position closed at {current_price:.9f}, profit taken")
                        self.log_exit("Sell Signal - Take Profit")
                    self.reset_position_state()
                else:
                    self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
                    alert_message = f"""Close {self.asset}"""
                    self.send_alert(alert_message)
                    self.reset_position_state()
                    self.buy_executed = False
            else:
                if self.p.debug == True:
                    print(
                        f"| - Awaiting take profit target.\n"
                        f"| - Current close price: {self.data.close[0]:.12f},\n"
                        f"| - Average entry price: {self.average_entry_price:.12f},\n"
                        f"| - Take profit price: {self.take_profit_price:.12f}")
        self.conditions_checked = True