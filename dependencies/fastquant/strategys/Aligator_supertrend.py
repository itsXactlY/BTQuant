from datetime import datetime
from fastquant.strategys.base import BaseStrategy
from fastquant.strategys.custom_indicators.SuperTrend import SuperTrend
from fastquant.strategys.custom_indicators.WilliamsAligator import WilliamsAlligator

class AliG_STrend(BaseStrategy):
    params = (
        ('dca_threshold', 20.5),
        ('take_profit', 5),
        ('percent_sizer', 0.045),
        ('premium', 0.003),
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
                    self.calc_averages()
                    self.buy_executed = True
                    alert_message = f"""\nBuy Alert arrived!\nExchange: {self.exchange}\nAction: buy {self.asset}\nEntry Price: {self.data.close[0]:.9f}\nTake Profit: {self.take_profit_price:.9f}"""
                    self.send_alert(alert_message)
                elif self.p.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
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
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
        self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed:
            current_price = round(self.data.close[0], 9)
            avg_price = round(self.average_entry_price, 9)
            tp_price = round(self.take_profit_price, 9)

            if current_price >= tp_price and current_price >= avg_price:
                if self.params.backtest:
                    self.close()
                    self.reset_position_state()
                    self.buy_executed = False
                    if self.p.debug:
                        print(f"Position closed at {current_price:.9f}, profit taken")
                else:
                    self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
                    alert_message = f"""Close {self.asset}"""
                    self.send_alert(alert_message)
                    self.reset_position_state()
                    self.buy_executed = False
            else:
                if self.p.debug == True:
                    print(
                        f"| - Avoiding sell at a loss or below take profit.\n"
                        f"| - Current close price: {self.data.close[0]:.12f},\n "
                        f"| - Average entry price: {self.average_entry_price:.12f},\n "
                        f"| - Take profit price: {self.take_profit_price:.12f}")
        self.conditions_checked = True