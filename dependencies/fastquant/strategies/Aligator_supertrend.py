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
        self.supertrend = SuperTrend(plotname='Supertrend: ')
        self.alligator = WilliamsAlligator(plotname='WilliamsAlligator: ')
        self.DCA = True
        self.buy_executed = False
        self.conditions_checked = False  # Flag to ensure conditions are checked only once per candle
        

    def next(self):
        BaseStrategy.next(self)
        self.print_counter += 1

        # Reset conditions_checked flag for the new candle
        self.conditions_checked = False

        if self.live_data and self.buy_executed:
            if self.print_counter % 5 == 0:  # reduce logging spam
                print(f'|\n| {datetime.now()}\n| Price: {self.data.close[0]:.12f} Entry: {self.average_entry_price:.12f} TakeProfit: {self.take_profit_price:.12f}')

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.supertrend.lines.super_trend[0] > 0 and \
                    self.alligator.lines.jaw[0] > 0 and \
                    self.alligator.lines.teeth[0] > 0 and \
                    self.alligator.lines.lips[0] > 0:

                self.entry_prices.append(self.data.close[0])
                print(self.amount)
                self.sizes.append(self.amount)
                self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                self.load_trade_data()
                self.buy_executed = True
                self.conditions_checked = True

    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if self.supertrend.lines.super_trend[0] > 0 and \
                    self.alligator.lines.jaw[0] > 0 and \
                    self.alligator.lines.teeth[0] > 0 and \
                    self.alligator.lines.lips[0] > 0:

                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.calc_averages()
                    print(f'DCA Buy at {self.data.close[0]:.12f} - Take Profit {self.take_profit_price:.12f}')
                    self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.conditions_checked = True

    def sell_or_cover_condition(self):
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            average_entry_price = sum(self.entry_prices) / len(self.entry_prices) if self.entry_prices else 0
            # print(average_entry_price, self.take_profit_price)

            # Avoid selling at a loss or below the take profit price
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                self.log(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )
                return

            print(f"| - Taking Profit reached at {self.take_profit_price:.12f} Closing Position.")
            self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            self.reset_position_state()
            self.buy_executed = False