from fastquant.strategies.base import BaseStrategy, BuySellArrows

class Pancakeswap_dca_mm(BaseStrategy):
    params = (
        ('stlen', 7),
        ('stmult', 7.0),
        ("dca_deviation", 2.25),
        ("take_profit", 5),
        ('percent_sizer', 0.1),
        ('backtest', None)
    )
    def __init__(self, **kwargs):
        BuySellArrows(self.data0, barplot=True)
        super().__init__(**kwargs)
        self.DCA = True

        self.stake = 1 # temporary for the BaseStrategy, not used here on WEB3.
        self.stake_to_use = self.stake
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.params.backtest == False:
                self.entry_prices.append(self.data.close[0])
                print(f'\nBUY EXECUTED AT {self.data.close[0]:.12f}\n')
                self.sizes.append(self.amount)
                self.enqueue_web3order('buy', amount=self.amount)
                self.calc_averages()
                self.buy_executed = True
                self.conditions_checked = True


    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):  
                print(f'DCA Buy Condition Met: {self.data.close[0]:.12f} < {self.entry_prices[-1] * (1 - self.params.dca_deviation):.12f}')
                if self.params.backtest == False:
                    self.enqueue_web3order('buy', amount=self.amount)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    print(f'\nDCA-BUY EXECUTED AT {self.data.close[0]:.12f}\n')
                    self.calc_averages()
                    self.buy_executed = True
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
                self.enqueue_web3order('sell')
                print(f'Sell at {self.dataclose[0]:.12f}')
                self.buy_executed = False
                self.conditions_checked = True
                self.reset_position_state()
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)

    def next(self):
        self.conditions_checked = False  # Reset at the start of every next iteration
        BaseStrategy.next(self)

