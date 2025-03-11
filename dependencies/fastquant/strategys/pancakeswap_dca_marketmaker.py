from fastquant.strategys.base import BaseStrategy, BuySellArrows

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
                self.sizes.append(self.usdt_amount)
                self.enqueue_web3order('buy', amount=self.usdt_amount)
                self.calc_averages()
                self.buy_executed = True
                self.conditions_checked = True


    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):  
                print(f'DCA Buy Condition Met: {self.data.close[0]:.12f} < {self.entry_prices[-1] * (1 - self.params.dca_deviation):.12f}')
                if self.params.backtest == False:
                    self.enqueue_web3order('buy', amount=self.usdt_amount)
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.usdt_amount)
                    print(f'\nDCA-BUY EXECUTED AT {self.data.close[0]:.12f}\n')
                    self.calc_averages()
                    self.buy_executed = True
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
                    self.enqueue_web3order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
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

