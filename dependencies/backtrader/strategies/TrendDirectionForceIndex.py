from backtrader.indicators.TrendDirectionForceIndex import TrendDirectionForceIndex
from .base import BaseStrategy, bt

class TDFStrategy(BaseStrategy):
    params = \
    (
        ('tdf_period', 13),
        ('buy_threshold', 0.5),
        ('sell_threshold', -0.5),
        ('dca_threshold', 2.25),
        ('take_profit', 3),
        ('percent_sizer', 0.01), # 0.01 -> 1%
        ('debug', True),
        ('warmup_period_me', 15),
        ("backtest", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tdf = TrendDirectionForceIndex(self.data, period=self.params.tdf_period)
        self.DCA = True
        self.print_counter = 0

    def buy_or_short_condition(self):
        if self.tdf.lines.ntdf[0] > self.params.buy_threshold:
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

    def dca_or_short_condition(self):
        if self.buy_executed and self.tdf.lines.ntdf[0] > self.params.buy_threshold:
            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):    
                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    # self.load_trade_data()
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

    def sell_or_cover_condition(self):
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
                print(f'\n\n\nSELL EXECUTED AT {self.data.close[0]}\n\n\n')
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()

            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    def stop(self):
        self.order_queue.put(None)  # Signal the order thread to stop
        self.order_thread.join()  # Wait for the order thread to finish