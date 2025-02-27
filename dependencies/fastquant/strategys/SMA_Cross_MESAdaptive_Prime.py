import backtrader as bt
from fastquant.strategys.base import BaseStrategy
from fastquant.strategys.custom_indicators.MesaAdaptiveMovingAverage import MAMA

class SMA_Cross_MESAdaptivePrime(BaseStrategy, bt.SignalStrategy):
    params = (
        ('fast', 13),
        ('slow', 37),
        ('dca_deviation', 1.5),
        ('take_profit_percent', 1),
        ('percent_sizer', 0.045), # 4.5%
        ('debug', False),
        ("backtest", None)
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Simple Moving Averages
        self.sma17 = bt.ind.SMA(period=17)
        self.sma47 = bt.ind.SMA(period=47)
        # Mesa Adaptive Moving Average
        self.mama = MAMA(self.data, fast=self.p.fast, slow=self.p.slow)
        # Crossover signal
        self.crossover = bt.ind.CrossOver(self.sma17, self.sma47)
        # Momentum
        self.momentum = bt.ind.Momentum(period=42)
        self.DCA = True

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if self.crossover > 0 and self.momentum > 0 and self.mama.lines.MAMA > self.mama.lines.FAMA:
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
            if self.crossover > 0 and self.momentum > 0 and self.mama.lines.MAMA > self.mama.lines.FAMA:  
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):    
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


