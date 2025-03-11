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

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
            if self.buy_executed and not self.conditions_checked:
                if self.crossover > 0 and self.momentum > 0 and self.mama.lines.MAMA > self.mama.lines.FAMA:  
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


