import backtrader as bt
from backtrader.indicators.SuperTrend import SuperTrend
from backtrader.indicators.ASI import AccumulativeSwingIndex
from .base import BaseStrategy, BuySellArrows

class aLcas_STrend_AccumulativeSwingIndex(BaseStrategy):
    params = (
        ('stlen', 7),
        ('stmult', 7.0),
        ("dca_deviation", 1.5),
        ("take_profit", 2),
        ('percent_sizer', 0.1),
        ('backtest', None)
    )
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        BuySellArrows(self.data0, barplot=True)
        self.DCA = True
        
        self.asi_short = AccumulativeSwingIndex(period=7)
        self.asi_long = AccumulativeSwingIndex(period=14)
        # Indicators
        self.sttrend = SuperTrend(self.data, period=self.p.stlen, multiplier=self.p.stmult)

        # Buy/Sell Signals
        self.stLong = bt.ind.CrossOver(self.data.close, self.sttrend)
        self.stShort = bt.ind.CrossDown(self.data.close, self.sttrend)

    def buy_or_short_condition(self):
        # Signallogik unveraendert. Nur die Ausfuehrung: create_order() statt
        # des alten Live-first self.buy()/enqueue_order()-Zweigs, der bei
        # backtest=None in KEINEM Zweig landete und daher nie handelte.
        if not self.buy_executed and not self.conditions_checked:
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1]:
                if self.create_order(action='BUY') is not None:
                    self.buy_executed = True
                    self.conditions_checked = True
                    return True
        return False

    def dca_or_short_condition(self):
        if self.buy_executed and not self.conditions_checked:
            if self.stLong and self.asi_short[0] > self.asi_short[-1] and self.asi_short[0] > 5 and self.asi_long[0] > self.asi_long[-1]:
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
                    if self.create_order(action='BUY') is not None:
                        self.conditions_checked = True
                        return True
        return False

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

            for order in list(self.active_orders):
                self.close_order(order)
            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True
            return True
        return False


# Loader-Vertrag: import_strategy() macht getattr(module, <Dateiname>).
# Die Konzept-Klasse darunter behaelt ihren Namen, das Modul ist der Einstieg.
aLcas_Strend_ASI = aLcas_STrend_AccumulativeSwingIndex
