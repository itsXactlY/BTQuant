import backtrader as bt
import backtrader.indicators as btind


class RSI_Momentum_Trend_Continuation_20260927_054951(bt.Strategy):
    params = (
        ("fast", 10),
        ("slow", 30),
    )

    def __init__(self):
        self.rsi = btind.rsi(self.data, period=self.params.fast)
        self.sma = btind.sma(self.data, period=self.params.slow)
        super().__init__()

    def next(self):
        fast = self.params.fast
        slow = self.params.slow

        rsi_lookback = min(fast, len(self.rsi) - 1)
        sma_lookback = min(slow, len(self.sma) - 1)

        rsi_current = self.rsi[0]
        rsi_previous = self.rsi[-rsi_lookback]
        sma_current = self.sma[0]
        sma_previous = self.sma[-sma_lookback]

        entry_condition = (
            rsi_current > rsi_previous
            and self.data.close[0] > sma_current
            and sma_current > sma_previous
        )

        exit_condition = (
            rsi_current < rsi_previous
            or self.data.close[0] < sma_current
            or sma_current < sma_previous
        )

        if not self.position:
            if entry_condition:
                self.buy()
                if self.risk_management.stop_loss is not None:
                    stop_price = self.data.close[0] * (
                        1 - self.risk_management.stop_loss
                    )
                    self.sell(
                        price=stop_price,
                        exectype=bt.Order.Stop,
                    )
        elif exit_condition:
            self.close()
