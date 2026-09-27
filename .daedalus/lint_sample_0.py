import backtrader as bt
import backtrader.indicators as btind


class EMA_Crossover_Momentum_with_RSI_Confirmation_20260927_054930(bt.Strategy):
    params = (
        ("fast", 10),
        ("slow", 30),
    )

    def __init__(self):
        super().__init__()
        self.ema = btind.EMA(self.data, period=self.p.fast)
        self.slow_ema = btind.EMA(self.data, period=self.p.slow)
        self.rsi = btind.RSI(self.data)
        self.stop_order = None

    def next(self):
        fast_ema = self.ema.line[0]
        slow_ema = self.slow_ema.line[0]
        previous_fast_ema = self.ema.line[-1]
        previous_slow_ema = self.slow_ema.line[-1]
        rsi = self.rsi.line[0]
        previous_rsi = self.rsi.line[-1]

        bullish_cross = (
            previous_fast_ema <= previous_slow_ema
            and fast_ema > slow_ema
        )
        bearish_cross = (
            previous_fast_ema >= previous_slow_ema
            and fast_ema < slow_ema
        )
        rsi_confirmation = rsi > previous_rsi
        rsi_loss_of_momentum = rsi < previous_rsi

        if self.position:
            if bearish_cross or rsi_loss_of_momentum:
                self.close()
        elif bullish_cross and rsi_confirmation:
            self.buy()

        risk_management = getattr(self, "risk_management", None)
        stop_loss = getattr(risk_management, "stop_loss", None)

        if stop_loss is not None and self.position:
            if self.stop_order is None:
                price = (
                    self.data.close[0] - stop_loss
                    if self.position.size > 0
                    else self.data.close[0] + stop_loss
                )
                self.stop_order = self.sell(
                    exectype=bt.Order.Stop,
                    price=price,
                )

        if not self.position and self.stop_order is not None:
            self.stop_order = None
