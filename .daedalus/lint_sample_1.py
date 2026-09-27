import backtrader as bt
import backtrader.indicators as btind


class ATR_Channel_Breakout_with_Volume_Surge_20260927_054943(bt.Strategy):
    description = "Volatility expansion with participation"

    params = (
        ("fast", 10),
        ("slow", 30),
    )

    risk_management = {
        "profile": "moderate",
        "stop_loss": None,
    }

    def __init__(self):
        self.atr = btind.ATR(self.data)
        self.fast_sma = btind.SMA(self.data, period=self.p.fast)
        self.slow_sma = btind.SMA(self.data, period=self.p.slow)
        super().__init__()

    def next(self):
        fast = self.p.fast
        slow = self.p.slow

        upper_channel = self.slow_sma + (self.atr * fast)
        lower_channel = self.fast_sma - (self.atr * slow)
        average_volume = self.data.volume.mean(fast)

        entry_long = (
            self.data.close[0] > upper_channel[0]
            and self.data.volume[0] > average_volume[0]
        )
        entry_short = (
            self.data.close[0] < lower_channel[0]
            and self.data.volume[0] > average_volume[0]
        )

        exit_long = self.data.close[0] < self.fast_sma[0]
        exit_short = self.data.close[0] > self.slow_sma[0]

        if not self.position:
            if entry_long:
                if self.risk_management.get("stop_loss") is not None:
                    self.buy(stoploss=self.risk_management["stop_loss"])
                else:
                    self.buy()
            elif entry_short:
                if self.risk_management.get("stop_loss") is not None:
                    self.sell(stoploss=self.risk_management["stop_loss"])
                else:
                    self.sell()

        elif self.position:
            if self.position.size > 0 and exit_long:
                self.close()
            elif self.position.size < 0 and exit_short:
                self.close()
