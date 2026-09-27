from backtrader.strategies.base import BaseStrategy, bt


class Continuous_Variance_Confirmed_Trend_CVC_BPV_20260927_081756(BaseStrategy):
    """Jump vs continuous variation filter on an ATR-normalised short trend.

    Continuous share = |cumulative displacement| / summed hourly range over the
    lookback window. High share => the move is drift driven, low share => jump
    driven, so we stand aside. Only plain line arithmetic (operators, never line
    methods) is built in __init__; shifted reads happen in Python helpers.
    """

    params = (
        ('percent_sizer', 0.95),
        ('lookback_hours', 48),
        ('cv_threshold', 0.7),
        ('drift_span', 24),
        ('atr_period', 14),
        ('low_share_exit', 0.4),
        ('atr_z_max', 2.0),
        ('atr_z_window', 48),
        ('atr_stop_mult', 2.0),
        ('risk_per_trade', 0.005),
        ('max_drawdown', 0.15),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.atr = bt.ind.ATR(period=self.p.atr_period)
        self.drift = bt.ind.EMA(self.data.close, period=self.p.drift_span)
        self.total_variation = bt.ind.SumN(
            self.data.high - self.data.low, period=self.p.lookback_hours
        )
        self.atr_mean = bt.ind.SMA(self.atr, period=self.p.atr_z_window)
        self.atr_sq_mean = bt.ind.SMA(
            self.atr * self.atr, period=self.p.atr_z_window
        )

        self.in_position = False
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.peak_price = 0.0
        self.equity_peak = 0.0
        self.halted = False
        self.last_sign = 0
        self._warm = self.p.lookback_hours + self.p.drift_span + 2

    # ------------------------------------------------------------- helpers
    def _v(self, line, ago=0):
        """Last known float of a line, safe against warm-up, nan and inf."""
        try:
            val = float(line[0] if ago == 0 else line[-ago])
        except (TypeError, ValueError, IndexError):
            return 0.0
        if val != val or val in (float('inf'), float('-inf')):
            return 0.0
        return val

    def _ready(self):
        try:
            return len(self.data) > self._warm
        except TypeError:
            return False

    def _continuous_share(self):
        total = self._v(self.total_variation)
        if total <= 0.0:
            return 0.0
        net = self._v(self.data.close) - self._v(
            self.data.close, self.p.lookback_hours
        )
        if net < 0.0:
            net = -net
        share = net / total
        if share < 0.0:
            return 0.0
        if share > 1.0:
            return 1.0
        return share

    def _atr_z(self):
        atr = self._v(self.atr)
        mean = self._v(self.atr_mean)
        var = self._v(self.atr_sq_mean) - mean * mean
        if var <= 0.0:
            return 0.0
        sigma = var ** 0.5
        if sigma <= 0.0:
            return 0.0
        return (atr - mean) / sigma

    def _drift_sign(self):
        change = self._v(self.drift) - self._v(self.drift, 1)
        if change > 0.0:
            return 1
        if change < 0.0:
            return -1
        return 0

    def _trail_stop_hit(self):
        if self.stop_price <= 0.0:
            return False
        return self._v(self.data.close) <= self.stop_price

    def _drawdown_halted(self):
        try:
            value = float(self.getbrokervalue())
        except (TypeError, ValueError, IndexError):
            return False
        if self.equity_peak <= 0.0:
            self.equity_peak = value
            return False
        if value > self.equity_peak:
            self.equity_peak = value
        return (self.equity_peak - value) / self.equity_peak >= self.p.max_drawdown

    def _inverse_vol_size(self):
        """Portfolio fraction affordable with risk_per_trade against a 2xATR stop."""
        atr = self._v(self.atr)
        if atr <= 0.0:
            return 0.0
        try:
            value = float(self.getbrokervalue())
        except (TypeError, ValueError, IndexError):
            return 0.0
        if value <= 0.0:
            return 0.0
        risk_budget = self.p.risk_per_trade * value
        risk_per_unit = self.p.atr_stop_mult * atr
        fraction = (risk_budget / risk_per_unit) / value
        if fraction <= 0.0:
            return 0.0
        if fraction > 1.0:
            return 1.0
        return fraction

    # ---------------------------------------------------------- conditions
    def buy_or_short_condition(self):
        if not self._ready():
            return False
        sign = self._drift_sign()
        prev = self.last_sign
        self.last_sign = sign
        if self.halted or self.in_position:
            return False
        if sign <= 0 or prev > 0:
            return False
        if self._continuous_share() <= self.p.cv_threshold:
            return False
        if self._atr_z() > self.p.atr_z_max:
            return False
        if self._inverse_vol_size() <= 0.0:
            return False
        if self._drawdown_halted():
            self.halted = True
            return False
        price = self._v(self.data.close)
        if price <= 0.0:
            return False
        self.entry_price = price
        self.peak_price = price
        self.stop_price = price - self.p.atr_stop_mult * self._v(self.atr)
        self.create_order(action='BUY')
        self.in_position = True
        return True

    def sell_or_cover_condition(self):
        sign = self._drift_sign()
        self.last_sign = sign
        if not self.in_position:
            return False
        price = self._v(self.data.close)
        if self._drawdown_halted():
            self.halted = True
            return self._close()
        if price > self.peak_price:
            self.peak_price = price
            cand = price - self.p.atr_stop_mult * self._v(self.atr)
            if cand > self.stop_price:
                self.stop_price = cand
        if self._trail_stop_hit():
            return self._close()
        if self._continuous_share() < self.p.low_share_exit:
            return self._close()
        if sign < 0:
            return self._close()
        return False

    def _close(self):
        self.create_order(action='SELL')
        self.in_position = False
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.peak_price = 0.0
        return True
