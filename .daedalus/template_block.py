        params_dict = dict(params)
        params_dict.setdefault("stop_atr_mult", 2.0)
        params_dict.setdefault("rsi_period", params.get("rsi_period", 14))
        params_dict.setdefault("ema_period", params.get("ema_period", 21))
        params_dict.setdefault("sma_period", params.get("sma_period", 20))
        params_dict.setdefault("atr_period", params.get("atr_period", 14))
        params_dict.setdefault("ema_fast_period", params.get("ema_fast", 12))
        params_dict.setdefault("ema_slow_period", params.get("ema_slow", 26))
        # BaseStrategy sizes every order from percent_sizer. Leave it at 0 and
        # create_order sends size 0 — a strategy that never trades.
        params_dict.setdefault("percent_sizer", 0.95)
        param_lines = []
        for k, v in params_dict.items():
            if isinstance(v, (int, float)):
                param_lines.append(f'        ("{k}", {v}),')
            else:
                param_lines.append(f'        ("{k}", {repr(v)}),')
        params_tuple = "\n".join(param_lines) if param_lines else "        # no params"

        # ---- indicators ----
        indicator_lines = ["        super().__init__(**kwargs)"]
        indicator_set = {ind.lower().replace("-", "").replace(" ", "") for ind in indicators}
        if "rsi" in indicator_set:
            indicator_lines.append(
                "        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)"
            )
        if "macd" in indicator_set:
            indicator_lines.append(
                "        self.macd = bt.ind.MACD(self.data.close)"
            )
        if "atr" in indicator_set:
            indicator_lines.append(
                "        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)"
            )
        if "ema" in indicator_set:
            indicator_lines.append(
                "        self.ema = bt.ind.EMA(self.data.close, period=self.p.ema_period)"
            )
            # Always also expose fast/slow EMAs — most entry/exit conditions
            # reference ema_fast > ema_slow or similar crossovers.
            indicator_lines.append(
                "        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast_period)"
            )
            indicator_lines.append(
                "        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow_period)"
            )
        if "sma" in indicator_set:
            indicator_lines.append(
                "        self.sma = bt.ind.SMA(self.data.close, period=self.p.sma_period)"
            )
            indicator_lines.append(
                "        self.sma_fast = bt.ind.SMA(self.data.close, period=self.p.ema_fast_period)"
            )
            indicator_lines.append(
                "        self.sma_slow = bt.ind.SMA(self.data.close, period=self.p.ema_slow_period)"
            )
        if "wma" in indicator_set:
            indicator_lines.append(
                "        self.wma = bt.ind.WMA(self.data.close, period=self.p.ema_period)"
            )
        if "bollinger" in indicator_set or "bbands" in indicator_set:
            indicator_lines.append(
                "        self.bb = bt.ind.BollingerBands(self.data.close, period=20)"
            )
        if "stochastic" in indicator_set:
            indicator_lines.append(
                "        self.stoch = bt.ind.Stochastic(self.data)"
            )
        if "cci" in indicator_set:
            indicator_lines.append(
                "        self.cci = bt.ind.CCI(self.data)"
            )
        if "adx" in indicator_set:
            indicator_lines.append(
                "        self.adx = bt.ind.ADX(self.data)"
            )

        # Fallback: no recognised indicators at all → use ATR + EMA so the
        # strategy has SOMETHING to compute.
        if indicator_lines == ["        super().__init__(**kwargs)"]:
            indicator_lines.extend([
                "        self.atr = bt.ind.ATR(self.data, period=self.p.atr_period)",
                "        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast_period)",
                "        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow_period)",
                "        self.rsi = bt.ind.RSI(self.data.close, period=self.p.rsi_period)",
            ])
            indicator_set.update(["atr", "ema", "rsi"])

        # ---- compile conditions ----
        compiled_entry = [self._compile_condition(c) for c in entry] or ["True"]
        compiled_exit = [self._compile_condition(c) for c in exit_] or ["False"]

        # Join with `and` so all conditions must hold (or with `or` if
        # the hypothesis uses commas — but the spec already splits them
        # into a list).
        entry_expr = " and ".join(f"({c})" for c in compiled_entry)
        exit_expr = " or ".join(f"({c})" for c in compiled_exit)

        # ---- generate the condition methods ----
        # BaseStrategy.next() does the sizing and then calls one of these two.
        # The base enforces NO exit of its own, so the stop lives here.
        body = f'''    def buy_or_short_condition(self):
        if self.buy_executed:
            return False
        if {entry_expr}:
            self.create_order(action="BUY")
            return True
        return False

    def sell_or_cover_condition(self):
        if not self.active_orders:
            return False
        fired = False
        if {exit_expr}:
            fired = True
        if (not fired and self.p.stop_atr_mult > 0
                and hasattr(self, "atr") and self.first_entry_price):
            stop_price = (self.first_entry_price
                          - self.atr[0] * self.p.stop_atr_mult)
            if self.data.close[0] <= stop_price:
                fired = True
        if fired:
            for tracker in list(self.active_orders):
                self.close_order(tracker)
            return True
        return False'''

        return f'''"""Auto-generated strategy: {cls_name}
Source hypothesis: {spec.get("hypothesis_id", "?")}
Description: {spec.get("description", "")}
Indicators: {indicators}
Entry conditions: {entry}
Exit conditions: {exit_}
"""
from backtrader.strategies.base import BaseStrategy, bt


class {cls_name}(BaseStrategy):
    """{spec.get("description", "")[:200]}"""

    params = (
{params_tuple}
    )

    def __init__(self, **kwargs):
{chr(10).join(indicator_lines)}

{body}
'''
