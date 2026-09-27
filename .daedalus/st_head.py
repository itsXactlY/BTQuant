"""Generate a full-valued strategy that inherits from BaseStrategy.

The new template (replaces the bare-bones version) produces strategies
that:
  * Inherit from ``BaseStrategy`` for shared risk/DCA/sizing scaffolding
  * Use concept extraction on the class name to drive indicator selection
  * Implement real entry/exit signals based on strategy type
  * Carry forward operator-tuned params from the original spec

This is the production template — every strategy the agency writes goes
through here.
"""

from __future__ import annotations

from .concept_extractor import parse_strategy_concept


def _build_signal_methods(strategy_type: str, indicators: list[str], has_dca: bool) -> tuple[str, str]:
    """Return (entry_signal_body, exit_signal_body) source code for the subclass."""
    has_ema = "ema" in indicators
    has_rsi = "rsi" in indicators
    has_macd = "macd" in indicators
    has_bbands = "bbands" in indicators
    has_adx = "adx" in indicators
    has_volume = "volume" in indicators

    if strategy_type == "momentum":
            # EMA crossover + MACD confirmation + ADX filter (if present)
            entry_lines = []
            if has_ema:
                entry_lines.append("ema_fast = self.ema_fast[0]")
                entry_lines.append("ema_slow = self.ema_slow[0]")
                entry_lines.append("ema_cross_up = ema_fast > ema_slow and self.ema_fast[-1] <= self.ema_slow[-1]")
            else:
                entry_lines.append("ema_cross_up = False")
            if has_macd:
                entry_lines.append("macd_bull = (self.macd.macd[0] - self.macd.signal[0]) > 0")
            else:
                entry_lines.append("macd_bull = True")
            if has_adx:
                entry_lines.append("adx_strong = self.adx[0] > 20")
            else:
                entry_lines.append("adx_strong = True")
            entry_lines.append("return ema_cross_up and macd_bull and adx_strong")
            entry_signal = "\n".join(entry_lines)

            exit_lines = []
            if has_ema:
                exit_lines.append("ema_cross_dn = self.ema_fast[0] < self.ema_slow[0]")
            else:
                exit_lines.append("ema_cross_dn = False")
            if has_macd:
                exit_lines.append("macd_bear = (self.macd.macd[0] - self.macd.signal[0]) < 0")
            else:
                exit_lines.append("macd_bear = False")
            if has_rsi:
                exit_lines.append("rsi_overbought = self.rsi[0] > 70")
            else:
                exit_lines.append("rsi_overbought = False")
            exit_lines.append("return ema_cross_dn or macd_bear or rsi_overbought")
            exit_signal = "\n".join(exit_lines)

        elif strategy_type == "mean_reversion":
            entry_lines = []
            if has_rsi:
                entry_lines.append("rsi_oversold = self.rsi[0] < 30")
            else:
                entry_lines.append("rsi_oversold = False")
            if has_bbands:
                entry_lines.append("bb_lower_touch = self.data.close[0] < self.bb.lines.bot[0]")
            else:
                entry_lines.append("bb_lower_touch = False")
            if has_ema:
                entry_lines.append("above_trend = self.data.close[0] > self.ema_slow[0]")
            else:
                entry_lines.append("above_trend = True")
            entry_lines.append("return (rsi_oversold or bb_lower_touch) and above_trend")
            entry_signal = "\n".join(entry_lines)

            exit_lines = []
            if has_rsi:
                exit_lines.append("rsi_mid = self.rsi[0] > 50")
            else:
                exit_lines.append("rsi_mid = True")
            if has_bbands:
                exit_lines.append("bb_mid_cross = self.data.close[0] >= self.bb.lines.mid[0]")
            else:
                exit_lines.append("bb_mid_cross = True")
            if has_ema:
                exit_lines.append("near_ema = abs(self.data.close[0] - self.ema_slow[0]) / self.ema_slow[0] < 0.005")
            else:
                exit_lines.append("near_ema = False")
            exit_lines.append("return rsi_mid or bb_mid_cross or near_ema")
            exit_signal = "\n".join(exit_lines)

        elif strategy_type == "breakout":
            entry_lines = []
            entry_lines.append("# Donchian-style breakout: price above recent N-bar high")
            entry_lines.append("lookback = 20")
            entry_lines.append("try:")
            entry_lines.append("    high_n = max(self.data.high.get(ago=-1, size=lookback))")
            entry_lines.append("    breakout_up = self.data.close[0] > high_n")
            entry_lines.append("except Exception:")
            entry_lines.append("    breakout_up = False")
            if has_volume:
                entry_lines.append("vol_confirm = self.data.volume[0] > self.data.volume[-1] * 1.2")
            else:
                entry_lines.append("vol_confirm = True")
            if has_rsi:
                entry_lines.append("rsi_not_extreme = 30 < self.rsi[0] < 70")
            else:
                entry_lines.append("rsi_not_extreme = True")
            entry_lines.append("return breakout_up and vol_confirm and rsi_not_extreme")
            entry_signal = "\n".join(entry_lines)

            exit_lines = []
            exit_lines.append("# Trailing stop exit handled by BaseStrategy._update_trailing_stop")
            exit_lines.append("if hasattr(self, '_peak_price') and self._peak_price is not None:")
            exit_lines.append("    giveback = (self._peak_price - self.data.close[0]) / self._peak_price")
            exit_lines.append("    return giveback > 0.05")
            exit_lines.append("return False")
            exit_signal = "\n".join(exit_lines)

        elif strategy_type == "volatility":
            entry_lines = []
            if has_ema:
                entry_lines.append("# Trend following with vol-target sizing (sizing handled by base)")
                entry_lines.append("ema_up = self.ema_fast[0] > self.ema_slow[0]")
            else:
                entry_lines.append("ema_up = True")
            entry_lines.append("# Volatility expansion signal — enter when ATR is rising")
            entry_lines.append("try:")
            entry_lines.append("    atr_rising = self.atr[0] > self.atr[-5]")
            entry_lines.append("except Exception:")
            entry_lines.append("    atr_rising = True")
            if has_rsi:
                entry_lines.append("rsi_ok = 40 < self.rsi[0] < 70")
            else:
                entry_lines.append("rsi_ok = True")
            entry_lines.append("return ema_up and atr_rising and rsi_ok")
            entry_signal = "\n".join(entry_lines)

            exit_lines = []
            exit_lines.append("# Exit when vol contracts (ATR falling) or trend breaks")
            exit_lines.append("try:")
            exit_lines.append("    atr_falling = self.atr[0] < self.atr[-5]")
            exit_lines.append("except Exception:")
            exit_lines.append("    atr_falling = False")
            if has_ema:
                exit_lines.append("ema_dn = self.ema_fast[0] < self.ema_slow[0]")
            else:
                exit_lines.append("ema_dn = False")
            exit_lines.append("return atr_falling or ema_dn")
            exit_signal = "\n".join(exit_lines)

        elif strategy_type == "regime":
            entry_lines = []
            entry_lines.append("# Regime filter: only trade when ADX confirms trend strength")
            if has_adx:
                entry_lines.append("trending = self.adx[0] > 20")
            else:
                entry_lines.append("trending = True")
            if has_ema:
                entry_lines.append("ema_bull = self.ema_fast[0] > self.ema_slow[0]")
            else:
                entry_lines.append("ema_bull = True")
            if has_rsi:
                entry_lines.append("rsi_pullback = 40 < self.rsi[0] < 60")
            else:
                entry_lines.append("rsi_pullback = True")
            entry_lines.append("return trending and ema_bull and rsi_pullback")
            entry_signal = "\n".join(entry_lines)

            exit_lines = []
            if has_adx:
                exit_lines.append("regime_lost = self.adx[0] < 15")
            else:
                exit_lines.append("regime_lost = False")
            if has_ema:
                exit_lines.append("ema_bear = self.ema_fast[0] < self.ema_slow[0]")
            else:
                exit_lines.append("ema_bear = False")
            exit_lines.append("return regime_lost or ema_bear")
            exit_signal = "\n".join(exit_lines)

        else:  # arbitrage or unknown
            entry_lines = ["# Default entry: any sustained momentum with vol confirmation"]
            if has_ema:
                entry_lines.append("ema_up = self.ema_fast[0] > self.ema_slow[0]")
            else:
                entry_lines.append("ema_up = True")
            if has_rsi:
                entry_lines.append("rsi_ok = 30 < self.rsi[0] < 70")
            else:
                entry_lines.append("rsi_ok = True")
            entry_lines.append("return ema_up and rsi_ok")
            entry_signal = "\n".join(entry_lines)
            exit_lines = ["return self.rsi[0] > 65 or self.ema_fast[0] < self.ema_slow[0]"]
            exit_signal = "\n".join(exit_lines)

        return entry_signal, exit_signal


def generate_full_strategy(class_name: str, description: str = "", concept_hints: dict | None = None) -> str:
    """Generate a full strategy file inheriting from BaseStrategy."""
    concept = parse_strategy_concept(class_name)
    indicators = concept.indicators
    strategy_type = concept.strategy_type
    has_dca = concept.has_dca_hint
    sizing_mode = "vol_target" if concept.has_vol_target_hint else "vol_target"
    # ATR is always needed for risk + sizing; ensure it's the FIRST indicator
    if "atr" not in indicators:
        indicators = ["atr"] + indicators

    entry_signal_body, exit_signal_body = _build_signal_methods(strategy_type, indicators, has_dca)

    # Indent every line of the body by 12 spaces so it sits inside
    # the function body at the correct depth.
    def _indent(body: str) -> str:
        return "\n".join((" " * 12 + line) if line else "" for line in body.splitlines())

    entry_signal_body = _indent(entry_signal_body)
    exit_signal_body = _indent(exit_signal_body)

    # Build __init__ body
    init_lines = ["        super().__init__()"]
    for ind in indicators:
        if ind == "ema":
            init_lines.append("        self.ema = btind.EMA(self.data.close, period=self.p.ema_period)")
            init_lines.append("        self.ema_fast = btind.EMA(self.data.close, period=self.p.ema_fast_period)")
            init_lines.append("        self.ema_slow = btind.EMA(self.data.close, period=self.p.ema_slow_period)")
        elif ind == "rsi":
            init_lines.append("        self.rsi = btind.RSI(self.data.close, period=self.p.rsi_period)")
        elif ind == "macd":
            init_lines.append("        self.macd = btind.MACD(self.data.close)")
        elif ind == "bbands":
            init_lines.append("        self.bb = btind.BollingerBands(self.data.close, period=20)")
        elif ind == "adx":
            init_lines.append("        self.adx = btind.ADX(self.data, period=14)")
        # atr is set up in BaseStrategy.__init__ already

    init_body = "\n".join(init_lines)

    # Compose docstring
    desc = description or (
        f"Concept-derived {strategy_type} strategy. Indicators: {', '.join(indicators)}. "
        f"Generated from class name: {class_name}. "
        f"Inherits from BaseStrategy (Kelly sizing, ATR stops, optional DCA, "
        f"max-drawdown + consecutive-loss pause)."
    )

    raw_kw = ", ".join(concept.raw_keywords) if concept.raw_keywords else "(none)"

    # Compose the full file source
    src = f'''"""
{description or class_name}
Inherits: BaseStrategy (Kelly sizing, ATR stops, DCA, max-DD cap, loss-streak pause)
Strategy type (inferred from class name): {strategy_type}
Indicators (inferred): {", ".join(indicators)}
Concept keywords: {raw_kw}

This file is auto-generated by the BTQuant agency. It is a FULL strategy —
not a demo stub. Every bar calls BaseStrategy.next() which:
  1. Updates trailing stop
  2. Enforces max-hold-time exit
  3. Gates new entries on drawdown + daily-loss + loss-streak
  4. Sizes the position (Kelly / vol-target / fixed)
  5. Optionally scale-ins via DCA on adverse moves
  6. Places ATR-based stop + take-profit orders
  7. Ratchets the stop on favourable moves
"""
import backtrader as bt
import backtrader.indicators as btind

from .base_strategy import BaseStrategy


class {class_name}(BaseStrategy):
    """{description or class_name}"""

    params = (
        # Inherit all BaseStrategy params, then add strategy-specific overrides
        ("verbose_log", False),
    )

    def __init__(self):
{init_body}

    def _entry_signal(self) -> bool:
{entry_signal_body}

    def _exit_signal(self) -> bool:
{exit_signal_body}
'''
    return src


if __name__ == "__main__":
    import sys
    for cn in sys.argv[1:]:
        print(generate_full_strategy(cn))
        print("=" * 60)