"""Verify the AST lint accepts real LLM output and still rejects stubs."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.strategy_factory import StrategyFactory  # noqa: E402

f = StrategyFactory()
HERE = Path("/home/alca/projects/PubBTQuant/.daedalus")

print("=== real LLM samples (rejected by the old regex lint) ===")
for i in range(3):
    code = (HERE / f"lint_sample_{i}.py").read_text()
    ok, err = f._lint_generated_code(code, {"strategy_name": f"Sample_{i}"})
    print(f"  sample_{i}: ok={ok} err={err}")

print("=== the file that died with NameError: btind ===")
broken = (Path("/home/alca/projects/PubBTQuant/autonomous_agency/strategies") /
          "ATR_Channel_Breakout_with_Volume_Surge_20260927_054510.py").read_text()
print("  ", f._lint_generated_code(broken, {"strategy_name": "ATR_X"}))

print("=== heredoc-wrapped LLM reply ===")
from autonomous_agency.strategy_factory import _strip_heredoc_markers  # noqa: E402
heredoc = ("cat > strategy.py <<'EOF'\n"
           "import backtrader as bt\n"
           "import backtrader.indicators as btind\n"
           "class S(bt.Strategy):\n"
           "    def __init__(self):\n        super().__init__()\n"
           "        self.rsi = btind.RSI(self.data, period=14)\n"
           "    def next(self):\n        if self.rsi[0] < 30:\n"
           "            self.buy()\n"
           "EOF\n")
print("  ", f._lint_generated_code(_strip_heredoc_markers(heredoc), {"strategy_name": "S"}))

print("=== stubs must stay rejected ===")
CASES = {
    "pass": "import backtrader as bt\nclass S(bt.Strategy):\n    def next(self):\n        pass\n",
    "bare return": "import backtrader as bt\nclass S(bt.Strategy):\n    def next(self):\n        return\n",
    "docstring only": 'import backtrader as bt\nclass S(bt.Strategy):\n    def next(self):\n        """nothing"""\n',
    "no next()": "import backtrader as bt\nclass S(bt.Strategy):\n    def start(self):\n        self.buy()\n",
    "missing btind": "import backtrader as bt\nclass S(bt.Strategy):\n    def next(self):\n        a = btind.SMA(self.data)\n        self.buy()\n",
    "blank-line real": ("import backtrader as bt\nimport backtrader.indicators as btind\n"
                        "class S(bt.Strategy):\n    def next(self):\n"
                        "        a = self.sma[0]\n\n        if a > 1:\n            self.buy()\n"
                        "\n        else:\n            self.close()\n"),
    "helper method real": ("import backtrader as bt\nimport backtrader.indicators as btind\n"
                           "class S(bt.Strategy):\n    def _fire(self, n):\n"
                           "        if n > 0:\n            self.buy()\n\n        else:\n            self.close()\n"
                           "    def next(self):\n        self._fire(self.sma[0])\n"),
    "HULL buy every bar": ("import backtrader as bt\nclass S(bt.Strategy):\n"
                           "    def next(self):\n        x = self.data.close[0]\n"
                           "        y = x * 2\n        self.buy()\n"),
    "HULL long but unguarded": ("import backtrader as bt\nclass S(bt.Strategy):\n"
                                "    def next(self):\n        a = self.data.close[0]\n"
                                "        b = a + 1\n        c = b * 3\n        d = c - a\n"
                                "        e = d / 2\n        self.buy()\n"),
    "HULL indicator in next": ("import backtrader as bt\n"
                               "import backtrader.indicators as btind\n"
                               "class S(bt.Strategy):\n    def next(self):\n"
                               "        s = btind.SMA(self.data, period=10)\n"
                               "        if s[0] > 1:\n            self.buy()\n"),
    "bogus indicator name": ("import backtrader as bt\n"
                             "import backtrader.indicators as btind\n"
                             "class S(bt.Strategy):\n    def __init__(self):\n"
                             "        super().__init__()\n"
                             "        self.rsi = btind.RSIndicator(self.data)\n"
                             "    def next(self):\n        if self.rsi[0] > 50:\n"
                             "            self.buy()\n"),
    "legit strategy ok": ("import backtrader as bt\n"
                         "import backtrader.indicators as btind\n"
                         "class S(bt.Strategy):\n    def __init__(self):\n"
                         "        super().__init__()\n"
                         "        self.rsi = btind.RSI(self.data, period=14)\n"
                         "        self.ema = btind.EMA(self.data, period=20)\n"
                         "    def next(self):\n        if not self.position:\n"
                         "            if self.rsi[0] > 55 and self.ema[0] > self.data.close[0]:\n"
                         "                self.buy()\n        elif self.rsi[0] < 45:\n"
                         "            self.close()\n"),
    "HULL bb.bottom": ("import backtrader as bt\n"
                       "import backtrader.indicators as btind\n"
                       "class S(bt.Strategy):\n    def __init__(self):\n"
                       "        super().__init__()\n"
                       "        self.bb = btind.BollingerBands(self.data, period=20)\n"
                       "    def next(self):\n        if self.bb.bottom[0] < 0:\n"
                       "            self.buy()\n"),
    "HULL data.vol": ("import backtrader as bt\n"
                      "import backtrader.indicators as btind\n"
                      "class S(bt.Strategy):\n    def __init__(self):\n"
                      "        super().__init__()\n"
                      "        self.sma = btind.SMA(self.data.volume, period=20)\n"
                      "    def next(self):\n        if self.data.vol[0] > self.sma[0]:\n"
                      "            self.buy()\n"),
    "legit bb.bb.bot": ("import backtrader as bt\n"
                        "import backtrader.indicators as btind\n"
                        "class S(bt.Strategy):\n    def __init__(self):\n"
                        "        super().__init__()\n"
                        "        self.bb = btind.BollingerBands(self.data, period=20)\n"
                        "    def next(self):\n        if self.bb.bot[0] < 0:\n"
                        "            self.buy()\n"),
    "legit macd.hist": ("import backtrader as bt\n"
                        "import backtrader.indicators as btind\n"
                        "class S(bt.Strategy):\n    def __init__(self):\n"
                        "        super().__init__()\n"
                        "        self.macd = btind.MACD(self.data)\n"
                        "    def next(self):\n        if self.macd.hist[0] > 0:\n"
                        "            self.buy()\n"),
}
for name, code in CASES.items():
    ok, err = f._lint_generated_code(code, {"strategy_name": "S"})
    print(f"  {name:20s} ok={ok} {err}")
