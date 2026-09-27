
import logging, sys
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
from autonomous_agency.strategy_factory import StrategyFactory

f = StrategyFactory()
NAMES = [
    "EMA Crossover Momentum with RSI Confirmation",
    "RSI Momentum Trend Continuation",
    "Dual Moving Average Trend Follower with Volatility Filter",
    "Bollinger Band Mean Reversion on Volatility Compression",
    "ATR Channel Breakout with Volume Surge",
    "MACD Momentum Reversal Strategy",
]
import hashlib
for n in NAMES:
    spec = {"strategy_name": n, "hypothesis_id": "probe-"+n[:8],
            "description": "probe", "indicators": ["ema","rsi"]}
    code = f._generate_concept_driven_code(spec)
    body = code.split("class ",1)[1]
    # normalize away class name / header for the fingerprint
    import re
    norm = re.sub(r"\s+", " ", body.split('"""',2)[-1])
    h = hashlib.sha256(norm.encode()).hexdigest()[:10]
    gates = re.findall(r"gate_(\w+?)_\d+ = ", code)
    print(f"{n[:44]:44s} fp={h} gates={gates} min_votes={'min_votes' in code}")
    compile(code, n, "exec")
