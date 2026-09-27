
import sys
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.concept_extractor import parse_strategy_concept
NAMES = [
 "EMA Crossover Momentum with RSI Confirmation",
 "Bollinger Band Mean Reversion on Volatility Compression",
 "RSI Momentum Trend Continuation",
 "ATR Channel Breakout with Volume Surge",
 "MACD Momentum Reversal Strategy",
 "Dual Moving Average Trend Follower with Volatility Filter",
 "Kalman Filter Mean Reversion with Hawkes Regime Switch",
 "Wavelet Decomposition Trend Follower",
]
for n in NAMES:
    c = parse_strategy_concept(n)
    print(f"{n[:46]:46s} type={c.strategy_type:14s} inds={','.join(c.indicators):18s} logic={','.join(c.logic_keywords)}")
