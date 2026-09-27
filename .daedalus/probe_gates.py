
import pandas as pd, numpy as np
df = pd.read_parquet("/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet")
df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
c,h,l,o = df.close, df.high, df.low, df.open

# replicate backtrader indicators
delta = c.diff()
gain = delta.clip(lower=0); loss = (-delta.clip(upper=0))
rsi = 100 - 100/(1 + gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
                      / loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean())
pc = c.shift(1)
tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
ema_slow = c.ewm(span=26, adjust=False).mean()
sma20 = c.rolling(20).mean()
hom_range = c.rolling(10).max() - c.rolling(10).min()

g = {}
g["base(rsi<35 & >0.95*ema26)"] = (rsi<35) & (c > ema_slow*0.95)
g["gate_homology(range10>3atr)"] = hom_range > atr*3
g["gate_microstructure(range/open>2%)"] = (h-l)/o > 0.02
g["gate_wasserstein(|c-sma20|>1.5atr)"] = (c-sma20).abs() > atr*1.5
g["gate_reversion(rsi<30)"] = rsi < 30
all4 = g["base(rsi<35 & >0.95*ema26)"] & g["gate_homology(range10>3atr)"] & \
      g["gate_microstructure(range/open>2%)"] & g["gate_wasserstein(|c-sma20|>1.5atr)"] & g["gate_reversion(rsi<30)"]
n = len(df)
print(f"bars = {n}")
for k,v in g.items():
    print(f"  {k:38s} fires {int(v.sum()):6d} times ({100*v.sum()/n:.4f}%)")
print(f"  {'ALL FIVE ANDed (as generated)':38s} fires {int(all4.sum()):6d} times ({100*all4.sum()/n:.4f}%)")
print()
print(" 1m bar range as % of open: p50=%.4f%% p99=%.4f%% p99.9=%.4f%% max=%.3f%%" % (
    tuple(100*np.percentile((h-l)/o*100,[50,99,99.9]))+(100*((h-l)/o).max(),)))
