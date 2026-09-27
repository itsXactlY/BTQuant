
import pandas as pd, numpy as np
df = pd.read_parquet("/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet")
c = df["Close"].astype(float)
e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
d = e12 - e26
cross_up = (d > 0) & (d.shift(1) <= 0)
delta = c.diff()
gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
rsi = 100 - 100/(1 + gain/loss)
n = int(cross_up.sum())
print("bars:", len(c), "ema(12,26) cross_up events:", n)
sub = rsi[cross_up].dropna()
print("rsi at cross bars: n=%d  frac in (40,70): %.4f" % (len(sub), ((sub>40)&(sub<70)).mean()))
print("rsi min/max at cross bars: %.1f / %.1f" % (sub.min(), sub.max()))
print("rsi overall frac in (40,70): %.4f" % (rsi.gt(40).mul(rsi.lt(70)).mean()))
hl = (df["High"]-df["Low"])/df["Open"]
print("concept gate (high-low)/open > 0.02 fires: %.6f of bars" % (hl>0.02).mean())
