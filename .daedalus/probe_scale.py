
import pandas as pd, numpy as np, glob, os
df = pd.read_parquet("/home/alca/projects/PubBTQuant/.btq_cache/BTC_1m_USDT_2ee8fdb96a31.parquet")
df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
r = (df.high-df.low)/df.open*100
p = np.percentile(r,[50,99,99.9])
print("BTC 1m intrabar range pct of open: p50={:.3f} p99={:.3f} p99.9={:.3f} max={:.2f}".format(p[0],p[1],p[2],r.max()))
print()
print("=== cached datasets: how often does a 2% intrabar range fire? ===")
for path in sorted(glob.glob("/home/alca/projects/PubBTQuant/.btq_cache/*.parquet")):
    try:
        d = pd.read_parquet(path).rename(columns={"Open":"open","High":"high","Low":"low","Close":"close"})
        if "open" not in d.columns: continue
        rr = (d.high-d.low)/d.open
        print("  {:<44s} n={:6d}  >2pct: {:5d} ({:.3f}%)  max={:.2f}%".format(
            os.path.basename(path)[:44], len(d), int((rr>0.02).sum()), 100*(rr>0.02).mean(), 100*rr.max()))
    except Exception as e:
        print("  {:<44s} ERR {}".format(os.path.basename(path)[:44], e))
