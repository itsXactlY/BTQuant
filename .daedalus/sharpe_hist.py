
import glob, json
vals=[]
for j in glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/results/*_backtest_results.json"):
    try:
        d=json.load(open(j))
        vals.append((d.get("sharpe_ratio"), d.get("num_trades",0)))
    except Exception: pass
nz=[v for v in vals if v[0] not in (0,0.0,None)]
print("total parseable:", len(vals), " sharpe != 0:", len(nz))
print("samples:", nz[:8])
