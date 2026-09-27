
import sys, pathlib
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.strategy_factory import _find_uninitialised_attrs as f
d = pathlib.Path("/home/alca/projects/PubBTQuant/autonomous_agency/strategies")
files = sorted(d.glob("*.py"))
tot=len(files); hol=0; names={}
for fp in files:
    m = f(fp.read_text(errors="ignore"))
    if m:
        hol+=1
        for n in m: names[n]=names.get(n,0)+1
print(f"files={tot} flagged_hollow={hol} ({100*hol/max(tot,1):.1f}%)")
print("top missing attrs:", sorted(names.items(), key=lambda x:-x[1])[:15])
