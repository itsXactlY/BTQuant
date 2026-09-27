
import sys, importlib, importlib.util, os
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
import autonomous_agency.strategy_factory as sf
importlib.reload(sf)

f = sf.StrategyFactory.__new__(sf.StrategyFactory)
import logging; f.logger = logging.getLogger("probe")
try: f._llm_client = None
except Exception: pass

NAME = "Persistent Homology Microstructure Topology with Wasserstein Mean Reversion and Ergodic Entropic Allocation"
code = sf.StrategyFactory._generate_concept_driven_code(f, {"hypothesis_id": "probe", "strategy_name": NAME})
p = "/home/alca/projects/PubBTQuant/.daedalus/probe_vote_strategy.py"
open(p, "w").write(code)
print("generated", len(code), "bytes ->", p)
import ast; ast.parse(code); print("parses OK")
for ln in code.splitlines():
    s = ln.strip()
    if s.startswith("gate_") or "_votes" in s or "min_votes" in s or "if base_signal" in s:
        print("   ", s)
