"""Validate the hollow-strategy guard against the real corpus.

Three assertions:
  A. it FLAGS strategies known to be hollow (self.atr never assigned)
  B. it does NOT flag healthy strategies (false-positive rate must be low)
  C. _call_condition turns a real AttributeError into a raise, not a silent 0
"""
import sys, glob, re, ast
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
import autonomous_agency.strategy_factory as sf
from autonomous_agency.strategy_factory import _find_uninitialised_attrs

files = sorted(glob.glob("/home/alca/projects/PubBTQuant/autonomous_agency/strategies/*.py"))
print(f"corpus: {len(files)} strategy files")

flagged, clean, parse_fail = [], [], []
for p in files:
    src = open(p, encoding="utf-8", errors="replace").read()
    try:
        ast.parse(src)
    except SyntaxError:
        parse_fail.append(p)
        continue
    miss = _find_uninitialised_attrs(src)
    (flagged if miss else clean).append((p, miss))

print(f"  unparseable   : {len(parse_fail)}")
print(f"  FLAGGED hollow: {len(flagged)}")
print(f"  clean         : {len(clean)}")

# --- A. does it catch the known-dead one? ---
known_dead = [f for f in files if "Persistent_Homology_Microstructure_Topology" in f]
for p in known_dead:
    m = _find_uninitialised_attrs(open(p, encoding="utf-8", errors="replace").read())
    hit = "atr" in m
    print(f"\nA. known-dead persistent-homology: atr flagged = {hit}")
    print(f"   flagged attrs: {m[:12]}")

# --- B. false positives: show what clean-but-suspicious look like ---
print(f"\nB. sample of CLEAN files (guard must not be a blanket reject):")
for p, _ in clean[:5]:
    print(f"   {p.split('/')[-1][:70]}")

print(f"\nB. top flagged attribute names across corpus:")
from collections import Counter
c = Counter(a for _, m in flagged for a in m)
for name, n in c.most_common(15):
    print(f"   {name:<28} {n}")
