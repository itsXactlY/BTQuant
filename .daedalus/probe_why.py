
import sys
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
from autonomous_agency.strategy_factory import _find_uninitialised_attrs, _BASE_SAFE_ATTRS
p = "/home/alca/projects/PubBTQuant/autonomous_agency/strategies/Active_Inference_Free_Energy_Momentum_with_Hysteretic_Precision_Gating_and_Natural_Gradient_Wasserstein_Kelly_AIFEM_HPG_NGWK_20260623_085319.py"
src = open(p, encoding="utf-8", errors="replace").read()
print("guard says missing:", _find_uninitialised_attrs(src))
print("'atr' in _BASE_SAFE_ATTRS:", "atr" in _BASE_SAFE_ATTRS)
import re
print("assigned-regex hits for atr:", re.findall(r"self\.(\w+)\s*(?:\[[^\]]*\]\s*)?(?:=[^=]|\+=)", src)[:40])
