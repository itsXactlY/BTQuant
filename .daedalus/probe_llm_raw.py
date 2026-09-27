
import json, sys, logging
sys.path.insert(0, "/home/alca/projects/PubBTQuant")
logging.basicConfig(level=logging.WARNING)
from autonomous_agency.strategy_factory import StrategyFactory
from autonomous_agency.ai_interface import StrategyHypothesis
from autonomous_agency.llm_adapter import get_default_client

h = StrategyHypothesis(id="probe-llm", name="EMA Crossover Momentum with RSI Confirmation",
    description="Fast/slow EMA cross confirmed by RSI regime", indicators=["ema","rsi"],
    entry_conditions=["c1"], exit_conditions=["c2"], parameters={"fast":10,"slow":30},
    rationale="probe", mathematical_beauty_score=0.5, expected_regime="trending", risk_profile="moderate")
f = StrategyFactory()
spec = f._create_strategy_spec(h)
sysp = ("You are a code printer. You output Python code only. DO NOT THINK. "
        "Your response must start with ```python on the very first character.")
c = get_default_client()
payload = c._build_payload([{"role":"system","content":sysp},
                           {"role":"user","content":f._build_code_generation_prompt(spec)}],
                          None, 0.2, 8000)
data = c._post(payload, timeout=240)
ch = (data.get("choices") or [{}])[0]
msg = ch.get("message", {})
print("finish_reason:", ch.get("finish_reason"))
print("usage:", json.dumps(data.get("usage"), indent=None)[:300])
print("message keys:", sorted(msg.keys()))
for k in sorted(msg.keys()):
    v = msg[k]
    print(f"  {k}: type={type(v).__name__} len={len(str(v))} head={str(v)[:120]!r}")
print("top-level keys:", sorted(data.keys()))
