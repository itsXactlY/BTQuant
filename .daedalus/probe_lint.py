"""Capture REAL LLM codegen output and compare the current lint vs an AST lint.

Writes each raw sample to .daedalus/lint_sample_<i>.py so the rejection reason
can be inspected without re-calling the model.
"""
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")
logging.basicConfig(level=logging.WARNING)

from autonomous_agency.strategy_factory import StrategyFactory  # noqa: E402
from autonomous_agency.ai_interface import StrategyHypothesis  # noqa: E402

HERE = Path("/home/alca/projects/PubBTQuant/.daedalus")

HYPOTHESES = [
    ("EMA Crossover Momentum with RSI Confirmation",
     "Fast/slow EMA cross confirmed by RSI regime", ["ema", "rsi"]),
    ("ATR Channel Breakout with Volume Surge",
     "Volatility expansion with participation", ["atr", "sma"]),
    ("RSI Momentum Trend Continuation",
     "Buy strength confirmed by trend filter", ["rsi", "sma"]),
]

# The exact regex the shipped lint uses to slice out next().
RX = re.compile(r"def\s+next\s*\(\s*self[^)]*\)\s*:\s*(?:\n[ \t]+[^\n]*)+\n")


def current_lint_reason(code: str) -> str:
    m = RX.search(code)
    if not m:
        return "NO MATCH"
    src = m.group(0)
    nc = [l.strip() for l in src.splitlines()[1:]
          if l.strip() and not l.strip().startswith("#")]
    sub = [l for l in nc if l != "return"]
    if len(sub) < 4:
        return f"TOO THIN substantive={len(sub)}"
    if not any(c in src for c in ("self.buy(", "self.sell(", "self.close(")):
        return "NO ORDER CALL"
    return "ok"


for i, (name, desc, inds) in enumerate(HYPOTHESES):
    h = StrategyHypothesis(
        id=f"probe-lint-{i}", name=name, description=desc, indicators=inds,
        entry_conditions=["c1"], exit_conditions=["c2"],
        parameters={"fast": 10, "slow": 30}, rationale="probe",
        mathematical_beauty_score=0.5, expected_regime="trending",
        risk_profile="moderate",
    )
    f = StrategyFactory()
    spec = f._create_strategy_spec(h)
    from autonomous_agency.llm_adapter import get_default_client
    c = get_default_client()
    try:
        text = c.chat(
            [{"role": "system", "content":
              "You are a code printer. You output Python code only. "
              "DO NOT THINK. DO NOT EXPLAIN. Your response must start with "
              "```python on the very first character."},
             {"role": "user", "content": f._build_code_generation_prompt(spec)}],
            temperature=0.2, max_tokens=8000, timeout=240,
        )
    except Exception as e:
        print(f"[{i}] LLM CALL FAILED: {e}")
        continue
    code = f._extract_python_block(text)
    if not code:
        print(f"[{i}] no python block, len(text)={len(text or '')}")
        continue
    p = HERE / f"lint_sample_{i}.py"
    p.write_text(code)
    ok, err = f._lint_generated_code(code, spec)
    print(f"[{i}] {name[:38]:38s} chars={len(code):5d} "
          f"shipped_lint={'OK' if ok else 'REJECT:' + err}  "
          f"regex_only={current_lint_reason(code)}  "
          f"blank_lines={code.count(chr(10) + chr(10))}  "
          f"imports_btind={'btind' in code.split('class ')[0]}  "
          f"-> {p}")
