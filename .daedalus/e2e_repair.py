"""E2E: force the repair branch — attempt 1 is rejected, the model must fix it.

Verifies the multi-message conversation survives the daedalus-CLI backend,
which is the part the unit-level gate test cannot cover.
"""
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(levelname)s %(name)s: %(message)s")

from autonomous_agency.ai_interface import StrategyHypothesis
from autonomous_agency.strategy_factory import StrategyFactory

h = StrategyHypothesis(
    id="e2e-repair-1",
    name="RSI Dip Reversion Repair Probe",
    description="Buy oversold RSI dips below 35, exit when RSI recovers above 60. percent_sizer 0.95.",
    indicators=["rsi"],
    entry_conditions=["RSI(14) crosses below 35"],
    exit_conditions=["RSI(14) crosses above 60"],
    parameters={"percent_sizer": 0.95, "rsi_period": 14},
    rationale="Mean reversion after short-term oversold extremes.",
    mathematical_beauty_score=0.7,
    expected_regime="ranging",
    risk_profile="moderate",
)

sf = StrategyFactory.__new__(StrategyFactory)
sf.logger = logging.getLogger("repair")
sf._last_validation_error = ""
sf.config = None

# Sabotage attempt 1 exactly the way the real bug showed up: pretend the
# harness ran the class and it died on a bogus indicator kwarg.
real_validate = sf._validate_and_format_code
calls = {"n": 0}


def sabotage(code, hypothesis):
    calls["n"] += 1
    if calls["n"] == 1:
        sf._last_validation_error = (
            "TypeError: RelativeStrengthIndex.__init__() got an unexpected "
            "keyword argument '_ma'"
        )
        sf.logger.warning("INJECTED failure on attempt 1")
        return None
    return real_validate(code, hypothesis)


sf._validate_and_format_code = sabotage

spec = sf._create_strategy_spec(h)
code = sf._generate_code_with_llm(spec, h)
print("=" * 60)
print("attempts:", calls["n"])
print("RESULT:", "repaired code returned" if code else "None (repair failed too)")
if code:
    print("--- generated ---")
    print(code)
    print("--- independent re-check:", repr(sf._smoke_run_generated_code(code)))
