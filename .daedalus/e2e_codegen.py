"""E2E: one real LLM codegen through the smoke gate + repair loop."""
import logging, sys
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(levelname)s %(name)s: %(message)s")

from autonomous_agency.ai_interface import StrategyHypothesis
from autonomous_agency.strategy_factory import StrategyFactory

h = StrategyHypothesis(
    id="e2e-probe-1",
    name="RSI Dip Reversion Probe",
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
sf.logger = logging.getLogger("e2e")
sf._last_validation_error = ""
sf.config = None

spec = sf._create_strategy_spec(h)
code = sf._generate_code_with_llm(spec, h)
print("=" * 60)
print("RESULT:", "code returned" if code else "None (fell through to concept path)")
if code:
    print("--- generated ---")
    print(code)
    err = sf._smoke_run_generated_code(code)
    print("--- independent re-check:", repr(err))
