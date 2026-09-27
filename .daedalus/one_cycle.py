"""One real evolution cycle with the daedalus-CLI LLM backend.

Not `--mode full` (that loops forever): init the components, force exactly
one cycle, print what came out.
"""
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from autonomous_agency.config import load_config_from_env
from autonomous_agency.orchestrator import PerpetualOrchestrator


async def main():
    config = load_config_from_env()
    orch = PerpetualOrchestrator(config)
    await orch._initialize_components()
    print("=" * 60, "\nINIT OK — running ONE evolution cycle\n", "=" * 60)
    await orch.force_evolution_cycle()
    st = orch.status
    print("=" * 60)
    print(f"generation        : {st.current_generation}")
    print(f"total strategies  : {st.total_strategies}")
    print(f"active strategies : {st.active_strategies}")
    print(f"alerts            : {st.alerts}")


asyncio.run(main())
