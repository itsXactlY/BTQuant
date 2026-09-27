"""Re-run only the batch hypotheses the LLM path failed on, now that the prompt
carries the real indicator signatures. Same code path as generate_batch.py,
just an explicit index list so the 4 that already landed are not regenerated.

Usage: regen_failed.py 3 4 5 6
"""
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/alca/projects/PubBTQuant")
sys.path.insert(0, "/home/alca/projects/PubBTQuant/.daedalus")

logging.basicConfig(level=logging.WARNING, stream=sys.stderr,
                    format="%(levelname)s %(name)s: %(message)s")

from generate_batch import make_hypothesis          # noqa: E402
from autonomous_agency.strategy_factory import StrategyFactory   # noqa: E402
from autonomous_agency.sweep import class_name_of, is_tradeable   # noqa: E402

OUT = Path("/home/alca/projects/PubBTQuant/.daedalus/regen_results.json")


def main():
    idxs = [int(a) for a in sys.argv[1:]] or [3, 4, 5, 6]
    out = []
    for n, i in enumerate(idxs, 1):
        h = make_hypothesis(i)
        sf = StrategyFactory()
        sf.logger = logging.getLogger("regen")
        sf._last_validation_error = ""
        t0 = time.time()
        try:
            code = sf._generate_code_with_llm(sf._create_strategy_spec(h), h)
        except Exception as e:
            rec = {"i": i, "hypothesis": h.name, "result": "exception",
                   "error": f"{type(e).__name__}: {e}"}
            out.append(rec)
            print(f"[{n}/{len(idxs)}] {h.name}: EXCEPTION {rec['error'][:70]}", flush=True)
            continue
        if not code:
            out.append({"i": i, "hypothesis": h.name, "result": "no_code",
                        "error": sf._last_validation_error})
            print(f"[{n}/{len(idxs)}] {h.name}: no code "
                  f"({sf._last_validation_error[:90]})", flush=True)
            OUT.write_text(json.dumps(out, indent=1))
            continue
        p = Path(sf.output_dir) / f"{h.name.replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code)
        out.append({"i": i, "hypothesis": h.name, "result": "ok",
                    "path": str(p.resolve()), "tradeable": is_tradeable(p),
                    "seconds": round(time.time() - t0, 1)})
        print(f"[{n}/{len(idxs)}] {h.name}: {len(code)} B, tradeable="
              f"{out[-1]['tradeable']}, {time.time()-t0:.0f}s", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
