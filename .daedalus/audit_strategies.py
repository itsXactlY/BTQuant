"""Static audit of the generated strategy library.

A backtest that produces 0 trades has two very different causes: a strategy
whose next() exists but whose conditions never fire, and a strategy with no
next() at all -- in which case it inherits the base class's no-op and can never
trade, no matter how good the idea is. Only the second is detectable without
running anything, and it is the one worth knowing before spending CPU.

Also flags stock-backtrader API that the fork does not have, since that is what
the codegen keeps hallucinating.
"""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

STRATEGIES = Path("/home/alca/projects/PubBTQuant/autonomous_agency/strategies")

# APIs the LLM reaches for that do not exist in this fork.
FORK_MISSING = [
    (re.compile(r"\bbt\.indicators\.Cross\b|\bindicators\.Cross\b(?!Over)"), "indicators.Cross (fork has CrossOver only)"),
    (re.compile(r"\bCross\w*\(.*_lines\.(?!\w+\))"), "Cross* with _lines="),
    (re.compile(r"get\([^)]*lookahead\s*="), "LineBuffer.get(lookahead=) does not exist"),
    (re.compile(r"\bRSI\([^)]*_ma\s*="), "RSI(_ma=) does not exist"),
    (re.compile(r"\bEMA\([^)]*_period\s*="), "EMA(_period=) does not exist"),
    (re.compile(r"\bbr\.order\([^)]*(order_target_pct|target=)"), "br.order(target=) is not a thing"),
    (re.compile(r"\bself\.broker\.get_cash\(\)\b"), "broker.get_cash exists; getcash() does not"),
    (re.compile(r"\bimport pandas as pd\b"), "pandas is not in the venv"),
    (re.compile(r"\bimport numpy as np\b"), "numpy is not in the venv"),
]


def audit(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    row = {"file": path.name, "bytes": path.stat().st_size}
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        row.update(ok=False, reason=f"SyntaxError: {e.msg}")
        return row

    cls = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                nm = b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
                if "Strategy" in nm:
                    cls = node
                    break
        if cls is not None:
            break
    if cls is None:
        row.update(ok=False, reason="no Strategy class")
        return row

    methods = {m.name for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    row["class"] = cls.name
    row["inherits"] = ", ".join(
        (b.id if isinstance(b, ast.Name) else getattr(b, "attr", "?")) for b in cls.bases)

    if "next" not in methods:
        row.update(ok=False, reason="no next() in the class")
        return row

    # Does next() reach an order at all -- directly or via a helper?
    next_fn = next(m for m in cls.body
                   if isinstance(m, ast.FunctionDef) and m.name == "next")
    next_src = ast.get_source_segment(text, next_fn) or ""
    row["next_lines"] = len(next_src.splitlines())
    ordered = bool(re.search(r"self\.(buy|sell|close|order)\b", next_src))
    # helper methods next() calls that place orders
    helpers = set(re.findall(r"self\.(_\w+)\(", next_src))
    helper_ordered = False
    for h in helpers:
        m = next((x for x in cls.body
                  if isinstance(x, ast.FunctionDef) and x.name == h), None)
        if m is None:
            continue
        s = ast.get_source_segment(text, m) or ""
        if re.search(r"self\.(buy|sell|close|order)\b", s):
            helper_ordered = True
    row["orders_direct"] = ordered
    row["orders_via_helper"] = helper_ordered
    if not (ordered or helper_ordered):
        row.update(ok=False, reason="next() never places an order")
        return row

    row["ok"] = True
    row["reason"] = ""
    return row


def main():
    files = sorted(STRATEGIES.glob("*.py"))
    rows = [audit(f) for f in files]
    ok = [r for r in rows if r.get("ok")]
    reasons = Counter(r.get("reason", "") for r in rows if not r.get("ok"))
    print(f"files: {len(rows):,}   runnable-by-static-check: {len(ok):,} "
          f"({100*len(ok)/max(len(rows),1):.1f}%)")
    print("\nreasons a file is not runnable:")
    for reason, n in reasons.most_common():
        print(f"  {n:>5}  {reason}")

    api = Counter()
    for f in files:
        t = f.read_text(errors="ignore")
        for rx, label in FORK_MISSING:
            if rx.search(t):
                api[label] += 1
    print("\nstock-backtrader API the fork lacks (files affected):")
    for label, n in api.most_common():
        print(f"  {n:>5}  {label}")

    with open("/home/alca/projects/PubBTQuant/.daedalus/strategy_audit.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\nwrote .daedalus/strategy_audit.json")


if __name__ == "__main__":
    main()
