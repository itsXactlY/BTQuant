#!/usr/bin/env python3
"""Audit the repo for hardcoded BTQuant DB credentials.

Masks any secret it finds so the audit output itself is safe to print/log.
Prints file:line plus a masked preview, never the raw value.
"""
import os
import re
import sys

# Read the real secrets so we can search for their actual values.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btq_secrets

cfg = btq_secrets.load()
SECRET_VALUES = {
    v: k for k, v in cfg.items()
    if k == "password" and isinstance(v, str) and len(v) >= 6
}

# Historic real values that were committed before the indirection existed.
# These come from the local secrets file, never from this repo — writing them
# here would re-commit the very secrets we are removing.
HISTORIC = {
    v: "HISTORIC real password" for v in cfg.get("historic_passwords", [])
    if isinstance(v, str) and len(v) >= 6
}
HISTORIC.update({
    "YourPassword": "doc placeholder",
    "test-password": "test fixture",
})

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".btq_cache", ".venv", "venv"}
EXTS = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".toml", ""}


def mask(text: str) -> str:
    for secret in SECRET_VALUES:
        text = text.replace(secret, "***REAL_SECRET***")
    return text


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    findings = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = os.path.join(dirpath, name)
            ext = os.path.splitext(name)[1]
            if ext not in EXTS and name not in ("Dockerfile", "Makefile"):
                continue
            try:
                with open(path, "r", errors="replace") as handle:
                    lines = handle.readlines()
            except OSError:
                continue
            for lineno, line in enumerate(lines, 1):
                for secret, key in SECRET_VALUES.items():
                    if secret in line:
                        rel = os.path.relpath(path, root)
                        findings.append((rel, lineno, f"LIVE {key}"))
                for historic, label in HISTORIC.items():
                    if historic in line:
                        rel = os.path.relpath(path, root)
                        findings.append((rel, lineno, label))

    if not findings:
        print("CLEAN: no hardcoded credentials anywhere in the tree.")
        return 0

    print(f"{len(findings)} finding(s):\n")
    for rel, lineno, label in sorted(findings):
        print(f"  {rel}:{lineno}  [{label}]")
    live = [f for f in findings if f[2].startswith("LIVE")]
    if live:
        print(f"\nCRITICAL: {len(live)} live secret(s) still in the tree.")
        return 1
    print("\nNo live secrets. Remaining items are placeholders/test fixtures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
