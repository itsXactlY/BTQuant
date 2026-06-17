#!/usr/bin/env python3
"""Compatibility wrapper for the current BTQuant UI refinement routine.

The original 100-pass harness has been superseded by iterate_350.py:
- 100 deterministic baseline passes
- 250 FREE OpenRouter model-persona review passes
- 350 total passes recorded in index.html and ITERATION_REPORT.md
"""
from iterate_350 import main

if __name__ == "__main__":
    main()
