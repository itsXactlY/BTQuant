
#!/usr/bin/env python3
"""
Cheatcode Strategy BTQuant Implementation
"""

import sys
import os

# Add the strategy file to path
sys.path.append(os.path.dirname(__file__))
from cheatcode_strategy import CheatcodeStrategy
from backtrader.utils.backtest import backtest

if __name__ == '__main__':
    try:
        backtest(
            CheatcodeStrategy, 
            coin='PENDLE',
            collateral='USDT',
            start_date="2024-01-01", 
            end_date="2027-01-08", 
            interval="1h",
            init_cash=1000,
            plot=True, 
            quantstats=False
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
