#!/home/alca/projects/BTQuant/.btq/bin/python
from typing import List
import os
import subprocess
from datetime import datetime

class ScreenManager:
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.join(os.getcwd(), 'trading_logs')
        self.date_str = datetime.now().strftime('%Y%m%d')
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary log directories"""
        os.makedirs(os.path.join(self.base_dir, self.date_str), exist_ok=True)

    def create_screen_command(self, coin: str) -> List[str]:
        """Generate screen command for a specific coin"""
        screen_name = f"trading_{coin}"
        log_dir = os.path.join(self.base_dir, self.date_str, coin)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{coin}_trading.log')
        
        # Create the Python command that will be executed
        python_script = f"""
import sys
from fastquant import livetrade_crypto_binance as livetrade

try:
    livetrade(
        coin='{coin}',
        collateral='USDT',
        strategy='STScalp',
        exchange='mimic',
        account='binance_sub2',
        asset='{coin}/USDT',
        amount=11
    )
except Exception as e:
    print(f"Error trading {coin}: {{str(e)}}")
    sys.exit(1)
"""
        
        # Save the Python script
        script_path = os.path.join(log_dir, f'{coin}_trade.py')
        with open(script_path, 'w') as f:
            f.write(python_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Construct screen command
        return [
            'screen',
            '-dmS', screen_name,
            '-L', '-Logfile', log_file,
            'python3', script_path
        ]

def main():
    coins = ['BTC', 'LINK', 'ETH', 'BNB', 'CAKE', 'XRP', '1000CAT', 'FLOKI', 'PEPE', 'DOGE', 'BCH', 'LTC']
    manager = ScreenManager()
    
    print("Starting trading sessions...")
    
    for coin in coins:
        cmd = manager.create_screen_command(coin)
        print(f"Starting {coin} trading session...")
        subprocess.run(cmd)
        print(f"Waiting 10 seconds before starting next coin...")
        if coin != coins[-1]:  # Don't sleep after last coin
            subprocess.run(['sleep', '10'])

    print("\nAll trading sessions started!")
    print("\nTo view running sessions:")
    print("  screen -ls")
    print("\nTo attach to a session:")
    print("  screen -r trading_COINNAME")
    print("\nTo detach from a session:")
    print("  Press Ctrl+A then D")
    print("\nTo kill all sessions:")
    print("  killall screen")

if __name__ == "__main__":
    main()