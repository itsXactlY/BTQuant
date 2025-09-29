
# BTQuant BSC Trading Configuration - Keep it Simple Stupid!

import os

class SimpleBSCConfig:
    # your wallet (set these in environment or here)
    # i hate getting this wrong so double check!
    # TODO implement proper secret key encryption
    WALLET_PRIVATE_KEY = os.getenv("BSC_PRIVATE_KEY", "")
    WALLET_ADDRESS = os.getenv("BSC_WALLET_ADDRESS", "")

    # FREE RPC with MEV protection - we use this as our primary endpoint
    PRIMARY_RPC = "https://bscrpc.pancakeswap.finance"

    # backup free RPCs 
    BACKUP_RPCS = [
        "https://bsc-rpc.publicnode.com",
        "https://rpc.merkle.io/bsc", 
        "https://bsc-mainnet.public.blastapi.io"
    ]

    # trading pairs (contract addresses)
    TRADING_TOKENS = {
        # Stablecoins (safe for beginners to start with)
        "USDT": "0x55d398326f99059fF775485246999027B3197955",
        "USDC": "0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d", 
        "BUSD": "0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56",

        # Major tokens (more volatile, higher profits)
        "WBNB": "0xbb4CdB9CBd36B01bD1cBaeBF2De08d9173bc095c",
        "CAKE": "0x0E09FaBB73Bd3Ade0a17ECC321fD13a19e81cE82",
        "ETH": "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
    }

    # conservative trading settings (for beginners)
    TRADING_RULES = {
        "MIN_TRADE_USD": 10,      # Start small
        "MAX_TRADE_USD": 50,      # Dont risk too much  
        "STOP_LOSS": 0.01,        # 1% stop loss
        "PROFIT_TARGET": 0.02,    # 2% profit target
        "MAX_SLIPPAGE": 0.01,     # 1% slippage
        "MAX_DAILY_TRADES": 5,    # Dont overtrade
        "DAILY_LOSS_LIMIT": 0.10  # Stop if down 10% in a day
    }

    # gas settings (BSC is cheap but optimize anyway)
    GAS_CONFIG = {
        "GAS_LIMIT": 200000,      # Standard for swaps
        "GAS_PRICE_MULTIPLIER": 1.1,  # 10% above recommended
        "MAX_GAS_PRICE_GWEI": 20  # Dont pay too much
    }

# validate configuration
def validate_config():
    config = SimpleBSCConfig()

    if not config.WALLET_PRIVATE_KEY:
        print("⚠️ WARNING: No wallet private key configured!")
        print("Set BSC_PRIVATE_KEY environment variable")
        return False

    if not config.WALLET_ADDRESS:
        print("⚠️ WARNING: No wallet address configured!")
        print("Set BSC_WALLET_ADDRESS environment variable")  
        return False

    print("✅ Configuration looks good!")
    print(f"Wallet: {config.WALLET_ADDRESS[:6]}...{config.WALLET_ADDRESS[-4:]}")
    print(f"RPC: {config.PRIMARY_RPC}")
    print(f"Max trade size: ${config.TRADING_RULES['MAX_TRADE_USD']}")
    return True

if __name__ == "__main__":
    validate_config()
