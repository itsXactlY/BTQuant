import asyncio
import websockets
import json
import base64
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict

@dataclass
class PoolState:
    token_address: str = "7ZqzGzTNg5tjK1CHTBdGFHyKjBtXdfvAobuGgdt4pump"
    token_vault: str = "GFuqAXDm3LxphnUsaa6raLxtoD5YvcNuoG8Ri3aYQA92"
    sol_vault: str = "3sQ5fXczzdP7X1Dju5UG7L7pzfWf28VkzwZveUvasyot"
    token_decimals: int = 6
    sol_decimals: int = 9

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    token_sol_price: float
    token_usd_price: float
    sol_usd_price: float

class RaydiumOHLCVCollector:
    def __init__(self):
        self.ws_url = "wss://solana-rpc.publicnode.com"
        self.binance_ws_url = "wss://stream.binance.com:9443/stream?streams=solusdt@trade"
        self.pool = PoolState()
        self.current_candle: Candle = None
        self.candle_history: List[Candle] = []
        self.last_update = None
        self.sol_price = 0
        self.reconnect_delay = 1  # Start with 1 second delay
        self.usdc_sol_pool = "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"  # USDC/SOL pool


    async def subscribe_to_accounts(self):
        """Subscribe to both HOBA pool and USDC/SOL pool"""
        subscription_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "accountSubscribe",
            "params": [
                self.pool.token_vault,
                {"encoding": "base64", "commitment": "confirmed"}
            ]
        }

        sol_price_subscription = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "accountSubscribe",
            "params": [
                self.usdc_sol_pool,
                {"encoding": "base64", "commitment": "confirmed"}
            ]
        }
        
        async with websockets.connect(self.ws_url) as websocket:
            # Subscribe to both pools
            await websocket.send(json.dumps(subscription_message))
            await websocket.send(json.dumps(sol_price_subscription))
            
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    # Check which subscription this is
                    if 'params' in data:
                        account = data['params']['result']['value']['data'][0]
                        subscription_id = data['params']['subscription']
                        
                        if subscription_id == 1:  # HOBA pool
                            await self.process_account_update(data)
                        elif subscription_id == 2:  # USDC/SOL pool
                            await self.update_sol_price(data)
                            
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    await asyncio.sleep(1)

    async def update_sol_price(self, data: Dict):
        """Process USDC/SOL pool data to get SOL price"""
        try:
            account_data = base64.b64decode(data['params']['result']['value']['data'][0])
            
            print("\nSOL/USDC Pool Raw Data:")
            # Print several potential positions
            for i in range(64, 200, 8):
                value = int.from_bytes(account_data[i:i+8], 'little')
                print(f"Bytes {i}-{i+7}: {value}")
            
            # Try different positions for USDC/SOL pool
            usdc_amount = int.from_bytes(account_data[64:72], 'little') / (10 ** 6)
            sol_amount = int.from_bytes(account_data[72:80], 'little') / (10 ** 9)
            
            print(f"USDC Amount: {usdc_amount}")
            print(f"SOL Amount: {sol_amount}")
            
            if sol_amount > 0:
                self.sol_price = usdc_amount / sol_amount
                print(f"Updated SOL price: ${self.sol_price:.2f}")
                
        except Exception as e:
            print(f"Error updating SOL price: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    async def process_account_update(self, data: Dict):
        try:
            if 'params' in data:
                account_data = base64.b64decode(data['params']['result']['value']['data'][0])
                
                print("\nHOBA Pool Raw Data:")
                # Print several potential positions
                for i in range(64, 200, 8):
                    value = int.from_bytes(account_data[i:i+8], 'little')
                    print(f"Bytes {i}-{i+7}: {value}")
                
                # Try different positions for token amounts
                token_raw = int.from_bytes(account_data[64:72], 'little')
                sol_raw = int.from_bytes(account_data[72:80], 'little')
                
                # Also try alternative positions
                alt_token_raw = int.from_bytes(account_data[168:176], 'little')
                alt_sol_raw = int.from_bytes(account_data[176:184], 'little')
                
                print(f"\nPotential Raw Values:")
                print(f"Position 64-72/72-80:")
                print(f"Token Raw: {token_raw}")
                print(f"SOL Raw: {sol_raw}")
                print(f"\nPosition 168-176/176-184:")
                print(f"Alt Token Raw: {alt_token_raw}")
                print(f"Alt SOL Raw: {alt_sol_raw}")
                
                # Convert to actual amounts with decimals
                token_amount = token_raw / (10 ** self.pool.token_decimals)
                sol_amount = sol_raw / (10 ** self.pool.sol_decimals)
                
                print(f"\nFinal Amounts:")
                print(f"HOBA Amount: {token_amount}")
                print(f"SOL Amount: {sol_amount}")
                
                # Calculate prices
                if token_amount > 0 and sol_amount > 0:
                    token_sol_price = sol_amount / token_amount
                    token_usd_price = token_sol_price * self.sol_price if self.sol_price > 0 else 0
                    
                    print(f"\nPrice Calculations:")
                    print(f"HOBA/SOL Price: {token_sol_price:.8f}")
                    print(f"HOBA/USD Price: {token_usd_price:.8f}")
                    print(f"SOL/USD Price: {self.sol_price:.2f}")
                else:
                    token_sol_price = 0
                    token_usd_price = 0
                
                # Calculate volume
                volume = 0
                if self.last_update:
                    volume = abs(token_amount - self.last_update[0]) + abs(sol_amount - self.last_update[1])
                self.last_update = (token_amount, sol_amount)

                await self.update_candle(token_sol_price, token_usd_price, volume)
                
        except Exception as e:
            print(f"Error processing update: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    async def update_candle(self, token_sol_price: float, token_usd_price: float, volume: float):
        current_time = datetime.now()

        if not self.current_candle or (current_time - self.current_candle.timestamp).seconds >= 1:
            if self.current_candle:
                self.candle_history.append(self.current_candle)
            
            self.current_candle = Candle(
                timestamp=current_time,
                open=token_sol_price,
                high=token_sol_price,
                low=token_sol_price,
                close=token_sol_price,
                volume=volume,
                token_sol_price=token_sol_price,
                token_usd_price=token_usd_price,
                sol_usd_price=self.sol_price
            )
        else:
            self.current_candle.high = max(self.current_candle.high, token_sol_price)
            self.current_candle.low = min(self.current_candle.low, token_sol_price)
            self.current_candle.close = token_sol_price
            self.current_candle.volume += volume
            self.current_candle.token_sol_price = token_sol_price
            self.current_candle.token_usd_price = token_usd_price
            self.current_candle.sol_usd_price = self.sol_price

        print(f"\n=== Candle {self.current_candle.timestamp} ===")
        print(f"Open: {self.current_candle.open:.16f} SOL")
        print(f"High: {self.current_candle.high:.16f} SOL")
        print(f"Low: {self.current_candle.low:.16f} SOL")
        print(f"Close: {self.current_candle.close:.16f} SOL")
        print(f"Volume: {self.current_candle.volume:.6f}")
        print(f"Token/SOL: {self.current_candle.token_sol_price:.16f}")
        print(f"Token/USD: {self.current_candle.token_usd_price:.16f}")
        print(f"SOL/USD: {self.current_candle.sol_usd_price:.2f}")

    async def main(self):
        """Main entry point"""
        print(f"""
Raydium OHLCV Collector
======================
Pair: HOBA/SOL
Token Address: {self.pool.token_address}
Token Vault: {self.pool.token_vault}
SOL Vault: {self.pool.sol_vault}
        """)
        
        # Create tasks for both WebSocket connections
        solana_task = asyncio.create_task(self.subscribe_to_accounts())
        
        # Run both tasks concurrently
        try:
            await asyncio.gather(solana_task)
        except Exception as e:
            print(f"Error in main: {e}")
            if solana_task:
                solana_task.cancel()

if __name__ == "__main__":
    collector = RaydiumOHLCVCollector()
    try:
        asyncio.run(collector.main())
    except KeyboardInterrupt:
        print("\nShutting down collector...")
    except Exception as e:
        print(f"Fatal error: {e}")