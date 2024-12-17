# raydium_prototype.py
import asyncio
import aiohttp
import logging
import json
from solana.rpc.api import Client
from solana.rpc.api import Pubkey as PublicKey
from solana.rpc.commitment import Commitment
from typing import Optional
import requests
import time
from base58 import b58decode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RaydiumPrototype')

# Constants
RAYDIUM_API_URL = "https://api.raydium.io/v2/sdk/liquidity/mainnet.json"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
TOKEN_MINT = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

class RaydiumAPI:
    def __init__(self):
        self.url = RAYDIUM_API_URL

    async def get_pool_info(self, token_mint: str) -> dict:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()
            pools_data = response.json()

            logger.info("Successfully fetched Raydium pools data")
            
            for pool in pools_data.get('official', []):
                if pool.get('baseMint') == token_mint or pool.get('quoteMint') == token_mint:
                    logger.info(f"Found pool for token {token_mint}")
                    return pool  # Return the entire pool data

            logger.warning(f"No pool found for token {token_mint}")
            return None

        except Exception as e:
            logger.error(f"Error fetching pool info: {e}")
            return None

class SolanaClient:
    def __init__(self, rpc_url: str):
        self.client = Client(rpc_url)

    async def get_account_info(self, pubkey: PublicKey, encoding: str = "base64", commitment: str = "confirmed"):
        try:
            result = await self.client.get_account_info(
                pubkey,
                encoding=encoding,
                commitment=commitment
            )
            return result
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return None

class RaydiumPrototype:
    def __init__(self, rpc_url: str):
        self.solana_client = SolanaClient(rpc_url)
        self.raydium_api = RaydiumAPI()
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY

    async def get_pool_data(self, pool_id: str):
        try:
            # Remove the extra characters from the public key
            pool_id = pool_id[:44].rstrip("=")
            
            # Decode the base58 string
            decoded = b58decode(pool_id)
            
            # Check if the decoded string has a length of 32
            if len(decoded) != 32:
                raise ValueError(f"Invalid pool ID: {pool_id}")
            
            # Create a PublicKey object from the decoded string
            pubkey = PublicKey(decoded)
            
            # Get account info with explicit encoding
            result = await self.solana_client.get_account_info(
                pubkey,
                encoding="base64",
                commitment="confirmed"
            )
                
            if result and hasattr(result, 'value') and result.value:
                account_data = result.value.data
                if account_data:
                    if isinstance(account_data, str):
                        import base64
                        account_data = base64.b64decode(account_data)
                        logger.info(f"Raw account data length: {len(account_data)} bytes")
                        return account_data
                    else:
                        logger.warning(f"Unexpected data type: {type(account_data)}")
            
            logger.warning(f"No account data found for pool ID: {pool_id}")
            return None

        except ValueError as ve:
            logger.error(f"Error creating PublicKey: {ve}")
            # Try alternative method using bytes
            try:
                decoded = b58decode(pool_id)
                if len(decoded) != 32:
                    padded = decoded.rjust(32, b'\0')  # Pad with zeros if necessary
                    pubkey = PublicKey(padded)
                else:
                    pubkey = PublicKey(decoded)
                # ... rest of the code ...
            except Exception as e:
                logger.error(f"Alternative method failed: {e}")
                return None

        except Exception as e:
            logger.error(f"Error fetching pool data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def get_pool_data_with_retry(self, pool_id: str) -> Optional[bytes]:
        """Fetch pool data with retries"""
        for attempt in range(self.max_retries):
            try:
                data = await self.get_pool_data(pool_id)
                if data:
                    return data
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                
        return None

    async def monitor_pool(self, pool_info: dict, interval: int = 5):
        """Monitor pool with given interval"""
        try:
            while True:
                try:
                    amm_id = pool_info.get('authority')
                    if not amm_id:
                        raise ValueError("No valid AMM ID available in the provided pool info")

                    logger.info(f"Fetching data for AMM ID: {amm_id} (length: {len(amm_id)})")
                    pool_data = await self.get_pool_data_with_retry(amm_id)

                    if pool_data:
                        logger.info(f"Received pool data, size: {len(pool_data)} bytes")
                        logger.info(f"Data preview (hex): {pool_data[:32].hex()}")
                        
                        # Add parsing logic here once we get valid data
                        
                    else:
                        logger.warning("No pool data received after retries")

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    logger.error(f"Error type: {type(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
                await asyncio.sleep(interval)

        except Exception as e:
            logger.error(f"Error in monitor_pool: {e}")

    async def get_sol_price(self) -> float:
        try:
            response = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd')
            response.raise_for_status()
            price = response.json()['solana']['usd']
            logger.info(f"Current SOL price: ${price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching SOL price: {e}")
            return None

    def validate_pubkey(self, pubkey_str: str) -> bool:
        """Validate a Solana public key string"""
        try:
            # Try to create a PublicKey object
            PublicKey(pubkey_str)
            return True
        except Exception as e:
            logger.error(f"Public key validation failed: {e}")
            return False


async def main():
    # RPC URL - Consider using a dedicated RPC provider for production
    rpc_url = SOLANA_RPC_URL
    
    # RAY token mint address
    token_mint = TOKEN_MINT
    
    raydium = RaydiumPrototype(rpc_url)
    
    try:
        logger.info(f"Fetching pool info for token: {token_mint}")
        pool_info = await raydium.raydium_api.get_pool_info(token_mint)
        
        if pool_info:
            logger.info("Pool Info:")
            logger.info(json.dumps(pool_info, indent=2))

            sol_price = await raydium.get_sol_price()
            if sol_price:
                logger.info(f"Current SOL price: ${sol_price}")

            logger.info("Starting pool monitor...")
            await raydium.monitor_pool(pool_info)
        else:
            logger.error("No pool information found")
    
    except KeyboardInterrupt:
        logger.info("Stopping monitor...")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())