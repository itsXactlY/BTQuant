import requests
from queue import Queue
from .raydium_feed import RaydiumData

class TokenInfo:
    def __init__(self, token_address):
        self.token_address = token_address
        self.market_id = ''
        self.price = 0
        self.token_vault_ui_amount = 0
        self.sol_vault_address = ''
        self.token_vault_address = ''
        self.sol_address = ''
        self.token_decimals = ''

def get_request(request_uri: str):
    response = requests.get(request_uri)

    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_price_in_usd():
    try:
        url = 'https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd'
        response = requests.get(url)
        data = response.json()
        return data['solana']['usd']
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None

def get_amm_token_pool_data(token_address: str)->TokenInfo:
    ray_uri = "https://api-v3.raydium.io/pools"
    ray_uri_marketid_uri = ray_uri + "/info/mint?mint1=" + token_address + "&poolType=all&poolSortField=default&sortType=desc&pageSize=1&page=1"

    #Make the API call
    data = get_request(ray_uri_marketid_uri)
    
    if len(data) > 0:
        try:
            token_info = TokenInfo(token_address)
            token_info.market_id = data['data']['data'][0]['id']
            token_info.price = data['data']['data'][0]['price']

            pool_info_uri = ray_uri + "/key/ids?ids=" + token_info.market_id

            data = get_request(pool_info_uri)

            if len(data) > 0:
                mintA = data['data'][0]['mintA']
                mintB = data['data'][0]['mintB']
                vaultA = data['data'][0]['vault']['A']
                vaultB = data['data'][0]['vault']['B']

                if mintA['address'] == token_address:
                    token_info.sol_address = mintB['address']
                    token_info.token_decimals = mintA['decimals'] 
                    token_info.token_vault_address = vaultA
                    token_info.sol_vault_address = vaultB
                else:
                    token_info.sol_address = mintA['address']
                    token_info.token_decimals = mintB['decimals'] 
                    token_info.sol_vault_address = vaultA
                    token_info.token_vault_address = vaultB
                    token_info.price = 1/token_info.price

                solana_price_in_usd = get_price_in_usd()
                if solana_price_in_usd is not None:
                    token_price_in_usd = token_info.price * solana_price_in_usd
                    print(f"Token Address: {token_info.token_address}")
                    print(f"Market ID: {token_info.market_id}")
                    print(f"Price in SOL: {token_info.price:.6f}")
                    print(f"Price in USD: ${token_price_in_usd:.6f}")
                    print(f"SOL Address: {token_info.sol_address}")
                    print(f"Token Decimals: {token_info.token_decimals}")
                    print(f"Token Vault Address: {token_info.token_vault_address}")
                    print(f"SOL Vault Address: {token_info.sol_vault_address}")
                return token_info
        except Exception as e:
            print(str(e))
    return

class RaydiumStore:
    def __init__(self, token_address):
        self.token_address = token_address
        self.token_info = get_amm_token_pool_data(token_address)
        self.message_queue = Queue()

        if not self.token_info:
            raise ValueError("Unable to fetch token pool data.")
    
    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = RaydiumData(store=self, start_date=start_date)
        return self._data

    def fetch_latest_price(self):
        # Periodically fetch price from Raydium API
        try:
            new_info = get_amm_token_pool_data(self.token_address)
            if new_info:
                self.token_info.price = new_info.price
        except Exception as e:
            print(f"Error updating price: {e}")
