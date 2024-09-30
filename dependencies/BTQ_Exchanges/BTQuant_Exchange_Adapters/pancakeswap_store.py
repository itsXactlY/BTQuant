from .pancakeswap_feed import Web3, FACTORY_ABI, PAIR_ABI, Queue, threading, time, TimeFrame, PancakeSwapData
import threading
import requests
from queue import Queue
from backtrader.dataseries import TimeFrame
from .pancakeswap_feed import PancakeSwapData

class PancakeSwapStore(object):
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): '1s',
        (TimeFrame.Minutes, 1): '1m',
        (TimeFrame.Minutes, 3): '3m',
        (TimeFrame.Minutes, 5): '5m',
        (TimeFrame.Minutes, 15): '15m',
        (TimeFrame.Minutes, 30): '30m',
        (TimeFrame.Minutes, 60): '1h',
        (TimeFrame.Minutes, 120): '2h',
        (TimeFrame.Minutes, 240): '4h',
        (TimeFrame.Minutes, 360): '6h',
        (TimeFrame.Minutes, 480): '8h',
        (TimeFrame.Minutes, 720): '12h',
        (TimeFrame.Days, 1): '1d',
        (TimeFrame.Days, 3): '3d',
        (TimeFrame.Weeks, 1): '1w',
        (TimeFrame.Months, 1): '1M',
    }

    def __init__(self, coin_refer, coin_target):
        self.coin_refer = coin_refer
        self.coin_target = coin_target
        self.ws_url = "wss://bsc-rpc.publicnode.com"
        self.w3 = None
        self.factory_address = Web3.to_checksum_address('0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73')
        self.bnb_address = Web3.to_checksum_address(coin_target)
        self.factory_contract = None
        self.message_queue = Queue()
        self.websocket = None
        self.websocket_thread = None
        self.w3 = Web3(Web3.LegacyWebSocketProvider(self.ws_url))
        self.factory_contract = self.w3.eth.contract(address=self.factory_address, abi=FACTORY_ABI)

    def getdata(self, start_date=None):
        if not hasattr(self, '_data'):
            self._data = PancakeSwapData(store=self, token_address=self.coin_refer, start_date=start_date)
        return self._data

    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    def start_socket(self, token_address):
        def run_socket():
            print("Starting WebSocket connection...")
            pair_address = self.get_pair_address(token_address)
            if pair_address == '0x0000000000000000000000000000000000000000':
                print("No liquidity pair found for this token with BNB")
                return

            pair_contract = self.w3.eth.contract(address=pair_address, abi=PAIR_ABI)
            token0 = pair_contract.functions.token0().call()
            is_token0 = token_address.lower() == token0.lower()

            bnb_price_usd = self.get_bnb_price_in_usd()
            if bnb_price_usd is None:
                print(f"Couldn't fetch BNB price in USD.")
                return

            last_price = None
            high = low = open_price = close_price = None
            volume = 0
            start_time = time.time()

            while True:
                try:
                    current_time = time.time()
                    if current_time - start_time >= 1:  # 1-second candlestick
                        reserves = pair_contract.functions.getReserves().call()
                        reserve0, reserve1, _ = reserves

                        if is_token0:
                            token_reserve, bnb_reserve = reserve0, reserve1
                        else:
                            bnb_reserve, token_reserve = reserve0, reserve1

                        current_price_in_bnb = (bnb_reserve / (10**18)) / (token_reserve / (10**18))
                        current_price_in_usd = current_price_in_bnb * bnb_price_usd

                        if last_price is None:
                            open_price = high = low = current_price_in_usd
                        else:
                            high = max(high, current_price_in_usd)
                            low = min(low, current_price_in_usd)

                        volume += abs(current_price_in_usd - last_price) if last_price else 0
                        close_price = current_price_in_usd

                        price_data = {
                            'timestamp': int(current_time),
                            'open': open_price,
                            'high': high,
                            'low': low,
                            'close': close_price,
                            'volume': volume
                        }
                        self.message_queue.put(price_data)

                        # Reset for next candlestick
                        open_price = current_price_in_usd
                        high = low = current_price_in_usd
                        volume = 0
                        last_price = current_price_in_usd
                        start_time = current_time

                    time.sleep(0.5)  # Sleep for 500ms to reduce CPU usage

                except Exception as e:
                    print(f"Error in data fetching: {e}")
                    print("Attempting to reconnect in 5 seconds...")
                    time.sleep(5)

        self.websocket_thread = threading.Thread(target=run_socket, daemon=True)
        self.websocket_thread.start()

    def stop_socket(self):
        if self.websocket:
            self.websocket.close()
            print("WebSocket connection closed.")

    def fetch_ohlcv(self, token_address, interval, since=None):
        # This method is left as a placeholder.
        # Fetching historical OHLCV data from BSC would require additional off-chain data sources
        # or complex on-chain data aggregation, which is beyond the scope of this example.
        print("Warning: fetch_ohlcv not implemented for BSC. Returning empty list.")
        return []

    def get_bnb_price_in_usd(self):
        try:
            url = 'https://api.coingecko.com/api/v3/simple/price?ids=binancecoin&vs_currencies=usd'
            response = requests.get(url)
            data = response.json()
            return data['binancecoin']['usd']
        except Exception as e:
            print(f"Error fetching BNB price: {e}")
            return None

    def get_pair_address(self, token_address):
        token_address = Web3.to_checksum_address(token_address)
        pair_address = self.factory_contract.functions.getPair(token_address, self.bnb_address).call()
        return pair_address

