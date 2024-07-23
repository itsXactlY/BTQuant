import time

from functools import wraps
from math import floor

from backtrader.dataseries import TimeFrame
from binance import Client, ThreadedWebsocketManager
from binance.enums import *
from binance.exceptions import BinanceAPIException
from requests.exceptions import ConnectTimeout, ConnectionError

from .binance_broker import BinanceBroker
from .binance_feed import BinanceData

class BinanceStore(object):
    _GRANULARITIES = {
        (TimeFrame.Seconds, 1): KLINE_INTERVAL_1SECOND,
        (TimeFrame.Minutes, 1): KLINE_INTERVAL_1MINUTE,
        (TimeFrame.Minutes, 3): KLINE_INTERVAL_3MINUTE,
        (TimeFrame.Minutes, 5): KLINE_INTERVAL_5MINUTE,
        (TimeFrame.Minutes, 15): KLINE_INTERVAL_15MINUTE,
        (TimeFrame.Minutes, 30): KLINE_INTERVAL_30MINUTE,
        (TimeFrame.Minutes, 60): KLINE_INTERVAL_1HOUR,
        (TimeFrame.Minutes, 120): KLINE_INTERVAL_2HOUR,
        (TimeFrame.Minutes, 240): KLINE_INTERVAL_4HOUR,
        (TimeFrame.Minutes, 360): KLINE_INTERVAL_6HOUR,
        (TimeFrame.Minutes, 480): KLINE_INTERVAL_8HOUR,
        (TimeFrame.Minutes, 720): KLINE_INTERVAL_12HOUR,
        (TimeFrame.Days, 1): KLINE_INTERVAL_1DAY,
        (TimeFrame.Days, 3): KLINE_INTERVAL_3DAY,
        (TimeFrame.Weeks, 1): KLINE_INTERVAL_1WEEK,
        (TimeFrame.Months, 1): KLINE_INTERVAL_1MONTH,
    }

    # def __init__(self, api_key, api_secret, coin_refer, coin_target, testnet=False, retries=5):
    def __init__(self, coin_refer, coin_target, exchange, account, testnet=False, retries=5):
        self.binance = Client(testnet=testnet)
        self.binance_socket = ThreadedWebsocketManager(testnet=testnet)
        self.binance_socket.daemon = True
        self.binance_socket.start()
        self.coin_refer = coin_refer
        self.coin_target = coin_target
        self.symbol = coin_refer + coin_target
        self._symbol = coin_refer+'/'+coin_target
        self.retries = retries

        self._cash = 0
        self._value = 0

        self._step_size = None
        self._tick_size = None

        self._broker = BinanceBroker(store=self)
        self._data = None
        self._dataname = self.symbol
        
    def _format_value(self, value, step):
        precision = step.find('1') - 1
        if precision > 0:
            return '{:0.0{}f}'.format(value, precision)
        return floor(int(value))
        
    def retry(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attempt in range(1, self.retries + 1):
                time.sleep(60 / 1200) # API Rate Limit
                try:
                    return func(self, *args, **kwargs)
                except (BinanceAPIException, ConnectTimeout, ConnectionError) as err:
                    if isinstance(err, BinanceAPIException) and err.code == -1021:
                        # Recalculate timestamp offset between local and Binance's server
                        res = self.binance.get_server_time()
                        self.binance.timestamp_offset = res['serverTime'] - int(time.time() * 1000)
                    
                    if attempt == self.retries:
                        raise
        return wrapper

    def format_price(self, price):
        return self._format_value(price, self._tick_size)
    
    def format_quantity(self, size):
        return self._format_value(size, self._step_size)
    
    def getbroker(self):
        return self._broker

    def getdata(self, start_date=None):
        if not self._data:
            self._data = BinanceData(store=self, start_date=start_date)
        return self._data

    def get_interval(self, timeframe, compression):
        return self._GRANULARITIES.get((timeframe, compression))

    @retry
    def get_symbol_info(self, symbol):
        print(f'Trying to get Symbol Info via websocket for {symbol}')
        return self.binance.get_symbol_info(symbol)

    def stop_socket(self):
        self.binance_socket.stop()
        self.binance_socket.join(5)