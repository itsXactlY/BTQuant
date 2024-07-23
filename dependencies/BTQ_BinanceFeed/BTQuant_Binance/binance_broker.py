from collections import deque
from backtrader.broker import BrokerBase
from binance.enums import *

class BinanceBroker(BrokerBase):
    params = (('use_positions', False), )

    def __init__(self, store):
        super(BinanceBroker, self).__init__()
        self.notifs = deque()
        self.positions = {}
        self.open_orders = list()
        self._store = store
        self._symbol = self._store._symbol
        # print(f'init {self._symbol}')
        self._store = store
        self._store.binance_socket.start_user_socket(self._handle_user_socket_message)

    def _handle_user_socket_message(self, msg):
        if msg['e'] == 'error':
            raise msg

    def getvalue(self, datas=None):
        self.value = self._store._value
        return self.value

    def format_price(self, value):
        return self._store.format_price(value)

    def get_asset_balance(self, asset):
        return self._store.get_asset_balance(asset)

    def getcash(self):
        self.cash = self._store._cash
        return self.cash

    def get_notification(self):
        if not self.notifs:
            return None

        return self.notifs.popleft()

    def notify(self, order):
        self.notifs.append(order)
