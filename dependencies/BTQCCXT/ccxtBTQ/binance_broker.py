import datetime as dt
from collections import deque
from math import copysign

from backtrader.broker import BrokerBase
from backtrader.order import Order, OrderBase
from backtrader.position import Position
from binance.enums import *
import requests
import json
from dontcommit import jrr_order_history
from ccxt import BaseError

class BinanceOrder(OrderBase):
    def __init__(self, owner, data, exectype, binance_order):
        self.owner = owner
        self.data = data
        self.exectype = exectype
        self.ordtype = self.Buy if binance_order['side'] == SIDE_BUY else self.Sell

        # Market order price is zero
        if self.exectype == Order.Market:
            self.size = float(binance_order['executedQty'])
            self.price = sum(float(fill['price']) for fill in binance_order['fills']) / len(binance_order['fills'])  # Average price
        else:
            self.size = float(binance_order['origQty'])
            self.price = float(binance_order['price'])
        self.binance_order = binance_order

        super(BinanceOrder, self).__init__()
        self.accept()


class BinanceBroker(BrokerBase):
    _ORDER_TYPES = {
        Order.Market: 'market',
        Order.Limit: 'limit',
        Order.Stop: 'stop',  # stop-loss for kraken, stop for bitmex
        Order.StopLimit: 'stop limit'
    }

    params = (('use_positions', True), )

    def __init__(self, store):
        super(BinanceBroker, self).__init__()

        self.notifs = deque()
        self.positions = {}

        self.open_orders = list()
        
        self._store = store
        
        self._symbol = self._store._symbol
        print(f'init {self._symbol}')

        self._store = store
        self._store.binance_socket.start_user_socket(self._handle_user_socket_message)

    # def start(self):
    #     if self.p.use_positions:
    #         self.load_and_calc_trade_data()

    def update_position(self, data, size, price):
        print('updating Position!')
        symbol = self._symbol
        print(f'SYMBOL: {self._symbol}')
        if symbol in self.positions:
            self.positions[self._symbol].size = size
            self.positions[self._symbol].price = price
            print(size=size, price=price)
            Position(size=size, price=price)
        else:
            self.positions[self._symbol] = Position(size=size, price=price)
        print(f"Position updated for {symbol}: size={size}, price={price}")
        # self.notify(f"Position updated for {symbol}: size={size}, price={price}")

    def get_position(self, data):
        symbol = data._name
        if symbol in self.positions:
            return self.positions[symbol]
        return None

    def _execute_order(self, order, date, executed_size, executed_price):
        pos = self.getposition(order.data, clone=False)
        if pos:
            pos.update(copysign(executed_size, order.size), executed_price)
        else:
            self.update_position(order.data, executed_size, executed_price)
            pos = self.getposition(order.data, clone=False)
            if pos:
                print(f"New position created: {pos.size} {pos.price}")
                self.notify(order)

        order.execute(
            date,
            executed_size,
            executed_price,
            0, 0.0, 0.0,
            0, 0.0, 0.0,
            0.0, 0.0,
            0, 0.0)
        pos = self.getposition(order.data, clone=False)
        pos.update(copysign(executed_size, order.size), executed_price)

    def _handle_user_socket_message(self, msg):
        """https://binance-docs.github.io/apidocs/spot/en/#payload-order-update"""
        if msg['e'] == 'executionReport':
            if msg['s'] == self._store.symbol:
                for o in self.open_orders:
                    if o.binance_order['orderId'] == msg['i']:
                        if msg['X'] in [ORDER_STATUS_FILLED, ORDER_STATUS_PARTIALLY_FILLED]:
                            date = dt.datetime.fromtimestamp(msg['T'] / 1000)
                            executed_size = float(msg['l'])
                            executed_price = float(msg['L'])
                            self._execute_order(o, dt, executed_size, executed_price)
                        self._set_order_status(o, msg['X'])

                        if o.status not in [Order.Accepted, Order.Partial]:
                            self.open_orders.remove(o)
                        self.notify(o)
        elif msg['e'] == 'error':
            raise msg

    def _set_order_status(self, order, binance_order_status):
        if binance_order_status == ORDER_STATUS_CANCELED:
            order.cancel()
        elif binance_order_status == ORDER_STATUS_EXPIRED:
            order.expire()
        elif binance_order_status == ORDER_STATUS_FILLED:
            order.completed()
        elif binance_order_status == ORDER_STATUS_PARTIALLY_FILLED:
            order.partial()
        elif binance_order_status == ORDER_STATUS_REJECTED:
            order.reject()

    def _submit(self, owner, data, side, exectype, size, price):
        type = self._ORDER_TYPES.get(exectype, ORDER_TYPE_MARKET)

        binance_order = self._store.create_order(side, type, size, price)
        order = BinanceOrder(owner, data, exectype, binance_order)
        if binance_order['status'] in [ORDER_STATUS_FILLED, ORDER_STATUS_PARTIALLY_FILLED]:
            self._execute_order(
                order,
                dt.datetime.fromtimestamp(binance_order['transactTime'] / 1000),
                float(binance_order['executedQty']),
                float(binance_order['price']))
        self._set_order_status(order, binance_order['status'])
        if order.status == Order.Accepted:
            self.open_orders.append(order)
        self.notify(order)
        return order

    def cancel(self, order):
        order_id = order.binance_order['orderId']
        self._store.cancel_order(order_id)

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

    def getposition(self, data, clone=True):
        symbol = data._name  # Assuming this is the symbol name for the data feed
        try:
            pos = self.positions[symbol]
            if clone:
                pos = pos.clone()
            return pos
        except KeyError:
            print(f"Position not found for symbol: {symbol}")
            return None

    def getvalue(self, datas=None):
        self.value = self._store._value
        return self.value

    def notify(self, order):
        self.notifs.append(order)

    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            **kwargs):
        return self._submit(owner, data, SIDE_BUY, exectype, size, price)

    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             **kwargs):
        return self._submit(owner, data, SIDE_SELL, exectype, size, price)

    def load_trade_data(self, data):
        try:
            response = requests.get(url=jrr_order_history)
            response.raise_for_status()
            orders = response.text.strip().split('\n')
            orders.reverse()

            found_sell = False
            for order in orders:
                if not order.strip():  # Skip empty strings
                    continue

                try:
                    order_data = json.loads(order)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON: {order}")
                    continue

                action = order_data.get('Action')
                asset = order_data.get('Asset')

                if action == 'sell' and asset == self._store._symbol:
                    found_sell = True
                    continue

                if not found_sell and action == 'buy' and asset == self._store._symbol:
                    _amount = order_data.get(self._store.coin_refer, 0.0)
                    entry_price = order_data.get('Price', 0.0)
                    self.update_position(data, size=_amount, price=entry_price)
            # Handle the last order
            if orders and orders[0].strip():
                try:
                    last_order_data = json.loads(orders[0])
                    usdt_value = last_order_data.get('USDT', 0.0)
                    print(f"Free USDT: {usdt_value:.9f}")
                    self.stake_to_use = usdt_value
                except json.JSONDecodeError:
                    print("Error parsing the last order, resetting position state.")
                    
            return _amount, entry_price, True

        except requests.exceptions.RequestException as e:
            print(f"Error fetching trade data: {e}")

    def get_spot_positions_by_name(self, symbol=None):
        try:
            # Fetch SPOT balances
            balances = self._store.get_balance()

            open_position_amount = None
            open_position_entry = None
            position_found = False

            # Find the balance for the specified symbol
            for balance in balances:
                if balance['asset'] == symbol:
                    open_position_amount = balance['free'] + balance['locked']
                    open_position_entry = None  # SPOT balances do not have an entry price
                    position_found = True
                    break

            return open_position_amount, open_position_entry, position_found
        except BaseError as e:
            print(f"Error: {e}")
            return None, None, False