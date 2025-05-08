from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
import datetime
from backtrader import BrokerBase, Order
from backtrader.utils.py3 import queue
from backtrader.stores.ccxtstore import CCXTStore, CCXTOrder

class SimpleOrder:
    """A simplified order class that doesn't rely on BackTrader internals"""
    
    def __init__(self, symbol, side, size, price, order_id=None):
        self.symbol = symbol
        self.side = side
        self.size = float(size)
        self.price = float(price)
        self.id = order_id or f'manual-{int(time.time())}'
        self.status = 'closed'
        self.executed = self.size
        self.executed_price = self.price
        self.executed_value = self.executed * self.executed_price
        self.created = datetime.datetime.now()
        
        # Create a simplified ccxt_order structure for compatibility
        self.ccxt_order = {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'amount': self.size,
            'price': self.price,
            'cost': self.size * self.price,
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.datetime.now().isoformat(),
            'status': 'closed',
            'type': 'manual',
            'info': {'manual_entry': True}
        }
        
    def __str__(self):
        return f"{self.side.upper()} {self.size} {self.symbol} @ {self.price}"

class SimpleTrade(SimpleOrder):
    """A simple trade record from exchange history"""
    
    @classmethod
    def from_exchange_trade(cls, trade):
        """Create SimpleTrade object from exchange trade data"""
        return cls(
            symbol=trade['symbol'],
            side=trade['side'],
            size=float(trade['amount']),
            price=float(trade['price']),
            order_id=trade.get('order'),
            timestamp=trade['timestamp']
        )
    
    def __init__(self, symbol, side, size, price, order_id=None, timestamp=None):
        super().__init__(symbol, side, size, price, order_id)
        self.timestamp = timestamp or int(time.time() * 1000)
        
    def __str__(self):
        dt = datetime.datetime.fromtimestamp(self.timestamp / 1000)
        dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        return f"{self.side.upper()} {self.size} {self.symbol} @ {self.price} ({dt_str})"

class CCXTBroker(BrokerBase):
    '''Broker implementation for CCXT cryptocurrency trading library.
    This class maps the orders/positions from CCXT to the
    internal API of ``backtrader``.
    '''
    order_types = {Order.Market: 'market',
                   Order.Limit: 'limit',
                   Order.Stop: 'stop',
                   Order.StopLimit: 'stop limit'}
    
    def __init__(self, exchange, currency, config, retries=5):
        super(CCXTBroker, self).__init__()
        self.store = CCXTStore.get_store(exchange, config, retries)
        self.currency = currency
        self.notifs = queue.Queue()  # holds orders which are notified
        self.orders = {}  # maps order_id to order objects
        self.executed_orders = []  # list of executed orders
        
    def getcash(self):
        return self.store.getcash(self.currency)
        
    def getvalue(self, datas=None):
        return self.store.getvalue(self.currency)
        
    def get_notification(self):
        try:
            return self.notifs.get(False)
        except queue.Empty:
            return None
            
    def notify(self, order):
        self.notifs.put(order)
        
    def getposition(self, data):
        currency = data.symbol.split('/')[0]
        return self.store.getposition(currency)
        
    def get_value(self, datas=None, mkt=False, lever=False):
        return self.store.getvalue(self.currency)
        
    def get_cash(self):
        return self.store.getcash(self.currency)
        
    def _submit(self, owner, data, exectype, side, amount, price, params):
        order_type = self.order_types.get(exectype)
        _order = self.store.create_order(symbol=data.symbol, order_type=order_type, side=side,
                                        amount=amount, price=price, params=params)
        order = CCXTOrder(owner, data, amount, _order)
        self.orders[order.ccxt_order['id']] = order
        self.notify(order)
        return order
        
    def buy(self, owner, data, size, price=None, plimit=None,
            exectype=None, valid=None, tradeid=0, oco=None,
            trailamount=None, trailpercent=None,
            **kwargs):
        return self._submit(owner, data, exectype, 'buy', size, price, kwargs)
        
    def sell(self, owner, data, size, price=None, plimit=None,
             exectype=None, valid=None, tradeid=0, oco=None,
             trailamount=None, trailpercent=None,
             **kwargs):
        return self._submit(owner, data, exectype, 'sell', size, price, kwargs)
        
    def cancel(self, order):
        result = self.store.cancel_order(order)
        if order.ccxt_order['id'] in self.orders:
            del self.orders[order.ccxt_order['id']]
        return result
        
    def get_orders_open(self, safe=False):
        return self.store.fetch_open_orders()
    
    def get_order_status(self, order):
        """Fetch current status of an order from the exchange"""
        return self.store.fetch_order_status(order.ccxt_order['id'], order.symbol)
    
    def check_orders(self):
        """Update status of all tracked orders"""
        for order_id, order in list(self.orders.items()):
            try:
                status = self.get_order_status(order)
                if status == 'closed' or status == 'filled':
                    # Order is filled
                    self.executed_orders.append(order)
                    del self.orders[order_id]
                    # Update the order with filled information and notify
                    order.ccxt_order['status'] = status
                    self.notify(order)
                elif status == 'canceled' or status == 'expired' or status == 'rejected':
                    # Order is no longer active
                    del self.orders[order_id]
                    order.ccxt_order['status'] = status
                    self.notify(order)
            except Exception as e:
                print(f"Error checking order {order_id}: {e}")

    def load_initial_positions(self, data):
        """Load initial positions from exchange for startup"""
        symbol = data.symbol
        currency = symbol.split('/')[0]
        
        # Get current position
        position = self.store.getposition(currency)
        
        # If we have a position, create a manual entry
        if position > 0:
            print(f"Found position of {position} {currency}")
            
            try:
                # Create a simplified manual order to represent our position
                ticker = self.store.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                manual_order = SimpleOrder(
                    symbol=symbol, 
                    side='buy', 
                    size=position, 
                    price=current_price
                )
                
                # Store for reference
                self.executed_orders.append(manual_order)
                
                print(f"Manually initialized position: {position} {currency} at reference price {current_price}")
                return [manual_order]
                    
            except Exception as e:
                print(f"Error setting up manual position: {e}")
                import traceback
                traceback.print_exc()
        
        return []

    def get_position_info(self, data):
        """Get current position info for a symbol"""
        symbol = data.symbol
        currency = symbol.split('/')[0]
        
        position_size = self.store.getposition(currency)
        
        if position_size > 0:
            # For a current position, use current price as reference
            try:
                ticker = self.store.exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                return {
                    'size': position_size,
                    'price': current_price,
                    'value': position_size * current_price
                }
            except Exception as e:
                print(f"Error fetching ticker: {e}")
                return {
                    'size': position_size,
                    'price': 0,
                    'value': 0
                }
        
        return {
            'size': 0,
            'price': 0,
            'value': 0
        }

# Custom order class to handle manual positions
class ManualPositionOrder(CCXTOrder):
    def __init__(self, owner, data, size, ccxt_order):
        self.owner = owner
        self.data = data
        self.size = float(size)
        self.ccxt_order = ccxt_order
        self.symbol = data.symbol if data else ccxt_order.get('symbol')
        self.ordtype = self.Buy if ccxt_order['side'] == 'buy' else self.Sell
        self.status = Order.Completed
        self.executed = self.size
        self.executed_price = float(ccxt_order.get('price', 0))
        self.executed_value = self.executed * self.executed_price
        self.created = datetime.datetime.now()
        
        # Skip the original __init__ which tries to access data.close[0]
        Order.__init__(self)