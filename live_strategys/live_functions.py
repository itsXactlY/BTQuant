from datetime import datetime
import backtrader as bt
import requests
from live_strategys.jrr_orders import *
from dontcommit import *
import json

import threading
import queue
import time

import logging
# Configure logging
log_file = 'BaseStrategy.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

trade_log_file = f'BaseStrategy_Trade_Monitor.log'
trade_logger = logging.getLogger('TradeMonitor')
trade_handler = logging.FileHandler(trade_log_file)
trade_handler.setLevel(logging.DEBUG)
trade_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
trade_handler.setFormatter(trade_formatter)
trade_logger.addHandler(trade_handler)

class BaseStrategy(bt.Strategy):
    params = (
        ("printlog", False),
        ('debug', False),
        ('take_profit', 0),
        ('exchange', None),
        ('account', None),
        ('asset', None),
        ('amount', None),
        ('coin', None),
        ('collateral', None),
        ("backtest", None)
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataclose = self.datas[0].close
        self.order = None
        self.DCA = False
        self.entry_price = None
        self.take_profit_price = None
        self.buy_executed = False
        self.average_entry_price = None
        self.entry_prices = []
        self.sizes = []
        self.print_counter = 0
        self.stake_to_use = None # new default set later if stays empty handed to its gunfight :<
        self.stake_to_sell = None
        self.stake = None
        self.amount = self.p.amount
        self.conditions_checked = False
        self.order_queue = queue.Queue()
        self.order_thread = threading.Thread(target=self.process_orders)
        self.order_thread.daemon = True
        self.order_thread.start()
        self.last_order_time = 0
        self.order_cooldown = 5
        if self.params.backtest == False:
            # JackRabbitRelay Init
            self.exchange = self.p.exchange
            self.account = self.p.account
            self.asset = self.p.asset
            self.rabbit = JrrOrderBase()

    def process_orders(self):
        while True:
            order = self.order_queue.get()
            print(order)
            if order is None:
                break
            action, params = order
            if action == 'buy':
                print('-'*99)
                print('BUY REQUEST')
                print('-'*99)
                self.rabbit.send_jrr_buy_request(**params)
            elif action == 'sell':
                print('-'*99)
                print('SELL REQUEST')
                print('-'*99)
                self.rabbit.send_jrr_close_request(**params)
                self.reset_position_state()
            self.order_queue.task_done()

    def enqueue_order(self, action, **params):
        current_time = time.time()
        if current_time - self.last_order_time >= self.order_cooldown:
            self.order_queue.put((action, params))
            self.last_order_time = current_time          

    def stop(self):
        self.order_queue.put(None)
        self.order_thread.join()

    def buy_or_short_condition(self):
        return False
    
    def dca_or_short_condition(self):
        return False

    def sell_or_cover_condition(self):
        return False

    def load_trade_data(self):
        try:
            file_path = f"/home/JackrabbitRelay2/Data/Mimic/{self.account}.history"
            with open(file_path, 'r') as file:
                orders = file.read().strip().split('\n')
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

                if action == 'sell' and asset == self.asset:
                    found_sell = True
                    continue

                if not found_sell and action == 'buy' and asset == self.asset:
                    # _amount = order_data.get(self.p.coin, 0.0)
                    entry_price = order_data.get('Price', 0.0)
                    self.entry_prices.append(entry_price)
                    self.sizes.append(self.p.amount)

            if self.entry_prices and self.sizes:
                print(f"Loaded {len(self.entry_prices)} buy orders after the last sell.")
                self.calc_averages()
                self.buy_executed = True
                self.entry_price = self.average_entry_price
            else:
                if found_sell:
                    print("No buy orders found after the last sell.")
                else:
                    print("No executed sell orders found.")
            
            if orders and orders[0].strip():
                try:
                    last_order_data = json.loads(orders[0])
                    usdt_value = last_order_data.get('USDT', 0.0)
                    print(f"Free USDT: {usdt_value:.9f}")
                    self.stake_to_use = usdt_value
                except json.JSONDecodeError:
                    print("Error parsing the last order, resetting position state.")
                    self.reset_position_state()
                    self.stake_to_use = 1000.0 # new Default :<
            else:
                self.reset_position_state()

        except FileNotFoundError:
            print(f"History file not found for account {self.account}.")
            self.reset_position_state()
            self.stake_to_use = 1000.0  # Default stake when file is not found
        except PermissionError:
            print(f"Permission denied when trying to access the history file for account {self.account}.")
            self.reset_position_state()
            self.stake_to_use = 1000.0  # Default stake when permission is denied
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trade data: {e}")
            self.reset_position_state()
            self.stake_to_use = 1000.0  # Default stake when there's a request exception
        except Exception as e:
            print(f"Unexpected error occurred while loading trade data: {e}")
            self.reset_position_state()
            self.stake_to_use = 1000.0  # Default stake for any other unexpected errors
            
    def calc_averages(self):
        total_value = sum(entry_price * size for entry_price, size in zip(self.entry_prices, self.sizes))
        total_size = sum(self.sizes)
        if total_size:
            self.average_entry_price = total_value / total_size
            self.take_profit_price = self.average_entry_price * (1 + self.params.take_profit / 100)
        else:
            self.average_entry_price = None
            self.take_profit_price = None

        if self.entry_prices:
            if self.p.backtest == False:
                print(f"Calculated average_entry_price: {self.average_entry_price:.9f} and take_profit_price: {self.take_profit_price:.9f}")
            self.buy_executed = True
        else:
            print("No positions exist. Entry and Take Profit prices reset to None")

    def start(self):
        if self.params.backtest == False:
            ptu()
            print('STUXNET INITATED...')
            self.load_trade_data()

    def next(self):
        if self.params.backtest == False:
            if self.order:
                return
            if self.live_data == True:
                self.stake = self.stake_to_use * self.p.percent_sizer / self.dataclose # TODO figure out why no default stake is set on empty history
                
                if not self.buy_executed:
                    self.buy_or_short_condition()
                elif self.DCA == True:
                    self.sell_or_cover_condition()
                    self.dca_or_short_condition()
                elif self.DCA == False and self.buy_executed:
                    self.sell_or_cover_condition()
                    
                if self.live_data == True and self.buy_executed and self.p.debug:
                    self.print_counter += 1
                    if self.print_counter % 1 == 60: # reduce logging spam
                        if self.p.take_profit == 0: self.take_profit_price = 0.00
                        print(f'| {datetime.utcnow()}\n|{'-'*99}¬\n| Position Report\n| Price: {self.data.close[0]:.9f}\n| Entry: {self.average_entry_price:.9f}\n| TakeProfit: {self.take_profit_price:.9f}\n|{'-'*99}¬')
        
        elif self.params.backtest == True:
            self.stake = self.broker.getcash() * self.p.percent_sizer / self.dataclose
            
            if not self.buy_executed:
                    self.buy_or_short_condition()
            elif self.DCA == True:
                self.sell_or_cover_condition()
                self.dca_or_short_condition()
            elif self.DCA == False and self.buy_executed:
                self.sell_or_cover_condition()

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg= 'Data Status: {}'.format(data._getstatusname(status))
        print(dt,dn,msg)
        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False

    def log(self, txt, dt=None):
        if self.p.printlog:
            if dt is not None:
                if isinstance(dt, float):
                    dt_str = f"dt:.2f"
                else:
                    dt_str = dt.isoformat()
                print(f"{dt_str} {txt}")
            else:
                print(txt)
    
    def log_trade(self, message):
        trade_logger.info(message)

    def reset_position_state(self):
        self.buy_executed = False
        self.entry_price = None
        self.entry_prices = []
        self.average_entry_price = None
        self.take_profit_price = None
        self.sizes = []
        self.stop_loss_price = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

class MyPandasData(bt.feeds.PandasData):
    params = (
        ('Datetime', 'datetime'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None)
    )

import numpy as np
class CustomSQN(bt.Analyzer):
    params = (('compressionFactor', 1e6),)

    def __init__(self):
        super(CustomSQN, self).__init__()
        self.pnl = []
        self.count = 0

    def notify_trade(self, trade):
        if trade.status == trade.Closed:
            self.pnl.append(trade.pnlcomm / self.p.compressionFactor)
            self.count += 1

    def stop(self):
        if self.count > 1:
            pnl_array = np.array(self.pnl)
            sqrt_n = np.sqrt(len(pnl_array))
            sqn = sqrt_n * np.mean(pnl_array) / np.std(pnl_array, ddof=1)
            self.rets['sqn'] = sqn
            self.rets['trades'] = self.count
        else:
            self.rets['sqn'] = 0.0
            self.rets['trades'] = self.count
