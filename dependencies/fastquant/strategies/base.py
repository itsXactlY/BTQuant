from datetime import datetime
import backtrader as bt
import requests
from fastquant.strategies.jrr_orders import *
from fastquant.strategies.pancakeswap_orders import PancakeSwapV2DirectOrderBase as _web3order
from fastquant.dontcommit import *
import json
import threading
import queue
import time
import numpy as np

# import logging
# from logging.handlers import RotatingFileHandler

from fastquant.config import (
    INIT_CASH,
    COMMISSION_PER_TRANSACTION,
    GLOBAL_PARAMS,
    BUY_PROP,
    SELL_PROP,
    SHORT_MAX,
)

class BuySellArrows(bt.observers.BuySell):
    plotlines = dict(buy=dict(marker='$\u21E7$', markersize=16.0),
                     sell=dict(marker='$\u21E9$', markersize=16.0))

class BaseStrategy(bt.Strategy):
    params = (
        ('exchange', None),
        ('account', None),
        ('asset', None),
        ('amount', None),
        ('coin', None),
        ('collateral', None),
        ('debug', False),
        ('backtest', True),
        ('is_training', None),
        ('use_stoploss', None),
        ("strategy_logging", True),
        ("periodic_logging", False),
        ("transaction_logging", True),
        ("strategy_logging", True),
        ("pnl", None),
        ("final_value", None),
        ("channel", ""),
        ("symbol", ""),
        ("allow_short", False),
        ("short_max", SHORT_MAX),
        ("add_cash_amount", 0),
        ("add_cash_freq", "M"),
        ("invest_div", True),
        ("init_cash", INIT_CASH),
        ("buy_prop", BUY_PROP),
        ("sell_prop", SELL_PROP),
        ("fractional", False),
        ("slippage", 0.001),
        ("single_position", None),
        ("commission", COMMISSION_PER_TRANSACTION),
        ("stop_loss", 0),  # Zero means no stop loss
        ("stop_trail", 0),  # Zero means no stop loss
        ("take_profit", 0.1),  # Zero means no take profit -> Default: 1.0%
        ("percent_sizer", 0.95), # # Zero means no percentage usage per buy -> Default: 95% (keep 5% to cover fees, etc.)
        ("allow_short", False),
        ("short_max", SHORT_MAX),
        ("add_cash_amount", 0),
        ("order_cooldown", 5)

    )
    
    def __init__(self, **kwargs):
        BuySellArrows(self.data0, barplot=True)
        super().__init__(**kwargs)
        self.dataclose = self.datas[0].close
        self.order = None
        self.DCA = False
        self.entry_price = None
        self.take_profit_price = None
        self.stop_loss_price = None
        self.buy_executed = False
        self.average_entry_price = None
        self.entry_prices = []
        self.sizes = []
        self.stake_to_use = None # new default set later if stays empty handed to its gunfight :<
        self.stake_to_sell = None
        self.stake = None
        self.amount = self.p.amount
        self.conditions_checked = False
                
        self.last_order_time = 0
        self.order_cooldown = self.p.order_cooldown
        self.position_count = 0
        self.position_history = []
        # Variables to track DCA and positions
        self.initial_buy_price = None
        self.buy_count = 0
        self.average_buy_price = 0

        self.init_cash = self.params.init_cash
        self.buy_prop = self.params.buy_prop
        self.sell_prop = self.params.sell_prop
        self.strategy_logging = self.params.strategy_logging
        self.periodic_logging = self.params.periodic_logging
        self.transaction_logging = self.params.transaction_logging
        self.strategy_logging = self.params.strategy_logging
        self.slippage = self.params.slippage
        self.single_position = self.params.single_position
        self.commission = self.params.commission
        self.stop_loss = self.params.stop_loss
        self.stop_trail = self.params.stop_trail
        self.take_profit = self.params.take_profit
        self.allow_short = self.params.allow_short
        self.short_max = self.params.short_max
        self.add_cash_amount = self.params.add_cash_amount
        self.percent_sizer = self.params.percent_sizer
        # Attribute that tracks how much cash was added over time
        self.total_cash_added = 0

        if self.params.backtest == False and self.p.exchange.lower() != "pancakeswap":
            # JackRabbitRelay Init
            self.exchange = self.p.exchange
            self.account = self.p.account
            self.asset = self.p.asset
            self.rabbit = JrrOrderBase()

            self.order_queue = queue.Queue()
            self.order_thread = threading.Thread(target=self.process_orders)
            self.order_thread.daemon = True
            self.order_thread.start()
        elif self.params.backtest == False:
            self.pcswap = _web3order(coin=self.p.coin, collateral=self.p.collateral)
            self.web3order_queue = queue.Queue()
            self.web3order_thread = threading.Thread(target=self.process_web3orders)
            self.web3order_thread.daemon = True
            self.web3order_thread.start()

        if self.strategy_logging and self.p.backtest:
            self.log("===Global level arguments===")
            self.log("init_cash : {}".format(self.init_cash))
            self.log("buy_prop : {}".format(self.buy_prop))
            self.log("sell_prop : {}".format(self.sell_prop))
            self.log("percent_sizer : {}".format(self.percent_sizer))
            self.log("commission : {}".format(self.commission))
            self.log("stop_loss : {}".format(self.stop_loss))
            self.log("stop_trail : {}".format(self.stop_trail))
            self.log("take_profit : {}".format(self.take_profit))
            self.log("allow_short : {}".format(self.allow_short))

        self.order_history = {
            "dt": [],
            "type": [],
            "price": [],
            "size": [],
            "order_value": [],
            "portfolio_value": [],
            "commission": [],
            "pnl": [],
        }

        self.periodic_history = {
            "dt": [],
            "portfolio_value": [],
            "cash": [],
            "size": [],
        }
        self.order_history_df = None
        self.periodic_history_df = None

        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print("%s, %s" % (dt.isoformat(), txt))

    def update_order_history(self, order):
        self.order_history["dt"].append(self.datas[0].datetime.datetime(0))
        self.order_history["type"].append("buy" if order.isbuy() else "sell")
        self.order_history["price"].append(order.executed.price)
        self.order_history["size"].append(order.executed.size)
        self.order_history["order_value"].append(order.executed.value)
        self.order_history["portfolio_value"].append(self.broker.getvalue())
        self.order_history["commission"].append(order.executed.comm)
        self.order_history["pnl"].append(order.executed.pnl)

    def update_periodic_history(self):
        self.periodic_history["dt"].append(self.datas[0].datetime.datetime(0))
        self.periodic_history["portfolio_value"].append(self.broker.getvalue())
        self.periodic_history["cash"].append(self.broker.getcash())
        self.periodic_history["size"].append(self.position.size)


    @property
    def current_price(self):
        return self.data.close[0]

    @property
    def position_value(self):
        return self.position_size * self.current_price

    @property
    def wallet_balance(self):
        return self.broker.cash

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
            if action == 'sell':
                self.last_order_time = time.time()
            else:
                self.last_order_time = time.time()        

    def process_web3orders(self):
        while True:
            order = self.web3order_queue.get()
            print(order)
            if order is None:
                break
            action, params = order
            if action == 'buy':
                print('-'*99)
                print('WEB3 BUY REQUEST')
                print('-'*99)
                try:
                    self.pcswap.send_pcs_buy_request(**params)
                except Exception as e: 
                    print(e)
            elif action == 'sell':
                print('-'*99)
                print('WEB3 SELL REQUEST')
                print('-'*99)
                try:
                    self.pcswap.send_pcs_close_request(**params)
                except Exception as e: 
                    print(e)
                self.reset_position_state()
            self.web3order_queue.task_done()

    def enqueue_web3order(self, action, **params):
        current_time = time.time()
        if current_time - self.last_order_time >= self.order_cooldown:
            self.web3order_queue.put((action, params))
            if action == 'sell':
                self.last_order_time = time.time()
            else:
                self.last_order_time = time.time()

    def buy_or_short_condition(self):
        return False
    
    def dca_or_short_condition(self):
        return False
    
    def check_stop_loss(self):
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
        if self.params.backtest == False and self.p.exchange.lower() != "pancakeswap":
            ptu()
            print(f"STUXNET INITATED FOR {self.p.exchange} ...")
            self.load_trade_data()
        elif self.params.backtest == False:
            ptu()
            print('DEX Exchange Detected - Dont chase the Rabbit.')
        

    def next(self):
        self.conditions_checked = False
        if self.params.backtest == False:
            if self.live_data == True:
                self.stake = self.stake_to_use * self.p.percent_sizer / self.dataclose # TODO figure out why no default stake is set on empty history
                
                if not self.buy_executed:
                    self.buy_or_short_condition()
                elif self.DCA == True and self.buy_executed:
                    self.sell_or_cover_condition()
                    self.dca_or_short_condition()
                elif self.DCA == False and self.buy_executed:
                    self.sell_or_cover_condition()
                    
                if self.live_data == True and self.buy_executed and self.p.debug:
                    self.print_counter += 1
                    if self.print_counter % 1 == 60: # reduce logging spam
                        print(f'| {datetime.utcnow()}\n|{"-"*99}¬\n| Position Report\n| Price: {self.data.close[0]:.9f}\n| Entry: {self.average_entry_price:.9f}\n| TakeProfit: {self.take_profit_price:.9f}\n|{"-"*99}¬')
        
        elif self.params.backtest == True:
            self.stake = self.broker.getcash() * self.p.percent_sizer / self.dataclose
            if not self.buy_executed:
                self.buy_or_short_condition()
            elif self.DCA == True and self.buy_executed:
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
    
    def reset_position_state(self):
        self.buy_executed = False
        self.entry_price = None
        self.entry_prices = []
        self.average_entry_price = None
        self.take_profit_price = None
        self.sizes = []
        self.position_count = 0
        self.stop_loss_price = 0.0

    def notify_order(self, order):
        if self.p.backtest:
            if order.status in [order.Submitted, order.Accepted]:
                return

            if order.status in [order.Completed]:
                # Update order history whenever an order is completed
                self.update_order_history(order)
                if order.isbuy():
                    self.action = "buy"
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm

                else:  # Sell
                    self.action = "sell"

                self.bar_executed = len(self)

                if self.transaction_logging:
                    self.log(
                        "%s EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f, Size: %.2f"
                        % (
                            self.action.upper(),
                            order.executed.price,
                            order.executed.value,
                            order.executed.comm,
                            order.executed.size,
                        )
                    )

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                if self.transaction_logging:
                    if not self.periodic_logging:
                        self.log("Cash %s Value %s" % (self.cash, self.value))
                    self.log("Order Canceled/Margin/Rejected")
                    self.log("Canceled: {}".format(order.status == order.Canceled))
                    self.log("Margin: {}".format(order.status == order.Margin))
                    self.log("Rejected: {}".format(order.status == order.Rejected))

            # Write down: no pending order
            self.order = None

    def notify_trade(self, trade):
        if self.p.backtest:
            if not trade.isclosed:
                return
            if self.transaction_logging:
                self.log(
                    "OPERATION PROFIT, GROSS: %.2f, NET: %.2f" % (trade.pnl, trade.pnlcomm)
                )

    def notify_cashvalue(self, cash, value):
        # Update cash and value every period
        if self.periodic_logging:
            self.log("Cash %s Value %s" % (cash, value))
        self.cash = cash
        self.value = value

    def stop(self):
        if self.p.backtest:
            # Saving to self so it's accessible later during optimization
            self.final_value = self.broker.getvalue()
            # Note that PnL is the final portfolio value minus the initial cash balance minus the total cash added
            self.pnl = round(self.final_value - self.init_cash - self.total_cash_added, 2)
            if self.strategy_logging:
                self.log("Final Portfolio Value: {}".format(self.final_value))
                self.log("Final PnL: {}".format(self.pnl))
            self.order_history_df = pd.DataFrame(self.order_history)
            self.periodic_history_df = pd.DataFrame(self.periodic_history)

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

import backtrader.utils as btu
class CustomPandasData(bt.feeds.PandasData):
    lines = ('target',)
    params = (
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('target', -1),
    )
    
    def _load(self):
        if self._idx >= len(self.p.dataname):
            return False

        for col in self.p.dataname.columns:
            if col == 'TimestampStart':
                self.lines.datetime[0] = btu.date2num(self.p.dataname[col].iloc[self._idx])
            elif col == 'Open':
                self.lines.open[0] = self.p.dataname[col].iloc[self._idx]
            elif col == 'High':
                self.lines.high[0] = self.p.dataname[col].iloc[self._idx]
            elif col == 'Low':
                self.lines.low[0] = self.p.dataname[col].iloc[self._idx]
            elif col == 'Close':
                self.lines.close[0] = self.p.dataname[col].iloc[self._idx]
            elif col == 'Volume':
                self.lines.volume[0] = self.p.dataname[col].iloc[self._idx]
            elif col == 'target':
                self.lines.target[0] = self.p.dataname[col].iloc[self._idx]

        self._idx += 1
        return True

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
