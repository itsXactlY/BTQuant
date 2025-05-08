from datetime import datetime
import backtrader as bt
import traceback
from backtrader.brokers.jrrbroker import *
from backtrader.brokers.pancakeswap_orders import PancakeSwapV2DirectOrderBase as _web3order
from backtrader.dontcommit import *
import json
import threading
import queue
import time
import numpy as np
import shutil
import os
import uuid
import sys


# TODO fix order pickup & calculations once again on restart

order_lock = threading.Lock()

class BuySellArrows(bt.observers.BuySell):
    def next(self):
        super().next()
        
        if self.lines.buy[0]:
            self.lines.buy[0] -= self.data.low[0] * 0.02
            
        if self.lines.sell[0]:
            self.lines.sell[0] += self.data.high[0] * 0.02
            
    plotlines = dict(
        buy=dict(marker='$\u21E7$', markersize=8.0),
        sell=dict(marker='$\u21E9$', markersize=8.0)
    )

INIT_CASH = 100_000.0

class BaseStrategy(bt.Strategy):
    params = (
        ('init_cash', INIT_CASH),
        ('exchange', None),
        ('account', None),
        ('asset', None),
        ('amount', None),
        ('coin', None),
        ('collateral', None),
        ('debug', False),
        ('backtest', None),
        ('is_training', None),
        ('use_stoploss', None),
        ("pnl", None),
        ("final_value", None),
        ("channel", ""),
        ("symbol", ""),
        ("stop_loss", 0),
        ("stop_trail", 0),
        ("take_profit", 0),
        ("percent_sizer", 0),
        ("order_cooldown", 0),
        ("enable_alerts", False),
        ("alert_channel", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.p.backtest == True:
            BuySellArrows(self.data0, barplot=True)
        self.dataclose = self.datas[0].close
        self.order = None
        self.DCA = False
        self.entry_price = None
        self.take_profit_price = None
        self.first_entry_price = None
        self.stop_loss_price = None
        self.buy_executed = None
        self.average_entry_price = None
        self.entry_prices = []
        self.sizes = []
        self.stake_to_use = None
        self.stake_to_sell = None
        self.stake = None
        self.amount = self.p.amount
        self.conditions_checked = None
        self.print_counter = 0
                
        self.last_order_time = 0
        self.order_cooldown = self.p.order_cooldown
        self.position_count = 0
        self.position_history = []
        # Variables to track DCA and positions
        self.initial_buy_price = None
        self.buy_count = 0
        self.average_buy_price = 0

        self.init_cash = self.params.init_cash
        self.total_cash_added = 0
        self.total_pnl = 0.0
        self.total_wins = 0
        self.total_losses = 0
        self.total_trades = 0
        self.win_rate = 0
        self.final_value = 0.0

        self.short_entry_prices = []
        self.short_sizes = []
        self.average_short_price = None
        self.first_short_entry_price = None
        self.short_take_profit_price = None
        self.short_average_entry_price = 0
        self.short_executed = False
        self.short_count = 0


        if self.params.backtest == False:
            self.init_live_trading()
        

        # temp delete me
        self.account = self.p.account
        self.asset = self.p.asset

    def init_live_trading(self):
        """Initialize live trading components based on exchange type"""
        if self.p.exchange.lower() == "pancakeswap":
            self._init_pancakeswap()
        elif self.p.exchange.lower() == "raydium":
            self._init_raydium()
        elif self.p.exchange.lower() == "mimic":
            self._init_standard_exchange()
        else:
            print('Using CCXT Exchange - nothing todo here')

    def _init_alert_system(self, coin_name=".__!_"):
        """Initialize alert system with Telegram and Discord services if enabled"""
        if not self.p.enable_alerts:
            print("Alert system disabled (not enabled via configuration)")
            return None

        try:
            base_session_file = ".base.session"
            new_session_file = f"{coin_name}_{uuid.uuid4().hex}.session"

            if not os.path.exists(base_session_file):
                raise FileNotFoundError(f"Base session file '{base_session_file}' not found.")
            shutil.copy(base_session_file, new_session_file)
            print(f"✅ Copied base session to {new_session_file}")

            self.alert_loop = asyncio.new_event_loop()

            self.telegram_service = TelegramService(
                api_id=telegram_api_id,
                api_hash=telegram_api_hash,
                session_file=new_session_file,
                channel_id=self.p.alert_channel or telegram_channel_debug
            )

            async def init_services():
                await self.telegram_service.initialize(loop=self.alert_loop)
                return AlertManager(
                    [self.telegram_service, DiscordService(discord_webhook_url)],
                    loop=self.alert_loop
                )

            def run_alert_loop():
                asyncio.set_event_loop(self.alert_loop)
                self.alert_manager = self.alert_loop.run_until_complete(init_services())
                self.alert_loop.run_forever()

            self.alert_thread = threading.Thread(target=run_alert_loop, daemon=True)
            self.alert_thread.start()

            time.sleep(2)
            print("✅ Alert system initialized successfully")
            return self.alert_manager

        except Exception as e:
            print(f"❌ Error initializing alert system: {str(e)}")
            return None

    def _init_standard_exchange(self):
        """Initialize standard exchange trading with JackRabbitRelay"""
        # Initialize the alert system only if alerts are enabled
        alert_manager = self._init_alert_system()
        time.sleep(1)

        self.exchange = self.p.exchange
        self.account = self.p.account
        self.asset = self.p.asset
        self.rabbit = JrrOrderBase(alert_manager=alert_manager)

        self.order_queue = queue.Queue()
        self.order_thread = threading.Thread(target=self.process_orders)
        self.order_thread.daemon = True
        self.order_thread.start()

    def send_alert(self, message: str):
        """Helper method to safely send alerts if enabled"""
        if self.p.enable_alerts and hasattr(self, 'alert_manager') and self.alert_manager is not None:
            try:
                self.alert_manager.send_alert(message)
            except Exception as e:
                print(f"Error sending alert: {str(e)}")
        else:
            pass

    def _init_pancakeswap(self):
        """Initialize PancakeSwap trading"""
        self.pcswap = _web3order(coin=self.p.coin, collateral=self.p.collateral)
        self.web3order_queue = queue.Queue()
        self.web3order_thread = threading.Thread(target=self.process_web3orders)
        self.web3order_thread.daemon = True
        self.web3order_thread.start()

    def _init_raydium(self):
        """Initialize Raydium trading"""
        print('Raydium integration not implemented yet')
        raise NotImplementedError("Raydium trading not yet implemented")


    def process_orders(self):
        while True:
            order = self.order_queue.get()
            if order is None:
                break
            action, params = order
            if action == 'buy':
                self.rabbit.send_jrr_buy_request(**params)
            elif action == 'sell':
                self.rabbit.send_jrr_close_request(**params)
                self.reset_position_state()
            # elif action == 'short':
            #     self.rabbit.send_jrr_short_request(**params)
            # elif action == 'cover':
            #     self.rabbit.send_jrr_cover_request(**params)
                self.reset_position_state()
            self.order_queue.task_done()

    def enqueue_order(self, action, **params):
        with order_lock:
            current_time = time.time()
            if current_time - self.last_order_time >= self.order_cooldown:
                self.order_queue.put((action, params))
                self.last_order_time = current_time
                return True
        return False

    
    def process_web3orders(self):
        while True:
            order = self.web3order_queue.get()
            print(order)
            if order is None:
                break
            action, params = order
            if action == 'buy':
                try:
                    self.pcswap.send_pcs_buy_request(**params)
                except Exception as e: 
                    print(e)
            elif action == 'sell':
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

    def calculate_position_size(self):
        """Calculate the position size based on available USDT and current price"""
        if self.p.exchange.lower() == 'binance':
            min_order_value = 5.50
        elif self.p.exchange.lower() == 'mexc':
            min_order_value = 1.10
        else:
            min_order_value = 10

        usdt_to_use = self.stake_to_use * self.p.percent_sizer

        if hasattr(self, 'dataclose') and len(self.dataclose) > 0 and self.dataclose[0] > 0:
            self.amount = usdt_to_use / self.dataclose[0]
            order_value = self.amount * self.dataclose[0]

            if order_value < min_order_value:
                self.amount = min_order_value / self.dataclose[0]
                usdt_to_use = min_order_value
                print(f"Adjusted to minimum order value: ${min_order_value}")
            
            self.amount = round(self.amount, 8)
            self.usdt_amount = round(self.amount * self.dataclose[0], 2)
            
            print(f"Calculated position size: {self.amount} units worth {self.usdt_amount:.2f} USDT")
        else:
            self.amount = min_order_value / 1000
            self.usdt_amount = min_order_value
            print(f"No price data available. Using default amount: {self.amount}")
        
        return self.amount

    '''
    Keep the Average Entry price(s) for later FUTURES trading usage - reworked version below for SPOT market
    def calc_averages(self):
        _amount = [price * size for price, size in zip(self.entry_prices, self.sizes)]
        total_value = sum(_amount)
        total_size = sum(self.sizes)
        
        if self.p.debug:
            print(f"Debug :: amount of price×size: {_amount}")
            print(f"Debug :: Total value: {total_value}, Total size: {total_size}")
        
        if total_size:
            self.average_entry_price = total_value / total_size
            self.take_profit_price = self.average_entry_price * (1 + self.params.take_profit / 100)
        else:
            self.average_entry_price = None
            self.take_profit_price = None
            
        if self.entry_prices:
            if self.p.backtest == False and self.p.debug:
                print(f"Calculated average_entry_price: {self.average_entry_price:.9f} and take_profit_price: {self.take_profit_price:.9f}")
            self.buy_executed = True
        else:
            print("No positions exist. Entry and Take Profit prices reset to None")'''

    def calc_averages(self):
        _amount = [price * size for price, size in zip(self.entry_prices, self.sizes)]
        total_value = sum(_amount)
        total_size = sum(self.sizes)
        
        if self.p.debug:
            print(f"Debug :: amount of price×size: {_amount}")
            print(f"Debug :: Total value: {total_value}, Total size: {total_size}")
        
        if total_size:
            self.average_entry_price = total_value / total_size
            
            if not hasattr(self, 'first_entry_price') or self.first_entry_price is None:
                self.first_entry_price = self.entry_prices[0] if self.entry_prices else None
        
            if self.first_entry_price:
                self.take_profit_price = self.first_entry_price * (1 + self.params.take_profit / 100)
            else:
                self.take_profit_price = self.average_entry_price * (1 + self.params.take_profit / 100)
        else:
            self.average_entry_price = None
            self.take_profit_price = None
            self.first_entry_price = None
        
        if self.entry_prices:
            if self.p.backtest == False and self.p.debug:
                print(f"Calculated average_entry_price: {self.average_entry_price:.9f}")
                print(f"Using first entry price for take profit: {self.first_entry_price:.9f}")
                print(f"Take profit price: {self.take_profit_price:.9f}")
            self.buy_executed = True
        else:
            print("No positions exist. Entry and Take Profit prices reset to None")

    def calc_short_averages(self):
        _amount = [price * size for price, size in zip(self.short_entry_prices, self.short_sizes)]
        total_value = sum(_amount)
        total_size = sum(self.short_sizes)

        if self.p.debug:
            print(f"Debug :: SHORT amount of price×size: {_amount}")
            print(f"Debug :: SHORT total value: {total_value}, total size: {total_size}")

        if total_size:
            self.average_short_price = total_value / total_size

            if not self.first_short_entry_price:
                self.first_short_entry_price = self.short_entry_prices[0] if self.short_entry_prices else None

            if self.first_short_entry_price:
                self.short_take_profit_price = self.first_short_entry_price * (1 - self.params.take_profit / 100)
            else:
                self.short_take_profit_price = self.average_short_price * (1 - self.params.take_profit / 100)

            if self.p.backtest == False and self.p.debug:
                print(f"SHORT average entry: {self.average_short_price:.9f}")
                print(f"SHORT first entry: {self.first_short_entry_price:.9f}")
                print(f"SHORT take profit: {self.short_take_profit_price:.9f}")
            self.short_executed = True
        else:
            self.average_short_price = None
            self.short_take_profit_price = None
            self.first_short_entry_price = None
            print("No short positions exist. Entry and TP prices reset.")

    def load_trade_data(self):
        try:
            if self.p.exchange.lower() == 'mimic':
                file_path = f"/home/JackrabbitRelay2/Data/Mimic/{self.account}.history"
                if sys.platform != "win32":
                    os.sync()
                with open(file_path, 'r') as file:
                    orders = [line for line in file.read().strip().split('\n') if line.strip()]
                
                if self.p.debug:
                    print("Debug :: Processing order history")
                
                for order_str in reversed(orders):
                    try:
                        order_data = json.loads(order_str)
                        if self.p.debug:
                            print(f"Debug :: Order data: {order_data}")
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON: {order_str}")
                        continue
                        
                    action = order_data.get('Action')
                    asset = order_data.get('Asset')
                    
                    if self.p.debug:
                        print(f"Debug :: Checking order - Action: {action}, Asset: {asset}")
                    
                    if action == 'sell' and asset == self.asset:
                        if self.p.debug:
                            print("Debug :: Found sell order - breaking")
                        break
                        
                    if action == 'buy' and asset == self.asset:
                        entry_price = order_data.get('Price', 0.0)
                        amount = order_data.get('Amount', 0.0)
                        
                        if self.p.debug:
                            print(f"Debug :: Adding buy order - Price: {entry_price}, Amount: {amount}")
                        
                        self.entry_prices.append(entry_price)
                        self.sizes.append(amount)
                        self.buy_executed = True
                        self.DCA = True
                        print(f"STATE AFTER LOAD: buy_executed={self.buy_executed}, DCA={self.DCA}")
                
                if self.p.debug:
                    print(f"Debug :: Loaded entries - Prices: {self.entry_prices}, Sizes: {self.sizes}")
                
                '''
                Keep in place for FUTURES trading rework
                if self.entry_prices and self.sizes:
                    print(f"Loaded {len(self.entry_prices)} buy orders after the last sell")
                    self.calc_averages()
                    self.entry_price = self.average_entry_price
                    self.buy_executed = True'''
                if self.entry_prices and self.sizes:
                    print(f"Loaded {len(self.entry_prices)} buy orders after the last sell")
                    self.first_entry_price = self.entry_prices[0]
                    self.calc_averages()
                    self.entry_price = self.average_entry_price
                    self.buy_executed = True
                else:
                    print("No buy orders found after the last sell")
                if orders and orders[-1].strip():
                    try:
                        last_order_data = json.loads(orders[-1])
                        usdt_value = last_order_data.get('USDT', 0.0)
                        print(f"Free USDT: {usdt_value:.9f}")
                        self.stake_to_use = usdt_value
                        print(f"Last modified: {os.path.getmtime(file_path)}")
                    except json.JSONDecodeError:
                        print("Error parsing the last order, resetting position state")
                        self.stake_to_use = 1000.0
                        self.reset_position_state()
                else:
                    self.reset_position_state()
                    self.stake_to_use = 1000.0
            elif self.p.exchange.lower() != 'mimic':
                cash = self.broker.getcash()
                self.stake_to_use = cash
                print(f"Available USDT: {self.stake_to_use}")
                
                for data in self.datas:
                    symbol = data.symbol
                    position_info = self.broker.get_position_info(data)
                    
                    if position_info['size'] > 0:
                        # We have a position, log it
                        print(f"Found existing position: {position_info['size']} units at reference price {position_info['price']}")
                        
                        # Get the actual entry price from trade history
                        entry_info = self.broker.get_entry_price(symbol)
                        
                        if entry_info and entry_info['trade']:
                            entry_trade = entry_info['trade']
                            self.entry_price = entry_trade.price
                            self.entry_size = entry_trade.size
                            self.entry_time = datetime.fromtimestamp(entry_trade.timestamp / 1000)
                            
                            print(f"Found entry trade: {entry_trade}")
                            print(f"Entry price: {self.entry_price}")
                            self.average_entry_price = entry_trade.price
                            self.take_profit_price = entry_trade.price * (1 + self.params.take_profit / 100)
                            print(f"Takeprofit price: {self.take_profit_price}")
                            # self.calc_averages()
                            self.buy_executed = True
                            self.DCA = True
                            
        except Exception as e:
            print(f"Unexpected error occurred while loading trade data: {e}")
            traceback.print_exc()
            self.reset_position_state()
            self.stake_to_use = 1000.0
    
    def start(self):
        # from backtrader.dontcommit import ptu
        # from backtrader.brokers.jrrbroker import 
        if self.params.backtest == False and self.p.exchange.lower() != "pancakeswap":
            # ptu()
            print(f"BTQuant initialized for {self.p.exchange}")
            self.load_trade_data()
        elif self.params.backtest == False:
            # ptu()
            print('DEX Exchange Detected - Dont chase the Rabbit.')


    def next(self):
        self.conditions_checked = False
        if self.params.backtest == False and self.live_data == True:

            if self.p.debug:
                print(f"Debug :: live_data={getattr(self, 'live_data', False)}, buy_executed={self.buy_executed}, DCA={self.DCA}, print_counter={self.print_counter}")
                print(f"NEXT STATE CHECK: buy_executed={self.buy_executed}, DCA={self.DCA}")

            if self.buy_executed and self.p.debug:
                self.print_counter += 1
                if self.print_counter % 10 == 0:
                    print(f'| {datetime.now()}'
                        f'\n|{"-"*99}¬'
                        f'\n| Position Report'
                        f'\n| Price: {self.data.close[0]:.9f}'
                        f'\n| Entry: {self.average_entry_price:.9f}'
                        f'\n| TakeProfit: {self.take_profit_price:.9f}'
                        f'\n|{"-"*99}¬')
                
            if not self.buy_executed:
                if self.buy_or_short_condition():
                    return
            elif self.DCA == True and self.buy_executed:
                if self.sell_or_cover_condition():
                    return
                self.dca_or_short_condition()
            elif self.DCA == False and self.buy_executed:
                self.sell_or_cover_condition()


        elif self.params.backtest == True:
            self.stake = self.broker.getcash() * self.p.percent_sizer / self.dataclose
            if not self.buy_executed:
                if self.buy_or_short_condition():
                    return
            elif self.DCA == True and self.buy_executed:
                if self.sell_or_cover_condition():
                    return
                if self.broker.getcash() < 10.0:
                    print('Rejected Margin - decrease percent sizer or increase DCA deviation')
                    return
                self.dca_or_short_condition()
            elif self.DCA == False and self.buy_executed:
                if self.broker.getcash() < 10.0:
                    print('Rejected Margin - decrease percent sizer or increase DCA deviation')
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
        if self.p.debug:
            print("RESET CALLED FROM:", traceback.extract_stack()[-2][2])
        self.buy_executed = False
        self.entry_price = None
        self.entry_prices = []
        self.average_entry_price = None
        self.take_profit_price = None
        self.first_entry_price = None
        self.sizes = []
        self.position_count = 0
        self.stop_loss_price = 0.0
        
        if self.p.debug:
            self.current_cycle_buys = 0
            self.position_start_time = None
            self.total_buys = 0
            self.max_buys_per_cycle = 0
            self.trade_cycles = 0
            self.total_profit_usd = 0
            self.last_profit_usd = 0

    def reset_short_position_state(self):
        self.short_executed = False
        self.short_entry_prices.clear()
        self.short_sizes.clear()
        self.first_short_entry_price = None
        self.short_average_entry_price = 0

    def notify_order(self, order):
        if self.p.backtest:
            if order.status in [order.Submitted, order.Accepted]:
                # return

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

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                if self.transaction_logging:
                    if not self.periodic_logging:
                        print("Cash %s Value %s" % (self.cash, self.value))
                    print("Order Canceled/Margin/Rejected")
                    print("Canceled: {}".format(order.status == order.Canceled))
                    print("Margin: {}".format(order.status == order.Margin))
                    print("Rejected: {}".format(order.status == order.Rejected))
            self.order = None

    def notify_trade(self, trade):
        # Only process closed trades
        if trade.isclosed:
            self.total_trades += 1
            self.total_pnl += trade.pnl
            # Check if it's a win or a loss
            if trade.pnl > 0:
                self.total_wins += 1
            else:
                self.total_losses += 1
            
            # Calculate win rate
            self.win_rate = (self.total_wins / self.total_trades) * 100 if self.total_trades > 0 else 0

    def notify_cashvalue(self, cash, value):
        self.cash = cash
        self.value = value

    def stop(self):
        if self.p.backtest:
            self.final_value = self.broker.getvalue()

            # Print the backtest summary (optional)
            print("\n=== STRATEGY BACKTEST RESULTS ===")
            print("+-------------------------------------+-----------------+-------------+-----------+------------+----------------+--------------+---------------+")
            print("| Strategy                            | Initial Capital | Final Value | Total P&L | Return (%) | Avg Return (%) | Win Rate (%) | Wins | Losses |")
            print("+-------------------------------------+-----------------+-------------+-----------+------------+----------------+--------------+---------------+")
            print("| {0:<35} | {1:<15} | {2:<11} | {3:<9} | {4:<10} | {5:<14} | {8:<12} | {9:<4} | {10:<6} |".format(
                self.__class__.__name__,
                1000,
                round(self.final_value, 2),
                round(self.total_pnl, 2),
                round(self.total_pnl / 1000 * 100, 2),
                round(self.total_pnl / self.total_trades, 6) if self.total_trades > 0 else 0,
                'N/A',
                'N/A',
                round(self.win_rate, 2),
                self.total_wins,
                self.total_losses
            ))
            print("+-------------------------------------+-----------------+-------------+-----------+------------+----------------+--------------+---------------+")
        
        elif self.p.backtest == False:
            """Clean up resources when strategy stops"""
            if hasattr(self, 'alert_loop') and self.alert_loop:
                self.alert_loop.call_soon_threadsafe(self.alert_loop.stop)
            
            if hasattr(self, 'alert_thread') and self.alert_thread:
                self.alert_thread.join(timeout=5)
                
            if hasattr(self, 'telegram_service') and self.telegram_service.client:
                async def disconnect():
                    await self.telegram_service.client.disconnect()
                
                if self.alert_loop and self.alert_loop.is_running():
                    asyncio.run_coroutine_threadsafe(disconnect(), self.alert_loop)
            
            super().stop()

    # def log_entry(self):
    #     trade_logger.info("-" * 100)
    #     self.total_buys += 1
    #     self.current_cycle_buys += 1
    #     self.max_buys_per_cycle = max(self.max_buys_per_cycle, self.current_cycle_buys)

    #     trade_logger.info(f"{datetime.utcnow()} - Buy executed: {self.data._name}")
    #     trade_logger.info(f"Entry price: {self.entry_prices[-1]:.12f}")
    #     trade_logger.info(f"Position size: {self.sizes[-1]}")
    #     trade_logger.info(f"Current cash: {self.broker.getcash():.2f}")
    #     trade_logger.info(f"Current portfolio value: {self.broker.getvalue():.2f}")
    #     trade_logger.info("*" * 100)

    # def log_exit(self, exit_type):
    #     trade_logger.info("-" * 100)
    #     trade_logger.info(f"{datetime.utcnow()} - {exit_type} executed: {self.data._name}")
        
    #     position_size = sum(self.sizes)
    #     exit_price = self.data.close[0]
    #     profit_usd = (exit_price - self.first_entry_price) * position_size
    #     self.last_profit_usd = profit_usd
    #     self.total_profit_usd += profit_usd
    #     self.trade_cycles += 1
        
    #     trade_logger.info(f"Exit price: {exit_price:.12f}")
    #     trade_logger.info(f"Average entry price: {self.average_entry_price:.12f}")
    #     trade_logger.info(f"Position size: {position_size}")
    #     trade_logger.info(f"Profit for this cycle (USD): {profit_usd:.2f}")
    #     trade_logger.info(f"Total profit (USD): {self.total_profit_usd:.2f}")
    #     trade_logger.info(f"Trade cycles completed: {self.trade_cycles}")
    #     trade_logger.info(f"Average profit per cycle (USD): {self.total_profit_usd / self.trade_cycles:.2f}")
    #     trade_logger.info(f"Time elapsed: {datetime.utcnow() - self.start_time}")
    #     if self.position_start_time:
    #         trade_logger.info(f"Position cycle time: {datetime.utcnow() - self.position_start_time}")
    #     trade_logger.info(f"Maximum buys per cycle: {self.max_buys_per_cycle}")
    #     trade_logger.info(f"Total buys: {self.total_buys}")
    #     trade_logger.info("*" * 100)

    #     self.current_cycle_buys = 0
    #     self.position_start_time = None
    #     self.total_buys = 0
    #     self.max_buys_per_cycle = 0
    #     self.trade_cycles = 0
    #     self.total_profit_usd = 0
    #     self.last_profit_usd = 0

class CustomPandasData(bt.feeds.PandasData):
    params = (
        ('Datetime', 'datetime'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None)
    )


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
