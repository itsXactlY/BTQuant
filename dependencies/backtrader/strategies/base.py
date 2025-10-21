from datetime import datetime
import backtrader as bt
import traceback
import threading
import queue
import time
import numpy as np
import shutil
import os
import uuid
import asyncio
import csv
import multiprocessing
import os

from backtrader import transparencypatch
optimized_patch = transparencypatch.TransparencyPatch()

def activate_patch(debug: bool = False):
    optimized_patch.debug = debug
    optimized_patch.apply_indicator_patch()

def capture_patch(strategy):
    optimized_patch.capture_patch_fast(strategy)

# === Console Formatting Utilities ===
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    COLORAMA = True
except Exception:
    COLORAMA = False
    class _NoColor:
        def __getattr__(self, name): return ''
    Fore = Style = _NoColor()

def cinfo(msg):   return f"{Fore.CYAN}ℹ {msg}{Style.RESET_ALL}" if COLORAMA else f"[i] {msg}"
def cgood(msg):   return f"{Fore.GREEN}✔ {msg}{Style.RESET_ALL}" if COLORAMA else f"[OK] {msg}"
def cwarn(msg):   return f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}" if COLORAMA else f"[!] {msg}"
def cerr(msg):    return f"{Fore.RED}✘ {msg}{Style.RESET_ALL}" if COLORAMA else f"[x] {msg}"

def chead(title, char='━', color=Fore.MAGENTA):
    line = char * max(10, len(title) + 8)
    if COLORAMA:
        return f"{color}{line}\n  {title}\n{line}{Style.RESET_ALL}"
    return f"{line}\n  {title}\n{line}"

def csep(char='─', width=80, color=Fore.BLUE):
    if COLORAMA:
        return f"{color}{char*width}{Style.RESET_ALL}"
    return char * width

def fmt_num(val, prec=8, width=None, plus=False):
    if isinstance(val, (int, float)):
        s = f"{val:+.{prec}f}" if plus else f"{val:.{prec}f}"
    else:
        s = str(val)
    if width:
        return s.rjust(width)
    return s

def fmt_pct(p, width=None, colorize=True):
    if p is None:
        s = "--"
        return s.rjust(width) if width else s
    s = f"{p:+.2f}%"
    if colorize and COLORAMA:
        if p > 0:  s = f"{Fore.GREEN}{s}{Style.RESET_ALL}"
        elif p < 0: s = f"{Fore.RED}{s}{Style.RESET_ALL}"
        else: s = f"{Fore.YELLOW}{s}{Style.RESET_ALL}"
    return s.rjust(width) if width else s

order_lock = threading.Lock()
INIT_CASH = 1000.0


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
        ('backtest', True),
        ('bulk', False),      # NEW: Flag for bulk operations
        ('optuna', False),    # NEW: Flag for optuna optimization
        ("capture_data", False), # NEW: Flag for Indicator Chain Transparency Matrix
        ('quantstats', None),
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
        
        # Core data references
        self.dataclose = self.datas[0].close
        self.symbol = self.p.asset or self.p.symbol
        
        # Auto-initialize all tracking variables - strategies don't need to do this
        self._init_position_tracking()
        self._init_metrics()
        
        # Backtesting setup
        if self.p.backtest:
            BuySellArrows(self.data0, barplot=True)
        
        if hasattr(self.datas[0], '_dataname'):
            symbol = self.datas[0]._dataname
            self.active_orders = OrderTracker.load_active_orders_from_csv(
                symbol=symbol,
                backtest=self.params.backtest,
                bulk=self.params.bulk,
                optuna=self.params.optuna
            )
        
        if self.p.capture_data:
            activate_patch(debug=False)
        
        # Live trading setup
        if not self.p.backtest:
            self.init_live_trading()

    def _init_position_tracking(self):
        """Initialize all position tracking variables - strategies don't need to touch these"""
        self.conditions_checked = False
        self.entry_price = None
        self.take_profit_price = None
        self.first_entry_price = None
        self.stop_loss_price = None
        self.buy_executed = False
        self.average_entry_price = None
        self.usdt_amount = None
        self.stake_to_use = None
        self.stake_to_sell = None
        self.stake = None
        self.order = None
        self.DCA = False
        
        # Order tracking - centralized
        self.entry_prices = []
        self.sizes = []
        self.short_entry_prices = []
        self.short_sizes = []
        self.active_orders = []
        
        # Timing and cooldown
        self.last_order_time = 0
        self.order_cooldown = self.p.order_cooldown
        self.position_count = 0
        self.position_history = []
        
        # Short position tracking
        self.average_short_price = None
        self.first_short_entry_price = None
        self.short_take_profit_price = None
        self.short_average_entry_price = 0
        self.short_executed = False
        self.short_count = 0
        
        # Debug/reporting
        self.print_counter = 0
        self.live_data = False

    def _init_metrics(self):
        """Initialize performance metrics"""
        self.init_cash = self.params.init_cash
        self.total_cash_added = 0
        self.total_pnl = 0.0
        self.total_wins = 0
        self.total_losses = 0
        self.total_trades = 0
        self.win_rate = 0
        self.final_value = 0.0

    # ============================================================================
    # STRATEGY INTERFACE - Override these in your strategy
    # ============================================================================
    
    def buy_or_short_condition(self):
        """Override this method to implement buy/short entry logic
        Return True if order was placed, False otherwise"""
        return False
    
    def dca_or_short_condition(self):
        """Override this method to implement DCA logic
        Return True if order was placed, False otherwise"""
        return False
    
    def sell_or_cover_condition(self):
        """Override this method to implement sell/cover exit logic
        Return True if order was placed, False otherwise"""
        return False
    
    def check_stop_loss(self):
        """Override this method to implement custom stop loss logic
        Return True if stop loss hit, False otherwise"""
        return False

    # ============================================================================
    # HELPER METHODS - Available for strategies to use
    # ============================================================================
    
    def _determine_size(self):
        """Calculate position size based on backtesting or live trading mode"""
        if self.p.backtest:
            available_cash = self.broker.get_cash()
            return (available_cash * self.p.percent_sizer) / self.dataclose[0] if self.dataclose[0] > 0 else 0
        else:
            self.calculate_position_size()
            return self.usdt_amount

    def create_order(self, action='BUY', size=None, price=None):
        """
        Unified order creation that handles OrderTracker automatically
        Strategies call this instead of manually creating OrderTracker instances
        
        Args:
            action: 'BUY' or 'SELL'
            size: position size (auto-calculated if None)
            price: entry price (uses current close if None)
        
        Returns:
            OrderTracker instance if successful, None otherwise
        """
        if size is None:
            size = self._determine_size()
        
        if price is None:
            price = self.data.close[0]
        
        if size <= 0:
            if self.p.debug:
                print(cwarn(f"Invalid size {size}, order not created"))
            return None
        
        current_dt = self.data.datetime.datetime(0)
        
        order_tracker = OrderTracker(
            entry_price=price,
            size=size,
            take_profit_pct=self.params.take_profit,
            symbol=self.data._dataname,
            order_type=action,
            backtest=self.params.backtest,
            bulk=self.params.bulk,
            optuna=self.params.optuna,
            data_datetime=current_dt
        )
        
        self.active_orders.append(order_tracker)
        self.entry_prices.append(price)
        self.sizes.append(size)
        
        if not self.first_entry_price:
            self.first_entry_price = price
        
        self.buy_executed = True
        self.calc_averages()
        
        # Place actual order
        if action == 'BUY':
            self.order = self.buy(size=size, exectype=bt.Order.Market)
        else:
            self.order = self.sell(size=size, exectype=bt.Order.Market)
        
        if self.p.debug:
            print(cgood(f"{action} order created: {size:.8f} @ {price:.8f}"))
        
        return order_tracker

    def close_order(self, order_tracker, exit_price=None):
        """
        Close a specific order and update tracking
        
        Args:
            order_tracker: OrderTracker instance to close
            exit_price: exit price (uses current close if None)
        """
        if exit_price is None:
            exit_price = self.data.close[0]
        
        current_dt = self.data.datetime.datetime(0)
        order_tracker.close_order(exit_price, exit_datetime=current_dt)
        
        # Execute the sell
        self.order = self.sell(size=order_tracker.size, exectype=bt.Order.Market)
        
        if self.p.debug:
            profit_pct = ((exit_price / order_tracker.entry_price) - 1) * 100
            print(cgood(f"Order closed: {profit_pct:.2f}% profit"))
        
        # Remove from active orders
        if order_tracker in self.active_orders:
            self.active_orders.remove(order_tracker)
        
        # Update tracking
        self.entry_prices = [o.entry_price for o in self.active_orders]
        self.sizes = [o.size for o in self.active_orders]
        
        if not self.active_orders:
            self.reset_position_state()
        else:
            self.calc_averages()

    def calc_averages(self):
        """Calculate average entry price and take profit from active orders"""
        if not self.active_orders:
            self.average_entry_price = None
            self.take_profit_price = None
            return
        
        total_value = sum(o.entry_price * o.size for o in self.active_orders)
        total_size = sum(o.size for o in self.active_orders)
        
        if total_size > 0:
            self.average_entry_price = total_value / total_size
            if self.first_entry_price:
                self.take_profit_price = self.first_entry_price * (1 + self.params.take_profit / 100)
            self.buy_executed = True

    def reset_position_state(self):
        """Reset all position tracking - called automatically after full exit"""
        if self.p.debug:
            print(cinfo("Resetting position state"))
        
        self.buy_executed = False
        self.entry_price = None
        self.entry_prices = []
        self.average_entry_price = None
        self.take_profit_price = None
        self.first_entry_price = None
        self.sizes = []
        self.active_orders = []
        self.position_count = 0
        self.stop_loss_price = 0.0

    def reset_short_position_state(self):
        """Reset short position tracking"""
        self.short_executed = False
        self.short_entry_prices.clear()
        self.short_sizes.clear()
        self.first_short_entry_price = None
        self.short_average_entry_price = 0

    # ============================================================================
    # LIVE TRADING - Automatically configured
    # ============================================================================
    
    def init_live_trading(self):
        """Initialize live trading components based on exchange type"""
        if self.p.exchange.lower() == "pancakeswap":
            self._init_pancakeswap()
        elif self.p.exchange.lower() == "mimic":
            self._init_jrr_exchange()
        else:
            print(cwarn('No JackRabbitRelay / Web3 exchange detected'))

    def _init_jrr_exchange(self):
        """Initialize standard exchange trading with JackRabbitRelay"""
        from backtrader.brokers.jrrbroker import JrrOrderBase
        
        alert_manager = self._init_alert_system() if self.p.enable_alerts else None
        time.sleep(1)
        
        self.exchange = self.p.exchange
        self.account = self.p.account
        self.asset = self.p.asset
        self.rabbit = JrrOrderBase(alert_manager=alert_manager)
        
        self.order_queue = queue.Queue()
        self.order_thread = threading.Thread(target=self.process_jrr_orders, daemon=True)
        self.order_thread.start()

    def _init_pancakeswap(self):
        """Initialize PancakeSwap trading"""
        from backtrader.brokers.pancakeswap_orders import PancakeSwapV2DirectOrderBase as _web3order
        self.pcswap = _web3order(coin=self.p.coin, collateral=self.p.collateral)
        self.web3order_queue = queue.Queue()
        self.web3order_thread = threading.Thread(target=self.process_web3orders, daemon=True)
        self.web3order_thread.start()
        print(cgood("PancakeSwap Web3 trading initialized"))

    def _init_alert_system(self, session=".__!_"):
        """Initialize alert system with Telegram and Discord services if enabled"""
        if not self.p.enable_alerts:
            return None
        
        try:
            base_session_file = ".base.session"
            new_session_file = f"{session}_{uuid.uuid4().hex}.session"
            
            if not os.path.exists(base_session_file):
                raise FileNotFoundError(f"Base session file '{base_session_file}' not found.")
            shutil.copy(base_session_file, new_session_file)
            print(cgood(f"✅ Copied base session to {new_session_file}"))
            
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
            print(cgood("✅ Alert system initialized"))
            return self.alert_manager
        
        except Exception as e:
            print(cerr(f"❌ Error initializing alert system: {str(e)}"))
            return None

    def send_alert(self, message: str):
        """Helper method to safely send alerts if enabled"""
        if self.p.enable_alerts and hasattr(self, 'alert_manager') and self.alert_manager:
            try:
                self.alert_manager.send_alert(message)
            except Exception as e:
                print(cerr(f"Error sending alert: {str(e)}"))

    def process_jrr_orders(self):
        """Process JRR order queue"""
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
            self.order_queue.task_done()

    def enqueue_order(self, action, **params):
        """Add order to queue with cooldown check"""
        with order_lock:
            current_time = time.time()
            if current_time - self.last_order_time >= self.order_cooldown:
                self.order_queue.put((action, params))
                self.last_order_time = current_time
                return True
        return False

    def process_web3orders(self):
        """Process Web3 order queue"""
        while True:
            order = self.web3order_queue.get()
            if order is None:
                break
            action, params = order
            if action == 'buy':
                try:
                    self.pcswap.send_pcs_buy_request(**params)
                    print(cgood("Web3 buy request sent"))
                except Exception as e: 
                    print(cerr(f"Web3 buy error: {e}"))
            elif action == 'sell':
                try:
                    self.pcswap.send_pcs_close_request(**params)
                    print(cgood("Web3 close request sent"))
                except Exception as e:
                    print(cerr(f"Web3 close error: {e}"))
                self.reset_position_state()
            self.web3order_queue.task_done()

    def enqueue_web3order(self, action, **params):
        """Add Web3 order to queue with cooldown check"""
        current_time = time.time()
        if current_time - self.last_order_time >= self.order_cooldown:
            self.web3order_queue.put((action, params))
            self.last_order_time = time.time()

    def calculate_position_size(self):
        """Calculate the position size based on available balance and current price"""
        if self.p.exchange.lower() == 'binance':
            min_order_value = 5.50
        elif self.p.exchange.lower() == 'mexc':
            min_order_value = 1.10
        elif self.p.exchange.lower() == 'pancakeswap':
            min_order_value = 0.00001
        else:
            min_order_value = 10
        
        if self.p.exchange.lower() == 'pancakeswap':
            if hasattr(self, 'pcswap') and self.pcswap:
                actual_bnb_balance = self.pcswap.get_collateral_balance()
                bnb_to_use = actual_bnb_balance * self.p.percent_sizer
                gas_reserve = 0.001
                if bnb_to_use > actual_bnb_balance - gas_reserve:
                    bnb_to_use = max(actual_bnb_balance - gas_reserve, 0)
                if bnb_to_use < min_order_value:
                    print(cerr(f"Insufficient BNB. Need {min_order_value}, have {bnb_to_use:.8f}"))
                    return 0
                self.amount = bnb_to_use
                self.usdt_amount = bnb_to_use
                return self.amount
            else:
                print(cerr("PancakeSwap order base not initialized"))
                return 0
        else:
            usdt_to_use = self.broker.getcash() * self.p.percent_sizer / self.dataclose
            if hasattr(self, 'dataclose') and len(self.dataclose) > 0 and self.dataclose[0] > 0:
                self.amount = usdt_to_use / self.dataclose[0]
                order_value = self.amount * self.dataclose[0]
                if order_value < min_order_value:
                    self.amount = min_order_value / self.dataclose[0]
                    usdt_to_use = min_order_value
                self.amount = round(self.amount, 8)
                self.usdt_amount = round(self.amount * self.dataclose[0], 8)
            else:
                self.amount = min_order_value / 1000
                self.usdt_amount = min_order_value
            return self.amount

    def load_trade_data(self):
        """Load existing positions from CSV or API"""
        try:
            if self.p.exchange.lower() == 'mimic':
                pass  # Implement later
            elif self.p.exchange.lower() not in ('mimic', 'pancakeswap'):
                cash = self.broker.getcash()
                self.stake_to_use = cash
                print(cinfo(f"Available USDT: {self.stake_to_use}"))
                
                if not hasattr(self, 'active_orders') or self.active_orders is None:
                    self.active_orders = []
                
                for data in self.datas:
                    symbol = self.symbol
                    print(cinfo(f"Loading trade data for symbol: {symbol}"))

                    try:
                        loaded_orders = OrderTracker.load_active_orders_from_csv(symbol)
                        if loaded_orders:
                            self.active_orders = loaded_orders
                            self.buy_executed = True
                            self.entry_prices = [order.entry_price for order in self.active_orders]
                            self.sizes = [order.size for order in self.active_orders]
                            self.calc_averages()
                            print(cgood(f"Loaded {len(loaded_orders)} orders from CSV"))
                            continue
                        else:
                            print(cwarn(f"No active orders in CSV, falling back to API"))
                    except Exception as e:
                        print(cerr(f"Error loading from CSV: {e}"))
                    
                    position_infos = self.broker.get_all_positions(data)
                    if not position_infos:
                        print(cwarn(f"No positions found for {symbol}"))
                        continue
                    
                    print(cinfo(f"Found {len(position_infos)} open positions"))
                    
                    for position_info in position_infos:
                        if position_info['size'] > 0:
                            entry_info = self.broker.get_entry_price(symbol, position_info['id'])
                            if entry_info and entry_info['trade']:
                                entry_trade = entry_info['trade']
                                entry_price = entry_trade.price
                                entry_size = entry_trade.size
                                entry_time = datetime.fromtimestamp(entry_trade.timestamp / 1000)
                                
                                self.entry_prices.append(entry_price)
                                self.sizes.append(entry_size)
                                
                                if not self.first_entry_price:
                                    self.first_entry_price = entry_price
                                
                                order_tracker = OrderTracker(
                                    entry_price=entry_price,
                                    size=entry_size,
                                    take_profit_pct=self.params.take_profit,
                                    symbol=symbol,
                                    order_type="BUY",
                                    backtest=self.params.backtest
                                )
                                order_tracker.timestamp = entry_time
                                order_tracker.tracker_id = str(position_info['id'])
                                self.active_orders.append(order_tracker)
                    
                    if self.entry_prices:
                        self.buy_executed = True
                        self.DCA = True
                        self.calc_averages()
                        print(cgood(f"Loaded {len(self.active_orders)} positions from API"))
            
            elif self.p.exchange.lower() == 'pancakeswap':
                if not hasattr(self, 'active_orders') or self.active_orders is None:
                    self.active_orders = []
                
                for data in self.datas:
                    symbol = self.symbol
                    try:
                        loaded_orders = OrderTracker.load_active_orders_from_csv(symbol)
                        if loaded_orders:
                            print(cgood(f"Loaded {len(loaded_orders)} orders from CSV"))
                            self.active_orders = loaded_orders
                            self.buy_executed = True
                            self.entry_prices = [order.entry_price for order in self.active_orders]
                            self.sizes = [order.size for order in self.active_orders]
                            self.calc_averages()
                        else:
                            print(cwarn("No active orders found, starting fresh"))
                    except Exception as e:
                        print(cerr(f"Error loading from CSV: {e}"))
        
        except Exception as e:
            print(cerr(f"Error loading trade data: {e}"))
            traceback.print_exc()
            self.reset_position_state()

    # ============================================================================
    # CORE LIFECYCLE METHODS
    # ============================================================================
    
    def start(self):
        """Called once at the start of the backtest/live session"""
        if not self.params.backtest:
            if self.p.exchange.lower() == "pancakeswap":
                print(chead(f"BTQuant initialized for {self.p.exchange}"))
                self.load_trade_data()
            else:
                print(chead(f"BTQuant initialized for {self.p.exchange}"))
                self.load_trade_data()
                
                try:
                    for data in self.datas:
                        symbol = data.symbol
                        currency = symbol.split('/')[0]
                        
                        try:
                            pos_obj = self.broker.getposition(data)
                            actual_position = pos_obj.size if hasattr(pos_obj, 'size') else float(pos_obj)
                        except:
                            try:
                                actual_position = float(self.broker.store.getposition(currency))
                            except:
                                actual_position = 0.0
                        
                        print(cinfo(f"Validated position for {currency}: {actual_position}"))
                        
                        if actual_position > 0 and (not self.buy_executed or not self.entry_prices):
                            print(cwarn(f"Ensuring position tracking for {actual_position} {currency}"))
                            try:
                                position_info = self.broker.get_position_info(data)
                                if position_info['size'] > 0:
                                    self.entry_price = position_info['price']
                                    self.entry_prices = [self.entry_price]
                                    self.sizes = [actual_position]
                                    self.first_entry_price = self.entry_price
                                    self.buy_executed = True
                                    self.DCA = True
                                    self.calc_averages()
                                    
                                    order_tracker = OrderTracker(
                                        entry_price=self.entry_price,
                                        size=actual_position,
                                        take_profit_pct=self.params.take_profit,
                                        symbol=symbol,
                                        order_type="BUY",
                                        backtest=False
                                    )
                                    self.active_orders.append(order_tracker)
                                    print(cgood("Added existing position to tracking"))
                            except Exception as inner_e:
                                print(cerr(f"Error initializing tracking: {inner_e}"))
                except Exception as e:
                    print(cerr(f"Error during validation: {e}"))
                    traceback.print_exc()

    def next(self):
        """Called for every data point - this is where strategy logic runs"""
        if self.p.capture_data:
            capture_patch(self)
        
        self.conditions_checked = False
        
        # Debug output
        if self.p.debug and hasattr(self, 'live_data') and self.live_data:
            if self.buy_executed:
                self.print_counter += 1
            if self.print_counter % 15 == 0:
                self.stake = self.broker.getcash() * self.p.percent_sizer / self.dataclose
                pos = "OPEN" if self.buy_executed else "FLAT"
                dca = "DCA" if self.DCA else "-"
                price = fmt_num(self.data.close[0], 6)
                cash = fmt_num(self.broker.getcash(), 2)
                print(f"{Fore.WHITE if COLORAMA else ''}⏱ {datetime.now().strftime('%H:%M:%S')} | {self.symbol or ''} | {pos} {dca} | Px {price} | Cash {cash}{Style.RESET_ALL if COLORAMA else ''}")
                
                if self.buy_executed and self.average_entry_price and self.take_profit_price:
                    print(chead("Position Report", char='─', color=Fore.BLUE))
                    print(f"Price:        {fmt_num(self.data.close[0], prec=9)}")
                    print(f"Entry (avg):  {fmt_num(self.average_entry_price, prec=9)}")
                    print(f"Take Profit:  {fmt_num(self.take_profit_price, prec=9)}")
                    print(csep('─', 60, color=Fore.BLUE))
        
        # Execute strategy logic based on mode
        if hasattr(self, 'live_data') and self.live_data:
            self._execute_strategy_logic()
        elif self.params.backtest:
            self.stake = self.broker.getcash() * self.p.percent_sizer / self.dataclose if self.dataclose[0] > 0 else 0
            self._execute_strategy_logic()

    def _execute_strategy_logic(self):
        """
        Centralized strategy execution logic - same for backtest and live
        Automatically handles the order flow, strategies just override the condition methods
        """
        # Check cash before DCA in backtest mode
        if self.params.backtest and self.broker.getcash() < 10.0:
            if self.p.debug:
                print(cwarn('Insufficient cash for new orders'))
            # Still allow exits
            if self.buy_executed:
                self.sell_or_cover_condition()
            self.conditions_checked = True
            return
        
        if not self.buy_executed:
            # No position - look for entry
            if self.buy_or_short_condition():
                self.conditions_checked = True
                return
        
        elif self.DCA and self.buy_executed:
            # Position open with DCA enabled
            # First check exits
            if self.sell_or_cover_condition():
                self.conditions_checked = True
                return
            # Then check for DCA opportunity
            self.dca_or_short_condition()
        
        elif self.buy_executed:
            # Position open without DCA
            self.sell_or_cover_condition()
        
        self.conditions_checked = True

    # ============================================================================
    # REPORTING & VISUALIZATION
    # ============================================================================
    
    def report_positions(self):
        """Print a compact, perfectly aligned ACTIVE POSITIONS table."""
        if not self.active_orders:
            print(f"\n{Fore.YELLOW if COLORAMA else ''}📊 No active positions to display{Style.RESET_ALL if COLORAMA else ''}\n")
            return
        
        data_rows = []
        now = datetime.now()
        price = self.data.close[0]
        
        for t in self.active_orders:
            pnl = (price / t.entry_price - 1) * 100
            hours = (now - t.timestamp).total_seconds() / 3600 if t.timestamp else 0
            
            if hours < 1:
                time_s = f"{hours * 60:.0f}m"
            elif hours < 24:
                time_s = f"{hours:.0f}h"
            else:
                time_s = f"{hours / 24:.1f}d"
            
            data_rows.append({
                'entry': f"{t.entry_price:.8f}",
                'size': f"{t.size:,.4f}",
                'tp': f"{t.take_profit_price:.8f}",
                'pnl': f"{pnl:+.2f}%",
                'pnl_val': pnl,
                'time': time_s
            })
        
        w_entry = max(12, max(len(r['entry']) for r in data_rows))
        w_size = max(12, max(len(r['size']) for r in data_rows))
        w_tp = max(13, max(len(r['tp']) for r in data_rows))
        w_pnl = max(7, max(len(r['pnl']) for r in data_rows))
        w_time = max(6, max(len(r['time']) for r in data_rows))
        
        total_width = w_entry + w_size + w_tp + w_pnl + w_time + 18
        
        print()
        print(f"{Fore.CYAN if COLORAMA else ''}{'═' * total_width}{Style.RESET_ALL if COLORAMA else ''}")
        title = "📈 ACTIVE POSITIONS"
        print(f"{Fore.CYAN if COLORAMA else ''}║{Style.RESET_ALL if COLORAMA else ''} {Fore.WHITE + Style.BRIGHT if COLORAMA else ''}{title.center(total_width - 4)}{Style.RESET_ALL if COLORAMA else ''} {Fore.CYAN if COLORAMA else ''}║{Style.RESET_ALL if COLORAMA else ''}")
        print(f"{Fore.CYAN if COLORAMA else ''}{'═' * total_width}{Style.RESET_ALL if COLORAMA else ''}")
        
        top = f"╭{'─' * (w_entry + 2)}┬{'─' * (w_size + 2)}┬{'─' * (w_tp + 2)}┬{'─' * (w_pnl + 2)}┬{'─' * (w_time + 2)}╮"
        print(f"{Fore.BLUE if COLORAMA else ''}{top}{Style.RESET_ALL if COLORAMA else ''}")
        
        h1 = "💰 Entry Price".center(w_entry)
        h2 = "📊 Size".center(w_size + 1)
        h3 = "🎯 Take Profit".center(w_tp + 1)
        h4 = "📈 P/L".center(w_pnl + 1)
        h5 = "⏱️ Time".center(w_time + 3)
        print(f"│{h1}│{h2}│{h3}│{h4}│{h5}│")
        
        mid = f"├{'─' * (w_entry + 2)}┼{'─' * (w_size + 2)}┼{'─' * (w_tp + 2)}┼{'─' * (w_pnl + 2)}┼{'─' * (w_time + 2)}┤"
        print(f"{Fore.BLUE if COLORAMA else ''}{mid}{Style.RESET_ALL if COLORAMA else ''}")
        
        for row in data_rows:
            c1 = row['entry'].rjust(w_entry)
            c2 = row['size'].rjust(w_size)
            c3 = row['tp'].rjust(w_tp)
            c4 = row['pnl'].rjust(w_pnl)
            c5 = row['time'].rjust(w_time)
            
            if COLORAMA:
                if row['pnl_val'] > 0:
                    c4 = f"{Fore.GREEN}{c4}{Style.RESET_ALL}"
                elif row['pnl_val'] < 0:
                    c4 = f"{Fore.RED}{c4}{Style.RESET_ALL}"
                else:
                    c4 = f"{Fore.YELLOW}{c4}{Style.RESET_ALL}"
            
            print(f"│ {c1} │ {c2} │ {c3} │ {c4} │ {c5} │")
        
        bot = f"╰{'─' * (w_entry + 2)}┴{'─' * (w_size + 2)}┴{'─' * (w_tp + 2)}┴{'─' * (w_pnl + 2)}┴{'─' * (w_time + 2)}╯"
        print(f"{Fore.BLUE if COLORAMA else ''}{bot}{Style.RESET_ALL if COLORAMA else ''}")
        
        total = sum(t.entry_price * t.size for t in self.active_orders)
        avg = sum((price / t.entry_price - 1) * 100 for t in self.active_orders) / len(self.active_orders)
        summary = f"📊 {len(self.active_orders)} positions │ Total ${total:,.2f} │ Avg P/L {avg:+.2f}%"
        
        if COLORAMA:
            print(f"\n{Fore.CYAN + Style.BRIGHT}{summary}{Style.RESET_ALL}\n")
        else:
            print(f"\n{summary}\n")

    # ============================================================================
    # BACKTRADER NOTIFICATION HANDLERS
    # ============================================================================
    
    def notify_data(self, data, status, *args, **kwargs):
        """Handle data status changes"""
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}'.format(data._getstatusname(status))
        print(dt, dn, msg)
        self.live_data = (data._getstatusname(status) == 'LIVE')

    def notify_order(self, order):
        """Handle order execution notifications"""
        if order.status in [order.Completed]:
            if order.isbuy():
                # Update the last order tracker with the broker's order reference
                if hasattr(self, 'active_orders') and self.active_orders:
                    # Backtrader might assign an order ID, but we use tracker_id
                    pass
                self.action = "buy"
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.action = "sell"
            
            self.bar_executed = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if self.p.debug:
                print(cerr(f"Order {order.status}: Canceled={order.status == order.Canceled}, "
                          f"Margin={order.status == order.Margin}, Rejected={order.status == order.Rejected}"))
        
        self.order = None

    def notify_trade(self, trade):
        """Handle trade close notifications"""
        if trade.isclosed:
            self.total_trades += 1
            self.total_pnl += trade.pnl
            
            if trade.pnl > 0:
                self.total_wins += 1
            else:
                self.total_losses += 1
            
            self.win_rate = (self.total_wins / self.total_trades) * 100 if self.total_trades > 0 else 0

    def notify_cashvalue(self, cash, value):
        """Handle cash/value updates"""
        self.cash = cash
        self.value = value

    def stop(self):
        """Called when strategy stops - print results and cleanup"""
        in_bulk_mode = (
            "FORK" in multiprocessing.current_process().name.upper() or
            "SPAWN" in multiprocessing.current_process().name.upper()
        )
        if in_bulk_mode or self.p.optuna:
            return
        if self.p.backtest and not self.p.quantstats:
            self.final_value = self.broker.getvalue()
            print("\n" + "=" * 120)
            print(chead("STRATEGY BACKTEST RESULTS", char='═'))
            print("=" * 120)
            print("+-------------------------------------+-----------------+-------------+-----------+------------+----------------+--------------+--------+")
            print("| Strategy                            | Initial Capital | Final Value | Total P&L | Return (%) | Avg Return (%) | Win Rate (%) | Trades |")
            print("+-------------------------------------+-----------------+-------------+-----------+------------+----------------+--------------+--------+")
            print("| {0:<35} | {1:<15} | {2:<11} | {3:<9} | {4:<10} | {5:<14} | {6:<12} | {7:<6} |".format(
                self.__class__.__name__[:35],
                f"${self.init_cash:,.2f}",
                f"${self.final_value:,.2f}",
                f"${self.total_pnl:,.2f}",
                f"{(self.total_pnl / self.init_cash * 100):.2f}%",
                f"${(self.total_pnl / self.total_trades):.2f}" if self.total_trades > 0 else "N/A",
                f"{self.win_rate:.2f}%",
                f"{self.total_wins}/{self.total_losses}"
            ))
            print("+-------------------------------------+-----------------+-------------+-----------+------------+----------------+--------------+--------+")
            print("=" * 120 + "\n")
        
        elif not self.p.backtest:
            # Clean up live trading resources
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


# ============================================================================
# SUPPORTING CLASSES
# ============================================================================

class CustomData(bt.feeds.PolarsData):
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


class BuySellArrows(bt.observers.BuySell):
    def next(self):
        super().next()
        
        if self.lines.buy[0]:
            self.lines.buy[0] -= self.data.low[0] * 0.02
        
        if self.lines.sell[0]:
            self.lines.sell[0] += self.data.high[0] * 0.02
    
    plotlines = dict(
        buy=dict(marker='$⇧$', markersize=8.0),
        sell=dict(marker='$⇩$', markersize=8.0)
    )


import os
import csv
import traceback
from datetime import datetime

class OrderTracker:
    """
    Tracks individual orders/positions with automatic CSV persistence.
    Completely independent - strategies don't need to manage this directly.
    Supports separate CSV files per symbol.
    
    CSV persistence is DISABLED during:
    - Backtests (backtest=True)
    - Bulk operations (bulk=True)
    - Optuna optimizations (optuna=True)
    """
    _order_counter = 0
    _persistence_enabled = True  # Global flag to disable CSV operations

    def __init__(self, entry_price, size, take_profit_pct, symbol=None, order_type="BUY", 
                 backtest=False, bulk=False, optuna=False, data_datetime=None):
        self.entry_price = entry_price
        self.size = size
        self.take_profit_price = entry_price * (1 + take_profit_pct / 100)
        self.executed = True
        self.backtest = backtest
        self.bulk = bulk
        self.optuna = optuna

        OrderTracker._order_counter += 1
        self.tracker_id = f"order_{OrderTracker._order_counter}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        self.timestamp = data_datetime if data_datetime else datetime.now()

        if symbol is None or (hasattr(symbol, 'is_empty') and symbol.is_empty()):
            symbol = "UNKNOWN"
        elif hasattr(symbol, '__len__') and len(symbol) == 0:
            symbol = "UNKNOWN"
        elif isinstance(symbol, (str, int, float)) and not symbol:
            symbol = "UNKNOWN"
        self.symbol = symbol
        self.order_type = order_type
        self.closed = False
        self.exit_price = None
        self.exit_timestamp = None
        self.profit_pct = None

        # Only save to CSV if NOT in backtest/bulk/optuna mode
        if self._should_persist():
            self.save_to_csv()

    def _should_persist(self):
        """
        Determine if CSV persistence should be enabled.
        Returns False for backtest, bulk, optuna, or if globally disabled.
        """
        # Check global flag first
        if not OrderTracker._persistence_enabled:
            return False
        
        # Disable for backtest, bulk, or optuna
        if self.backtest or self.bulk or self.optuna:
            return False
        
        return True

    def _get_csv_file(self):
        """Return the CSV filename for this symbol"""
        safe_symbol = self.symbol.replace("/", "_") if self.symbol else "unknown"
        return f"{safe_symbol}_order_tracker.csv"

    def close_order(self, exit_price, exit_datetime=None):
        self.closed = True
        self.exit_price = exit_price
        self.exit_timestamp = exit_datetime if exit_datetime else datetime.now()
        self.profit_pct = ((exit_price / self.entry_price) - 1) * 100 if self.order_type == "BUY" else ((self.entry_price / exit_price) - 1) * 100
        
        # Only update CSV if persistence is enabled
        if self._should_persist():
            self.update_csv()

    def save_to_csv(self):
        """Save order to CSV - only called if _should_persist() returns True"""
        csv_file = self._get_csv_file()
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            fieldnames = ['tracker_id', 'symbol', 'order_type', 'entry_price', 'size',
                          'take_profit_price', 'timestamp', 'closed', 'exit_price',
                          'exit_timestamp', 'profit_pct']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({
                'tracker_id': self.tracker_id,
                'symbol': self.symbol,
                'order_type': self.order_type,
                'entry_price': f"{self.entry_price:.8f}",
                'size': f"{self.size:.10f}",
                'take_profit_price': f"{self.take_profit_price:.8f}",
                'timestamp': self.timestamp.isoformat(),
                'closed': self.closed,
                'exit_price': f"{self.exit_price:.8f}" if self.exit_price else '',
                'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else '',
                'profit_pct': f"{self.profit_pct:.4f}" if self.profit_pct else ''
            })
            f.flush()
            os.fsync(f.fileno())

    def update_csv(self):
        """Update CSV - only called if _should_persist() returns True"""
        try:
            csv_file = self._get_csv_file()
            if not os.path.isfile(csv_file):
                self.save_to_csv()
                return

            temp_file = f"{csv_file}.tmp"
            modified = False

            with open(csv_file, 'r', newline='') as infile, open(temp_file, 'w', newline='') as outfile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    if row['tracker_id'] == self.tracker_id:
                        row['closed'] = 'True'
                        row['exit_price'] = f"{self.exit_price:.8f}"
                        row['exit_timestamp'] = self.exit_timestamp.isoformat()
                        row['profit_pct'] = f"{self.profit_pct:.4f}"
                        modified = True
                    writer.writerow(row)

            os.replace(temp_file, csv_file)

            if not modified:
                self.save_to_csv()

        except Exception as e:
            print(f"Error updating CSV: {e}")
            traceback.print_exc()

    @classmethod
    def load_active_orders_from_csv(cls, symbol=None, backtest=False, bulk=False, optuna=False):
        """
        Load active orders from CSV.
        Returns empty list if in backtest/bulk/optuna mode.
        """
        # Never load from CSV during backtest, bulk, or optuna
        if backtest or bulk or optuna or not cls._persistence_enabled:
            return []

        active_orders = []
        try:
            if not symbol:
                return active_orders

            csv_file = f"{symbol.replace('/', '_')}_order_tracker.csv"
            if not os.path.isfile(csv_file):
                return active_orders

            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['closed'].lower() == 'false':
                        order = cls.__new__(cls)
                        order.entry_price = float(row['entry_price'])
                        order.size = float(row['size'])
                        order.take_profit_price = float(row['take_profit_price'])
                        order.executed = True
                        order.tracker_id = row['tracker_id']
                        order.timestamp = datetime.fromisoformat(row['timestamp']) if row['timestamp'] else datetime.now()
                        order.symbol = row['symbol']
                        order.order_type = row['order_type']
                        order.closed = False
                        order.exit_price = None
                        order.exit_timestamp = None
                        order.profit_pct = None
                        order.backtest = backtest
                        order.bulk = bulk
                        order.optuna = optuna

                        active_orders.append(order)
            return active_orders
        except Exception as e:
            print(f"Error loading orders from CSV: {e}")
            traceback.print_exc()
            return []

    @classmethod
    def disable_persistence(cls):
        """Globally disable CSV persistence (useful for bulk/optuna operations)"""
        cls._persistence_enabled = False

    @classmethod
    def enable_persistence(cls):
        """Re-enable CSV persistence"""
        cls._persistence_enabled = True