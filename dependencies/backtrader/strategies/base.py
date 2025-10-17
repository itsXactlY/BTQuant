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

from backtrader import transparencypatch
optimized_patch = transparencypatch.TransparencyPatch()

def activate_patch(debug: bool = False):
    optimized_patch.debug = debug
    optimized_patch.apply_indicator_patch()

def capture_patch(strategy):
    optimized_patch.capture_patch_fast(strategy)

# === Console Formatting Utilities (non-invasive) ===
try:
    from colorama import init as colorama_init, Fore, Style
    colorama_init(autoreset=True)
    COLORAMA = True
except Exception:
    # Fallback if colorama not available
    COLORAMA = False
    class _NoColor:
        def __getattr__(self, name): return ''
    Fore = Style = _NoColor()

def cinfo(msg):   # info
    return f"{Fore.CYAN}ℹ {msg}{Style.RESET_ALL}" if COLORAMA else f"[i] {msg}"

def cgood(msg):   # success
    return f"{Fore.GREEN}✔ {msg}{Style.RESET_ALL}" if COLORAMA else f"[OK] {msg}"

def cwarn(msg):   # warning
    return f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}" if COLORAMA else f"[!] {msg}"

def cerr(msg):    # error
    return f"{Fore.RED}✘ {msg}{Style.RESET_ALL}" if COLORAMA else f"[x] {msg}"

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

def fmt_hh(hours, width=None):
    s = f"{hours:.1f} h"
    return s.rjust(width) if width else s

def fmt_label(text, width, align='center', color=None):
    t = text.center(width) if align == 'center' else (text.ljust(width) if align=='left' else text.rjust(width))
    if color and COLORAMA:
        return f"{color}{t}{Style.RESET_ALL}"
    return t

def table_sep(col_widths):
    segs = []
    for w in col_widths:
        segs.append("-" * w)
    line = "+-" + "-+-".join(segs) + "-+"
    return line

def table_row(values, col_widths, align='right'):
    cells = []
    for v, w in zip(values, col_widths):
        s = str(v)
        if align == 'right':
            cells.append(s.rjust(w))
        elif align == 'left':
            cells.append(s.ljust(w))
        else:
            cells.append(s.center(w))
    return "| " + " | ".join(cells) + " |"


order_lock = threading.Lock()
INIT_CASH = 100_000.0

## HUGE TODO :: Cleanup

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
        ("capture_data", False),
        ("alert_channel", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.p.backtest == True:
            BuySellArrows(self.data0, barplot=True)
        if self.p.capture_data == True:
            activate_patch(debug=False)
        self.dataclose = self.datas[0].close
        self.symbol = self.p.asset

        self.conditions_checked = False
        self.entry_price = None
        self.take_profit_price = None
        self.first_entry_price = None
        self.stop_loss_price = None
        self.buy_executed = None
        self.average_entry_price = None
        self.usdt_amount = None
        self.stake_to_use = None
        self.stake_to_sell = None
        self.stake = None
        self.order = None
        self.DCA = False

        self.entry_prices = []
        self.sizes = []
        self.short_entry_prices = []
        self.short_sizes = []
        self.active_orders = []
        
        self.amount = self.p.amount
        self.conditions_checked = None
        self.print_counter = 0
 
        self.last_order_time = 0
        self.order_cooldown = self.p.order_cooldown
        self.position_count = 0
        self.position_history = []

        self.init_cash = self.params.init_cash
        self.total_cash_added = 0
        self.total_pnl = 0.0
        self.total_wins = 0
        self.total_losses = 0
        self.total_trades = 0
        self.win_rate = 0
        self.final_value = 0.0

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

    def _init_alert_system(self, session=".__!_"):
        """Initialize alert system with Telegram and Discord services if enabled"""
        if not self.p.enable_alerts:
            print(cinfo("Alert system disabled (not enabled via configuration)"))
            return None

        try:
            base_session_file = ".base.session"
            new_session_file = f"{session}_{uuid.uuid4().hex}.session"

            if not os.path.exists(base_session_file):
                raise FileNotFoundError(f"Base session file '{base_session_file}' not found.")
            shutil.copy(base_session_file, new_session_file)
            print(cgood(f"✅ Copied base session to {new_session_file}"))

            self.alert_loop = asyncio.new_event_loop()

            # Note: expects these names to exist in the environment/context
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
            print(cgood("✅ Alert system initialized successfully"))
            return self.alert_manager

        except Exception as e:
            print(cerr(f"❌ Error initializing alert system: {str(e)}"))
            return None

    def send_alert(self, message: str):
        """Helper method to safely send alerts if enabled"""
        if self.p.enable_alerts and hasattr(self, 'alert_manager') and self.alert_manager is not None:
            try:
                self.alert_manager.send_alert(message)
            except Exception as e:
                print(cerr(f"Error sending alert: {str(e)}"))
        else:
            pass

    def init_live_trading(self):
        """Initialize live trading components based on exchange type"""
        if self.p.exchange.lower() == "pancakeswap":
            self._init_pancakeswap()
        elif self.p.exchange.lower() == "mimic":
            self._init_jrr_exchange()
        else:
            print(cwarn('No JackRabbitRelay / Web3 exchange detected - no trading will be done on them.'))

    def _init_jrr_exchange(self):
        from backtrader.brokers.jrrbroker import JrrOrderBase
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

    def _init_pancakeswap(self):
        """Initialize PancakeSwap trading"""
        from backtrader.brokers.pancakeswap_orders import PancakeSwapV2DirectOrderBase as _web3order
        self.pcswap = _web3order(coin=self.p.coin, collateral=self.p.collateral)
        self.web3order_queue = queue.Queue()
        self.web3order_thread = threading.Thread(target=self.process_web3orders)
        self.web3order_thread.daemon = True
        self.web3order_thread.start()
        print(cgood("PancakeSwap Web3 trading initialized"))

    def process_jrr_orders(self):
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
            print(cinfo(f"Web3 order received: {order}"))
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
        current_time = time.time()
        if current_time - self.last_order_time >= self.order_cooldown:
            self.web3order_queue.put((action, params))
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
        elif self.p.exchange.lower() == 'pancakeswap':
            min_order_value = 0.00001
        else:
            min_order_value = 10

        # For PancakeSwap, get actual wallet balance in BNB
        if self.p.exchange.lower() == 'pancakeswap':
            if hasattr(self, 'pcswap') and self.pcswap:
                actual_bnb_balance = self.pcswap.get_collateral_balance()
                print(f"🔍 Actual BNB balance: {actual_bnb_balance:.8f} BNB")
                
                # Calculate BNB to use based on percent_sizer
                bnb_to_use = actual_bnb_balance * self.p.percent_sizer
                
                # Reserve some BNB for gas fees (e.g., 0.001 BNB)
                gas_reserve = 0.001
                if bnb_to_use > actual_bnb_balance - gas_reserve:
                    bnb_to_use = max(actual_bnb_balance - gas_reserve, 0)
                    print(f"⚠️ Adjusted for gas reserve: {bnb_to_use:.8f} BNB")
                
                if bnb_to_use < min_order_value:
                    print(f"❌ Insufficient BNB. Need at least {min_order_value} BNB, have {bnb_to_use:.8f} BNB")
                    return 0
                
                self.amount = bnb_to_use
                self.usdt_amount = bnb_to_use  # For PancakeSwap, amount is in BNB
                
                print(f"✅ Using {self.amount:.8f} BNB for trade")
                return self.amount
            else:
                print("❌ PancakeSwap order base not initialized")
                return 0
        else:
            # For centralized exchange trading, use broker cash
            usdt_to_use = self.broker.getcash() * self.p.percent_sizer / self.dataclose

            if hasattr(self, 'dataclose') and len(self.dataclose) > 0 and self.dataclose[0] > 0:
                self.amount = usdt_to_use / self.dataclose[0]
                order_value = self.amount * self.dataclose[0]

                if order_value < min_order_value:
                    self.amount = min_order_value / self.dataclose[0]
                    usdt_to_use = min_order_value
                    print(f"⚠️ Adjusted to minimum order value: ${min_order_value}")
                
                self.amount = round(self.amount, 8)
                self.usdt_amount = round(self.amount * self.dataclose[0], 8)
                
                print(f"ℹ️ Calculated position size: {self.amount} units worth {self.usdt_amount:.8f} USDT")
            else:
                self.amount = min_order_value / 1000
                self.usdt_amount = min_order_value
                print(f"⚠️ No price data available. Using default amount: {self.amount}")
            
            return self.amount

    def _determine_size(self):
        if self.p.backtest:
            return self.stake
        else:
            self.calculate_position_size()
            return self.usdt_amount

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
        if not hasattr(self, 'active_orders') or not self.active_orders:
            self.average_entry_price = None
            self.take_profit_price = None
            return
        
        # Derive from active_orders
        self.entry_prices = [order.entry_price for order in self.active_orders]
        self.sizes = [order.size for order in self.active_orders]
        
        total_value = sum(p * s for p, s in zip(self.entry_prices, self.sizes))
        total_size = sum(self.sizes)
        
        if total_size > 0:
            self.average_entry_price = total_value / total_size
            if not self.first_entry_price:
                self.first_entry_price = self.entry_prices[0]
            self.take_profit_price = self.first_entry_price * (1 + self.params.take_profit / 100)
            self.buy_executed = True

    def load_trade_data(self):
        try:
            if self.p.exchange.lower() == 'mimic':
                # reimplementing later
                pass
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
                            
                            # if we successfully loaded orders from CSV, we skip the API check
                            # TODO :: add an cross check on exchange to validate the loaded orders
                            continue
                        else:
                            print(cwarn(f"No active orders found in CSV for {symbol}, falling back to API"))
                    except Exception as e:
                        print(cerr(f"Error loading active orders from CSV: {e} \nFalling back to API"))
                
                    position_infos = self.broker.get_all_positions(data)
                    
                    if not position_infos or len(position_infos) == 0:
                        print(cwarn(f"No positions found for {symbol} via API"))
                        continue
                    
                    print(cinfo(f"Found {len(position_infos)} open positions for {symbol} via API"))
                    
                    for position_info in position_infos:
                        if position_info['size'] > 0:
                            print(cinfo(f"Found position: {position_info['size']} units at reference price {position_info['price']}"))
                            
                            entry_info = self.broker.get_entry_price(symbol, position_info['id'])

                            if entry_info and entry_info['trade']:
                                entry_trade = entry_info['trade']
                                entry_price = entry_trade.price
                                entry_size = entry_trade.size
                                entry_time = datetime.fromtimestamp(entry_trade.timestamp / 1000)
                                
                                print(cinfo(f"Found entry trade: {entry_trade}"))
                                print(cinfo(f"Entry price: {entry_price}"))
                                
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
                                order_tracker.order_id = position_info['id']
                                
                                self.active_orders.append(order_tracker)

                    if self.entry_prices:
                        self.buy_executed = True
                        self.DCA = True
                        self.calc_averages()
                        print(cgood(f"Loaded {len(self.active_orders)} positions into active_orders tracking from API"))

            elif self.p.exchange.lower() == 'pancakeswap':
                if not hasattr(self, 'active_orders') or self.active_orders is None:
                    self.active_orders = []

                for data in self.datas:
                    symbol = self.symbol
                    print(cinfo(f"Loading trade data for symbol: {symbol}"))
                    try:
                        loaded_orders = OrderTracker.load_active_orders_from_csv(symbol)
                        if loaded_orders:
                            print(cgood(f"Loaded {len(loaded_orders)} active orders for {symbol} from CSV"))
                            self.active_orders = loaded_orders
                            self.buy_executed = True
                            self.entry_prices = [order.entry_price for order in self.active_orders]
                            self.sizes = [order.size for order in self.active_orders]
                            self.calc_averages()
                            
                            # if we successfully loaded orders from CSV, we skip the API check
                            # TODO :: add an cross check on exchange to validate the loaded orders
                            continue
                        else:
                            print(cwarn(f"No active orders found in CSV for {symbol}, Starting from scratch"))
                    except Exception as e:
                        print(cerr(f"Error loading active orders from CSV: {e} \nStarting from scratch"))
        except Exception as e:
            print(cerr(f"Unexpected error occurred while loading trade data: {e}"))
            traceback.print_exc()
            self.reset_position_state()
            self.stake_to_use = None

    def start(self):
        if self.params.backtest == False and self.p.exchange.lower() != "pancakeswap":
            print(chead(f"BTQuant initialized for {self.p.exchange}"))
            self.load_trade_data()
            
            try:
                for data in self.datas:
                    symbol = data.symbol
                    currency = symbol.split('/')[0]
                    
                    try:
                        pos_obj = self.broker.getposition(data)
                        if hasattr(pos_obj, 'size'):
                            actual_position = pos_obj.size
                        else:
                            actual_position = float(pos_obj)
                    except (AttributeError, TypeError):
                        try:
                            actual_position = float(self.broker.store.getposition(currency))
                        except:
                            actual_position = 0.0
                    
                    print(cinfo(f"Validated position for {currency}: {actual_position}"))
                    
                    if actual_position > 0 and (not self.buy_executed or not self.entry_prices):
                        print(cwarn(f"Ensuring position tracking is initialized for {actual_position} {currency}"))
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
                                    take_profit_pct=self.params.take_profit
                                )
                                self.active_orders.append(order_tracker)
                                print(cgood("Added existing position to active_orders tracking"))
                        except Exception as inner_e:
                            print(cerr(f"Error initializing position tracking: {inner_e}"))
            
            except Exception as e:
                print(cerr(f"Error during position validation: {e}"))
                import traceback
                traceback.print_exc()
        
        if self.params.backtest == False and self.p.exchange.lower() == "pancakeswap":
            print(chead(f"BTQuant initialized for {self.p.exchange}"))
            self.load_trade_data()
        
        elif self.params.backtest == False:
            print(cinfo('DEX Exchange Detected - Dont chase the Rabbit - PancakeSwap trading initialized - warmup and preload will take longer than usual. Sit tight.'))

    def next(self):
        if self.p.capture_data:
            capture_patch(self)
        self.conditions_checked = False
        
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
                
                print(chead("Position Report", char='─', color=Fore.BLUE))
                print(f"Price:        {fmt_num(self.data.close[0], prec=9)}")
                print(f"Entry (avg):  {fmt_num(self.average_entry_price, prec=9)}")
                print(f"Take Profit:  {fmt_num(self.take_profit_price, prec=9)}")
                print(csep('─', 60, color=Fore.BLUE))
        
        if hasattr(self, 'live_data') and self.live_data:
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
                    print(cerr('Rejected Margin - decrease percent sizer or increase DCA deviation'))
                    return
                self.dca_or_short_condition()
            elif self.DCA == False and self.buy_executed:
                if self.broker.getcash() < 10.0:
                    print(cerr('Rejected Margin - decrease percent sizer or increase DCA deviation'))
                self.sell_or_cover_condition()

    def report_positions(self):
        """Print a compact, perfectly aligned ACTIVE POSITIONS table."""
        if not self.active_orders:
            print(f"\n{Fore.YELLOW if COLORAMA else ''}📊 No active positions to display{Style.RESET_ALL if COLORAMA else ''}\n")
            return
        
        # Pre-format all data
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
        
        # Calculate column widths
        w_entry = max(12, max(len(r['entry']) for r in data_rows))
        w_size = max(12, max(len(r['size']) for r in data_rows))
        w_tp = max(13, max(len(r['tp']) for r in data_rows))
        w_pnl = max(7, max(len(r['pnl']) for r in data_rows))
        w_time = max(6, max(len(r['time']) for r in data_rows))
        
        # Build the table
        total_width = w_entry + w_size + w_tp + w_pnl + w_time + 18  # +14 for borders and spacing
        
        # Header
        print()
        print(f"{Fore.CYAN if COLORAMA else ''}{'═' * total_width}{Style.RESET_ALL if COLORAMA else ''}")
        title = "📈ACTIVE POSITIONS"
        print(f"{Fore.CYAN if COLORAMA else ''}║{Style.RESET_ALL if COLORAMA else ''} {Fore.WHITE + Style.BRIGHT if COLORAMA else ''}{title.center(total_width - 4)}{Style.RESET_ALL if COLORAMA else ''} {Fore.CYAN if COLORAMA else ''}║{Style.RESET_ALL if COLORAMA else ''}")
        print(f"{Fore.CYAN if COLORAMA else ''}{'═' * total_width}{Style.RESET_ALL if COLORAMA else ''}")
        
        # Top border
        top = f"╭{'─' * (w_entry + 2)}┬{'─' * (w_size + 2)}┬{'─' * (w_tp + 2)}┬{'─' * (w_pnl + 2)}┬{'─' * (w_time + 2)}╮"
        print(f"{Fore.BLUE if COLORAMA else ''}{top}{Style.RESET_ALL if COLORAMA else ''}")
        
        # Header row
        h1 = "💰 Entry Price".center(w_entry)
        h2 = "📊 Size".center(w_size + 1)
        h3 = "🎯 Take Profit".center(w_tp + 1)
        h4 = "📈 P/L".center(w_pnl + 1)
        h5 = "⏱️ Time".center(w_time + 3)
        print(f"│{h1}│{h2}│{h3}│{h4}│{h5}│")
        
        # Middle border
        mid = f"├{'─' * (w_entry + 2)}┼{'─' * (w_size + 2)}┼{'─' * (w_tp + 2)}┼{'─' * (w_pnl + 2)}┼{'─' * (w_time + 2)}┤"
        print(f"{Fore.BLUE if COLORAMA else ''}{mid}{Style.RESET_ALL if COLORAMA else ''}")
        
        # Data rows
        for row in data_rows:
            c1 = row['entry'].rjust(w_entry)
            c2 = row['size'].rjust(w_size)
            c3 = row['tp'].rjust(w_tp)
            c4 = row['pnl'].rjust(w_pnl)
            c5 = row['time'].rjust(w_time)
            
            # Color the P/L
            if COLORAMA:
                if row['pnl_val'] > 0:
                    c4 = f"{Fore.GREEN}{c4}{Style.RESET_ALL}"
                elif row['pnl_val'] < 0:
                    c4 = f"{Fore.RED}{c4}{Style.RESET_ALL}"
                else:
                    c4 = f"{Fore.YELLOW}{c4}{Style.RESET_ALL}"
            
            print(f"│ {c1} │ {c2} │ {c3} │ {c4} │ {c5} │")
        
        # Bottom border
        bot = f"╰{'─' * (w_entry + 2)}┴{'─' * (w_size + 2)}┴{'─' * (w_tp + 2)}┴{'─' * (w_pnl + 2)}┴{'─' * (w_time + 2)}╯"
        print(f"{Fore.BLUE if COLORAMA else ''}{bot}{Style.RESET_ALL if COLORAMA else ''}")
        
        # Footer
        total = sum(t.entry_price * t.size for t in self.active_orders)
        avg = sum((price / t.entry_price - 1) * 100 for t in self.active_orders) / len(self.active_orders)
        summary = f"📊 {len(self.active_orders)} positions │ Total ${total:,.2f} │ Avg P/L {avg:+.2f}%"
        
        if COLORAMA:
            print(f"\n{Fore.CYAN + Style.BRIGHT}{summary}{Style.RESET_ALL}\n")
        else:
            print(f"\n{summary}\n")

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
        if order.status in [order.Completed]:
            if order.isbuy():
                if hasattr(self, 'active_orders') and self.active_orders:
                    self.active_orders[-1].order_id = order.ref

        if order.status in [order.Submitted, order.Accepted]:
            # return
            if order.status in [order.Completed]:
                self.update_order_history(order)
                if order.isbuy():
                    self.action = "buy"
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm

                else:  # Sell
                    self.action = "sell"

                self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
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
        if self.p.backtest and self.p.quantstats == False:
            self.final_value = self.broker.getvalue()
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
        buy=dict(marker='$⇧$', markersize=8.0),   # Direct Unicode character
        sell=dict(marker='$⇩$', markersize=8.0)
    )


class OrderTracker:
    def __init__(self, entry_price, size, take_profit_pct, symbol=None, order_type="BUY", backtest=False):
        self.entry_price = entry_price
        self.size = size
        self.take_profit_price = entry_price * (1 + take_profit_pct / 100)
        self.executed = True
        self.order_id = None
        self.timestamp = datetime.now()
        self.backtest = backtest
        
        if symbol is None or symbol == "":
            if hasattr(self, 'datas') and self.datas and hasattr(self.datas[0], '_dataname'):
                symbol = self.datas[0]._dataname
        
        self.symbol = symbol
        self.order_type = order_type  # BUY or SELL
        self.closed = False
        self.exit_price = None
        self.exit_timestamp = None
        self.profit_pct = None

        if not self.backtest:
            self.save_to_csv()
    
    def close_order(self, exit_price):
        self.closed = True
        self.exit_price = exit_price
        self.exit_timestamp = datetime.now()
        self.profit_pct = ((exit_price / self.entry_price) - 1) * 100 if self.order_type == "BUY" else ((self.entry_price / exit_price) - 1) * 100

        if not self.backtest:
            self.update_csv()
            self.remove_from_csv()

    def save_to_csv(self):
        if self.backtest:
            return
        
        print(f"Current working directory: {os.getcwd()}")
        csv_file = "order_tracker.csv"
        file_exists = os.path.isfile(csv_file)
        
        with open(csv_file, 'a', newline='') as f:
            fieldnames = ['order_id', 'symbol', 'order_type', 'entry_price', 'size', 
                        'take_profit_price', 'timestamp', 'closed', 'exit_price', 
                        'exit_timestamp', 'profit_pct']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'order_id': self.order_id,
                'symbol': self.symbol,
                'order_type': self.order_type,
                'entry_price': self.entry_price,
                'size': self.size,
                'take_profit_price': self.take_profit_price,
                'timestamp': self.timestamp,
                'closed': self.closed,
                'exit_price': self.exit_price,
                'exit_timestamp': self.exit_timestamp,
                'profit_pct': self.profit_pct
            })
    
    def update_csv(self):
        if self.backtest:
            return
            
        try:
            temp_file = "order_tracker_temp.csv"
            modified = False
            
            with open("order_tracker.csv", 'r', newline='') as infile, open(temp_file, 'w', newline='') as outfile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in reader:
                    if (abs(float(row['entry_price']) - self.entry_price) < 0.0001 and 
                        abs(float(row['size']) - self.size) < 0.0001 and
                        row['closed'].lower() == 'false'):
                        row['closed'] = 'True'
                        row['exit_price'] = str(self.exit_price)
                        row['exit_timestamp'] = str(self.exit_timestamp)
                        row['profit_pct'] = str(self.profit_pct)
                        modified = True
                    writer.writerow(row)
            
            os.replace(temp_file, "order_tracker.csv")
            
            if not modified:
                print(cwarn(f"Warning: Could not find order to update in CSV: entry={self.entry_price}, size={self.size}"))
        except Exception as e:
            print(cerr(f"Error updating CSV: {e}"))
    
    def remove_from_csv(self):
        if self.backtest:
            return
            
        try:
            temp_file = "order_tracker_temp.csv"
            order_removed = False

            with open("order_tracker.csv", 'r', newline='') as infile, open(temp_file, 'w', newline='') as outfile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for row in reader:
                    if (abs(float(row['entry_price']) - self.entry_price) < 0.0001 and 
                        abs(float(row['size']) - self.size) < 0.0001 and
                        row['closed'].lower() == 'true'):
                        order_removed = True
                        continue
                    writer.writerow(row)
            
            os.replace(temp_file, "order_tracker.csv")
            
            if not order_removed:
                print(cwarn(f"Warning: Could not find closed order to remove in CSV: entry={self.entry_price}, size={self.size}"))
        except Exception as e:
            print(cerr(f"Error removing closed order from CSV: {e}"))

    @classmethod
    def load_active_orders_from_csv(cls, symbol=None, backtest=False):
        if backtest:
            return []
            
        active_orders = []
        try:
            if not os.path.isfile("order_tracker.csv"):
                return active_orders
                
            with open("order_tracker.csv", 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['closed'].lower() == 'false':
                        if symbol and row['symbol'] and row['symbol'] != symbol:
                            print(cinfo(f"Skipping order for {row['symbol']} (looking for {symbol})"))
                            continue

                        order = cls.__new__(cls)

                        order.entry_price = float(row['entry_price'])
                        order.size = float(row['size'])
                        order.take_profit_price = float(row['take_profit_price'])
                        order.executed = True
                        order.order_id = row['order_id'] if row['order_id'] else None
                        order.timestamp = datetime.fromisoformat(row['timestamp']) if row['timestamp'] else datetime.now()
                        order.symbol = row['symbol']
                        order.order_type = row['order_type']
                        order.closed = False
                        order.exit_price = None
                        order.exit_timestamp = None
                        order.profit_pct = None
                        order.backtest = backtest
                        
                        print(cinfo(f"Loading order for {order.symbol}: {order.size} @ {order.entry_price}"))
                        active_orders.append(order)
            return active_orders
        except Exception as e:
            print(cerr(f"Error loading orders from CSV: {e}"))
            return []
