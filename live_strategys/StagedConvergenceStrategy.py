from .base import BaseStrategy, bt
import backtrader as bt
from datetime import datetime

import logging

def setup_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
trade_logger = setup_logger('TradeLogger', 'StagedConvergenceStrategy_Trade_Monitor.log', level=logging.DEBUG)

class StagedConvergenceStrategy(BaseStrategy):
    params = (
        ('fast1', 5),
        ('slow1', 50),
        ('fast2', 10),
        ('slow2', 100),
        ('fast3', 20),
        ('slow3', 200),
        ('hold1', 50),
        ('hold2', 35),
        ('hold3', 0),
        ('dca_threshold', 3),
        ('take_profit', 4),  # 2% take profi
        ('percent_sizer', 0.15), # 0.01 -> 1%
        ('stop_loss', 20),   # 20% stop loss
        ('debug', False),
        ('backtest', None),
        ('use_stoploss', False),
    )

    def __init__(self):
        super().__init__()

        self.sma1_fast = bt.indicators.SMA(period=self.params.fast1)
        self.sma1_slow = bt.indicators.SMA(period=self.params.slow1)
        self.sma2_fast = bt.indicators.SMA(period=self.params.fast2)
        self.sma2_slow = bt.indicators.SMA(period=self.params.slow2)
        self.sma3_fast = bt.indicators.SMA(period=self.params.fast3)
        self.sma3_slow = bt.indicators.SMA(period=self.params.slow3)

        self.hold_counter1 = 0
        self.hold_counter2 = 0
        self.hold_counter3 = 0

        self.entry_price = None
        self.buy_executed = False
        self.conditions_checked = False
        self.DCA = True

        # FORENSIC LOGGING
        self.trade_cycles = 0
        self.total_profit_usd = 0
        self.last_profit_usd = 0
        self.start_time = datetime.utcnow()
        self.position_start_time = None
        self.max_buys_per_cycle = 0
        self.total_buys = 0
        self.current_cycle_buys = 0

    def buy_or_short_condition(self):
        buy_signal = (self.sma1_fast[0] > self.sma1_slow[0] and
                      self.sma2_fast[0] > self.sma2_slow[0] and
                      self.sma3_fast[0] > self.sma3_slow[0])

        if buy_signal and self.hold_counter1 == 0 and self.hold_counter2 == 0 and self.hold_counter3 == 0:
            if self.params.backtest == False:
                self.entry_prices.append(self.data.close[0])
                print(f'\n\n\nBUY EXECUTED AT {self.data.close[0]}\n\n\n')
                self.sizes.append(self.amount)
                self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss / 100)
                self.calc_averages()
                self.buy_executed = True
                self.conditions_checked = True
                self.position_start_time = datetime.utcnow()
                self.log_entry()
            elif self.params.backtest == True:
                self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss / 100)
                self.buy_executed = True
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(self.stake)
                self.calc_averages()
                self.position_start_time = datetime.utcnow()
                self.log_entry()

    def dca_or_short_condition(self):
        buy_signal = (self.sma1_fast[0] > self.sma1_slow[0] and
                      self.sma2_fast[0] > self.sma2_slow[0] and
                      self.sma3_fast[0] > self.sma3_slow[0])
        if buy_signal and self.hold_counter1 == 0 and self.hold_counter2 == 0 and self.hold_counter3 == 0:
            
            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):    
                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss / 100)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    self.log_entry()
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.stop_loss_price = self.data.close[0] * (1 - self.params.stop_loss / 100)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.log_entry()

    def check_stop_loss(self):
        if self.buy_executed and self.stop_loss_price is not None:
            current_price = self.data.close[0]
            if current_price <= self.stop_loss_price:                
                if self.params.backtest == False:
                    self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
                elif self.params.backtest == True:
                    self.close()
                
                print(f'STOP LOSS TRIGGERED {self.stop_loss_price:.12f}')
                
                self.log_exit("Stop Loss")
                self.reset_position_state()
                self.conditions_checked = True
                return True
        return False

    def sell_or_cover_condition(self):
        if self.p.debug:
            print(f'| - sell_or_cover_condition {self.data._name} Entry:{self.average_entry_price:.12f} TakeProfit: {self.take_profit_price:.12f}')
        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            average_entry_price = sum(self.entry_prices) / len(self.entry_prices) if self.entry_prices else 0

            # Avoid selling at a loss or below the take profit price
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                self.log(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )
                return

        sell_signal = (self.sma1_fast[0] < self.sma1_slow[0] and
                       self.sma2_fast[0] < self.sma2_slow[0] and
                       self.sma3_fast[0] < self.sma3_slow[0])

        if self.buy_executed and (self.data.close[0] >= self.take_profit_price): # Example: >= self.take_profit_price or sell_signal 
            if self.params.backtest == False:
                print(f'\n\n\nSELL EXECUTED AT {self.data.close[0]}\n\n\n')
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()

            self.log_exit("Sell Signal" if sell_signal else "Take Profit")
            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    def next(self):
        # Update holding periods
        if self.hold_counter1 > 0:
            self.hold_counter1 -= 1
        if self.hold_counter2 > 0:
            self.hold_counter2 -= 1
        if self.hold_counter3 > 0:
            self.hold_counter3 -= 1

        # Reset holding periods when signal changes
        if (self.sma1_fast[0] > self.sma1_slow[0]) != (self.sma1_fast[-1] > self.sma1_slow[-1]):
            self.hold_counter1 = self.params.hold1
        if (self.sma2_fast[0] > self.sma2_slow[0]) != (self.sma2_fast[-1] > self.sma2_slow[-1]):
            self.hold_counter2 = self.params.hold2

        if self.p.use_stoploss == True:
            self.check_stop_loss()
        BaseStrategy.next(self)

        # Reset conditions_checked flag for the new candle
        self.conditions_checked = False

    def log_entry(self):
        trade_logger.debug("-" * 100)
        self.total_buys += 1
        self.current_cycle_buys += 1
        self.max_buys_per_cycle = max(self.max_buys_per_cycle, self.current_cycle_buys)

        trade_logger.debug(f"{datetime.utcnow()} - Buy executed: {self.data._name}")
        trade_logger.debug(f"Entry price: {self.entry_prices[-1]:.12f}")
        trade_logger.debug(f"Position size: {self.sizes[-1]}")
        trade_logger.debug(f"Current cash: {self.broker.getcash():.2f}")
        trade_logger.debug(f"Current portfolio value: {self.broker.getvalue():.2f}")
        trade_logger.debug("*" * 100)

    def log_exit(self, exit_type):
        trade_logger.info("-" * 100)
        trade_logger.info(f"{datetime.utcnow()} - {exit_type} executed: {self.data._name}")
        
        position_size = sum(self.sizes)
        exit_price = self.data.close[0]
        profit_usd = (exit_price - self.average_entry_price) * position_size
        self.last_profit_usd = profit_usd
        self.total_profit_usd += profit_usd
        self.trade_cycles += 1
        
        trade_logger.info(f"Exit price: {exit_price:.12f}")
        trade_logger.info(f"Average entry price: {self.average_entry_price:.12f}")
        trade_logger.info(f"Position size: {position_size}")
        trade_logger.info(f"Profit for this cycle (USD): {profit_usd:.2f}")
        trade_logger.info(f"Total profit (USD): {self.total_profit_usd:.2f}")
        trade_logger.info(f"Trade cycles completed: {self.trade_cycles}")
        trade_logger.info(f"Average profit per cycle (USD): {self.total_profit_usd / self.trade_cycles:.2f}")
        trade_logger.info(f"Time elapsed: {datetime.utcnow() - self.start_time}")
        if self.position_start_time:
            trade_logger.info(f"Position cycle time: {datetime.utcnow() - self.position_start_time}")
        trade_logger.info(f"Maximum buys per cycle: {self.max_buys_per_cycle}")
        trade_logger.info(f"Total buys: {self.total_buys}")
        trade_logger.info("*" * 100)
        
        self.current_cycle_buys = 0
        self.position_start_time = None
