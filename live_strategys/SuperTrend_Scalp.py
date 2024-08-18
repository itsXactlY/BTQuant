import backtrader as bt
from live_strategys.live_functions import *
from custom_indicators.SuperTrend import SuperTrend

trade_logger = setup_logger('TradeLogger', 'SuperSTrendScalper_Trade_Monitor.log', level=logging.DEBUG)

class SuperSTrend_Scalper(BaseStrategy):
    params = (
        ("dca_threshold", 2.5),
        ("take_profit", 4),
        ('percent_sizer', 0.01), # 0.01 -> 1%
        # Trend Strenght
        ("adx_period", 13),
        ("adx_strength", 31),
        ("di_period", 14),
        ("adxth", 25),

        # Supertrends
        ("st_fast", 2),
        ('st_fast_multiplier', 3),
        ("st_slow", 6),
        ('st_slow_multiplier', 7),
        # RevFinder
        ('reversal_lookback', 10),
        ('reversal_malen', 40),
        ('reversal_mult', 2.2),
        ('reversal_rangethreshold', 0.9),
        ('debug', False),
        ("backtest", None)
        )

    def __init__(self):
        super().__init__()
        self.adx = bt.indicators.ADX(self.data, plot=False)
        self.plusDI = bt.indicators.PlusDI(self.data, plot=False)
        self.minusDI = bt.indicators.MinusDI(self.data, plot=False)
        self.supertrend_fast = SuperTrend(period=self.p.st_fast, multiplier=self.p.st_fast_multiplier, plotname='SuperTrend Fast: ', plot=True)
        self.supertrend_slow = SuperTrend(period=self.p.st_slow, multiplier=self.p.st_slow_multiplier, plotname='SuperTrend Slow: ', plot=True)
        self.supertrend_uptrend_signal = bt.indicators.CrossOver(self.supertrend_fast, self.supertrend_slow, plot=False)
        # self.supertrend_downtrend_signal = bt.indicators.CrossDown(self.supertrend_fast, self.supertrend_slow, plot=False) # NOT USED IN THIS EXAMPLE
        self.DCA = True
        self.buy_executed = False
        self.conditions_checked = False
        
        # Forensic Logging
        self.trade_cycles = 0
        self.total_profit_usd = 0
        self.last_profit_usd = 0
        self.start_time = datetime.utcnow()
        self.position_start_time = None
        self.max_buys_per_cycle = 0
        self.total_buys = 0
        self.current_cycle_buys = 0

    def buy_or_short_condition(self):
        if (
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):

                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    self.log_entry()
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.log_entry()

    def dca_or_short_condition(self):
        if (self.position and \
            self.adx[0] >= self.params.adxth and \
            self.minusDI[0] > self.params.adxth and \
            self.plusDI[0] < self.params.adxth and \
            self.supertrend_uptrend_signal
        ):
            
            if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):    
                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.enqueue_order('buy', exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                    self.log_entry()
                elif self.params.backtest == True:
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.log_entry()

    def sell_or_cover_condition(self):
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

        if self.buy_executed and self.data.close[0] >= self.take_profit_price:
            if self.data.close[0] < self.average_entry_price:
                print(f'Nothing Todo here. {self.average_entry_price, self.take_profit_price}')
                return
            
            if self.params.backtest == False:
                self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
            
            self.log_exit("Sell Signal - Take Profit")
            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    def next(self):
        BaseStrategy.next(self) 

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

    def stop(self):
        self.order_queue.put(None)
        self.order_thread.join()
        print('Final Portfolio Value: %.2f' % self.broker.getvalue())