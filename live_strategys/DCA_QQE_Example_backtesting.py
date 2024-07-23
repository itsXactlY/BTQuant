from live_strategys.live_functions import *
import backtrader as bt

trade_logger = setup_logger('TradeLogger', 'QQE_DCA_Example_Trade_Monitor.log', level=logging.DEBUG)

class VolumeOscillator(bt.Indicator):
    lines = ('osc',)
    params = (('shortlen', 5),
            ('longlen', 10))
    
    def __init__(self):
        shortlen, longlen = self.params.shortlen, self.params.longlen
        self.lines.short = bt.indicators.ExponentialMovingAverage(self.data.volume, period=shortlen)
        self.lines.long = bt.indicators.ExponentialMovingAverage(self.data.volume, period=longlen)
        self.lines.osc = (self.lines.short - self.lines.long) / self.lines.long * 100

    def next(self):
        self.osc[0] = (self.lines.short[0] - self.lines.long[0]) / self.lines.long[0] * 100

class QQEIndicator(bt.Indicator):
    params = (
        ("period", 6),
        ("fast", 5),
        ("q", 3.0),
    )
    lines = ("qqe_line",)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.fast)
        self.matr = bt.indicators.EMA(self.atr, period=int((self.p.period * 2) - 1))
        self.dar = bt.indicators.EMA(self.atr - self.p.q, period=int((self.p.period * 2) - 1))
        self.lines.qqe_line = self.rsi + self.dar

live = True

class QQE_DCA_Example(BaseStrategy):
    params = (
        ('dca_threshold', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.01), # 0.01 -> 1%
        ("ema_length", 20),
        ('hull_length', 53),
        ("printlog", True),
        ("backtest", None)
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.qqe = QQEIndicator(self.data)
        self.hma = bt.indicators.HullMovingAverage(self.data, period=self.p.hull_length)
        self.ema = bt.indicators.EMA(self.data.close, period=self.params.ema_length)
        self.volosc = VolumeOscillator(self.data)
        self.DCA = True
        self.buy_executed = False
        self.conditions_checked = False

    def buy_or_short_condition(self):
        if not self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
                (self.data.close[-1] > self.hma[0]) and \
                (self.volosc.osc[-1] > self.volosc.lines[0]
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
        if self.buy_executed and not self.conditions_checked:
            if (self.qqe.qqe_line[-1] > 0) and \
                (self.data.close[-1] > self.hma[0]) and \
                (self.volosc.osc[-1] > self.volosc.lines[0]
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

            if self.params.backtest == False:
                self.enqueue_order('sell', exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
            
            self.log_exit("Sell Signal - Take Profit")
            self.reset_position_state()
            self.buy_executed = False
            self.conditions_checked = True

    def next(self):
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

    def stop(self):
        self.order_queue.put(None)
        self.order_thread.join()
        print('Final Portfolio Value: %.2f' % self.broker.getvalue())