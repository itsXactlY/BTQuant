# This was an "Proof of Concept" and mutated into something more solid than i personally expected, when fed with one-second-data or an one-second-live-feed.
# Originally inspired by: https://www.tradingview.com/script/JNCGeDj7-Order-Chain-Kioseff-Trading/

import backtrader as bt
from .base import BaseStrategy, np, datetime
# import logging
# from logging.handlers import RotatingFileHandler

# def setup_logger(name, log_file, level=logging.INFO):
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler = RotatingFileHandler(log_file)
#     handler.setFormatter(formatter)
#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.addHandler(handler)
#     return logger

# trade_logger = setup_logger('TradeLogger', 'BaseStrategy_Trade_Monitor.log', level=logging.DEBUG)


class OrderChainIndicator(bt.Indicator):
    lines = ('order_chain',)
    plotinfo = dict(subplot=True)

    params = (
        ('chain', 150),
        ('ticks', 50),
        ('tick_size', 0.00010),
        ('max_levels', 1000),  # Max number of price levels to track
    )

    def __init__(self):
        self.market_orders = []
        self.levels = []
        self.ladder = []
        self.rel_data = {
            'Top': -float('inf'),
            'Bot': float('inf'),
            'Abs': 0,
            'Max': -float('inf')
        }
        self.vol = 0
        self.vol1 = 0
        self.c = 0
        self.c1 = 0

    def next(self):
        self.vol1 = self.vol
        self.vol = self.data.volume[0]
        self.c1 = self.c
        self.c = self.data.close[0]

        if not self.levels:
            self.levels = [
                self.c - self.p.tick_size * self.p.ticks,
                self.c,
                self.c + self.p.tick_size * self.p.ticks
            ]
            self.ladder = [{'delta': 0, 'dCol': 'white'} for _ in range(3)]
            self.rel_data['Top'] = self.levels[-1]
            self.rel_data['Bot'] = self.levels[0]

        # update top levels
        while self.c > self.levels[-1] and len(self.levels) < self.p.max_levels:
            new_level = self.levels[-1] + self.p.tick_size * self.p.ticks
            self.levels.append(new_level)
            self.ladder.append({'delta': 0, 'dCol': 'white'})
        self.rel_data['Top'] = self.levels[-1]

        # update bottom levels
        while self.c < self.levels[0] and len(self.levels) < self.p.max_levels:
            new_level = self.levels[0] - self.p.tick_size * self.p.ticks
            self.levels.insert(0, new_level)
            self.ladder.insert(0, {'delta': 0, 'dCol': 'white'})
        self.rel_data['Bot'] = self.levels[0]

        # trim excess levels if needed
        if len(self.levels) > self.p.max_levels:
            mid_index = len(self.levels) // 2
            self.levels = self.levels[mid_index - self.p.max_levels//2 : mid_index + self.p.max_levels//2]
            self.ladder = self.ladder[mid_index - self.p.max_levels//2 : mid_index + self.p.max_levels//2]

        if self.vol > self.vol1 and self.vol1 != 0 and self.c != self.c1:
            direction = np.sign(self.c - self.c1)
            col = 'green' if direction > 0 else 'red'
            vol_flow = (self.vol - self.vol1) * direction

            self.market_orders.append({
                'price': self.c,
                'volFlow': vol_flow,
                'location': 1,
                'bgcol': col
            })

            if len(self.market_orders) > self.p.chain:
                self.market_orders.pop(0)

            self.rel_data['Abs'] = max(self.rel_data['Abs'], abs(vol_flow))
            self.rel_data['Max'] = max(self.rel_data['Max'], self.c)

            index = self.get_level_index(self.c)
            if 0 <= index < len(self.ladder):
                self.ladder[index]['delta'] += vol_flow
                self.ladder[index]['dCol'] = 'green' if self.ladder[index]['delta'] > 0 else 'red' if self.ladder[index]['delta'] < 0 else 'white'

        self.lines.order_chain[0] = sum(order['volFlow'] for order in self.market_orders)

    def get_level_index(self, price):
        return min(range(len(self.levels)), key=lambda i: abs(self.levels[i] - price))

class Order_Chain_Kioseff_Trading(BaseStrategy):
    params = (
        ('chain', 150),
        ('ticks', 50),
        ('tick_size', 0.00010),
        ('signal_threshold', 100),  # 1000 threshold for generating signals
        ('debug', False),
        ('backtest', None),
        ("dca_threshold", 0.1),
        ("take_profit", 0.1),
        ('percent_sizer', 0.01), # 0.01 -> 1%
    )

    def __init__(self):
        super().__init__()
        self.order_chain = OrderChainIndicator(chain=self.p.chain, ticks=self.p.ticks, tick_size=self.p.tick_size)

        self.buy_executed = False
        self.conditions_checked = False 
        self.DCA = True
        
        ### FORENSIC LOGGING
        self.trade_cycles = 0
        self.total_profit_usd = 0
        self.last_profit_usd = 0
        self.start_time = datetime.utcnow()
        self.position_start_time = None
        self.max_buys_per_cycle = 0
        self.total_buys = 0
        self.current_cycle_buys = 0


    def buy_or_short_condition(self):
        if self.order_chain.lines.order_chain[0] > self.p.signal_threshold:
            if self.p.debug == True:
                print(f'| {datetime.utcnow()} - buy_or_short_condition {self.data._name}')
            if not self.buy_executed and not self.conditions_checked:
                if self.params.backtest == False:
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.amount)
                    self.load_trade_data()
                    self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                    self.calc_averages()
                    self.buy_executed = True
                    self.conditions_checked = True
                elif self.params.backtest == True:
                    if self.broker.getcash() <5.0:
                        if self.p.debug:
                            print('EMERGENCY: OUT OF CASH. ACCOUNT LIQUIDATED')
                        return
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market) 
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    
                    # ### FORENSIC LOGGING
                    # trade_logger.debug("-" * 100)
                    # self.total_buys += 1
                    # self.current_cycle_buys += 1
                    # self.max_buys_per_cycle = max(self.max_buys_per_cycle, self.current_cycle_buys)
                    # if self.position_start_time is None:
                    #     self.position_start_time = datetime.utcnow()

                    # trade_logger.debug(f"{datetime.utcnow()} - buy_or_short_condition triggered: {self.data._name}")
                    # trade_logger.debug(f"Current close price: {self.data.close[0]:.12f}")
                    # trade_logger.debug(f"Position size: {self.position.size}")
                    # trade_logger.debug(f"Current cash: {self.broker.getcash():.2f}")
                    # trade_logger.debug(f"Current COIN value in USD: {self.broker.getvalue():.2f}")
                    # trade_logger.debug(f"Entry prices: {self.entry_prices}")
                    # trade_logger.debug(f"Sizes: {self.sizes}")
                    # trade_logger.debug("*" * 100)

    def dca_or_short_condition(self):
        if self.order_chain.lines.order_chain[0] > self.p.signal_threshold:
            if self.p.debug and self.buy_executed:
                print(f'| {datetime.utcnow()} - dca_or_short_condition {self.data._name} Entry:{self.average_entry_price:.12f} TakeProfit: {self.take_profit_price:.12f}')
            if self.buy_executed and not self.conditions_checked:
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_threshold / 100):
                    if self.params.backtest == False:
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.amount)
                        self.load_trade_data()
                        self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                        self.calc_averages()
                        self.buy_executed = True
                        self.conditions_checked = True
                    elif self.params.backtest == True:
                        if self.broker.getcash() <5.0:
                            if self.p.debug:
                                print('EMERGENCY: OUT OF CASH. ACCOUNT LIQUIDATED')
                            return
                        self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                        self.buy_executed = True
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.stake)
                        self.calc_averages()
                        
                        # ### FORENSIC LOGGING
                        # trade_logger.debug("-" * 100)
                        # self.total_buys += 1
                        # self.current_cycle_buys += 1
                        # self.max_buys_per_cycle = max(self.max_buys_per_cycle, self.current_cycle_buys)
                        # if self.buy_executed:
                        #     trade_logger.debug(
                        #         f"{datetime.utcnow()} - dca_or_short_condition triggered: {self.data._name} "
                        #         f"Entry price: {self.average_entry_price:.12f} "
                        #         f"Take profit price: {self.take_profit_price:.12f}"
                        #     )

                        #     trade_logger.debug(f"Position size: {self.position.size}")
                        #     trade_logger.debug(f"Current cash: {self.broker.getcash():.2f}")
                        #     trade_logger.debug(f"Current COIN value in USD: {self.broker.getvalue():.2f}")
                        #     trade_logger.debug(f"Entry prices: {self.entry_prices}")
                        #     trade_logger.debug(f"Sizes: {self.sizes}")
                        #     trade_logger.debug("*" * 100)

    def sell_or_cover_condition(self):
        # if self.p.debug:
        #     trade_logger.debug(
        #         f"{datetime.utcnow()} - sell_or_cover_condition triggered: {self.data._name} "
        #         f"Entry price: {self.average_entry_price:.12f} "
        #         f"Take profit price: {self.take_profit_price:.12f}"
        #     )

        if self.buy_executed and self.data.close[0] >= self.take_profit_price or self.order_chain.lines.order_chain[0] < -self.p.signal_threshold:

            # Avoid selling at a loss or below the take profit price
            if round(self.data.close[0], 9) < round(self.average_entry_price, 9) or round(self.data.close[0], 9) < round(self.take_profit_price, 9):
                if self.p.debug == True: 
                    print(
                    f"| - Avoiding sell at a loss or below take profit. "
                    f"| - Current close price: {self.data.close[0]:.12f}, "
                    f"| - Average entry price: {self.average_entry_price:.12f}, "
                    f"| - Take profit price: {self.take_profit_price:.12f}"
                )
                return
            # trade_logger.info("-" * 100)
            # trade_logger.info(f"{datetime.utcnow()} - Sell operation executed: {self.data._name}")
            # trade_logger.debug(f"Current cash: {self.broker.getcash():.2f}")
            # trade_logger.debug(f"Current COIN value in USD: {self.broker.getvalue():.2f}")
            
            if self.params.backtest == False:
                self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
                
            ### FORENSIC LOGGING
            # Calculate profit for this trade cycle
            position_size = sum(self.sizes)
            profit_usd = (self.data.close[0] - self.average_entry_price) * position_size
            self.last_profit_usd = profit_usd
            self.total_profit_usd += profit_usd
            self.trade_cycles += 1
            
            # Log the sell operation and metrics
            # trade_logger.info(f"Sell price: {self.data.close[0]:.12f}")
            # trade_logger.info(f"Average entry price: {self.average_entry_price:.12f}")
            # trade_logger.info(f"Take profit price: {self.take_profit_price:.12f}")
            # trade_logger.info(f"Position size: {position_size}")
            # trade_logger.info(f"Profit for this cycle (USD): {profit_usd:.2f}")
            # trade_logger.info(f"Total profit (USD): {self.total_profit_usd:.2f}")
            # trade_logger.info(f"Trade cycles completed: {self.trade_cycles}")
            # trade_logger.info(f"Average profit per cycle (USD): {self.total_profit_usd / self.trade_cycles:.2f}")
            # trade_logger.info(f"Time elapsed: {datetime.utcnow() - self.start_time}")
            # if self.position_start_time:
            #     trade_logger.info(f"Position cycle time: {datetime.utcnow() - self.position_start_time}")
            # trade_logger.info(f"Maximum buys per cycle: {self.max_buys_per_cycle}")
            # trade_logger.info(f"Total buys: {self.total_buys}")
            # trade_logger.info("*" * 100)
            
            self.reset_position_state()
            self.conditions_checked = True
            self.current_cycle_buys = 0
            self.position_start_time = None
