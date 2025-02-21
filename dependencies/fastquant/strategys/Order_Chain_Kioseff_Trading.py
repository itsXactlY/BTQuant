# This was an "Proof of Concept" and mutated into something more solid than i personally expected, when fed with one-second-data or an one-second-live-feed.
# Originally inspired by: https://www.tradingview.com/script/JNCGeDj7-Order-Chain-Kioseff-Trading/

import backtrader as bt
from fastquant.strategys.base import BaseStrategy, np, BuySellArrows

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
        ('chain', 1500),
        ('ticks', 150),
        ('tick_size', 0.00010),
        ('signal_threshold', 1000),  # 1000 threshold for generating signals
        ('debug', False),
        ('backtest', None),
        ("dca_deviation", 2.5),
        ("take_profit_percent", 2),
        ('percent_sizer', 0.001), # 0.01 -> 1%
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order_chain = OrderChainIndicator(chain=self.p.chain, ticks=self.p.ticks, tick_size=self.p.tick_size)

        self.buy_executed = False
        self.conditions_checked = False 
        self.DCA = True

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
                    self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market) 
                    self.buy_executed = True
                    self.entry_prices.append(self.data.close[0])
                    self.sizes.append(self.stake)
                    self.calc_averages()
                    self.conditions_checked = True


    def dca_or_short_condition(self):
        if self.order_chain.lines.order_chain[0] > self.p.signal_threshold:
            if self.p.debug and self.buy_executed:
                print(f'| {datetime.utcnow()} - dca_or_short_condition {self.data._name} Entry:{self.average_entry_price:.12f} TakeProfit: {self.take_profit_price:.12f}')
            if self.buy_executed and not self.conditions_checked:
                if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
                    if self.params.backtest == False:
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.amount)
                        self.load_trade_data()
                        self.rabbit.send_jrr_buy_request(exchange=self.exchange, account=self.account, asset=self.asset, amount=self.amount)
                        self.calc_averages()
                        self.buy_executed = True
                        self.conditions_checked = True
                    elif self.params.backtest == True:
                        self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                        self.buy_executed = True
                        self.entry_prices.append(self.data.close[0])
                        self.sizes.append(self.stake)
                        self.calc_averages()
                        self.conditions_checked = True


    def sell_or_cover_condition(self):
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
            
            if self.params.backtest == False:
                self.rabbit.send_jrr_close_request(exchange=self.exchange, account=self.account, asset=self.asset)
            elif self.params.backtest == True:
                self.close()
            
            self.reset_position_state()
            self.conditions_checked = True