from .base import BaseStrategy, bt, np, datetime

class OrderChainIndicator(bt.Indicator):
    lines = ('order_chain',)
    plotinfo = dict(subplot=True)
    params = (
        ('chain', 150),
        ('ticks', 50),
        ('tick_size', 0.00010),
        ('max_levels', 1000),
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

        while self.c > self.levels[-1] and len(self.levels) < self.p.max_levels:
            new_level = self.levels[-1] + self.p.tick_size * self.p.ticks
            self.levels.append(new_level)
            self.ladder.append({'delta': 0, 'dCol': 'white'})
        self.rel_data['Top'] = self.levels[-1]

        while self.c < self.levels[0] and len(self.levels) < self.p.max_levels:
            new_level = self.levels[0] - self.p.tick_size * self.p.ticks
            self.levels.insert(0, new_level)
            self.ladder.insert(0, {'delta': 0, 'dCol': 'white'})
        self.rel_data['Bot'] = self.levels[0]

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
        ('signal_threshold', 1000),
        ('debug', False),
        ('backtest', None),
        ("dca_deviation", 2.5),
        ("take_profit", 2),
        ('percent_sizer', 0.25),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order_chain = OrderChainIndicator(chain=self.p.chain, ticks=self.p.ticks, tick_size=self.p.tick_size)
        self.DCA = True

    def buy_or_short_condition(self):
        if self.order_chain.lines.order_chain[0] > self.p.signal_threshold:
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        if (self.entry_prices and 
            self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100) and
            self.order_chain.lines.order_chain[0] > self.p.signal_threshold):
            self.create_order(action='BUY')
            return True
        return False
    
    def sell_or_cover_condition(self):
        current_price = self.data.close[0]
        for order_tracker in list(self.active_orders):
            if current_price >= order_tracker.take_profit_price:
                self.close_order(order_tracker)
                return True
        return False