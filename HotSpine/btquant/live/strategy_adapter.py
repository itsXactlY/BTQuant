class BTStrategyAdapter:
    def __init__(self, strategy_cls):
        self.strategy = strategy_cls.__new__(strategy_cls)
        self.strategy.datas = []
        self.strategy.broker = self
        self.strategy.position = 0
        strategy_cls.__init__(self.strategy)

    def buy(self, size=None, price=None):
        print("BUY", size, price)

    def sell(self, size=None, price=None):
        print("SELL", size, price)

    def on_trade(self, trade):
        self.strategy.data = trade
        self.strategy.next()
