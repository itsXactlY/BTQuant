from fastquant.strategys.base import BaseStrategy

class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy with a fixed budget
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.price = self.data.close
        self.order = None
        self.bought = False
        self.amount = 0
        self.initial_cash = self.broker.getcash()

    def next(self):
        if not self.bought and not self.position:
            cash = self.broker.getcash()
            # Calculate the maximum amount we can buy, accounting for commission
            max_size = cash / self.price[0]
            # Use 99% of the maximum to ensure we dont exceed available cash
            self.amount = 0.99 * max_size
            self.order = self.buy(size=self.amount)
            self.bought = True

    def stop(self):
        self.final_value = self.broker.getvalue()
        self.pnl = self.final_value - self.initial_cash
        print(f"Initial cash: ${self.initial_cash:.2f}")
        print(f"End value: ${self.final_value:.2f}")
        print(f"Profit: ${self.pnl:.2f}")
        print(f"Assets held: {self.amount:.8f}")
        print(f"Final Asset price: ${self.price[0]:.2f}")