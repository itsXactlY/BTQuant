from backtrader.strategies.base import BaseStrategy, bt

class SMA_Cross_Simple(BaseStrategy):
    """
    Simple SMA Crossover Strategy
    - Buy when short SMA crosses above long SMA
    - Sell when short SMA crosses below long SMA
    """
    
    params = (
        ('short_period', 10),
        ('long_period', 30),
        ('stop_loss_pct', 0.02),  # 2% stop loss
        ('take_profit_pct', 0.05),  # 5% take profit
        ('size', 0.1),  # Position size
        ('debug', False),
    )

    def __init__(self, **kwargs):
        try:
            super().__init__(**kwargs)
            
            # Core SMA indicators
            self.sma_short = bt.ind.SMA(period=self.p.short_period)
            self.sma_long = bt.ind.SMA(period=self.p.long_period)
            self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)
            
            # Track position state
            self.in_position = False
            self.entry_price = 0.0
            
            if self.p.debug:
                print(f"SMA_Cross_Simple initialized: short={self.p.short_period}, long={self.p.long_period}")
                
        except Exception as e:
            print(f"Strategy initialization error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def buy_or_short_condition(self):
        """Buy signal: short SMA crosses above long SMA"""
        # Crossover > 0 means short SMA just crossed above long SMA
        if self.crossover > 0 and not self.in_position:
            if self.p.debug:
                print(f"BUY SIGNAL: SMA{self.p.short_period} ({self.sma_short[0]:.4f}) crossed above SMA{self.p.long_period} ({self.sma_long[0]:.4f})")
            
            # Create buy order
            self.create_order(action='BUY')
            return True
        return False

    def sell_or_cover_condition(self):
        """Sell signal: short SMA crosses below long SMA OR stop loss/take profit hit"""
        if not self.in_position:
            return False
            
        current_price = self.data.close[0]
        
        # Check stop loss
        if self.entry_price > 0:
            drawdown = (current_price - self.entry_price) / self.entry_price
            if drawdown <= -self.p.stop_loss_pct:
                if self.p.debug:
                    print(f"STOP LOSS: Price {current_price:.4f} dropped {drawdown:.2%} from entry {self.entry_price:.4f}")
                self.create_order(action='SELL')
                return True
            
            # Check take profit
            if drawdown >= self.p.take_profit_pct:
                if self.p.debug:
                    print(f"TAKE PROFIT: Price {current_price:.4f} gained {drawdown:.2%} from entry {self.entry_price:.4f}")
                self.create_order(action='SELL')
                return True
        
        # Check SMA crossover for exit
        if self.crossover < 0:
            if self.p.debug:
                print(f"SELL SIGNAL: SMA{self.p.short_period} ({self.sma_short[0]:.4f}) crossed below SMA{self.p.long_period} ({self.sma_long[0]:.4f})")
            self.create_order(action='SELL')
            return True
            
        return False

    def next(self):
        """Process each new data bar"""
        super().next()
        
        # Check for buy signals first
        if self.buy_or_short_condition():
            return
            
        # Then check for sell signals
        if self.sell_or_cover_condition():
            return

    def stop(self):
        """Clean up and report performance"""
        super().stop()
        if self.p.debug:
            print("SMA_Cross_Simple strategy stopped")