import backtrader as bt

'''NOTE :: This is an very crude experimental implementation - and work in progress'''

class VuManchCipherB(bt.Indicator):
    lines = ('wt1', 'wt2', 'wtVwap', 'rsiMFI', 'rsi', 'stochK', 'stochD', 'tcVal',
             'wtCross', 'wtCrossUp', 'wtCrossDown', 'wtOversold', 'wtOverbought')

    params = (
        # WaveTrend parameters - adjusted for more sensitivity
        ('wtChannelLen', 20),     # Reduced from original 54
        ('wtAverageLen', 30),     # Reduced from original 72
        ('wtMALen', 3),           # WT MA Length

        # WT Overbought & Oversold levels - less extreme for more signals
        ('obLevel', 40),          # Reduced from original 53
        ('obLevel2', 50),         # Reduced from original 60
        ('obLevel3', 90),         # Reduced from original 00
        ('osLevel', -40),         # Increased from original -53
        ('osLevel2', -50),        # Increased from original -60
        ('osLevel3', -65),        # Increased from original -75

        ('rsiLen', 14),
        ('rsiOversold', 30),
        ('rsiOverbought', 60),
        ('rsiMFIperiod', 20),
        ('rsiMFIMultiplier', 150.0),
        ('rsiMFIPosY', 2.5),
        ('stochLen', 14),
        ('stochRsiLen', 14),
        ('stochKSmooth', 3),
        ('stochDSmooth', 3),
        ('tcLength', 10),
        ('tcFastLength', 12),    # Reduced from original 23
        ('tcSlowLength', 26),    # Reduced from original 30
        ('tcFactor', 0.5),
    )
    def __init__(self):
        # Set appropriate minimum periods
        self.addminperiod(max(self.p.wtChannelLen, 
                             self.p.wtAverageLen, 
                             self.p.rsiMFIperiod, 
                             self.p.stochLen + self.p.stochRsiLen, 
                             self.p.tcSlowLength))
        
        # Calculate WT components
        hlc3 = (self.data.high + self.data.low + self.data.close) / 3
        
        # WaveTrend calculation
        esa = bt.indicators.EMA(hlc3, period=self.p.wtChannelLen)
        de = bt.indicators.EMA(abs(hlc3 - esa), period=self.p.wtChannelLen)
        ci = (hlc3 - esa) / (0.015 * de)
        self.lines.wt1 = bt.indicators.EMA(ci, period=self.p.wtAverageLen)
        self.lines.wt2 = bt.indicators.SMA(self.lines.wt1, period=self.p.wtMALen)
        self.lines.wtVwap = self.lines.wt1 - self.lines.wt2
        
        # WT cross conditions
        self.lines.wtCross = bt.indicators.CrossOver(self.lines.wt1, self.lines.wt2)
        self.lines.wtCrossUp = bt.indicators.CrossUp(self.lines.wt1, self.lines.wt2)
        self.lines.wtCrossDown = bt.indicators.CrossDown(self.lines.wt1, self.lines.wt2)
        
        # Overbought/Oversold conditions
        self.lines.wtOversold = self.lines.wt2 <= self.p.osLevel
        self.lines.wtOverbought = self.lines.wt2 >= self.p.obLevel
        
        # RSI calculation
        self.lines.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsiLen)
        
        # RSI+MFI calculation - simplified
        self.lines.rsiMFI = self.lines.rsi  # Placeholder, not fully implemented
        
        # Stochastic RSI calculation
        src = self.data.close  # Simplified without log
        rsi_for_stoch = bt.indicators.RSI(src, period=self.p.stochRsiLen)
        
        # Manual implementation of Stochastic RSI - TODO :: Replace with my Stoch later
        highest_rsi = bt.indicators.Highest(rsi_for_stoch, period=self.p.stochLen)
        lowest_rsi = bt.indicators.Lowest(rsi_for_stoch, period=self.p.stochLen)
        
        # Safe division with normalization
        rsi_range = highest_rsi - lowest_rsi
        stoch_k = bt.If(rsi_range > 0, 
                        (rsi_for_stoch - lowest_rsi) / rsi_range * 100,
                        50)  # Default to midpoint if range is zero
        
        # Smooth the Stochastic
        self.lines.stochK = bt.indicators.SMA(stoch_k, period=self.p.stochKSmooth)
        self.lines.stochD = bt.indicators.SMA(self.lines.stochK, period=self.p.stochDSmooth)
        
        # Simplified Schaff Trend Cycle
        fast_ma = bt.indicators.EMA(self.data.close, period=self.p.tcFastLength)
        slow_ma = bt.indicators.EMA(self.data.close, period=self.p.tcSlowLength)
        macd = fast_ma - slow_ma
        self.lines.tcVal = bt.indicators.EMA(macd, period=2)  # Simplified STC
        
        # Buy signal - using CrossUp directly simplifies the logic
        self.buy_signal = bt.And(
            self.lines.wtCrossUp > 0,  # CrossUp returns 1 when crossing up
            self.lines.wtOversold > 0  # wtOversold is 1 when oversold condition is met
        )
        
        # Bottom signal - uses crossover and level check
        self.bottom_signal = bt.And(
            self.lines.wtCrossUp > 0,
            self.lines.wt2 < self.p.osLevel2
        )
        
        # Sell signal
        self.sell_signal = bt.And(
            self.lines.wtCrossDown > 0,  # CrossDown returns 1 when crossing down
            self.lines.wtOverbought > 0  # wtOverbought is 1 when overbought condition is met
        )
        
        # Top signal
        self.top_signal = bt.And(
            self.lines.wtCrossDown > 0,
            self.lines.wt2 > self.p.obLevel2
        )
    
    # Signal identification methods for strategy usage
    def is_buy_signal(self):
        return self.buy_signal[0]
    
    def is_sell_signal(self):
        return self.sell_signal[0]
    
    def is_bottom_signal(self):
        return self.bottom_signal[0]
    
    def is_top_signal(self):
        return self.top_signal[0]