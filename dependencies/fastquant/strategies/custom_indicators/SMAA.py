import backtrader as bt

class SSMA(bt.Indicator):
    lines = ('ssma',)
    params = (
        ('period', 20),
        ('smoothing', 0.5),
        ('sensitivity', 0.3),
    )
    
    def __init__(self):
        self.ma = bt.indicators.SimpleMovingAverage(self.data, period=self.p.period)
        self.smoothed_ma = 0
        self.ssma_value = 0
        
    def next(self):
        if len(self) <= self.p.period:
            self.smoothed_ma = self.ma[0]
            self.ssma_value = self.ma[0]
            self.lines.ssma[0] = self.ma[0]
        else:
            self.smoothed_ma = (self.ma[0] * (1 - self.p.smoothing) + 
                               self.smoothed_ma * self.p.smoothing)
            
            self.ssma_value = (self.smoothed_ma * (1 - self.p.sensitivity) + 
                              self.data[0] * self.p.sensitivity)
            
            self.lines.ssma[0] = self.ssma_value