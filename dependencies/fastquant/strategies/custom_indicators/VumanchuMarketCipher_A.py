import backtrader as bt

'''NOTE :: This is an very crude experimental implementation - and work in progress'''

class VuManchCipherA(bt.Indicator):
    '''
    VuManChu Cipher A indicator for Backtrader, based on TradingView script
    '''
    lines = ('wt1', 'wt2', 'ema1', 'ema2', 'ema3', 'ema4', 'ema5', 'ema6', 'ema7', 'ema8', 'rsi', 'rsi_mfi',
             'long_ema', 'red_cross', 'blue_triangle', 'red_diamond', 'blood_diamond', 'yellow_cross', 'bull_candle', 'short_ema')
    
    # Params based on the original TradingView script
    params = (
        # WaveTrend parameters
        ('wt_channel_len', 9),
        ('wt_average_len', 13),
        ('wt_ma_len', 3),
        # WaveTrend Oversold/Overbought levels
        ('ob_level', 53),
        ('ob_level2', 60),
        ('ob_level3', 100),
        ('os_level', -53),
        ('os_level2', -60),
        ('os_level3', -80),
        # EMA Ribbon parameters
        ('show_ribbon', True),
        ('ema1_len', 5),
        ('ema2_len', 11),
        ('ema3_len', 15),
        ('ema4_len', 18),
        ('ema5_len', 21),
        ('ema6_len', 24),
        ('ema7_len', 28),
        ('ema8_len', 34),
        # RSI parameters
        ('rsi_len', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 60),
        # RSI+MFI parameters
        ('rsi_mfi_show', True),
        ('rsi_mfi_period', 60),
        ('rsi_mfi_multiplier', 150),
    )
    
    def __init__(self):
        # WaveTrend components
        self.wt_ma_source = (self.data.high + self.data.low + self.data.close) / 3  # hlc3
        
        # WaveTrend
        self.esa = bt.indicators.EMA(self.wt_ma_source, period=self.p.wt_channel_len)
        self.de = bt.indicators.EMA(abs(self.wt_ma_source - self.esa), period=self.p.wt_channel_len)
        self.ci = (self.wt_ma_source - self.esa) / (0.015 * self.de)
        self.tci = bt.indicators.EMA(self.ci, period=self.p.wt_average_len)
        self.lines.wt1 = self.tci
        self.lines.wt2 = bt.indicators.SMA(self.lines.wt1, period=self.p.wt_ma_len)
        
        # EMA Ribbon
        self.lines.ema1 = bt.indicators.EMA(self.data.close, period=self.p.ema1_len)
        self.lines.ema2 = bt.indicators.EMA(self.data.close, period=self.p.ema2_len)
        self.lines.ema3 = bt.indicators.EMA(self.data.close, period=self.p.ema3_len)
        self.lines.ema4 = bt.indicators.EMA(self.data.close, period=self.p.ema4_len)
        self.lines.ema5 = bt.indicators.EMA(self.data.close, period=self.p.ema5_len)
        self.lines.ema6 = bt.indicators.EMA(self.data.close, period=self.p.ema6_len)
        self.lines.ema7 = bt.indicators.EMA(self.data.close, period=self.p.ema7_len)
        self.lines.ema8 = bt.indicators.EMA(self.data.close, period=self.p.ema8_len)
        
        # RSI
        self.lines.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_len)
        
        # RSI+MFI calculation
        self.rsi_mfi_mean = bt.indicators.SMA(
            (self.data.close - self.data.open) / (self.data.high - self.data.low) * self.p.rsi_mfi_multiplier,
            period=self.p.rsi_mfi_period
        )
        self.lines.rsi_mfi = self.rsi_mfi_mean
        
        # Signal indicators defined as lines
        self.lines.long_ema = bt.ind.CrossOver(self.lines.ema2, self.lines.ema8)
        self.lines.red_cross = bt.ind.CrossOver(self.lines.ema1, self.lines.ema2) * -1  # Crossunder
        self.lines.blue_triangle = bt.ind.CrossOver(self.lines.ema2, self.lines.ema3)
        self.wt_cross = bt.ind.CrossOver(self.lines.wt1, self.lines.wt2)
        
        # Pre-calculate short_ema signal
        self.ema8_over_ema2 = bt.ind.CrossOver(self.lines.ema8, self.lines.ema2)

    def next(self):
        # WaveTrend cross direction
        wt_cross_down = self.lines.wt2[0] - self.lines.wt1[0] >= 0
        
        # Red diamond: WT cross down
        self.lines.red_diamond[0] = 1 if self.wt_cross[0] != 0 and wt_cross_down else 0
        
        # Blood diamond: Red diamond + Red cross
        self.lines.blood_diamond[0] = 1 if self.lines.red_diamond[0] and self.lines.red_cross[0] != 0 else 0
        
        # Yellow cross pattern
        self.lines.yellow_cross[0] = 1 if (self.lines.red_diamond[0] and 
                                           self.lines.wt2[0] < 45 and 
                                           self.lines.wt2[0] > self.p.os_level3 and 
                                           self.lines.rsi[0] < 30 and 
                                           self.lines.rsi[0] > 15 and 
                                           self.lines.rsi_mfi[0] < -5) else 0
        
        # Bull candle
        self.lines.bull_candle[0] = 1 if (self.data.open[0] > self.lines.ema2[0] and 
                                          self.data.open[0] > self.lines.ema8[0] and 
                                          self.data.close[-1] > self.data.open[-1] and 
                                          self.data.close[0] > self.data.open[0] and 
                                          not self.lines.red_diamond[0] and 
                                          not self.lines.red_cross[0]) else 0
        
        # Short EMA - using the pre-calculated crossover
        self.lines.short_ema[0] = 1 if self.ema8_over_ema2[0] == 1 else 0
