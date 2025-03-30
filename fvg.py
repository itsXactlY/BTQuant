import backtrader as bt
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
from fastquant.strategies.base import BaseStrategy

'''class EnhancedFVGStrategy(BaseStrategy):
    params = (
        ('timeframes', {'1m': 1, '3m': 3, '5m': 5, '15m': 15}),
        ('volume_threshold', 1.5),     # Volume must be x times average
        ('mean_reversion_pct', 0.65),  # Target percentage of gap to fill
        ('volatility_filter', True),   # Apply volatility-based filters
        ('atr_periods', 14),           # Periods for ATR calculation
        ('atr_stop_mult', 1.5),        # ATR multiplier for stop placement
        ('profit_target_mult', 2.0),   # Risk:reward ratio
        ('max_active_fvgs', 5),        # Max FVGs to track per timeframe
        ('momentum_filter', True),      # Use momentum confirmation
        ('backtest', None),
    )

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Volatility metrics
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_periods)
        # self.daily_atr = bt.indicators.ATR(self.data, period=self.p.atr_periods)
        
        # Volume metrics
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Momentum indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=12, period_me2=26, period_signal=9)
        
        # FVG containers with priority scoring
        self.fvgs = {tf: [] for tf in self.p.timeframes}
        
        # Create multi-timeframe data
        self.datas_tf = {}
        for name, minutes in self.p.timeframes.items():
            if minutes == 1:  # Base timeframe
                self.datas_tf[name] = self.data0
            else:
                # Register resampled data
                data = self.cerebro.resampledata(self.data0, 
                                                timeframe=bt.TimeFrame.Minutes,
                                                compression=minutes)
                self.datas_tf[name] = data
        
        # Trade management
        self.active_trade = None
        self.stops = {}
        self.targets = {}
        self.entry_price = None
        self.entry_time = None
        self.fvg_signal = None
        self.trade_stats = {
            'wins': 0,
            'losses': 0,
            'total_profit': 0,
            'total_loss': 0
        }
        
        # Market regime identification
        self.regime = "neutral"  # Can be "trending", "ranging", "volatile", "neutral"
        self.last_regime_change = datetime.min
        
    def identify_market_regime(self):
        # Simple regime identification using price action and volatility
        close = self.data.close.get(size=50)
        if len(close) < 50:
            return
            
        # Trending vs ranging
        returns = np.diff(close) / close[:-1]
        hurst = self.calculate_hurst_exponent(returns)
        
        # Volatility regime
        recent_atr = self.atr[0]
        atr_history = [self.atr[-i] for i in range(20) if i < len(self.atr)]
        avg_atr = np.mean(atr_history) if atr_history else recent_atr
        
        # Determine regime
        if hurst > 0.6:  # Trending
            if recent_atr > 1.3 * avg_atr:
                self.regime = "volatile_trend"
            else:
                self.regime = "trending"
        else:  # Mean-reverting
            if recent_atr > 1.3 * avg_atr:
                self.regime = "volatile_range"
            else:
                self.regime = "ranging"
                
        # Apply adaptations based on regime
        if "volatile" in self.regime:
            self.p.atr_stop_mult = 2.0  # Wider stops in volatile markets
            self.p.profit_target_mult = 1.5  # Lower expectations
        elif self.regime == "trending":
            self.p.atr_stop_mult = 1.5
            self.p.profit_target_mult = 2.5  # Higher targets in trends
        else:  # ranging
            self.p.atr_stop_mult = 1.2
            self.p.profit_target_mult = 1.8
    
    def calculate_hurst_exponent(self, returns, max_lag=20):
        """Calculate Hurst exponent to determine if series is mean-reverting or trending"""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        
        if not tau or len(tau) < 2:
            return 0.5
            
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] / 2.0
    
    def next(self):
        # Update market regime every 4 hours
        current_time = self.data.datetime.datetime()
        if (current_time - self.last_regime_change).total_seconds() > 14400:  # 4 hours
            self.identify_market_regime()
            self.last_regime_change = current_time
        
        # Process each timeframe for FVG detection
        for tf_name, data in self.datas_tf.items():
            if len(data) < 4:
                continue
                
            # Detect and score FVGs
            self.detect_fvgs(tf_name, data)
            
            # Update existing FVGs status (mitigated/filled)
            self.update_fvgs(tf_name)
        
        # Manage existing positions
        if self.position:
            self.manage_position()
        else:
            # Look for new trading opportunities
            self.find_trading_signals()
    
    def detect_fvgs(self, tf_name, data):
        """Detect Fair Value Gaps and assign quality scores"""
        if len(data) < 4:
            return
            
        # Get OHLC values for past 3 bars
        o1, h1, l1, c1 = data.open[0], data.high[0], data.low[0], data.close[0]
        o2, h2, l2, c2 = data.open[-1], data.high[-1], data.low[-1], data.close[-1]
        o3, h3, l3, c3 = data.open[-2], data.high[-2], data.low[-2], data.close[-2]
        
        t1 = data.datetime.datetime(0)
        t2 = data.datetime.datetime(-1)
        t3 = data.datetime.datetime(-2)
        
        # Bullish FVG (gap up)
        if l1 > h3:
            gap_size = l1 - h3
            gap_score = self.score_fvg(gap_size, True, tf_name, data)
            
            if gap_score > 70:  # Quality threshold
                fvg = {
                    'type': 'bullish',
                    'open_time': t3,
                    'open': h3,
                    'close': l1,
                    'middle': (h3 + l1) / 2,
                    'size': gap_size,
                    'score': gap_score,
                    'mitigated': False,
                    'label': self.p.timeframes[tf_name],
                    'targets': {
                        '25%': h3 + gap_size * 0.25,
                        '50%': h3 + gap_size * 0.5,
                        '61.8%': h3 + gap_size * 0.618,
                        '100%': h3 + gap_size
                    }
                }
                self.fvgs[tf_name].append(fvg)
        
        # Bearish FVG (gap down)
        elif h1 < l3:
            gap_size = l3 - h1
            gap_score = self.score_fvg(gap_size, False, tf_name, data)
            
            if gap_score > 70:  # Quality threshold
                fvg = {
                    'type': 'bearish',
                    'open_time': t3,
                    'open': l3,
                    'close': h1,
                    'middle': (l3 + h1) / 2,
                    'size': gap_size,
                    'score': gap_score,
                    'mitigated': False,
                    'label': self.p.timeframes[tf_name],
                    'targets': {
                        '25%': l3 - gap_size * 0.25,
                        '50%': l3 - gap_size * 0.5,
                        '61.8%': l3 - gap_size * 0.618,
                        '100%': l3 - gap_size
                    }
                }
                self.fvgs[tf_name].append(fvg)
        
        # Limit to top N FVGs by score
        if len(self.fvgs[tf_name]) > self.p.max_active_fvgs:
            self.fvgs[tf_name] = sorted(self.fvgs[tf_name], key=lambda x: x['score'], reverse=True)[:self.p.max_active_fvgs]
    
    def score_fvg(self, gap_size, is_bullish, tf_name, data):
        """Score FVG quality based on multiple factors"""
        base_score = 50
        
        # 1. Gap size relative to ATR
        atr_value = self.atr[0]
        if gap_size > atr_value * 2:
            base_score += 15
        elif gap_size > atr_value:
            base_score += 10
        elif gap_size < atr_value * 0.5:
            base_score -= 10
        
        # 2. Volume confirmation
        current_vol = data.volume[0]
        avg_vol = np.mean([data.volume[-i] for i in range(1, 6) if i < len(data.volume)])
        if current_vol > avg_vol * self.p.volume_threshold:
            base_score += 15
        elif current_vol < avg_vol * 0.7:
            base_score -= 10
        
        # 3. Momentum alignment
        if self.p.momentum_filter:
            if is_bullish and self.rsi[0] > 50 and self.macd.macd[0] > self.macd.signal[0]:
                base_score += 10
            elif not is_bullish and self.rsi[0] < 50 and self.macd.macd[0] < self.macd.signal[0]:
                base_score += 10
            else:
                base_score -= 5
        
        # 4. Market regime alignment
        if (is_bullish and "trending" in self.regime) or (not is_bullish and "ranging" in self.regime):
            base_score += 10
        
        # 5. Timeframe weight - higher timeframes get priority
        tf_minutes = self.p.timeframes[tf_name]
        if tf_minutes >= 15:
            base_score += 15
        elif tf_minutes >= 5:
            base_score += 10
        elif tf_minutes >= 3:
            base_score += 5
        
        return min(100, max(0, base_score))  # Clamp to 0-100 range
    
    def update_fvgs(self, tf_name):
        """Update status of existing FVGs"""
        current_price = self.data.close[0]
        current_time = self.data.datetime.datetime(0)
        
        for i, fvg in enumerate(self.fvgs[tf_name]):
            if fvg['mitigated']:
                continue
                
            # Time-based decay (older FVGs lose relevance)
            age_hours = (current_time - fvg['open_time']).total_seconds() / 3600
            if age_hours > 24:  # Older than 24 hours
                fvg['score'] *= 0.95  # Decay score
            
            # Check if FVG is filled/mitigated
            if fvg['type'] == 'bullish' and current_price < fvg['open']:
                fvg['mitigated'] = True
                if self.position and self.fvg_signal == fvg:
                    self.close()  # Exit if our signal FVG is mitigated
            elif fvg['type'] == 'bearish' and current_price > fvg['open']:
                fvg['mitigated'] = True
                if self.position and self.fvg_signal == fvg:
                    self.close()  # Exit if our signal FVG is mitigated
            
            # Remove low-scoring FVGs
            if fvg['score'] < 30:
                self.fvgs[tf_name].pop(i)
                break
    
    def find_trading_signals(self):
        """Identify high-probability trading setups from FVGs"""
        if self.position:
            return  # Already in a position
        
        # Get top FVGs across all timeframes
        all_fvgs = []
        for tf_name, fvgs in self.fvgs.items():
            all_fvgs.extend(fvgs)
        
        # Sort by score
        all_fvgs = sorted([f for f in all_fvgs if not f['mitigated']], key=lambda x: x['score'], reverse=True)
        
        if not all_fvgs:
            return
        
        # Get best FVG
        best_fvg = all_fvgs[0]
        current_price = self.data.close[0]
        
        # Only trade if score exceeds threshold
        if best_fvg['score'] < 75:
            return
            
        # Evaluate setup quality
        setup_valid = self.validate_setup(best_fvg, current_price)
        if not setup_valid:
            return
            
        # Calculate position size and risk
        stop_price = self.calculate_stop(best_fvg)
        target_price = self.calculate_target(best_fvg, stop_price)
        risk_per_share = abs(current_price - stop_price)
        
        # Risk management
        account_value = self.broker.getvalue()
        risk_amount = account_value * 0.01  # Risk 1% per trade
        size = int(risk_amount / risk_per_share)
        
        if size < 1:
            return  # Not enough capital for minimum position
        
        # Execute trade
        if best_fvg['type'] == 'bullish' and current_price > best_fvg['middle']:
            self.buy(size=size)
            self.log(f"BUY {size} shares at {current_price:.2f}, Stop: {stop_price:.2f}, Target: {target_price:.2f}")
            self.entry_price = current_price
            self.stops[0] = stop_price
            self.targets[0] = target_price
            self.fvg_signal = best_fvg
            self.entry_time = self.data.datetime.datetime(0)
            
        elif best_fvg['type'] == 'bearish' and current_price < best_fvg['middle']:
            self.sell(size=size)
            self.log(f"SELL {size} shares at {current_price:.2f}, Stop: {stop_price:.2f}, Target: {target_price:.2f}")
            self.entry_price = current_price
            self.stops[0] = stop_price
            self.targets[0] = target_price
            self.fvg_signal = best_fvg
            self.entry_time = self.data.datetime.datetime(0)
    
    def validate_setup(self, fvg, current_price):
        """Validate trade setup with additional filters"""
        # Price must be in valid entry zone
        if fvg['type'] == 'bullish':
            if not (current_price > fvg['middle'] and current_price < fvg['close']):
                return False
        else:  # bearish
            if not (current_price < fvg['middle'] and current_price > fvg['close']):
                return False
        
        # Volume filter
        if self.data.volume[0] < self.volume_ma[0] * 0.8:
            return False
            
        # Volatility filter
        if self.p.volatility_filter and self.atr[0] < self.atr[-10] * 0.7:
            return False  # Too low volatility
            
        # Momentum alignment
        if self.p.momentum_filter:
            if fvg['type'] == 'bullish' and (self.rsi[0] < 40 or self.macd.macd[0] < 0):
                return False  # Bearish momentum
            elif fvg['type'] == 'bearish' and (self.rsi[0] > 60 or self.macd.macd[0] > 0):
                return False  # Bullish momentum
        
        # Market hours filter (avoid trading during low liquidity)
        hour = self.data.datetime.datetime(0).hour
        if hour < 2 or hour > 22:  # Assuming UTC, adjust for your market
            return False
        
        return True
    
    def calculate_stop(self, fvg):
        """Calculate adaptive stop loss level"""
        atr_value = self.atr[0]
        
        if fvg['type'] == 'bullish':
            # For bullish trades, stop below FVG open or ATR-based
            basic_stop = fvg['open'] - atr_value * 0.5
            return min(basic_stop, self.data.close[0] - atr_value * self.p.atr_stop_mult)
        else:
            # For bearish trades, stop above FVG open or ATR-based
            basic_stop = fvg['open'] + atr_value * 0.5
            return max(basic_stop, self.data.close[0] + atr_value * self.p.atr_stop_mult)
    
    def calculate_target(self, fvg, stop_price):
        """Calculate profit target based on R:R ratio"""
        risk = abs(self.data.close[0] - stop_price)
        reward = risk * self.p.profit_target_mult
        
        if fvg['type'] == 'bullish':
            return self.data.close[0] + reward
        else:
            return self.data.close[0] - reward
    
    def manage_position(self):
        """Manage open positions with adaptive exits"""
        current_price = self.data.close[0]
        position_age = (self.data.datetime.datetime(0) - self.entry_time).total_seconds() / 60  # in minutes
        
        # Get current stop and target
        stop = self.stops[0]
        target = self.targets[0]
        
        # Check for stop loss
        if (self.position.size > 0 and current_price < stop) or (self.position.size < 0 and current_price > stop):
            self.close()
            self.log(f"STOP triggered at {current_price:.2f}")
            self.trade_stats['losses'] += 1
            self.trade_stats['total_loss'] += abs(self.entry_price - current_price) * abs(self.position.size)
            return
            
        # Check for target hit
        if (self.position.size > 0 and current_price > target) or (self.position.size < 0 and current_price < target):
            self.close()
            self.log(f"TARGET hit at {current_price:.2f}")
            self.trade_stats['wins'] += 1
            self.trade_stats['total_profit'] += abs(self.entry_price - current_price) * abs(self.position.size)
            return
            
        # Trailing stop logic
        if self.position.size > 0:  # Long position
            # Move stop up if price advances
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            if profit_pct > 0.02:  # More than 2% profit
                new_stop = max(stop, self.entry_price + (current_price - self.entry_price) * 0.4)  # Trail by 60%
                if new_stop > stop:
                    self.stops[0] = new_stop
                    self.log(f"Trailing stop moved to {new_stop:.2f}")
                    
        else:  # Short position
            # Move stop down if price declines
            profit_pct = (self.entry_price - current_price) / self.entry_price
            
            if profit_pct > 0.02:  # More than 2% profit
                new_stop = min(stop, self.entry_price - (self.entry_price - current_price) * 0.4)  # Trail by 60%
                if new_stop < stop:
                    self.stops[0] = new_stop
                    self.log(f"Trailing stop moved to {new_stop:.2f}")
        
        # Time-based exit (avoid holding positions too long)
        if position_age > 240:  # 4 hours
            # If in profit but not reaching target, close portion
            if (self.position.size > 0 and current_price > self.entry_price) or \
               (self.position.size < 0 and current_price < self.entry_price):
                self.close(size=self.position.size * 0.5)
                self.log(f"Time-based partial exit at {current_price:.2f}")
        
        # Full exit after 8 hours regardless of profit/loss
        if position_age > 480:  # 8 hours
            self.close()
            self.log(f"Time-based full exit at {current_price:.2f}")
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.data.datetime.datetime(0)
        print(f'{dt.isoformat()}, {txt}')
'''

class EnhancedFVGStrategy(BaseStrategy):
    params = (
        ('timeframes', {'1m': 1, '3m': 3, '5m': 5, '15m': 15}),
        ('volume_threshold', 1.5),     # Volume must be x times average
        ('mean_reversion_pct', 0.65),  # Target percentage of gap to fill
        ('volatility_filter', True),   # Apply volatility-based filters
        ('atr_periods', 14),           # Periods for ATR calculation
        ('atr_stop_mult', 1.5),        # ATR multiplier for stop placement
        ('profit_target_mult', 2.0),   # Risk:reward ratio
        ('max_active_fvgs', 5),        # Max FVGs to track per timeframe
        ('momentum_filter', True),      # Use momentum confirmation
        ('backtest', None),
    )

    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Volatility metrics
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_periods)
        # self.daily_atr = bt.indicators.ATR(self.data, period=self.p.atr_periods, timeframe=bt.TimeFrame.Days)
        
        # Volume metrics
        self.volume_ma = bt.indicators.SMA(self.data.volume, period=20)
        
        # Momentum indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.macd = bt.indicators.MACD(self.data.close, period_me1=12, period_me2=26, period_signal=9)
        
        # FVG containers with priority scoring
        self.fvgs = {tf: [] for tf in self.p.timeframes}
        
        # Create multi-timeframe data
        self.datas_tf = {}
        for name, minutes in self.p.timeframes.items():
            if minutes == 1:  # Base timeframe
                self.datas_tf[name] = self.data0
            else:
                # Register resampled data
                data = self.cerebro.resampledata(self.data0, 
                                                timeframe=bt.TimeFrame.Minutes,
                                                compression=minutes)
                self.datas_tf[name] = data
        
        # Trade management
        self.active_trade = None
        self.stops = {}
        self.targets = {}
        self.entry_price = None
        self.entry_time = None
        self.fvg_signal = None
        self.trade_stats = {
            'wins': 0,
            'losses': 0,
            'total_profit': 0,
            'total_loss': 0
        }
        
        # Market regime identification
        self.regime = "neutral"  # Can be "trending", "ranging", "volatile", "neutral"
        self.last_regime_change = datetime.min
        self.last_fvg_check = datetime.min
        
    def next(self):
        """Main strategy logic executed on each bar"""
        current_time = self.data.datetime.datetime(0)
        
        # Update market regime every 4 hours
        if (current_time - self.last_regime_change).total_seconds() > 14400:
            self.identify_market_regime()
            self.last_regime_change = current_time
        
        # Process FVGs across timeframes - check every minute to avoid redundant processing
        if (current_time - self.last_fvg_check).total_seconds() >= 60:
            for tf_name, data in self.datas_tf.items():
                if len(data) < 4:
                    continue
                    
                # Detect and score new FVGs
                self.detect_fvgs(tf_name, data)
                
            self.last_fvg_check = current_time
        
        # Update status of existing FVGs with current price
        for tf_name in self.fvgs:
            self.update_fvgs(tf_name)
        
        # TRADING LOGIC EXECUTION
        if self.position:
            # Already in a position - manage it
            self.manage_position()
        else:
            # Look for new trading opportunities
            self.execute_trading_decision()
    
    def execute_trading_decision(self):
        """Core trading decision logic - finds and executes setups"""
        # Collect and sort all active, non-mitigated FVGs across timeframes
        all_active_fvgs = []
        for tf_name, fvgs in self.fvgs.items():
            for fvg in fvgs:
                if not fvg['mitigated'] and fvg['score'] >= 75:  # Only high-quality setups
                    all_active_fvgs.append(fvg)
        
        # Sort by score (highest first)
        all_active_fvgs.sort(key=lambda x: x['score'], reverse=True)
        
        if not all_active_fvgs:
            return  # No qualified setups
        
        # Get current price and top FVG
        current_price = self.data.close[0]
        best_fvg = all_active_fvgs[0]
        
        # Check for entry conditions
        entry_signal = False
        direction = None
        
        # ENTRY LOGIC - FVG Breakout with confirmation
        if best_fvg['type'] == 'bullish':
            # Bullish FVG entry when price breaks above middle line with momentum
            if (current_price > best_fvg['middle'] and 
                current_price < best_fvg['close'] and 
                self.data.close[-1] <= best_fvg['middle']):
                
                # Momentum confirmation
                if not self.p.momentum_filter or (self.rsi[0] > 50 and self.macd.macd[0] > self.macd.signal[0]):
                    entry_signal = True
                    direction = 'long'
        
        elif best_fvg['type'] == 'bearish':
            # Bearish FVG entry when price breaks below middle line with momentum
            if (current_price < best_fvg['middle'] and 
                current_price > best_fvg['close'] and 
                self.data.close[-1] >= best_fvg['middle']):
                
                # Momentum confirmation
                if not self.p.momentum_filter or (self.rsi[0] < 50 and self.macd.macd[0] < self.macd.signal[0]):
                    entry_signal = True
                    direction = 'short'
        
        # Execute trade if signal detected
        if entry_signal and self.validate_market_conditions():
            # Calculate stop loss
            stop_price = self.calculate_stop(best_fvg)
            
            # Calculate position size based on risk
            risk_per_share = abs(current_price - stop_price)
            account_value = self.broker.getvalue()
            risk_amount = account_value * 0.01  # Risk 1% per trade
            
            # Ensure minimum risk distance
            min_risk_distance = self.atr[0] * 0.5
            if risk_per_share < min_risk_distance:
                return  # Risk too small, avoid taking trade
            
            size = max(1, int(risk_amount / risk_per_share))
            
            # Calculate profit target
            target_price = self.calculate_target(best_fvg, stop_price)
            
            # Execute trade
            if direction == 'long':
                self.buy(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                self.log(f"BUY {size} shares at {current_price:.2f}, Stop: {stop_price:.2f}, Target: {target_price:.2f}")
            else:  # short
                self.sell(size=self.stake, price=self.data.close[0], exectype=bt.Order.Market)
                self.log(f"SELL {size} shares at {current_price:.2f}, Stop: {stop_price:.2f}, Target: {target_price:.2f}")
            
            # Record trade details
            self.entry_price = current_price
            self.entry_time = self.data.datetime.datetime(0)
            self.stops[0] = stop_price
            self.targets[0] = target_price
            self.fvg_signal = best_fvg
    
    def validate_market_conditions(self):
        """Validate overall market conditions for trading"""
        # Time filter - avoid trading during known low-liquidity periods
        hour = self.data.datetime.datetime(0).hour
        if hour < 2 or hour > 22:  # Adjust for your market's active hours
            return False
        
        # Volume filter - ensure sufficient liquidity
        if self.data.volume[0] < self.volume_ma[0] * 0.7:
            return False
        
        # Volatility filter - avoid extremely low or high volatility
        atr_avg = np.mean([self.atr[-i] for i in range(20) if i < len(self.atr)])
        if self.atr[0] < atr_avg * 0.5 or self.atr[0] > atr_avg * 3:
            return False
        
        # Check for extreme RSI conditions
        if self.rsi[0] < 20 or self.rsi[0] > 80:
            return False  # Avoid trading in extremely overbought/oversold conditions
        
        return True
    
    def detect_fvgs(self, tf_name, data):
        """Detect Fair Value Gaps and assign quality scores"""
        if len(data) < 4:
            return
            
        # Get OHLC values for past 3 bars
        o1, h1, l1, c1 = data.open[0], data.high[0], data.low[0], data.close[0]
        o2, h2, l2, c2 = data.open[-1], data.high[-1], data.low[-1], data.close[-1]
        o3, h3, l3, c3 = data.open[-2], data.high[-2], data.low[-2], data.close[-2]
        
        t1 = data.datetime.datetime(0)
        t2 = data.datetime.datetime(-1)
        t3 = data.datetime.datetime(-2)
        
        # Bullish FVG (gap up)
        if l1 > h3:
            gap_size = l1 - h3
            gap_score = self.score_fvg(gap_size, True, tf_name, data)
            
            if gap_score > 70:  # Quality threshold
                fvg = {
                    'type': 'bullish',
                    'open_time': t3,
                    'open': h3,
                    'close': l1,
                    'middle': (h3 + l1) / 2,
                    'size': gap_size,
                    'score': gap_score,
                    'mitigated': False,
                    'label': self.p.timeframes[tf_name],
                    'targets': {
                        '25%': h3 + gap_size * 0.25,
                        '50%': h3 + gap_size * 0.5,
                        '61.8%': h3 + gap_size * 0.618,
                        '100%': h3 + gap_size
                    }
                }
                self.fvgs[tf_name].append(fvg)
        
        # Bearish FVG (gap down)
        elif h1 < l3:
            gap_size = l3 - h1
            gap_score = self.score_fvg(gap_size, False, tf_name, data)
            
            if gap_score > 70:  # Quality threshold
                fvg = {
                    'type': 'bearish',
                    'open_time': t3,
                    'open': l3,
                    'close': h1,
                    'middle': (l3 + h1) / 2,
                    'size': gap_size,
                    'score': gap_score,
                    'mitigated': False,
                    'label': self.p.timeframes[tf_name],
                    'targets': {
                        '25%': l3 - gap_size * 0.25,
                        '50%': l3 - gap_size * 0.5,
                        '61.8%': l3 - gap_size * 0.618,
                        '100%': l3 - gap_size
                    }
                }
                self.fvgs[tf_name].append(fvg)
        
        # Limit to top N FVGs by score
        if len(self.fvgs[tf_name]) > self.p.max_active_fvgs:
            self.fvgs[tf_name] = sorted(self.fvgs[tf_name], key=lambda x: x['score'], reverse=True)[:self.p.max_active_fvgs]
    
    def score_fvg(self, gap_size, is_bullish, tf_name, data):
        """Score FVG quality based on multiple factors"""
        base_score = 50
        
        # 1. Gap size relative to ATR
        atr_value = self.atr[0]
        if gap_size > atr_value * 2:
            base_score += 15
        elif gap_size > atr_value:
            base_score += 10
        elif gap_size < atr_value * 0.5:
            base_score -= 10
        
        # 2. Volume confirmation
        current_vol = data.volume[0]
        avg_vol = np.mean([data.volume[-i] for i in range(1, 6) if i < len(data.volume)])
        if current_vol > avg_vol * self.p.volume_threshold:
            base_score += 15
        elif current_vol < avg_vol * 0.7:
            base_score -= 10
        
        # 3. Momentum alignment
        if self.p.momentum_filter:
            if is_bullish and self.rsi[0] > 50 and self.macd.macd[0] > self.macd.signal[0]:
                base_score += 10
            elif not is_bullish and self.rsi[0] < 50 and self.macd.macd[0] < self.macd.signal[0]:
                base_score += 10
            else:
                base_score -= 5
        
        # 4. Market regime alignment
        if (is_bullish and "trending" in self.regime) or (not is_bullish and "ranging" in self.regime):
            base_score += 10
        
        # 5. Timeframe weight - higher timeframes get priority
        tf_minutes = self.p.timeframes[tf_name]
        if tf_minutes >= 15:
            base_score += 15
        elif tf_minutes >= 5:
            base_score += 10
        elif tf_minutes >= 3:
            base_score += 5
        
        return min(100, max(0, base_score))  # Clamp to 0-100 range
    
    def update_fvgs(self, tf_name):
        """Update status of existing FVGs"""
        current_price = self.data.close[0]
        current_time = self.data.datetime.datetime(0)
        
        for i, fvg in enumerate(self.fvgs[tf_name]):
            if fvg['mitigated']:
                continue
                
            # Time-based decay (older FVGs lose relevance)
            age_hours = (current_time - fvg['open_time']).total_seconds() / 3600
            if age_hours > 24:  # Older than 24 hours
                fvg['score'] *= 0.95  # Decay score
            
            # Check if FVG is filled/mitigated
            if fvg['type'] == 'bullish' and current_price < fvg['open']:
                fvg['mitigated'] = True
                if self.position and self.fvg_signal == fvg:
                    self.close()  # Exit if our signal FVG is mitigated
            elif fvg['type'] == 'bearish' and current_price > fvg['open']:
                fvg['mitigated'] = True
                if self.position and self.fvg_signal == fvg:
                    self.close()  # Exit if our signal FVG is mitigated
            
            # Remove low-scoring FVGs
            if fvg['score'] < 30:
                self.fvgs[tf_name].pop(i)
                break
    
    def identify_market_regime(self):
        """Determine the current market regime"""
        # Get price data for analysis
        close = np.array([self.data.close[-i] for i in range(50) if i < len(self.data)])
        if len(close) < 50:
            return
            
        # Calculate returns
        returns = np.diff(close) / close[:-1]
        
        # Calculate Hurst exponent to determine trending vs mean-reverting
        hurst = self.calculate_hurst_exponent(returns)
        
        # Determine volatility regime
        recent_atr = self.atr[0]
        atr_history = [self.atr[-i] for i in range(20) if i < len(self.atr)]
        avg_atr = np.mean(atr_history) if atr_history else recent_atr
        
        # Define market regime
        if hurst > 0.6:  # Trending
            if recent_atr > 1.3 * avg_atr:
                self.regime = "volatile_trend"
            else:
                self.regime = "trending"
        else:  # Mean-reverting
            if recent_atr > 1.3 * avg_atr:
                self.regime = "volatile_range"
            else:
                self.regime = "ranging"
        
        # Adapt strategy parameters based on regime
        self.adapt_parameters_to_regime()
    
    def adapt_parameters_to_regime(self):
        """Adjust strategy parameters based on market regime"""
        if "volatile" in self.regime:
            # Wider stops, lower targets in volatile markets
            self.p.atr_stop_mult = 2.0
            self.p.profit_target_mult = 1.5
        elif self.regime == "trending":
            # Standard stops, higher targets in trends
            self.p.atr_stop_mult = 1.5
            self.p.profit_target_mult = 2.5
        else:  # ranging
            # Tighter stops, moderate targets in ranges
            self.p.atr_stop_mult = 1.2
            self.p.profit_target_mult = 1.8
    
    def calculate_hurst_exponent(self, returns, max_lag=20):
        """Calculate Hurst exponent to determine if series is mean-reverting or trending"""
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
        
        if not tau or len(tau) < 2:
            return 0.5
            
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] / 2.0
    
    def calculate_stop(self, fvg):
        """Calculate adaptive stop loss level"""
        atr_value = self.atr[0]
        current_price = self.data.close[0]
        
        if fvg['type'] == 'bullish':
            # For bullish FVGs - stop below FVG open or ATR-based
            base_stop = fvg['open'] - atr_value * 0.5
            atr_stop = current_price - atr_value * self.p.atr_stop_mult
            return min(base_stop, atr_stop)
        else:
            # For bearish FVGs - stop above FVG open or ATR-based
            base_stop = fvg['open'] + atr_value * 0.5
            atr_stop = current_price + atr_value * self.p.atr_stop_mult
            return max(base_stop, atr_stop)
    
    def calculate_target(self, fvg, stop_price):
        """Calculate profit target based on risk:reward ratio"""
        current_price = self.data.close[0]
        risk = abs(current_price - stop_price)
        reward = risk * self.p.profit_target_mult
        
        if fvg['type'] == 'bullish':
            # For bullish FVGs, target is above current price
            return current_price + reward
        else:
            # For bearish FVGs, target is below current price
            return current_price - reward
    
    def manage_position(self):
        """Manage open positions with adaptive exits"""
        current_price = self.data.close[0]
        current_time = self.data.datetime.datetime(0)
        position_age = (current_time - self.entry_time).total_seconds() / 60  # in minutes
        
        # Get current stop and target
        stop = self.stops[0]
        target = self.targets[0]
        
        # Check for stop loss hit
        if (self.position.size > 0 and current_price <= stop) or (self.position.size < 0 and current_price >= stop):
            self.close()
            self.log(f"STOP triggered at {current_price:.2f}")
            self.trade_stats['losses'] += 1
            self.trade_stats['total_loss'] += abs(self.entry_price - current_price) * abs(self.position.size)
            return
            
        # Check for target hit
        if (self.position.size > 0 and current_price >= target) or (self.position.size < 0 and current_price <= target):
            self.close()
            self.log(f"TARGET hit at {current_price:.2f}")
            self.trade_stats['wins'] += 1
            self.trade_stats['total_profit'] += abs(self.entry_price - current_price) * abs(self.position.size)
            return
        
        # Trailing stop logic for profitable trades
        if self.position.size > 0:  # Long position
            # Calculate current profit
            profit_pct = (current_price - self.entry_price) / self.entry_price
            
            # Trail stop once in profit
            if profit_pct > 0.02:  # 2% profit threshold
                # Move stop to lock in profit
                new_stop = max(stop, self.entry_price + (current_price - self.entry_price) * 0.4)
                if new_stop > stop:
                    self.stops[0] = new_stop
                    self.log(f"Trailing stop moved to {new_stop:.2f}")
        else:  # Short position
            # Calculate current profit
            profit_pct = (self.entry_price - current_price) / self.entry_price
            
            # Trail stop once in profit
            if profit_pct > 0.02:  # 2% profit threshold
                # Move stop to lock in profit
                new_stop = min(stop, self.entry_price - (self.entry_price - current_price) * 0.4)
                if new_stop < stop:
                    self.stops[0] = new_stop
                    self.log(f"Trailing stop moved to {new_stop:.2f}")
        
        # Time-based exit rules
        if position_age > 240:  # 4 hours
            # Take partial profits if in profit
            if ((self.position.size > 0 and current_price > self.entry_price) or 
                (self.position.size < 0 and current_price < self.entry_price)):
                self.close(size=self.position.size * 0.5)
                self.log(f"Time-based partial exit at {current_price:.2f}")
        
        # Hard time-based exit
        if position_age > 480:  # 8 hours
            self.close()
            self.log(f"Time-based full exit at {current_price:.2f}")
    
    def log(self, txt, dt=None):
        """Logging function"""
        dt = dt or self.data.datetime.datetime(0)
        print(f'{dt.isoformat()}, {txt}')
        
    def stop(self):
        """Called when backtest is complete"""
        total_trades = self.trade_stats['wins'] + self.trade_stats['losses']
        if total_trades > 0:
            win_rate = (self.trade_stats['wins'] / total_trades) * 100
            print(f'Strategy Performance:')
            print(f'Total Trades: {total_trades}')
            print(f'Win Rate: {win_rate:.2f}%')
            print(f'Total Profit: {self.trade_stats["total_profit"]:.2f}')
            print(f'Total Loss: {self.trade_stats["total_loss"]:.2f}')
            
            # if self.trade_stats["total_loss"] != 0:
            #     profit_factor = self.trade_stats["total_profit"] / self.trade_stats["total_loss"]
            #     print(f'Profit Factor: {profit_factor:.2f}')


from fastquant import get_database_data, backtest
data = get_database_data("BTC", "2024-01-01", "2024-04-05", "1m")
backtest(EnhancedFVGStrategy, data, init_cash=1000, backtest=True, plot=True, verbose=0)

