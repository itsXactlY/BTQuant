from sklearn.neighbors import NearestNeighbors
from backtrader.strategies.base import BaseStrategy, np, datetime, bt
from datetime import timedelta

class RationalQuadraticKernel(bt.indicators.PeriodN):
    lines = ('yhat',)
    params = (('h', 8), ('r', 8), ('x', 25))

    def __init__(self):
        super(RationalQuadraticKernel, self).__init__()
        self.addminperiod(self.p.x)

    def next(self):
        x = np.arange(len(self.data.get(size=self.p.x)))
        y = self.data.get(size=self.p.x)
        w = (1 + ((x - self.p.x)**2) / (2 * self.p.r * (self.p.h**2)))**(-self.p.r)
        
        sum_w = np.sum(w)
        if sum_w != 0:
            self.lines.yhat[0] = np.sum(w * y) / sum_w
        else:
            self.lines.yhat[0] = y[-1] if len(y) > 0 else 0

class NRK(BaseStrategy):
    params = (
        ('source', 'close'),
        ('neighbors_count', 8),
        ('max_bars_back', 100),
        ('use_volatility_filter', True),
        ('regime_threshold', -0.1),
        ('adx_threshold', 20),
        ('trade_with_kernel', True),
        ('kernel_lookback', 8),
        ('kernel_r', 8),
        ('kernel_x', 25),
        ('use_kernel_smoothing', True),
        ('kernel_smoothing_lag', 2),
        ('dca_deviation', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.01),
        ('debug', False),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.addminperiod(self.p.max_bars_back)
        self.source = getattr(self.data, self.p.source)
        self.atr = bt.indicators.ATR(self.data, period=14, plot=False)
        self.rsi = bt.indicators.RSI(self.data, period=14, plot=False)
        self.wt = bt.indicators.WilliamsR(self.data, period=10, plot=False)
        self.cci = bt.indicators.CCI(self.data, period=20, plot=False)
        self.adx = bt.indicators.ADX(self.data, period=14, plot=False)
        self.kernel_estimate = RationalQuadraticKernel(self.data.close, 
                                                       h=self.p.kernel_lookback, 
                                                       r=self.p.kernel_r, 
                                                       x=self.p.kernel_x)
        if self.p.use_kernel_smoothing:
            self.kernel_smooth = bt.indicators.SMA(self.kernel_estimate, period=self.p.kernel_smoothing_lag, plot=False)

        self.features = [self.rsi, self.wt, self.cci, self.adx, self.rsi]
        self.feature_data = []
        self.DCA = True
        self.last_minute = None

    def compute_ml_signal(self):
        current_features = []
        for f in self.features:
            try:
                current_features.append(f[0])
            except IndexError:
                current_features.append(np.nan)
        
        self.feature_data.append(current_features)
        if len(self.feature_data) > self.p.max_bars_back:
            self.feature_data.pop(0)
        if len(self.feature_data) < self.p.neighbors_count:
            return 0

        X = np.array(self.feature_data, dtype=float)
        if np.isnan(X).any():
            X = np.nan_to_num(X, nan=0.0)
        
        y = np.array([
            1 if self.source[-i] < self.source[0] 
            else -1 if self.source[-i] > self.source[0] 
            else 0 
            for i in range(1, len(X)+1)
        ])
        
        nn = NearestNeighbors(n_neighbors=min(self.p.neighbors_count, len(X)), metric='manhattan')
        nn.fit(X)
        distances, indices = nn.kneighbors([X[-1]])
        prediction = np.sum(y[indices[0]])
        signal = 1 if prediction > 0 else -1 if prediction < 0 else 0

        if self.p.use_volatility_filter and len(self.atr) > 20:
            atr_value = self.atr[0]
            atr_20_bars_ago = self.atr[-20]
            if atr_value <= atr_20_bars_ago:
                signal = 0

        if self.data.close[-20] != 0:
            regime = (self.data.close[0] / self.data.close[-20] - 1) * 100
        else:
            regime = 0
        
        if regime <= self.p.regime_threshold:
            signal = 0

        if self.adx[0] <= self.p.adx_threshold:
            signal = 0

        if self.p.trade_with_kernel:
            if self.p.use_kernel_smoothing:
                kernel_bullish = self.kernel_smooth[0] > self.kernel_estimate[0]
                kernel_bearish = self.kernel_smooth[0] < self.kernel_estimate[0]
            else:
                kernel_bullish = self.kernel_estimate[0] > self.kernel_estimate[-1]
                kernel_bearish = self.kernel_estimate[0] < self.kernel_estimate[-1]
            
            if (signal > 0 and not kernel_bullish) or (signal < 0 and not kernel_bearish):
                signal = 0
        
        return signal

    def buy_or_short_condition(self):
        signal = self.compute_ml_signal()
        if signal > 0:
            self.create_order(action='BUY')
            return True
        return False

    def dca_or_short_condition(self):
        if self.entry_prices and self.data.close[0] < self.entry_prices[-1] * (1 - self.params.dca_deviation / 100):
            signal = self.compute_ml_signal()
            if signal > 0:
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

    def _fetch(self):
        import pytz
        # data0 is your <Broker>Data -> has ._store and .symbol
        store = self.data._store
        symbol = store.symbol
        now = datetime.utcnow().replace(tzinfo=pytz.UTC)
        since = now - timedelta(minutes=5)
        since_ms = int(since.timestamp() * 1000)
        until_ms = int(now.timestamp() * 1000)
        klines = store.fetch_ohlcv(symbol, '1m', since=since_ms, until=until_ms)
        if not klines:
            return []
        last5 = klines[-5:]
        out = []
        for k in last5:
            ts = datetime.fromtimestamp(k[0] / 1000, tz=pytz.UTC)
            o, h, l, c, v = map(float, k[1:6])
            out.append(dict(
                dt=ts,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=v,
            ))
        return out

    def _log_view(self):
        mins = self._fetch()
        if mins:
            for i, mb in enumerate(mins):
                print(
                    f"  [{i}] {mb['dt']:%Y-%m-%d %H:%M} | "
                    f"O={mb['open']:.2f} H={mb['high']:.2f} "
                    f"L={mb['low']:.2f} C={mb['close']:.2f} "
                    f"V={mb['volume']:.6f}"
                )

    def next(self):
        super().next()

        if self.p.backtest:
            return

        if len(self.data) < 2:
            return

        dt_curr = self.data.datetime.datetime(0)
        dt_prev = self.data.datetime.datetime(-1)

        curr_key = (dt_curr.year, dt_curr.month, dt_curr.day, dt_curr.hour, dt_curr.minute)
        prev_key = (dt_prev.year, dt_prev.month, dt_prev.day, dt_prev.hour, dt_prev.minute)

        if curr_key == prev_key:
            return
        self.report_positions()
        self._log_view()
