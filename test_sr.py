from backtrader.strategies.base import BaseStrategy, np, bt

class SafeLogReturn(bt.indicators.PeriodN):
    """
    Simple, BTQ-compatible log-return indicator.

    logret[t] = ln(close[t] / close[t-period]) if prices > 0, else 0.
    """
    lines = ('logret',)
    params = (('period', 1),)

    def __init__(self):
        super().__init__()
        self.addminperiod(self.p.period + 1)

    def next(self):
        if len(self.data) <= self.p.period:
            self.lines.logret[0] = 0.0
            return

        cur = float(self.data[0])
        prev = float(self.data[-self.p.period])

        if cur > 0.0 and prev > 0.0:
            self.lines.logret[0] = float(np.log(cur / prev))
        else:
            self.lines.logret[0] = 0.0

class SRPullbackVolumeBands(bt.indicators.PeriodN):
    """
    Jackrabbit 'S/R Pullback OliverTwist' core indicator, BTQ-style.

    Computes:
    - support: lowest low over `period`
    - resistance: highest high over `period`
    - atr: ATR(atr_period) with SMA smoothing
    - volatility: stddev of SafeLogReturn over `volatility_period`
    - band_h: resistance - (atr + volatility)
    - band_m: (resistance + support) / 2
    - band_l: support + (atr + volatility)
    - vol_ma: SMA(volume, vol_ma_period)
    - vol_roc: % change in volume vs vol_roc_lag bars ago
    - vol_pct: (v - vol_ma) / (v + vol_ma) * 100
    - vol_buy: 1 if green candle & vol_pct > vol_pct_threshold & vol_roc > vol_roc_threshold
    - buy_signal: 1 on support+ATR+vol pullback + volume buy confirmation
    - sell_signal: -1 on resistance-ATR-vol pullback + volume buy confirmation
    """
    lines = (
        'support',
        'resistance',
        'atr',
        'volatility',
        'band_h',
        'band_m',
        'band_l',
        'vol_ma',
        'vol_roc',
        'vol_pct',
        'vol_buy',
        'buy_signal',
        'sell_signal',
    )

    params = (
        ('period', 197),              # S/R & volatility window, ~[100, 300]
        ('atr_period', 14),           # ATR length, ~[10, 20]
        ('vol_ma_period', 50),        # Volume SMA window, ~[30, 200]
        ('vol_roc_lag', 1),           # Volume ROC lag (bars), [1, 3]
        ('vol_pct_threshold', 30.0),  # Vol % above MA, ~[10, 80]
        ('vol_roc_threshold', 30.0),  # Volume ROC threshold %, ~[10, 80]
        ('volatility_period', 197),   # Volatility window, ~[100, 300]
    )

    def __init__(self):
        super().__init__()

        # Core price-based components
        self._support = bt.indicators.Lowest(self.data.low, period=self.p.period)
        self._resistance = bt.indicators.Highest(self.data.high, period=self.p.period)

        # ATR with SMA smoothing (no lookahead)
        self._atr = bt.indicators.ATR(self.data, period=self.p.atr_period, movav=bt.indicators.SMA)

        # Volatility: std dev of log-returns over volatility_period
        logret = SafeLogReturn(self.data.close, period=1)
        self._volatility = bt.indicators.StdDev(logret, period=self.p.volatility_period)

        # Volume-based components
        self._vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.vol_ma_period)

        # Enough history before meaningful signals
        self.addminperiod(
            max(
                self.p.period,
                self.p.atr_period + 1,
                self.p.volatility_period + 1,
                self.p.vol_ma_period + 1,
                self.p.vol_roc_lag + 1,
            )
        )

    def next(self):
        # Copy core lines
        self.lines.support[0] = self._support[0]
        self.lines.resistance[0] = self._resistance[0]
        self.lines.atr[0] = self._atr[0]
        self.lines.volatility[0] = self._volatility[0]
        self.lines.vol_ma[0] = self._vol_ma[0]

        vol = float(self.data.volume[0])
        vol_ma = float(self._vol_ma[0]) if self._vol_ma[0] is not None else float('nan')

        # Volume ROC (simple % vs vol_roc_lag bars ago)
        if len(self.data) > self.p.vol_roc_lag:
            prev_vol = float(self.data.volume[-self.p.vol_roc_lag])
            if prev_vol != 0.0:
                vol_roc = (vol - prev_vol) / abs(prev_vol) * 100.0
            else:
                vol_roc = 0.0
        else:
            vol_roc = 0.0

        self.lines.vol_roc[0] = vol_roc

        # Volume % above/below MA: (v - ma) / (v + ma) * 100
        denom = vol + (vol_ma if vol_ma == vol_ma else 0.0)  # NaN-safe
        if denom > 0.0 and vol_ma == vol_ma:
            vol_pct = (vol - vol_ma) / denom * 100.0
        else:
            vol_pct = 0.0

        self.lines.vol_pct[0] = vol_pct

        # ATR/volatility bands
        atr = float(self._atr[0])
        volat = float(self._volatility[0])
        res = float(self._resistance[0])
        sup = float(self._support[0])

        if atr == atr and volat == volat and res == res and sup == sup:
            band_h = res - (atr + volat)
            band_m = (res + sup) / 2.0
            band_l = sup + (atr + volat)
        else:
            band_h = band_m = band_l = float('nan')

        self.lines.band_h[0] = band_h
        self.lines.band_m[0] = band_m
        self.lines.band_l[0] = band_l

        # Volume "buy" confirmation: green candle + strong volume & ROC
        close0 = float(self.data.close[0])
        open0 = float(self.data.open[0])
        is_green = close0 > open0

        vol_buy = int(
            is_green
            and vol_pct > self.p.vol_pct_threshold
            and vol_roc > self.p.vol_roc_threshold
        )
        self.lines.vol_buy[0] = vol_buy

        # Crossing logic: price crossing the bands
        buy_cross = 0
        sell_cross = 0

        if len(self.data) > 1:
            close_prev = float(self.data.close[-1])
            band_l_prev = float(self.lines.band_l[-1])
            band_h_prev = float(self.lines.band_h[-1])

            if (
                band_l_prev == band_l_prev
                and band_h_prev == band_h_prev
                and band_l == band_l
                and band_h == band_h
            ):
                # Support pullback: close crosses from below to above the lower band
                if close_prev <= band_l_prev and close0 > band_l:
                    buy_cross = 1

                # Resistance pullback: close crosses from above to below the upper band
                if close_prev >= band_h_prev and close0 < band_h:
                    sell_cross = -1

        # Final signals: band cross + volume confirmation
        buy_signal = 1 if (buy_cross == 1 and vol_buy == 1) else 0
        sell_signal = -1 if (sell_cross == -1 and vol_buy == 1) else 0

        self.lines.buy_signal[0] = buy_signal
        self.lines.sell_signal[0] = sell_signal

class SRPullback_OliverTwist(BaseStrategy):
    """
    BTQuant-style 'S/R Pullback OliverTwist' strategy.

    - Long entry when SRPullbackVolumeBands.buy_signal == 1
      (support+ATR+vol pullback + volume confirmation).
    - DCA on adverse moves relative to last entry, controlled by dca_deviation.
    - Take-profit handled by BaseStrategy/OrderTracker via `take_profit` param.
    """

    params = (
        # --- Source / structure ---
        ('source', 'close'),

        # --- S/R + volume structure (Jackrabbit logic) ---
        ('period', 197),              # S/R & volatility window, ~[100, 300]
        ('atr_period', 14),           # ATR length, ~[10, 20]
        ('vol_ma_period', 50),        # Volume MA window, ~[30, 200]
        ('vol_roc_lag', 1),           # Volume ROC lag, [1, 3]
        ('vol_pct_threshold', 30.0),  # Vol % above MA, ~[10, 80]
        ('vol_roc_threshold', 30.0),  # Volume ROC threshold %, ~[10, 80]
        ('volatility_period', 197),   # Volatility window

        # --- Risk / execution knobs passed to BaseStrategy / OrderTracker ---
        ('order_value', 10.0),        # 10 USD per leg
        ('dca_deviation', 2.5),       # Adverse move % for DCA (1.5)
        ('take_profit', 4.5),         # TP % per leg (1.0 = 1%); BaseStrategy uses this
        ('percent_sizer', 0.02),      # Fraction of equity per entry; used by BaseStrategy

        # --- Misc ---
        ('debug', False),
        ('backtest', None),           # If truthy, skip any live-only logging in next()
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Strategy operates on chosen source (close by default)
        self.source = getattr(self.data, self.p.source)

        # Core S/R + volume indicator
        self.sr = SRPullbackVolumeBands(
            self.data,
            period=self.p.period,
            atr_period=self.p.atr_period,
            vol_ma_period=self.p.vol_ma_period,
            vol_roc_lag=self.p.vol_roc_lag,
            vol_pct_threshold=self.p.vol_pct_threshold,
            vol_roc_threshold=self.p.vol_roc_threshold,
            volatility_period=self.p.volatility_period,
            subplot=False
        )

        # Warmup to ensure indicators are ready
        self.addminperiod(self.p.period)

        # Internal helpers
        self.DCA = True
        self.last_signal = 0

    # --------------------------------------------------------------------- #
    # Signal Logic                                                          #
    # --------------------------------------------------------------------- #
    def compute_sr_signal(self):
        """
        Compute raw S/R pullback signal.

        Returns:
            1  -> long signal
            0  -> no trade
        (Shorts can be added later via self.sr.sell_signal if desired.)
        """
        buy_sig = int(self.sr.buy_signal[0]) if len(self.sr) > 0 else 0
        self.last_signal = buy_sig
        return buy_sig

    def _dollar_size(self, value=None):
        """Return size for a given dollar value at current price."""
        v = value if value is not None else self.p.order_value
        price = float(self.data.close[0])
        if price <= 0:
            return 0.0
        return v / price

    # --------------------------------------------------------------------- #
    # BaseStrategy hook methods                                             #
    # --------------------------------------------------------------------- #
    def buy_or_short_condition(self):
        """
        Called by BaseStrategy to decide if we should open a new position.

        Long-only:
        - If compute_sr_signal() > 0 -> create_order('BUY') with ~10 USD notional.
        """
        signal = self.compute_sr_signal()
        if signal > 0:
            size = self._dollar_size()      # ~10 USD
            if size > 0:
                self.create_order(action='BUY', size=size)
                return True
        return False

    def dca_or_short_condition(self):
        """
        Called by BaseStrategy to decide if we should add a DCA leg.

        - If we have previous entries and price moved dca_deviation% against
          last entry, and signal still points long -> create_order('BUY')
          with another ~10 USD.
        """
        if not self.entry_prices:
            return False

        current_price = float(self.data.close[0])
        last_entry = self.entry_prices[-1]

        threshold = last_entry * (1 - self.p.dca_deviation / 100.0)

        if current_price < threshold:
            signal = self.compute_sr_signal()
            if signal > 0:
                size = self._dollar_size()  # another 10 USD leg
                if size > 0:
                    self.create_order(action='BUY', size=size)
                    return True

        return False

    def sell_or_cover_condition(self):
        """
        Called by BaseStrategy to decide if we should exit / take-profit.

        Uses the OrderTracker take_profit_price (set by BaseStrategy based on
        `take_profit` param). Long-only semantics, same as NRK template.
        """
        current_price = self.data.close[0]

        for order_tracker in list(self.active_orders):
            # For long legs, TP when current_price >= TP
            if current_price >= order_tracker.take_profit_price:
                self.close_order(order_tracker)
                return True

        return False


# from backtrader import backtest, get_crypto_data
# _coin = "TAO"
# _collateral = 'USDT'
# _asset = f'{_coin}/{_collateral}'

# data = get_crypto_data(_asset,          # Pair
#                        "2025-01-01",    # Start date
#                        "2025-11-05",    # End date
#                         "1m",          # Timeframe
#                        "binance"        # Exchange
#                        )

# if __name__ == '__main__':
#     try:
#         backtest(SRPullback_OliverTwist, # Choose your strategy(s) here
#                 data=data,
#                 init_cash=1000,
#                 quantstats=True,
#                 plot=True,
#                 asset_name=_asset)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         print("Full traceback:")
        traceback.print_exc()


coinlist = ['AERO','ALTHEA', 'API3', 'CLOUD', 'EIGEN', 'HBAR', 'KTA', 'QNT', 'SCRT', 'SPX', 'TAO','TON', 'TRUMP', 'VIRTUAL','XRP', 'AERO', 'ALTHEA','EIGEN', 'KTA', 'RUNE','SCRT', 'SPX', 'XRP',]
from backtrader.utils.backtest import backtest, bulk_backtest
# if __name__ == '__main__':
#     try:
#         backtest(
#             SRPullback_OliverTwist,
#             coin='BTC',
#             collateral='USDT',
#             start_date="2025-01-01", 
#             # end_date="2025-02-01", 
#             interval="1m",
#             init_cash=1000,
#             plot=True, 
#             quantstats=True
#         )
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()

if __name__ == '__main__':
    try:
        results = bulk_backtest(
            strategy=SRPullback_OliverTwist,
            # =coinlist, # OPTIONAL :: use no Coin argument to use the whole available Database
            collateral='USDT',
            start_date="2025-01-01", 
            # end_date="2025-02-01", 
            interval="1m",
            init_cash=1000,
            plot=False,
            quantstats=True
        )
        print("Bulk backtest completed successfully.")
        print(f"Results: {results}")
    except Exception as e:
        print(f"An error occurred during the bulk backtest: {e}")