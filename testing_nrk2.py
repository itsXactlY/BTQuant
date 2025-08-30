from collections import deque
import polars as pl
import backtrader as bt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ========== Custom Backtrader Indicators ==========
from backtrader.indicators.MesaAdaptiveMovingAverage import MAMA
from backtrader.indicators.ChaikinMoneyFlow import ChaikinMoneyFlow
from backtrader.indicators.ChaikinVolatility import ChaikinVolatility
from backtrader.indicators.MADR import MADRIndicator
from backtrader.strategies.base import BaseStrategy, OrderTracker, datetime


# ========== Polars Indicator Functions ==========

def calculate_rsi(df: pl.DataFrame, period: int = 14) -> pl.Expr:
    delta = df["close"].diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=period)
    avg_loss = loss.ewm_mean(span=period)
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).alias("rsi")


def calculate_atr(df: pl.DataFrame, period: int = 14) -> pl.Expr:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    true_range = pl.max_horizontal([high_low, high_close, low_close])
    return true_range.ewm_mean(span=period).alias("atr")


def calculate_cci(df: pl.DataFrame, period: int = 20) -> pl.Expr:
    tp = ((df["high"] + df["low"] + df["close"]) / 3).alias("tp")
    sma = tp.rolling_mean(window_size=period)
    mad = tp.rolling_map(lambda s: float((s - s.mean()).abs().mean()), window_size=period)
    return ((tp - sma) / (0.015 * mad)).alias("cci")


def calculate_williams_r(df: pl.DataFrame, period: int = 10) -> pl.Expr:
    highest_high = df["high"].rolling_max(window_size=period)
    lowest_low = df["low"].rolling_min(window_size=period)
    return (-100 * (highest_high - df["close"]) / (highest_high - lowest_low)).alias("williams_r")


def calculate_adx(df: pl.DataFrame, period: int = 14) -> pl.Expr:
    high_diff = df["high"].diff()
    low_diff = df["low"].diff()
    plus_dm = pl.when((high_diff > low_diff) & (high_diff > 0)).then(high_diff).otherwise(0)
    minus_dm = pl.when((low_diff > high_diff) & (low_diff < 0)).then(-low_diff).otherwise(0)
    tr = calculate_atr(df, 1) * period
    plus_di = 100 * plus_dm.ewm_mean(span=period) / tr
    minus_di = 100 * minus_dm.ewm_mean(span=period) / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm_mean(span=period).alias("adx")


def calculate_cmf(df: pl.DataFrame, period: int = 20) -> pl.Expr:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        (df["high"] - df["low"]).replace(0, 1e-9)
    )
    mf = clv * df["volume"]
    cmf = mf.rolling_sum(window_size=period) / df["volume"].rolling_sum(window_size=period)
    return cmf.fill_null(0).alias("cmf")


def calculate_chaikin_volatility(df: pl.DataFrame, ema_period: int = 10, roc_period: int = 10) -> pl.Expr:
    hl_range = df["high"] - df["low"]
    ema_range = hl_range.ewm_mean(span=ema_period)
    roc = ema_range / ema_range.shift(roc_period) - 1
    return (roc * 100).fill_null(0).alias("cv")


def calculate_mama_fama(fast=20, slow=50):
    return pl.col("close").ewm_mean(span=fast), pl.col("close").ewm_mean(span=slow)


def calculate_madr_buy_signal(df: pl.DataFrame, window=14, atr_multiplier=0.45) -> pl.Expr:
    atr = calculate_atr(df, window)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling_mean(window_size=window)
    lower_band = sma - (atr * atr_multiplier)
    return (df["close"] < lower_band).alias("madr_buy")


# ========== KNN Signal Engine (Polars + sklearn) ==========

def calculate_rolling_knn_signals(df: pl.DataFrame,
                                  feature_cols: list[str],
                                  window: int,
                                  n_neighbors: int,
                                  horizon: int):
    df = df.with_columns(
        ((df["close"].shift(-horizon) / df["close"]) - 1).alias("forward_return")
    ).with_columns(
        pl.when(pl.col("forward_return") > 0).then(1)
        .when(pl.col("forward_return") < 0).then(-1)
        .otherwise(0)
        .alias("ml_label")
    )

    print(f"Calculating KNN signals for {len(df)} bars...")
    signals = np.zeros(len(df), dtype=np.int8)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="manhattan")

    for i in range(window, len(df)):
        if i % 1000 == 0:
            print(f"Processing bar {i}/{len(df)}")
        try:
            train_slice = df.slice(i - window, window)
            X_train = train_slice.select(feature_cols).to_numpy()
            y_train = train_slice["ml_label"].to_numpy()

            mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            X_train, y_train = X_train[mask], y_train[mask]

            if len(X_train) >= n_neighbors + 5:
                knn.fit(X_train, y_train)
                # FIXED way to get row i
                X_current = df[feature_cols].slice(i, 1).to_numpy().reshape(-1)
                if not np.isnan(X_current).any():
                    signals[i] = knn.predict([X_current])[0]
        except Exception:
            signals[i] = 0

    return signals


# ========== Data Preprocessing ==========
def preprocess_data(df: pl.DataFrame, params) -> pl.DataFrame:
    print("Starting data preprocessing...")
    df = df.clone()

    df = df.with_columns([
        calculate_rsi(df),
        calculate_cci(df),
        calculate_atr(df, params.atr_period),
        calculate_williams_r(df),
        calculate_adx(df),
        calculate_cmf(df, params.cmf_period),
        calculate_chaikin_volatility(df, params.cv_ema, params.cv_roc),
        calculate_madr_buy_signal(df),
    ])

    mama, fama = calculate_mama_fama(params.mama_fast, params.mama_slow)
    df = df.with_columns([mama.alias("mama"), fama.alias("fama")])

    cv_ma = df["cv"].rolling_mean(100).alias("cv_ma")
    cv_sd = df["cv"].rolling_std(100).alias("cv_sd")
    cv_z = ((df["cv"] - cv_ma) / cv_sd).fill_null(0).alias("cv_zscore")

    df = df.with_columns([
        cv_ma,
        cv_sd,
        cv_z,
        (df["atr"] / df["close"]).alias("atr_normalized"),
    ])

    # 👉 Add ML signals here
    feature_cols = ["rsi", "cci", "williams_r", "adx", "cmf", "atr_normalized"]
    signals = calculate_rolling_knn_signals(df, feature_cols,
                                            params.feature_window,
                                            params.neighbors_count,
                                            params.horizon)
    df = df.with_columns(pl.Series("ml_signal", signals))

    print("Data preprocessing completed!")
    return df

# ========== Strategy Class ==========
class NRK2_Optimized(BaseStrategy):
    params = (
        ("source", "close"),
        ("max_bars_back", 100),
        ("feature_window", 50),
        ("horizon", 5),
        ("neighbors_count", 7),
        ("use_regime", True),
        ("cmf_period", 20),
        ("cv_ema", 10),
        ("cv_roc", 10),
        ("mama_fast", 20),
        ("mama_slow", 50),
        ("target_annual_vol", 0.20),
        ("atr_period", 14),
        ("max_gross_exposure", 1.0),
        ("dca_deviation", 1.5),
        ("max_dcas", 3),
        ("take_profit", 2.0),
        ("dd_cut", 0.18),
        ("cv_block_z", 2.0),
        ("debug", True),
        ("backtest", None),
        ("use_precalc", True),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.addminperiod(max(self.p.max_bars_back, 50))

        self.use_precalc = (
            hasattr(self.data, "df")
            and self.p.use_precalc
            and isinstance(self.data.df, pl.DataFrame)
            and "ml_signal" in self.data.df.columns
        )

        if self.use_precalc:
            print("Using pre-calculated data")
            self.df = self.data.df
            self.current_idx = 0
            self.src = getattr(self.data, self.p.source)
        else:
            print("Using realtime mode (slower)")
            self.src = getattr(self.data, self.p.source)
            self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period, plot=False)
            self.rsi = bt.indicators.RSI(self.data, period=14, plot=False)
            self.cci = bt.indicators.CCI(self.data, period=20, plot=False)
            self.wt = bt.indicators.WilliamsR(self.data, period=10, plot=False)
            self.adx = bt.indicators.ADX(self.data, period=14, plot=False)
            self.cmf = ChaikinMoneyFlow(self.data, period=self.p.cmf_period)
            self.cv = ChaikinVolatility(ema_period=self.p.cv_ema, roc_period=self.p.cv_roc)
            self.madr = MADRIndicator(self.data, window=14, atr_window=14, atr_multiplier=0.45)
            self.mama = MAMA(self.data, fast=self.p.mama_fast, slow=self.p.mama_slow)
            self.cv_ma = bt.indicators.SMA(self.cv.cvi, period=100)
            self.cv_sd = bt.indicators.StdDev(self.cv.cvi, period=100)
            self.features = [self.rsi, self.cci, self.wt, self.adx, self.cmf.money_flow]
            self._X = deque(maxlen=self.p.feature_window)
            self._y = deque(maxlen=self.p.feature_window)
            self.knn = KNeighborsClassifier(n_neighbors=self.p.neighbors_count, weights="distance", metric="manhattan")
            self._knn_ready = False

        self.active_orders = []
        self.entry_prices, self.sizes = [], []
        self.first_entry_price = None
        self.buy_executed = False
        self.dca_count = 0
        self.equity_peak = None

    def get_current_data(self, field):
        if self.use_precalc and self.current_idx < len(self.df):
            val = self.df[field][self.current_idx]
            return val.item() if hasattr(val, "item") else val
        return 0

    def _forward_return_label(self, H):
        try:
            fwd = (self.src[0] / self.src[-H]) - 1.0
        except IndexError:
            return None
        return 1 if fwd > 0 else -1 if fwd < 0 else 0

    def _collect_features(self):
        if self.use_precalc:
            return np.array([
                self.get_current_data("rsi"),
                self.get_current_data("cci"),
                self.get_current_data("williams_r"),
                self.get_current_data("adx"),
                self.get_current_data("cmf"),
                self.get_current_data("atr_normalized"),
            ])
        vals = []
        for f in self.features:
            try: vals.append(float(f[0]))
            except: vals.append(0.0)
        vals += [float(self.atr[0] / max(1e-9, self.src[0])), float(self.adx[0])]
        return np.nan_to_num(np.array(vals), nan=0.0)

    def _regime_ok(self):
        if not self.p.use_regime:
            return True
        if self.use_precalc:
            return bool(self.get_current_data("cv_zscore") < self.p.cv_block_z)
        trend_up = self.mama.MAMA[0] > self.mama.FAMA[0]
        flow_ok = self.cmf.money_flow[0] > 0
        z = 0.0
        if self.cv_sd[0] > 0: z = (self.cv.cvi[0] - self.cv_ma[0]) / self.cv_sd[0]
        vol_ok = z < self.p.cv_block_z
        return trend_up and flow_ok and vol_ok

    def compute_ml_signal(self):
        if self.use_precalc:
            return int(self.get_current_data("ml_signal"))
        x = self._collect_features()
        self._X.append(x)
        lab = self._forward_return_label(self.p.horizon)
        if lab is not None: self._y.append(lab)
        if len(self._y) < max(30, self.p.neighbors_count + 5): return 0
        try: self.knn.fit(np.vstack(self._X), np.array(self._y)); self._knn_ready = True
        except: self._knn_ready = False; return 0
        try: return int(self.knn.predict([x])[0])
        except: return 0

    def next(self):
        super().next()
        if self.use_precalc: self.current_idx += 1
        sig = self.compute_ml_signal()
        # trading logic truncated for brevity, remains the same as your draft
        ...


# ========== Backtest Function ==========
def sanitize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure OHLCV columns are plain Float64 scalars (not list[f64])."""
    cols = ["open", "high", "low", "close", "volume"]
    out = df
    for col in cols:
        if col not in out.columns:
            continue
        # If column is list, take first element per row
        if out[col].dtype == pl.List(pl.Float64):
            out = out.with_columns(pl.col(col).list.first().alias(col))
        # Force cast to Float64
        out = out.with_columns(pl.col(col).cast(pl.Float64))
    return out

def fast_backtest(strategy_class,
                  coin="BTC",
                  start_date="2024-01-01",
                  end_date="2024-04-08",
                  interval="1m",
                  init_cash=1000,
                  plot=True):
    from backtrader.feeds.mssql_crypto import get_database_data

    print("Loading data...")
    data_df = get_database_data(coin, start_date, end_date, interval)
    data_df = data_df.rename({c: c.lower() for c in data_df.columns})
    print(f"Data loaded: {len(data_df)} rows")

    # 🔥 Sanitize OHLCV
    data_df = sanitize_ohlcv(data_df)
    print(data_df.dtypes)  # sanity check: all Float64 now

    processed_pl = preprocess_data(data_df, strategy_class.params)

    # feed PolarsData directly (no need to convert to Pandas)
    data_feed = bt.feeds.PolarsData(dataname=processed_pl)
    # attach the precomputed DF so the strategy can check `data.df.ml_signal`
    data_feed.df = processed_pl

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, use_precalc=True)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(init_cash)
    print("Running backtest...")
    result = cerebro.run()

    if plot:
        cerebro.plot()
    return result

NRK2 = NRK2_Optimized


if __name__ == "__main__":
    try:
        fast_backtest(
            NRK2_Optimized,
            coin="BTC",
            start_date="2024-01-01",
            end_date="2024-01-08",
            interval="1m",
            init_cash=1000,
            plot=True,
        )
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()