'''
Introduction to Machine Learning-Enhanced Algorithmic Trading with BTQuant

In this example, we will walk through the process of building a simple machine learning-driven trading strategy using BTQuant, 
in combination with scikit-learn for predictive modeling and Polars instead of slow "Pandas" for real fast, expressive data transformations.

Objectives:

In the end, you might get an understanding how to:

    Retrieve and prepare cryptocurrency OHLCV data from a SQL database.

    Engineer meaningful features from price data (e.g., moving averages, Bollinger Bands).

    Label data for supervised learning based on future returns.

    Train a K-Nearest Neighbors (KNN) classifier to predict price direction.

    Save and reuse machine learning models with joblib.

    Integrate the trained ML model into a custom BTQuant strategy for signal generation.

    Implement Dollar-Cost Averaging (DCA) and Take-Profit logic in your strategy.

This example uses DOGE/USDT 1-minute candlestick data between January 2024 and January 2025 to demonstrate:

    Data preprocessing using polars.

    Feature creation from moving averages, momentum, Bollinger Bands, and time-based features.

    Classification of future price changes into "buy", "hold", or "sell" classes.

    Training and testing of a KNeighborsClassifier model.

    Embedding this ML model into a live-capable BTQuant strategy called ML_Suite.

This is a realistic, end-to-end pipeline that mimics how modern quant traders build and deploy rule-based ML strategies, blending traditional technical indicators with supervised learning.

    Note to learners:
    While the strategy here is simplified for instructional purposes, the core ideas—like engineered features, label generation, model integration, and risk management—form the foundation of many real-world algorithmic trading systems.

Now, let's dive into the code and explore how each piece fits together to create a responsive, intelligent trading system.
'''

from datetime import datetime
import os
import numpy as np
import polars as pl
import backtrader as bt
from backtrader.feeds.mssql_crypto import get_database_data
from backtrader.strategies.base import BaseStrategy, OrderTracker
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier


_coin = 'BTC'
start_date = '2017-01-01'
end_date   = '2022-01-11'

data = get_database_data(_coin, start_date, end_date, '1m')

if data.schema['TimestampStart'] != pl.Datetime:
    data = data.with_columns(
        pl.col('TimestampStart')
          .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
          .alias('TimestampStart')
    )

hlc3 = (pl.col('High') + pl.col('Low') + pl.col('Close')) / 3

window_bbands = 20
n_std = 2

data = (
    data
    .with_columns([
        pl.col('Close').rolling_mean(10).alias('sma10'),
        pl.col('Close').rolling_mean(20).alias('sma20'),
        pl.col('Close').rolling_mean(50).alias('sma50'),
        pl.col('Close').rolling_mean(100).alias('sma100'),
        pl.col('Close').rolling_mean(200).alias('sma200'),
        hlc3.rolling_mean(window_bbands).alias('bb_mean'),
        hlc3.rolling_std(window_bbands).alias('bb_std')
    ])
    .with_columns([
        (pl.col('bb_mean') + n_std * pl.col('bb_std')).alias('bb_upper'),
        (pl.col('bb_mean') - n_std * pl.col('bb_std')).alias('bb_lower')
    ])
)

start_lit = pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
end_lit   = pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")

data = data.with_columns([
    ((pl.col('Close') - pl.col('sma10')) / pl.col('Close')).alias('X_SMA10'),
    ((pl.col('Close') - pl.col('sma20')) / pl.col('Close')).alias('X_SMA20'),
    ((pl.col('Close') - pl.col('sma50')) / pl.col('Close')).alias('X_SMA50'),
    ((pl.col('Close') - pl.col('sma100')) / pl.col('Close')).alias('X_SMA100'),
    ((pl.col('Close') - pl.col('sma200')) / pl.col('Close')).alias('X_SMA200'),
    ((pl.col('sma10') - pl.col('sma20')) / pl.col('Close')).alias('X_DELTA_SMA10'),
    ((pl.col('sma20') - pl.col('sma50')) / pl.col('Close')).alias('X_DELTA_SMA20'),
    ((pl.col('sma50') - pl.col('sma100')) / pl.col('Close')).alias('X_DELTA_SMA50'),
    ((pl.col('sma100') - pl.col('sma200')) / pl.col('Close')).alias('X_DELTA_SMA100'),
    pl.col('Close').pct_change(2).alias('X_MOM'),
    ((pl.col('bb_upper') - pl.col('Close')) / pl.col('Close')).alias('X_BB_upper'),
    ((pl.col('bb_lower') - pl.col('Close')) / pl.col('Close')).alias('X_BB_lower'),
    ((pl.col('bb_upper') - pl.col('bb_lower')) / pl.col('Close')).alias('X_BB_width'),
    (~pl.col('TimestampStart').dt.date().is_between(start_lit, end_lit)).alias('X_Sentiment'),
    pl.col('TimestampStart').dt.weekday().alias('X_day'),
    pl.col('TimestampStart').dt.hour().alias('X_hour')
])

from polars import when
plt = when(pl.col('Close').pct_change(48).shift(-48).abs() < 0.004).then(0)
plt = plt.when(pl.col('Close').pct_change(48).shift(-48) > 0).then(1)
plt = plt.otherwise(-1).alias('y')
data = data.with_columns(plt)

data = data.drop_nulls()

feature_cols = [c for c in data.columns if c.startswith('X_')]
X = data.select(feature_cols).to_numpy()
y = data.select('y').to_numpy().flatten()

n_train = int(len(X) * 0.4)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

model_path = f"{_coin.lower()}_knn_model.joblib"

if os.path.exists(model_path):
    clf = load(model_path)
    print("✅ Model loaded from disk. No retraining.")
else:
    clf = KNeighborsClassifier(7)
    clf.fit(X_train, y_train)
    dump(clf, model_path)
    print("💾 Model trained and saved to disk.")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import math

    pred = clf.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, pred))
    print('MSE:', mean_squared_error(y_test, pred))
    print('RMSE:', math.sqrt(mean_squared_error(y_test, pred)))
    print('R2:', r2_score(y_test, pred))


class ML_Suite(BaseStrategy):
    params = (
        ('dca_deviation', 1.5),
        ('take_profit', 2),
        ('percent_sizer', 0.05), # 0.05 -> 5%
        ('debug', False),
        ('backtest', None),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_cols = feature_cols
        self.n_train = n_train
        self.data_df = self.datas[0].p.dataname
        
        self.DCA = True
        
        self.model_loaded = False
        try:
            with open(model_path, "rb") as f:
                self.clf = load(f)
                self.model_loaded = True
        except FileNotFoundError:
            self.clf = clf
        
        print(f"Initialized NRK_ML with {len(self.feature_cols)} features and n_train={self.n_train}")

    def compute_ml_signal(self, idx):
        features = [
            'X_SMA10', 'X_SMA20', 'X_SMA50', 'X_SMA100', 'X_SMA200',
            'X_DELTA_SMA10', 'X_DELTA_SMA20', 'X_DELTA_SMA50', 'X_DELTA_SMA100',
            'X_MOM', 'X_BB_upper', 'X_BB_lower', 'X_BB_width',
            'X_Sentiment', 'X_day', 'X_hour'
        ]
        
        row = self.data_df.slice(idx, 1).select(features)
        if row.null_count().sum_horizontal().item() > 0:
            return None

        feat = np.array(row.to_numpy())
        return self.clf.predict(feat)[0]


    def buy_or_short_condition(self):
        idx = len(self.data) - 1
        if not self.model_loaded and idx < self.n_train:
            return False 

        signal = self.compute_ml_signal(idx)
        if signal is None:
            return
        if not self.buy_executed and signal > 0:
            size = self._determine_size()
            tracker = OrderTracker(
                entry_price=self.data.close[0],
                size=size,
                take_profit_pct=self.params.take_profit,
                symbol=getattr(self, 'symbol', None),
                order_type="BUY",
                backtest=self.params.backtest
            )
            tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.active_orders.append(tracker)
            self.entry_prices.append(self.data.close[0])
            self.sizes.append(size)
            self.order = self.buy(size=size, exectype=bt.Order.Market)
            if self.p.debug:
                print(f"Buy placed for {size} @ {self.data.close[0]:.4f}")
            self.buy_executed = True
            self.calc_averages()

    def dca_or_short_condition(self):
        if not self.entry_prices:
            return
        last_price = self.entry_prices[-1]
        if self.data.close[0] < last_price * (1 - self.params.dca_deviation / 100):
            idx = len(self.data) - 1
            if not self.model_loaded and idx < self.n_train:
                return False 

            signal = self.compute_ml_signal(idx)
            if signal is None:
                return
            if signal > 0:
                size = self._determine_size()
                tracker = OrderTracker(
                    entry_price=self.data.close[0],
                    size=size,
                    take_profit_pct=self.params.take_profit,
                    symbol=getattr(self, 'symbol', None),
                    order_type="BUY",
                    backtest=self.params.backtest
                )
                tracker.order_id = f"order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                self.active_orders.append(tracker)
                self.entry_prices.append(self.data.close[0])
                self.sizes.append(size)
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"DCA Buy placed for {size} @ {self.data.close[0]:.4f}")
                self.calc_averages()

    def sell_or_cover_condition(self):
        if not self.active_orders or not self.buy_executed:
            return
        current_price = self.data.close[0]
        to_remove = []
        for i, order in enumerate(self.active_orders):
            if current_price >= order.take_profit_price:
                self.order = self.sell(size=order.size, exectype=bt.Order.Market)
                if self.p.debug:
                    print(f"TP hit: Sold {order.size} @ {current_price:.4f} (entry: {order.entry_price:.4f})")
                order.close_order(current_price)
                to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.active_orders.pop(i)
        if not self.active_orders:
            self.reset_position_state()
            self.buy_executed = False
        else:
            self.entry_prices = [o.entry_price for o in self.active_orders]
            self.sizes = [o.size for o in self.active_orders]


    def next(self):
        super().next()
        if self.p.backtest == False:
            self.report_positions()
            dt = self.datas[0].datetime.datetime(0)
            print(f'Realtime: {datetime.now()} processing candle date: {dt}, with {self.data.close[0]}')


from backtrader.utils.backtest import backtest
backtest(ML_Suite, data, init_cash=1000, backtest=True, plot=True, quantstats=True, asset_name=f"{_coin}/USDT")