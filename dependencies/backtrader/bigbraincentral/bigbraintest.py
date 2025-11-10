from datetime import datetime, timedelta
import backtrader as bt
from backtrader.dataseries import TimeFrame
from backtrader.strategies.Aligator_supertrend import AliG_STrend
from backtrader.bigbraincentral.db_ohlcv_mssql import (
    MSSQLFeedConfig, BinanceDBData,
)

cfg = MSSQLFeedConfig(
    server="localhost",
    database="BTQ_MarketData",   # deine Live-DB
    username="SA",
    password="q?}33YIToo:H%xue$Kr*",
)

data = BinanceDBData(
    db_config=cfg,
    symbol="btcusdt",
    exchange="binance",
    timeframe=TimeFrame.Seconds,
    compression=1,
    fromdate=datetime.utcnow() - timedelta(hours=1),
    live=True,              # oder False für reinen Backtest
    poll_interval=0.25,
    mode="global",          # oder "per_pair"
    global_table="ohlcv",   # globale Tabelle
    # für per_pair z.B.:
    # mode="per_pair",
    # table_pattern="{symbol}_klines",
)


cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(AliG_STrend)
cerebro.run()
