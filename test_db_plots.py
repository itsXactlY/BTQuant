import optuna
import backtrader as bt
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX3 as StrategyClass
from backtrader.feeds.mssql_crypto import get_database_data, MSSQLData
from testing_optuna_newmacd import build_optuna_storage
from backtrader.dontcommit import connection_string as MSSQL_ODBC
storage = build_optuna_storage(MSSQL_ODBC)
study = optuna.load_study(study_name="junky_1m_jan2025", storage=storage)

trial_num = 34 # or None for best
trial = (study.best_trial if trial_num is None
         else next(t for t in study.get_trials(deepcopy=False) if t.number == trial_num))

raw_params = trial.params

def get_param_names(cls) -> set:
    names = set()
    try:
        names = set(cls.params._getkeys())  # type: ignore[attr-defined]
    except Exception:
        try:
            # legacy tuple-of-tuples style
            names = set(k for k, _ in cls.params)  # type: ignore[assignment]
        except Exception:
            # fallback: just trust trial params
            names = set(raw_params.keys())
    return names

param_names = get_param_names(StrategyClass)
params = {k: v for k, v in raw_params.items() if k in param_names}


symbol = "XRP"
spec_interval = "1m"
df = get_database_data(
    ticker=symbol,
    start_date="2025-01-01",
    end_date="2025-01-31",
    time_resolution=spec_interval,
    pair="USDT",
)
assert df is not None and not df.is_empty(), "No data returned"
df = df.sort("TimestampStart")

feed = MSSQLData(
    dataname=df,
    datetime="TimestampStart",
    open="Open",
    high="High",
    low="Low",
    close="Close",
    volume="Volume",
)


cerebro = bt.Cerebro(oldbuysell=True)
cerebro.adddata(feed)
cerebro.addstrategy(StrategyClass, backtest=True, **params)
cerebro.broker.setcash(1000)
cerebro.broker.setcommission(commission=0.00075)

cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

result = cerebro.run()[0]


dd = result.analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", 0.0)
sr = result.analyzers.sharpe.get_analysis().get("sharperatio")
ret = result.analyzers.returns.get_analysis().get("rnorm", 0.0)
trades = result.analyzers.trades.get_analysis().get("total", {}).get("total", 0)
print(f"Trial #{trial.number} | value={trial.value:.4f}")
print(f"Sharpe={sr}, MaxDD={dd:.2f}%, CAGR={ret:.2%}, Trades={trades}")
print(f"Final value: {cerebro.broker.getvalue():.2f}")

cerebro.plot(style='candles', volume=True)