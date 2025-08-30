from backtrader.utils.backtest import backtest
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX3 as strategy
import optuna
from testing_optuna_newmacd import build_optuna_storage
from backtrader.dontcommit import connection_string as MSSQL_ODBC
storage = build_optuna_storage(MSSQL_ODBC)
study = optuna.load_study(study_name="junk_1d", storage=storage)

trial_num = None # or None for best
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

param_names = get_param_names(strategy)
params = {k: v for k, v in raw_params.items() if k in param_names}


# --------------- Data spec ---------------
start="2025-01-01"
end="2026-01-31"
tf = "1m"
'''
specs = [
    DataSpec("BTC", start_date=start, end_date=end, interval=tf),
    DataSpec("ETH", start_date=start, end_date=end, interval=tf),
    DataSpec("LTC", start_date=start, end_date=end, interval=tf),
    DataSpec("XRP", start_date=start, end_date=end, interval=tf),
    DataSpec("BCH", start_date=start, end_date=end, interval=tf),
]
'''
if __name__ == '__main__':
    print(f"Using params: {params}")
    print(f"Param names: {param_names}")
    print(f"All raw params: {raw_params}")
    print(f"Trial number: {trial.number}")
    print(f"Trial value: {trial.value}")
    print(f"Trial state: {trial.state}")
    try:
        backtest(
        # optimize_backtest(
            strategy,
            coin='BTC',
            collateral='USDT',
            start_date=start, 
            end_date=end,
            interval=tf,
            init_cash=1000,
            plot=True,
            quantstats=False,
            param_names=param_names,
            params=params,
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Process interrupted by user.")