from backtrader.utils.backtest import backtest
from backtrader.strategies.MACD_ADX import Enhanced_MACD_ADX4 as strategy
# from backtrader.strategies.MACD_ADX import VectorMACD_ADX as strategy
import optuna
from testing_optuna_newmacd import build_optuna_storage
from backtrader.dontcommit import connection_string as MSSQL_ODBC
storage = build_optuna_storage(MSSQL_ODBC)
study = optuna.load_study(study_name="BullBearMarketBTC-ETH-LTC-XRP-BCH_1m_MACD_ADXV3", storage=storage)
# study = optuna.load_study(study_name="Optimized_1m_MTF_MACD_ADX_VEC_V2", storage=storage)

from rich.console import Console
console = Console()

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
bull_start = "2020-09-28"
bull_end = "2021-05-31"
bear_start = "2022-05-28"
bear_end = "2023-06-23"
# Optional holdout test period
test_bull_start="2023-06-12"
test_bull_end="2025-05-31"
tf = "15m"

if __name__ == '__main__':
    console.print(f"Using params: {params}")
    # print(f"All raw params: {raw_params}")
    console.print(f"Trial number: {trial.number}")
    # print(f"Trial value: {trial.value}")
    # print(f"Trial state: {trial.state}")
    try:
        backtest(
            strategy,
            coin='BTC',
            collateral='USDT',
            start_date=bear_end,
            # end_date=test_bull_end,
            interval=tf,
            plot=True,
            quantstats=True,
            exchange='MEXC', # enables can_short if strategy has it
            add_mtf_resamples=True,       # 5m/15m/60m-Guards
            params={**params,        # Optuna-Params
                    'qty_step': 0.00001,    # BTC
                    'price_tick': 0.1},
            debug=True
        )

    except Exception as e:
        console.print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        console.print("Process interrupted by user.")