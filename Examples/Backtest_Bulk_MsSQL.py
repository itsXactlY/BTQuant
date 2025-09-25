from backtrader.utils.backtest import bulk_backtest
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK

coinlist = ['1000CAT','AAVE', 'ACA', 'ACE', 'ACH', 'ACM', 'ACT', 'ACX']

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    try:
        results = bulk_backtest(
            strategy=NRK,
            # coins=coinlist, # OPTIONAL :: use no Coin argument to use the whole available Database
            collateral='USDT',
            start_date="2017-01-01",
            end_date="2026-01-08",
            interval="4h",
            init_cash=100,
            plot=False,
            quantstats=True
        )
        print("Bulk backtest completed successfully.")
        print(f"Results: {results}")
    except Exception as e:
        print(f"An error occurred during the bulk backtest: {e}")