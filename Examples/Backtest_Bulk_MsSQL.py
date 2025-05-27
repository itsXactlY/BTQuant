from backtrader.utils.backtest import bulk_backtest
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK

coinlist = ["SOL","TAO", "AAVE","BCH","SUI","TRUMP","MKR","AVAX","LINK","INJ","APT","QNT","RUNE","WIF","RENDER","VIRTUAL","EGLD","ADA","ATOM","NEAR","UNI","DOT","WLD","PENDLE","NXPC","TRB","ORDI","TIA","ENS","RAY","FET","KAITO","INIT","AR","ENA","ICP","BERA","CRV","FIL","TON","S","EIGEN","BANANA","ZRO","CAKE","ONDO","PEPE","WCT","HBAR","ARB"]

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    try:
        results = bulk_backtest(
            strategy=NRK,
            coins=coinlist,
            start_date="2025-01-01",
            end_date="2025-02-01",
            interval="1m",
            init_cash=1000,
            max_workers=8,
            save_results=True,
        )
        print("Bulk backtest completed successfully.")
        print(f"Results: {results}")
    except Exception as e:
        print(f"An error occurred during the bulk backtest: {e}")