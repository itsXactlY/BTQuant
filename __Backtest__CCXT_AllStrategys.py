from fastquant import backtest, get_crypto_data
crypto = get_crypto_data("BTC/USDT", "2024-01-01", "2024-08-08", "1m")


# backtest('bbands', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('rsi', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('smac', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('macd', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('emac', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('buynhold', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('qqe', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('OrChainKioseff', crypto, init_cash=1000, backtest=True, plot=True, verbose=1)
backtest('msa', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)
# backtest('STScalp', crypto, init_cash=1000, backtest=True, plot=True, verbose=0)