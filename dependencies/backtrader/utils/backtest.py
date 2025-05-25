import backtrader as bt
from collections.abc import Iterable
from datetime import datetime

INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075

def backtest(
    strategy,
    data,  # Treated as csv path is str, and dataframe of pd.DataFrame
    commission=COMMISSION_PER_TRANSACTION,
    init_cash=INIT_CASH,
    plot=True,
    quantstats=False,
    **kwargs,
):

    from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomPandasData
    from pprint import pprint
    
    kwargs = {
        k: v if isinstance(v, Iterable) and not isinstance(v, str) else [v]
        for k, v in kwargs.items()
    }


    from backtrader.feed import DataBase
    if isinstance(data, DataBase):
        data = data
    else:
        data = CustomPandasData(dataname=data)

    
    
    cerebro = bt.Cerebro(oldbuysell=True)
    cerebro.adddata(data)
    cerebro.addstrategy(strategy, backtest=True)

    # Add Strategy
    strat_names = []
    strat_name = None
    if not isinstance(strategy, str) and issubclass(strategy, bt.Strategy):
        strat_name = (
            strategy.__name__ if hasattr(strategy, "__name__") else str(strategy)
        )
    else:
        strat_name = strategy
    
    strat_names.append(strat_name)
    cerebro.broker.setcash(init_cash)

    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)
    
    cerebro.broker.setcommission(commission=commission)

    start = cerebro.broker.getvalue()
    strategy = cerebro.run()[0]

    max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()

    pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    import pandas as pd
    returns = pd.Series(returns)
    returns = returns.dropna()
    
    if quantstats:
        import quantstats_lumi as quantstats
        current_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"QuantStat_generated_on_{current_date}_{datetime.now()}.html"
        quantstats.reports.html(returns, output=filename, title=f'QuantStats_{current_date}')

    def print_trade_analyzer_results(trade_analyzer, indent=0):
        for key, value in trade_analyzer.items():
            if isinstance(value, dict):
                print_trade_analyzer_results(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    # Print all analyzer results
    print_trade_analyzer_results(trade_analyzer)
    pprint(f"Max Drawdown: {max_drawdown}")
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - init_cash
    pprint('Final Portfolio Value: ${}'.format(portvalue))
    pprint('P/L: ${}'.format(pnl))
    
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=False, barup='black', bardown='grey')
    
    return cerebro.broker.getvalue() - start