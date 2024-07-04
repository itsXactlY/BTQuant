from imports import *
from live_strategys.QQE_Hullband_VolumeOsc import QQE_Example
from live_strategys.DCA_QQE_Example_backtesting import QQE_DCA_Example
from live_strategys.SuperTrend_Scalp import SuperSTrend_Scalper

def backtest():
    running_backtest(True)

    # Load the data
    my_data_frame = pd.read_csv(rvn, index_col=0, parse_dates=True)
    my_data_frame = my_data_frame.sort_index()

    _start = "2022-01-01"
    _end = "2022-02-27"
    df = my_data_frame.loc[_start:_end]
    data = MyPandasData(dataname=df)
    
    cerebro = bt.Cerebro(oldbuysell=True)
    cerebro.adddata(data)
    cerebro.addstrategy(SuperSTrend_Scalper, backtest=True)
    
    startcash = 10000
    cerebro.broker.setcash(startcash)

    # Add analyzers
    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)
    
    start = cerebro.broker.getvalue()
    strategy = cerebro.run()[0]

    # Get analyzer results
    max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()['sqn']

    # Get returns from PyFolio analyzer
    pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    # Convert returns to a Pandas Series and drop NaN values
    returns = pd.Series(returns)
    returns = returns.dropna()

    # Format the filename with coin name, start date, end date, and current date
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    filename = f"QuantStat_{_start}_to_{_end}_generated_on_{current_date}.html"

    # Generate QuantStats report
    quantstats.reports.html(returns, output=filename, title=f'{current_date}_{_start}_to_{_end}_1m')
    
    def print_trade_analyzer_results(trade_analyzer, indent=0):
        for key, value in trade_analyzer.items():
            if isinstance(value, dict):
                print_trade_analyzer_results(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    # Print all analyzer results
    print_trade_analyzer_results(trade_analyzer)
    pprint(f"Max Drawdown: {max_drawdown}")
    pprint(f"SQN: {sqn}")
    portvalue = cerebro.broker.getvalue()
    pnl = portvalue - startcash
    pprint('Final Portfolio Value: ${}'.format(portvalue))
    pprint('P/L: ${}'.format(pnl))

    cerebro.plot(style='candles', numfigs=1, volume=True)

    return start - cerebro.broker.getvalue()

if __name__ == '__main__':
    backtest()
