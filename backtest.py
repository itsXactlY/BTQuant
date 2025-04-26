'''
############## IMPORTANT NOTE ABOUT IMPORTED STRATEGYS IN THIS FILE - LOAD OR IMPORT ONLY THAT PARTICULAR STRATEGY U USE! ##############
############## BACKTRADER WARMING UP EVERY POSSIBLE STRATEGY WHAT IS DECLARED AS IMPORT HERE! ##############
############## CAUSING ALOT OF WARMUP TIME, MEMORY CONSUMPTION, INDICATORS, AND EVERYTHING BEYONED (TIME IS MONEY!) ##############
'''
from backtrader.imports import CustomPandasData, bt, pd, TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer, CustomSQN, dt, quantstats, pprint
from backtrader.strategies.NearestNeighbors_RationalQuadraticKernel import NRK
from backtrader.feeds.mssql_crypto import get_database_data

rawdata = get_database_data("ETH", "2023-01-01", "2024-01-02", "1m")

def run_backtest():
    data = CustomPandasData(dataname=rawdata)
    
    cerebro = bt.Cerebro(oldbuysell=True)
    cerebro.adddata(data)
    cerebro.addstrategy(NRK, backtest=True)
    
    startcash = 1000
    cerebro.broker.setcash(startcash)

    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    
    # cerebro.addobserver(bt.observers.Value)
    # cerebro.addobserver(bt.observers.DrawDown)
    # cerebro.addobserver(bt.observers.Cash)
    
    start = cerebro.broker.getvalue()
    strategy = cerebro.run()[0]

    max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()

    pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    returns = pd.Series(returns)
    returns = returns.dropna()
    
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    filename = f"QuantStat_generated_on_{current_date}_{dt.datetime.now()}.html"

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
    pnl = portvalue - startcash
    pprint('Final Portfolio Value: ${}'.format(portvalue))
    pprint('P/L: ${}'.format(pnl))
    
    cerebro.plot(style='candles', numfigs=1, volume=False, barup='lightgreen', bardown='red')
    
    return start - cerebro.broker.getvalue()

if __name__ == '__main__':
    try:
        run_backtest()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
