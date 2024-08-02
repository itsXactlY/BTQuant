from live_strategys.QQE_Hullband_VolumeOsc import QQE_Example
from live_strategys.DCA_QQE_Example_backtesting import QQE_DCA_Example
from live_strategys.SuperTrend_Scalp import SuperSTrend_Scalper
from dontcommit import *
from imports import *
import time

def run_backtest():
    startdate = "2021-01-01" # 2021-01-01
    enddate = "2024-07-15" # 2024-07-15
    timeframe = "1s" # else 1m
    coin_name = "BTCUSDT_1s_klines"

    start_date = dt.datetime.strptime(startdate, "%Y-%m-%d")
    end_date = dt.datetime.strptime(enddate, "%Y-%m-%d")
    
    start_time = time.time()

    df = MSSQLData.get_data_from_db(connection_string, coin_name, timeframe, start_date, end_date)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time

    if df.empty:
        print("No data returned from the database. Please check your query and date range.")
        return

    print(f"Data extraction completed in {elapsed_time:.2f} seconds")
    print(f"Number of rows retrieved: {len(df)}")

    data = MSSQLData(dataname=df)
    
    cerebro = bt.Cerebro(oldbuysell=False, stdstats=True)
    cerebro.adddata(data)
    cerebro.addstrategy(SuperSTrend_Scalper, backtest=True)
    
    startcash = 1000
    cerebro.broker.setcash(startcash)

    # Add analyzers
    cerebro.addanalyzer(TimeReturn, _name='time_return')
    cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(DrawDown, _name='drawdown')
    cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(CustomSQN, _name='customsqn')
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # cerebro.addobserver(bt.observers.Value)
    cerebro.addobserver(bt.observers.DrawDown)
    cerebro.addobserver(bt.observers.Cash)
    
    start = cerebro.broker.getvalue()
    strategy = cerebro.run()[0]

    # Get analyzer results
    max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()
    # sqn = strategy.analyzers.sqn.get_analysis()['sqn']

    # Get returns from PyFolio analyzer
    pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    # Convert returns to a Pandas Series and drop NaN values
    returns = pd.Series(returns)
    returns = returns.dropna()
    
    # Format the filename with coin name, start date, end date, and current date
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    filename = f"QuantStat_{coin_name}_{startdate}_to_{enddate}_generated_on_{current_date}_{dt.datetime.now()}.html"

    # Generate QuantStats report
    quantstats.reports.html(returns, output=filename, title=f'{current_date}_{startdate}_to_{enddate}_1m')
    
    def print_trade_analyzer_results(trade_analyzer, indent=0):
        for key, value in trade_analyzer.items():
            if isinstance(value, dict):
                print_trade_analyzer_results(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    # Print all analyzer results
    print_trade_analyzer_results(trade_analyzer)
    pprint(f"Max Drawdown: {max_drawdown}")
    # pprint(f"SQN: {sqn}")
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