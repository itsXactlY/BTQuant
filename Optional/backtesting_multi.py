import os
import gc
import multiprocessing
import datetime as dt
import pandas as pd
import backtrader as bt
import quantstats
from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
from live_strategys.live_functions import CustomSQN
from live_strategys.QQE_Hullband_VolumeOsc import QQE_Example
from dontcommit import *

# Configuration
base_output_dir = "Bulk Testing" # Make sure folder exist
if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)

results_txt_file = os.path.join(base_output_dir, "results.txt")

def backtest(coin_name, startdate, enddate, timeframe, result_queue, semaphore):
    with semaphore:
        try:
            print(f"Starting backtest for {coin_name}")

            start_date = dt.datetime.strptime(startdate, "%Y-%m-%d")
            end_date = dt.datetime.strptime(enddate, "%Y-%m-%d")
            
            df = MSSQLData.get_data_from_db(connection_string, coin_name, timeframe, start_date, end_date)
            
            if df.empty:
                print("No data returned from the database. Please check your query and date range.")
                result_queue.put(None)
                return
            # pd.set_option('display.max_rows', None)  # Show all rows
            # pd.set_option('display.max_columns', None)  # Show all columns
            # print(df)
            # print(df.tail(1))
            data = MSSQLData(dataname=df)
            
            cerebro = bt.Cerebro(oldbuysell=False, stdstats=True)
            cerebro.adddata(data)
            cerebro.addstrategy(QQE_Example, backtest=True)
            
            startcash = 1000
            cerebro.broker.setcash(startcash)

            cerebro.addanalyzer(TimeReturn, _name='time_return')
            cerebro.addanalyzer(SharpeRatio, _name='sharpe_ratio')
            cerebro.addanalyzer(DrawDown, _name='drawdown')
            cerebro.addanalyzer(TradeAnalyzer, _name='trade_analyzer')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(CustomSQN, _name='customsqn')
            cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
            
            strategy = cerebro.run()[0]

            max_drawdown = strategy.analyzers.drawdown.get_analysis()['max']['drawdown']
            trade_analyzer = strategy.analyzers.trade_analyzer.get_analysis()

            pyfolio_analyzer = strategy.analyzers.getbyname('pyfolio')
            returns, _, _, _ = pyfolio_analyzer.get_pf_items()
            
            returns = pd.Series(returns).dropna()

            current_date = dt.datetime.now().strftime("%Y-%m-%d")
            portvalue = cerebro.broker.getvalue()
            pnl = portvalue - startcash
            
            # Create a subfolder for each coin pair
            coin_dir = os.path.join(base_output_dir, "___BULK___", coin_name)
            if not os.path.exists(coin_dir):
                os.makedirs(coin_dir)

            # Save QuantStats HTML report
            html_filename = f"QuantStat_{coin_name}_generated_on_{current_date}.html"
            html_filepath = os.path.join(coin_dir, html_filename)
            quantstats.reports.html(returns, output=html_filepath, title=f'{current_date}_{startdate}_to_{enddate}_1m')

            # Save results to a coin-specific txt file
            results_txt_file = os.path.join(coin_dir, f"results_{coin_name}.txt")
            
            result = {
                'coin_name': coin_name,
                'max_drawdown': max_drawdown,
                'trade_analyzer': trade_analyzer,
                'final_value': portvalue,
                'pnl': pnl,
                'html_filepath': html_filepath,
                'results_txt_file': results_txt_file
            }

            # Write results to the coin-specific txt file
            write_results_to_file(result, results_txt_file)

            # Clean up resources
            del cerebro
            del data
            del df
            gc.collect()

            return result_queue.put(result)
        except Exception as e:
            print(f"An error occurred while processing {coin_name}: {e}")
            import traceback
            print(f"Full traceback for {coin_name}:")
            traceback.print_exc()
            result_queue.put({'coin_name': coin_name, 'error': str(e)})


def run_backtests(multi=False, is_training=False):
    startdate = "2021-01-01" # 2021-04-14
    enddate = "2024-05-31" # 2022-03-24
    timeframe = "1m"

    if multi:
        pairs = MSSQLData.get_all_pairs(connection_string)
        result_queue = multiprocessing.Queue()
        semaphore = multiprocessing.Semaphore(8)

        processes = []

        for pair in pairs:
            p = multiprocessing.Process(target=backtest, args=(pair, startdate, enddate, timeframe, result_queue, semaphore, is_training))
            p.start()
            processes.append(p)

        results = []
        for p in processes:
            p.join()
            results.append(result_queue.get())

        result_queue.close()
        result_queue.join_thread()
    else:
        result_queue = multiprocessing.Queue()
        semaphore = multiprocessing.Semaphore(1)
        backtest("AERGOUSDT_klines", startdate, enddate, timeframe, result_queue, semaphore, is_training)
        results = [result_queue.get()]

    for result in results:
        if result:
            print_results(result)

def print_results(result):
    print(f"Results for {result['coin_name']}:")
    print(f"Max Drawdown: {result['max_drawdown']}")
    print(f"Final Portfolio Value: ${result['final_value']:.2f}")
    print(f"P/L: ${result['pnl']:.2f}")
    print(f"HTML report saved to: {result['html_filepath']}")
    print(f"Results txt file saved to: {result['results_txt_file']}")
    print("Trade Analyzer Results:")
    print_trade_analyzer_results(result['trade_analyzer'])
    print("\n")

def write_results_to_file(result, file_path):
    with open(file_path, "w") as file:
        file.write(f"Results for {result['coin_name']}:\n")
        file.write(f"Max Drawdown: {result['max_drawdown']}\n")
        file.write(f"Final Portfolio Value: ${result['final_value']:.2f}\n")
        file.write(f"P/L: ${result['pnl']:.2f}\n")
        file.write(f"HTML report saved to: {result['html_filepath']}\n")
        file.write("Trade Analyzer Results:\n")
        write_trade_analyzer_results(result['trade_analyzer'], file)
        file.write("\n\n")

def print_trade_analyzer_results(trade_analyzer, indent=0):
    for key, value in trade_analyzer.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_trade_analyzer_results(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")

def write_trade_analyzer_results(trade_analyzer, file, indent=0):
    for key, value in trade_analyzer.items():
        if isinstance(value, dict):
            file.write("  " * indent + f"{key}:\n")
            write_trade_analyzer_results(value, file, indent + 1)
        else:
            file.write("  " * indent + f"{key}: {value}\n")

   
if __name__ == '__main__':
    try:
        run_backtests(multi=True)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()