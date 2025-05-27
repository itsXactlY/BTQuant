import os
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
    asset_name=None,
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
        current_time = datetime.now().strftime("%H-%M-%S")
        
        if asset_name:
            coin_name = asset_name.replace('/', '_')
        elif hasattr(data, '_dataname'):
            coin_name = data._dataname
        else:
            coin_name = "Unknown_Asset"

        folder = os.path.join("QuantStats")
        os.makedirs(folder, exist_ok=True)

        filename = os.path.join(folder, f"{coin_name}_{current_date}_{current_time}.html")
        quantstats.reports.html(
            returns, 
            output=filename, 
            title=f'QuantStats_{coin_name}_{current_date}'
        )

    elif not quantstats:

        def print_trade_analyzer_results(trade_analyzer, indent=0):
            for key, value in trade_analyzer.items():
                if isinstance(value, dict):
                    print_trade_analyzer_results(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")

        print_trade_analyzer_results(trade_analyzer)
        pprint(f"Max Drawdown: {max_drawdown}")
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - init_cash
        pprint('Final Portfolio Value: ${}'.format(portvalue))
        pprint('P/L: ${}'.format(pnl))
    
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=False, barup='black', bardown='grey')
    
    return cerebro.broker.getvalue()

import concurrent.futures
import matplotlib
matplotlib.use('Agg') # Workaround for Quantstats matplotlib errors when multiprocessing
import gc
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

console = Console()

def fetch_all_data_sequential(coins, start_date, end_date, interval, progress_callback=None):
    from backtrader.feeds.mssql_crypto import get_database_data
    
    data_cache = {}

    for i, coin in enumerate(coins):
        try:
            data = get_database_data(coin, start_date, end_date, interval)
            data_cache[coin] = data

            if progress_callback:
                progress_callback(i + 1, len(coins), coin, "success")
                
        except Exception as e:
            data_cache[coin] = None
            if progress_callback:
                progress_callback(i + 1, len(coins), coin, "failed")
    
    return data_cache

def run_backtest_with_data(args):
    coin, data, strategy_class, init_cash, backtest_params = args

    if data is None:
        return {"coin": coin, "asset": f"{coin}", "error": "No data available", "status": "skipped"}

    try:
        asset = f'{coin}'
        
        result = backtest(
            strategy_class,
            data,
            init_cash=init_cash,
            **backtest_params,
            asset_name=asset
        )

        del data
        gc.collect()

        return {
            "coin": coin,
            "asset": asset,
            "result": result,
            "final_value": result,
            "pnl": result - init_cash,
            "return_pct": ((result - init_cash) / init_cash) * 100,
            "status": "success"
        }

    except Exception as e:
        console.print(f"[red]Error in {coin}: {str(e)}[/red]")
        return {"coin": coin, "asset": f"{coin}", "error": str(e), "status": "failed"}

def bulk_backtest(strategy, coins, start_date, end_date, interval, 
                 init_cash=1000, max_workers=8, save_results=True, 
                 output_file='backtest_results.json', **backtest_kwargs):

    default_backtest_params = {
        'backtest': True,
        'plot': False,
        'quantstats': True
    }
    default_backtest_params.update(backtest_kwargs)
    
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    with progress:
        fetch_task = progress.add_task("📊 Fetching data...", total=len(coins))
        
        def update_fetch_progress(current, total, coin, status):
            status_emoji = "✅" if status == "success" else "❌"
            progress.update(
                fetch_task, 
                completed=current,
                description=f"📊 Fetching data... {current}/{total} - Last: {coin} {status_emoji}"
            )
        
        data_cache = fetch_all_data_sequential(
            coins, start_date, end_date, interval, 
            progress_callback=update_fetch_progress
        )
        
        progress.update(fetch_task, completed=len(coins), description="📊 Data fetching complete")
        
        valid_pairs = [(coin, data) for coin, data in data_cache.items() if data is not None]
        skipped_coins = [coin for coin, data in data_cache.items() if data is None]
        
        progress.console.print(f"\n[green]✓[/green] Found data for [bold]{len(valid_pairs)}[/bold] coins, skipping [bold]{len(skipped_coins)}[/bold] coins")
        if skipped_coins:
            progress.console.print(f"[yellow]Skipped coins:[/yellow] {', '.join(skipped_coins)}")
        
        results = []
        
        for coin in skipped_coins:
            results.append({"coin": coin, "asset": f"{coin}", "error": "No data available", "status": "skipped"})
        
        if valid_pairs:
            backtest_task = progress.add_task("🚀 Running backtests...", total=len(valid_pairs))
            
            successful_count = 0
            failed_count = 0
            
            backtest_args = [
                (coin, data, strategy, init_cash, default_backtest_params)
                for coin, data in valid_pairs
            ]
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_coin = {
                    executor.submit(run_backtest_with_data, args): args[0] 
                    for args in backtest_args
                }
                
                for future in concurrent.futures.as_completed(future_to_coin):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["status"] == "success":
                            successful_count += 1
                        else:
                            failed_count += 1
                        
                        coin = result.get('coin', 'Unknown')
                        completed = successful_count + failed_count
                        
                        progress.update(
                            backtest_task, 
                            completed=completed,
                            description=f"🚀 Backtests: {completed}/{len(valid_pairs)} (✅{successful_count} ❌{failed_count}) - Last: {coin}"
                        )
                        
                    except Exception as exc:
                        coin = future_to_coin[future]
                        failed_count += 1
                        completed = successful_count + failed_count
                        
                        progress.console.print(f"[red]Exception for {coin}: {exc}[/red]")
                        results.append({"coin": coin, "asset": f"{coin}", "error": str(exc), "status": "failed"})
                        
                        progress.update(
                            backtest_task, 
                            completed=completed,
                            description=f"🚀 Backtests: {completed}/{len(valid_pairs)} (✅{successful_count} ❌{failed_count}) - Last: {coin} ❌"
                        )
            
            progress.update(backtest_task, completed=len(valid_pairs), description="🚀 All backtests complete")
    
    if save_results:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\n[green]Results saved to '{output_file}'[/green]")
    
    console.print(f"\n[bold green]🎉 Backtesting completed![/bold green]")
    return print_detailed_results(results)

def print_detailed_results(results):
    """Print detailed backtest results using only actual tracked values (portvalue and pnl)."""
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    skipped = [r for r in results if r["status"] == "skipped"]
    
    print(f"\n=== DETAILED RESULTS ===")
    
    if successful:
        print(f"\n🟢 SUCCESSFUL BACKTESTS ({len(successful)}):")
        print("-" * 60)
        print(f"{'Asset':<12} {'Final Value':<15} {'P&L':<15}")
        print("-" * 60)

        successful_sorted = sorted(successful, key=lambda x: x.get('pnl', 0), reverse=True)

        for r in successful_sorted:
            asset = r.get('asset', r.get('coin', 'N/A'))
            final_value = r.get('final_value')
            pnl = r.get('pnl')

            final_value_str = f"${final_value:,.2f}" if isinstance(final_value, (int, float)) else str(final_value)
            pnl_str = f"${pnl:,.2f}" if isinstance(pnl, (int, float)) else str(pnl)

            print(f"{asset:<12} {final_value_str:<15} {pnl_str:<15}")
    
    if failed:
        print(f"\n🔴 FAILED BACKTESTS ({len(failed)}):")
        for r in failed:
            print(f"  {r.get('asset', r.get('coin', 'N/A'))}: {r.get('error', 'Unknown error')}")
    
    if skipped:
        print(f"\n⚪ SKIPPED (No Data) ({len(skipped)}):")
        skipped_assets = [r.get('asset', r.get('coin', 'N/A')) for r in skipped]
        print(f"  {', '.join(skipped_assets)}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped (no data): {len(skipped)}")