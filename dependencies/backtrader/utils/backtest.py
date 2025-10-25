import backtrader as bt
from collections.abc import Iterable
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import gc
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

INIT_CASH = 100_000.0
COMMISSION_PER_TRANSACTION = 0.00075

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

def fetch_single_data(coin, start_date, end_date, interval, collateral="USDT"):
    """Fetch data for a single coin"""
    from backtrader.feeds.mssql_crypto import get_database_data
    
    try:
        data = get_database_data(coin, start_date, end_date, interval)
        data._dataname = f"{coin}{collateral}"
        return data
    except Exception as e:
        console.print(f"[red]Error fetching data for {coin}: {str(e)}[/red]")
        return None

def optimize_backtest(strategy, data=None, coin=None,
    start_date=None,
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=COMMISSION_PER_TRANSACTION,
    init_cash=INIT_CASH,
    asset_name=None,
    bulk=False,
    show_progress=True,
    quantstats=False,
    **kwargs):
    from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomData
    import pandas as pd
    
    if data is None:
        print(interval, coin)
        if not all([coin, start_date, end_date, interval]):
            raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")
        
        if show_progress and not bulk:
            console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")
        
        data = fetch_single_data(coin, start_date, end_date, interval, collateral)
        if data is None:
            raise ValueError(f"Failed to fetch data for {coin}")
        
        if show_progress and not bulk:
            console.print(f"✅ [bold green]Data fetched for {coin}[/bold green]")
    
    quantstats_option = kwargs.pop('quantstats', quantstats)

    if isinstance(quantstats_option, Iterable) and not isinstance(quantstats_option, (str, bytes)):
        quantstats_values = [bool(value) for value in quantstats_option]
    else:
        quantstats_values = [bool(quantstats_option)]

    quantstats_enabled = quantstats_values[0] if quantstats_values else False

    default_opt_params = {
        'breakout_period': [20, 40, 55],
        'adxth': [20, 25, 30],
        'vol_mult': [1.3, 1.6, 2.0],
        'init_sl_atr_mult': [1.0, 1.25, 1.5],
        'trail_atr_mult': [2.5, 3.0, 3.5],
        'dca_atr_mult': [0.8, 1.0, 1.2],
        'max_adds': [2, 3, 4],
    }

    default_opt_params.update(kwargs)

    opt_params = {}
    for key, value in default_opt_params.items():
        if isinstance(value, Iterable) and not isinstance(value, str):
            opt_params[key] = value
        else:
            opt_params[key] = [value]

    opt_params.setdefault('backtest', [True])
    opt_params['quantstats'] = quantstats_values

    from backtrader.feed import DataBase
    if isinstance(data, DataBase):
        data_feed = data
    else:
        data_feed = CustomData(dataname=data)

    cerebro = bt.Cerebro(oldbuysell=True)
    cerebro.adddata(data_feed)
    # cerebro.addstrategy(strategy, backtest=True)
    cerebro.optstrategy(
        strategy,
        **opt_params,
    )
    
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
    
    if asset_name:
        display_name = asset_name
    elif coin:
        display_name = f"{coin}/{collateral}"
    else:
        display_name = "Asset"
    
    if show_progress and not bulk:
        console.print(f"🔄 [bold blue]Starting backtest for {display_name}...[/bold blue]")
        
        with console.status(f"[bold green]Running backtest for {display_name}...") as status:
            strategy_result = cerebro.run()[0]
            
        console.print(f"✅ [bold green]Completed backtest for {display_name}[/bold green]")
    else:
        strategy_result = cerebro.run()[0]
    
    max_drawdown = strategy_result.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy_result.analyzers.trade_analyzer.get_analysis()
    pyfolio_analyzer = strategy_result.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    returns = pd.Series(returns)
    returns = returns.dropna()
    
    if quantstats_enabled:
        import quantstats_lumi as quantstats
        from datetime import datetime
        import os
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        
        if asset_name:
            coin_name = asset_name.replace('/', '_')
        elif coin:
            coin_name = f"{coin}_{collateral}"
        elif hasattr(data_feed, '_dataname'):
            coin_name = str(data_feed._dataname)
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
    
    if not bulk:
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        
        pnl_net = trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS - {display_name}")
        print(f"{'='*50}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades}")
        print(f"Losing Trades: {lost_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Net P&L: ${pnl_net:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - init_cash
        print(f"Final Portfolio Value: ${portvalue:.2f}")
        print(f"Total P/L: ${pnl:.2f}")
        print(f"Return: {(pnl / init_cash * 100):.2f}%")
        print(f"{'='*50}")
    
    return cerebro.broker.getvalue()

def backtest(
    strategy,
    data=None,
    coin=None,
    start_date=None,
    end_date="2030-12-31",
    interval=None,
    collateral="USDT",
    commission=COMMISSION_PER_TRANSACTION,
    init_cash=INIT_CASH,
    plot=True,
    quantstats=False,
    asset_name=None,
    bulk=False,
    show_progress=True,
    **kwargs,
):

    from backtrader.analyzers import TimeReturn, SharpeRatio, DrawDown, TradeAnalyzer
    from backtrader.strategies.base import CustomSQN, CustomData
    import pandas as pd
    
    if data is None:
        print(interval, coin)
        if not all([coin, start_date, end_date, interval]):
            raise ValueError("If data is not provided, coin, start_date, end_date, and interval are required")
        
        if show_progress and not bulk:
            console.print(f"🔄 [bold blue]Fetching data for {coin}...[/bold blue]")
        
        data = fetch_single_data(coin, start_date, end_date, interval, collateral)
        if data is None:
            raise ValueError(f"Failed to fetch data for {coin}")
        
        if show_progress and not bulk:
            console.print(f"✅ [bold green]Data fetched for {coin}[/bold green]")
    
    kwargs = {
        k: v if isinstance(v, Iterable) and not isinstance(v, str) else [v]
        for k, v in kwargs.items()
    }
    
    from backtrader.feed import DataBase
    if isinstance(data, DataBase):
        data_feed = data
    else:
        data_feed = CustomData(dataname=data)
    
    cerebro = bt.Cerebro(oldbuysell=True)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(strategy, backtest=True)
    
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
    
    if asset_name:
        display_name = asset_name
    elif coin:
        display_name = f"{coin}/{collateral}"
    else:
        display_name = "Asset"
    
    if show_progress and not bulk:
        console.print(f"🔄 [bold blue]Starting backtest for {display_name}...[/bold blue]")
        
        with console.status(f"[bold green]Running backtest for {display_name}...") as status:
            strategy_result = cerebro.run()[0]
            
        console.print(f"✅ [bold green]Completed backtest for {display_name}[/bold green]")
    else:
        strategy_result = cerebro.run()[0]
    
    max_drawdown = strategy_result.analyzers.drawdown.get_analysis()['max']['drawdown']
    trade_analyzer = strategy_result.analyzers.trade_analyzer.get_analysis()
    pyfolio_analyzer = strategy_result.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyfolio_analyzer.get_pf_items()
    
    returns = pd.Series(returns)
    returns = returns.dropna()
    
    if quantstats:
        import quantstats_lumi as quantstats
        from datetime import datetime
        import os
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H-%M-%S")
        
        if asset_name:
            coin_name = asset_name.replace('/', '_')
        elif coin:
            coin_name = f"{coin}_{collateral}"
        elif hasattr(data_feed, '_dataname'):
            coin_name = str(data_feed._dataname)
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
    
    if not bulk:
        total_trades = trade_analyzer.get('total', {}).get('total', 0)
        won_trades = trade_analyzer.get('won', {}).get('total', 0)
        lost_trades = trade_analyzer.get('lost', {}).get('total', 0)
        
        pnl_net = trade_analyzer.get('pnl', {}).get('net', {}).get('total', 0)
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"BACKTEST RESULTS - {display_name}")
        print(f"{'='*50}")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {won_trades}")
        print(f"Losing Trades: {lost_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Net P&L: ${pnl_net:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        portvalue = cerebro.broker.getvalue()
        pnl = portvalue - init_cash
        print(f"Final Portfolio Value: ${portvalue:.2f}")
        print(f"Total P/L: ${pnl:.2f}")
        print(f"Return: {(pnl / init_cash * 100):.2f}%")
        print(f"{'='*50}")
    
    if plot:
        cerebro.plot(style='candles', numfigs=1, volume=True, barup='black', bardown='grey')
    
    return cerebro.broker.getvalue()

def run_backtest_with_data(args):
    """Helper function for parallel execution in bulk_backtest"""
    coin, data, strategy_class, init_cash, backtest_params, collateral = args

    if data is None:
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": "No data available", "status": "skipped"}

    try:
        asset = f'{coin}/{collateral}'
        
        result = backtest(
            strategy_class,
            data=data,
            init_cash=init_cash,
            asset_name=asset,
            bulk=True,
            **backtest_params
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
        return {"coin": coin, "asset": f"{coin}/{collateral}", "error": str(e), "status": "failed"}

def bulk_backtest(strategy, coins, start_date, end_date, interval, 
                 collateral="USDT", init_cash=1000, max_workers=8, save_results=True, 
                 output_file='backtest_results.json', **backtest_kwargs):
    """
    Enhanced bulk backtest with unified progress tracking
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    default_backtest_params = {
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
        
        for coin, data in data_cache.items():
            if data is not None:
                data._dataname = f"{coin}{collateral}"
        
        progress.update(fetch_task, completed=len(coins), description="📊 Data fetching complete")
        
        valid_pairs = [(coin, data) for coin, data in data_cache.items() if data is not None]
        skipped_coins = [coin for coin, data in data_cache.items() if data is None]
        
        progress.console.print(f"\n[green]✓[/green] Found data for [bold]{len(valid_pairs)}[/bold] coins, skipping [bold]{len(skipped_coins)}[/bold] coins")
        if skipped_coins:
            progress.console.print(f"[yellow]Skipped coins:[/yellow] {', '.join(skipped_coins)}")
        
        results = []
        
        for coin in skipped_coins:
            results.append({"coin": coin, "asset": f"{coin}/{collateral}", "error": "No data available", "status": "skipped"})
        
        if valid_pairs:
            backtest_task = progress.add_task("🚀 Running backtests...", total=len(valid_pairs))
            
            successful_count = 0
            failed_count = 0
            
            backtest_args = [
                (coin, data, strategy, init_cash, default_backtest_params, collateral)
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
                        results.append({"coin": coin, "asset": f"{coin}/{collateral}", "error": str(exc), "status": "failed"})
                        
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
    """Print detailed backtest results"""
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
    
    return results