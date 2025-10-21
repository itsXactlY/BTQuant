# btq_cli.py - Universal BTQuant Command Line Interface

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================================
# ARGUMENT PARSER - Universal for all BTQuant operations
# ============================================================================

def create_parser():
    """Create comprehensive argument parser for BTQuant"""
    parser = argparse.ArgumentParser(
        prog='btq',
        description='BTQuant - Professional Backtesting & Optimization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single backtest
  btq backtest --coin BTC --interval 15m --plot
  
  # Multiple coins backtest
  btq backtest --coins BTC,ETH,BNB --interval 1h --start 2024-01-01
  
  # Bulk backtest (auto-discover all coins)
  btq bulk --interval 1h --workers 8
  
  # Optimize single coin
  btq optimize --coin BTC --trials 200 --workers 6 --aggressive
  
  # Optimize multiple coins (creates separate studies)
  btq optimize --coins BTC,ETH,DOGE --trials 100 --conservative
  
  # Multi-strategy optimization
  btq optimize --strategy MyStra1,MyStrat2 --coin BTC --trials 150
  
  # List available strategies/coins
  btq list strategies
  btq list coins --collateral USDT
        """
    )
    
    # === MODE (Required) ===
    parser.add_argument(
        'mode',
        choices=['backtest', 'bulk', 'optimize', 'list', 'live'],
        help='Operation mode'
    )
    
    # === COMMON ARGUMENTS ===
    common = parser.add_argument_group('Common Arguments')
    
    common.add_argument('--coin', '--symbol', type=str,
                       help='Single coin/symbol (e.g., BTC)')
    
    common.add_argument('--coins', '--symbols', type=str,
                       help='Comma-separated coins (e.g., BTC,ETH,BNB). For bulk mode, omit for all coins')
    
    common.add_argument('--strategy', '--strat', type=str,
                       help='Strategy class name or comma-separated list')
    
    common.add_argument('--collateral', '--pair', type=str, default='USDT',
                       help='Collateral/pair (default: USDT)')
    
    common.add_argument('--interval', '--timeframe', '--tf', type=str, default='15m',
                       help='Timeframe: 1m, 5m, 15m, 1h, 4h, 1d (default: 15m)')
    
    common.add_argument('--start', '--start-date', '--from', type=str,
                       help='Start date YYYY-MM-DD (default: earliest available)')
    
    common.add_argument('--end', '--end-date', '--to', type=str, default='2025-01-01',
                       help='End date YYYY-MM-DD (default: 2025-01-01)')
    
    # === CAPITAL & RISK ===
    capital = parser.add_argument_group('Capital & Risk')
    
    capital.add_argument('--cash', '--capital', '--init-cash', type=float, default=1000,
                        help='Initial capital (default: 1000)')
    
    capital.add_argument('--commission', '--comm', type=float, default=0.00075,
                        help='Commission rate (default: 0.00075 = 0.075%%)')
    
    capital.add_argument('--leverage', '--lev', type=int, default=1,
                        help='Leverage multiplier (default: 1 = no leverage)')
    
    capital.add_argument('--slippage', '--slip', type=float, default=5.0,
                        help='Slippage in basis points (default: 5.0)')
    
    # === OUTPUT & VISUALIZATION ===
    output = parser.add_argument_group('Output & Visualization')
    
    output.add_argument('--plot', '-p', action='store_true',
                       help='Show plot after backtest/optimization')
    
    output.add_argument('--quantstats', '--qs', '-q', action='store_true',
                       help='Generate QuantStats report')
    
    output.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug output')
    
    output.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    output.add_argument('--save', '--save-results', action='store_true',
                       help='Save results to file')
    
    output.add_argument('--output', '--output-file', '-o', type=str,
                       help='Output filename (auto-generated if not provided)')
    
    # === BULK OPTIONS ===
    bulk_opts = parser.add_argument_group('Bulk Options')
    
    bulk_opts.add_argument('--workers', '--jobs', '--max-workers', '-j', type=int, default=8,
                          help='Number of parallel workers (default: 8)')
    
    # === OPTIMIZATION OPTIONS ===
    opt = parser.add_argument_group('Optimization Options')
    
    opt.add_argument('--trials', '--n-trials', '-n', type=int, default=200,
                    help='Number of optimization trials (default: 200)')
    
    opt.add_argument('--opt-workers', '--n-jobs', type=int,
                    help='Parallel optimization workers (default: same as --workers)')
    
    opt.add_argument('--study-name', '--study', type=str,
                    help='Custom Optuna study name (auto-generated if not provided)')
    
    opt.add_argument('--aggressive', '--agg', action='store_true',
                    help='Use aggressive parameter space (more trades, higher risk tolerance)')
    
    opt.add_argument('--conservative', '--cons', action='store_true',
                    help='Use conservative parameter space (tighter DD control)')
    
    opt.add_argument('--multi-period', '--multi', action='store_true',
                    help='Run multi-period validation')
    
    opt.add_argument('--min-trades', type=int, default=30,
                    help='Minimum trades required for valid optimization (default: 30)')
    
    opt.add_argument('--pruner', choices=['hyperband', 'median', 'none'], default='hyperband',
                    help='Optuna pruner algorithm (default: hyperband)')
    
    opt.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
    
    # === STRATEGY PARAMETERS ===
    strat = parser.add_argument_group('Strategy Parameters (pass-through)')
    
    strat.add_argument('--params', type=str,
                      help='Strategy params as JSON string or key=value pairs')
    
    strat.add_argument('--take-profit', '--tp', type=float,
                      help='Take profit percentage')
    
    strat.add_argument('--stop-loss', '--sl', type=float,
                      help='Stop loss percentage')
    
    strat.add_argument('--position-size', '--size', type=float,
                      help='Position size as percent of capital')
    
    # === EXCHANGE & CACHE ===
    misc = parser.add_argument_group('Miscellaneous')
    
    misc.add_argument('--exchange', '--ex', type=str,
                     help='Exchange name (e.g., MEXC for shorting support)')
    
    misc.add_argument('--no-cache', action='store_true',
                     help='Disable data caching')
    
    misc.add_argument('--clear-cache', action='store_true',
                     help='Clear cache before running')
    
    misc.add_argument('--list-strategies', action='store_true',
                     help='List all available strategies')
    
    return parser


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_coins(coins_str: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated coins string"""
    if not coins_str:
        return None
    return [c.strip().upper() for c in coins_str.split(',')]


def parse_strategies(strat_str: Optional[str]) -> List[str]:
    """Parse comma-separated strategy string"""
    if not strat_str:
        return None
    return [s.strip() for s in strat_str.split(',')]


def parse_params(params_str: Optional[str]) -> dict:
    """Parse parameter string (JSON or key=value)"""
    if not params_str:
        return {}
    
    # Try JSON first
    try:
        import json
        return json.loads(params_str)
    except:
        pass
    
    # Try key=value format
    params = {}
    for pair in params_str.split(','):
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except:
                if value.lower() == 'true':
                    params[key] = True
                elif value.lower() == 'false':
                    params[key] = False
                else:
                    params[key] = value
    
    return params


def import_strategy(strategy_name: str):
    """Dynamically import strategy class"""
    try:
        # Try direct import from backtrader.strategies
        module = __import__(f'backtrader.strategies.{strategy_name}', fromlist=[strategy_name])
        return getattr(module, strategy_name)
    except:
        pass
    
    # Try from current directory
    try:
        module = __import__(strategy_name, fromlist=[strategy_name])
        return getattr(module, strategy_name)
    except:
        pass
    
    raise ImportError(f"Could not import strategy: {strategy_name}")


def list_available_strategies():
    """List all available strategies"""
    import pkgutil
    import backtrader.strategies
    
    strategies = []
    for importer, modname, ispkg in pkgutil.iter_modules(backtrader.strategies.__path__):
        try:
            module = __import__(f'backtrader.strategies.{modname}', fromlist=[modname])
            for item in dir(module):
                obj = getattr(module, item)
                if isinstance(obj, type) and hasattr(obj, 'params'):
                    strategies.append(item)
        except:
            pass
    
    return sorted(set(strategies))


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_backtest_mode(args):
    """Run single or multi-coin backtest"""
    from backtrader.utils.backtest import backtest
    
    # Parse coins
    coins = parse_coins(args.coins) if args.coins else ([args.coin] if args.coin else None)
    if not coins:
        console.print("[red]Error: --coin or --coins required for backtest mode[/red]")
        sys.exit(1)
    
    # Parse strategies
    strategies = parse_strategies(args.strategy) if args.strategy else None
    if not strategies:
        console.print("[red]Error: --strategy required[/red]")
        console.print("[yellow]Tip: Use 'btq list strategies' to see available strategies[/yellow]")
        sys.exit(1)
    
    # Parse params
    params = parse_params(args.params) if args.params else {}
    
    # Add CLI params to strategy params
    if args.take_profit:
        params['take_profit'] = args.take_profit
    if args.stop_loss:
        params['stop_loss'] = args.stop_loss
    if args.position_size:
        params['percent_sizer'] = args.position_size
    
    results = []
    
    for strategy_name in strategies:
        strategy_class = import_strategy(strategy_name)
        
        for coin in coins:
            console.print(f"\n{'='*60}")
            console.print(f"🚀 Backtesting: {strategy_name} on {coin}/{args.collateral}")
            console.print(f"{'='*60}\n")
            
            try:
                result = backtest(
                    strategy_class,
                    coin=coin,
                    collateral=args.collateral,
                    start_date=args.start,
                    end_date=args.end,
                    interval=args.interval,
                    init_cash=args.cash,
                    commission=args.commission,
                    plot=args.plot,
                    quantstats=args.quantstats,
                    debug=args.debug,
                    exchange=args.exchange,
                    slippage_bps=args.slippage,
                    params=params,
                )
                
                results.append({
                    'strategy': strategy_name,
                    'coin': coin,
                    'result': result,
                    'status': 'success'
                })
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                results.append({
                    'strategy': strategy_name,
                    'coin': coin,
                    'error': str(e),
                    'status': 'failed'
                })
    
    return results


def run_bulk_mode(args):
    """Run bulk backtest"""
    from backtrader.utils.backtest import bulk_backtest
    
    # Parse coins (None = all coins)
    coins = parse_coins(args.coins) if args.coins else None
    
    # Parse strategy
    strategy_name = args.strategy
    if not strategy_name:
        console.print("[red]Error: --strategy required[/red]")
        sys.exit(1)
    
    strategy_class = import_strategy(strategy_name)
    
    # Parse params
    params = parse_params(args.params) if args.params else {}
    
    console.print(f"\n{'='*60}")
    console.print(f"🚀 Bulk Backtest: {strategy_name}")
    if coins:
        console.print(f"Coins: {', '.join(coins)}")
    else:
        console.print("Coins: ALL (auto-discovery)")
    console.print(f"{'='*60}\n")
    
    results = bulk_backtest(
        strategy_class,
        coins=coins,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        collateral=args.collateral,
        init_cash=args.cash,
        max_workers=args.workers,
        save_results=args.save,
        output_file=args.output,
        params=params,
        commission=args.commission,
    )
    
    return results


def run_optimize_mode(args):
    """Run optimization"""
    from backtrader.utils.optimize import optimize, OptimizationConfig
    
    # Parse coins
    coins = parse_coins(args.coins) if args.coins else ([args.coin] if args.coin else None)
    if not coins:
        console.print("[red]Error: --coin or --coins required for optimize mode[/red]")
        sys.exit(1)
    
    # Parse strategy
    strategy_name = args.strategy
    if not strategy_name:
        console.print("[red]Error: --strategy required[/red]")
        sys.exit(1)
    
    strategy_class = import_strategy(strategy_name)
    
    # Determine parameter space
    param_space_mode = "aggressive" if args.aggressive else ("conservative" if args.conservative else "default")
    
    # Try to get param space function from strategy module
    try:
        module = __import__(f'backtrader.strategies.{strategy_name}', fromlist=[strategy_name])
        if args.aggressive and hasattr(module, 'param_space_aggressive'):
            param_space_fn = module.param_space_aggressive
        elif args.conservative and hasattr(module, 'param_space_conservative'):
            param_space_fn = module.param_space_conservative
        else:
            param_space_fn = None
    except:
        param_space_fn = None
    
    # Workers
    opt_workers = args.opt_workers if args.opt_workers else args.workers
    
    results = {}
    
    for coin in coins:
        # Generate study name
        if args.study_name:
            study_name = args.study_name
            if len(coins) > 1:
                study_name = f"{args.study_name}_{coin}"
        else:
            study_name = f"{strategy_name}_{param_space_mode}_{coin}_{args.interval}"
        
        console.print(f"\n{'='*60}")
        console.print(f"🔥 Optimizing: {strategy_name} on {coin}")
        console.print(f"Mode: {param_space_mode.upper()}")
        console.print(f"Study: {study_name}")
        console.print(f"{'='*60}\n")
        
        try:
            config = OptimizationConfig(
                strategy_class=strategy_class,
                coin=coin,
                interval=args.interval,
                start_date=args.start or "2023-01-01",
                end_date=args.end,
                collateral=args.collateral,
                init_cash=args.cash,
                commission=args.commission,
                exchange=args.exchange,
                n_trials=args.trials,
                n_jobs=opt_workers,
                study_name=study_name,
                pruner=None if args.pruner == 'none' else args.pruner,
                seed=args.seed,
                min_trades=args.min_trades,
                plot_best=args.plot,
                quantstats_best=args.quantstats,
            )
            
            study = optimize(config, param_space_fn=param_space_fn)
            results[coin] = study
            
        except Exception as e:
            console.print(f"[red]Optimization failed for {coin}: {e}[/red]")
            import traceback
            if args.debug:
                traceback.print_exc()
    
    return results


def run_list_mode(args):
    """List available resources"""
    if args.mode == 'list' and len(sys.argv) > 2:
        what = sys.argv[2]
        
        if what == 'strategies':
            strategies = list_available_strategies()
            
            table = Table(title="📚 Available Strategies")
            table.add_column("#", style="cyan")
            table.add_column("Strategy Name", style="green")
            
            for i, strat in enumerate(strategies, 1):
                table.add_row(str(i), strat)
            
            console.print(table)
            console.print(f"\n[cyan]Total: {len(strategies)} strategies[/cyan]")
            
        elif what == 'coins':
            from backtrader.utils.backtest import get_available_coins_from_database
            coins = get_available_coins_from_database(collateral=args.collateral)
            
            table = Table(title=f"💰 Available Coins ({args.collateral})")
            table.add_column("#", style="cyan")
            table.add_column("Coin", style="green")
            
            for i, coin in enumerate(coins, 1):
                table.add_row(str(i), coin)
            
            console.print(table)
            console.print(f"\n[cyan]Total: {len(coins)} coins[/cyan]")
            
        else:
            console.print(f"[red]Unknown list target: {what}[/red]")
            console.print("Usage: btq list [strategies|coins]")
    else:
        console.print("Usage: btq list [strategies|coins]")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main CLI entry point"""
    parser = create_parser()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        from backtrader.utils.backtest import CACHE_DIR
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            console.print(f"[green]Cache cleared: {CACHE_DIR}[/green]")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.mode == 'backtest':
            return run_backtest_mode(args)
        
        elif args.mode == 'bulk':
            return run_bulk_mode(args)
        
        elif args.mode == 'optimize':
            return run_optimize_mode(args)
        
        elif args.mode == 'list':
            return run_list_mode(args)
        
        elif args.mode == 'live':
            console.print("[red]❌ Live trading not yet implemented[/red]")
            sys.exit(1)
    
    except KeyboardInterrupt:
        console.print("\n\n⚠️ Operation cancelled by user")
        sys.exit(0)
    
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()