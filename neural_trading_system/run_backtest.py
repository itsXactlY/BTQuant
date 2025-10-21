import backtrader as bt
from strategy import NeuralTradingStrategy
import pandas as pd

def run_neural_backtest(
    data_path: str,
    model_path: str,
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    initial_cash: float = 10000,
    commission: float = 0.001,
    plot: bool = True
):
    """
    Run backtest using the neural trading strategy.
    
    Args:
        data_path: Path to OHLCV data with indicator internals
        model_path: Path to trained model checkpoint
        start_date: Backtest start date
        end_date: Backtest end date
        initial_cash: Starting capital
        commission: Trading commission (0.001 = 0.1%)
        plot: Whether to plot results
    """
    print("🚀 Starting Neural Strategy Backtest")
    print(f"Model: {model_path}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    
    # Load data
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    # Create Backtrader data feed
    data = bt.feeds.PandasData(
        dataname=df,
        datetime='timestamp',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    
    cerebro.adddata(data)
    
    # Add strategy
    cerebro.addstrategy(
        NeuralTradingStrategy,
        model_path=model_path,
        seq_len=100,
        confidence_threshold=0.6,
        position_size_mode='neural',
        debug=True
    )
    
    # Set broker parameters
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run backtest
    print("\n📊 Running backtest...")
    results = cerebro.run()
    strat = results[0]
    
    # Extract results
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    # Print results
    print("\n" + "="*60)
    print("🎯 NEURAL STRATEGY BACKTEST RESULTS")
    print("="*60)
    print(f"Final Portfolio Value: ${cerebro.broker.getvalue():,.2f}")
    print(f"Total Return: {returns.get('rtot', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    print(f"\nTrades:")
    print(f"  Total: {trades.get('total', {}).get('total', 0)}")
    print(f"  Won: {trades.get('won', {}).get('total', 0)}")
    print(f"  Lost: {trades.get('lost', {}).get('total', 0)}")
    
    if trades.get('total', {}).get('total', 0) > 0:
        win_rate = trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1) * 100
        print(f"  Win Rate: {win_rate:.2f}%")
    
    print("="*60)
    
    # Plot if requested
    if plot:
        cerebro.plot(style='candlestick', barup='green', bardown='red')
    
    return cerebro, results

if __name__ == '__main__':
    run_neural_backtest(
        data_path='data/btc_4h_indicators.csv',
        model_path='best_model.pt',
        start_date='2024-01-01',
        end_date='2024-12-31',
        initial_cash=10000,
        commission=0.001,
        plot=True
    )