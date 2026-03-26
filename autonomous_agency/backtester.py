"""
Automated Backtesting Engine for the Autonomous Quantitative Research Agency

This module provides comprehensive backtesting capabilities for generated strategies,
including multi-asset testing, walk-forward analysis, and performance evaluation.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import backtrader as bt
import quantstats_lumi as qs
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import config
from .strategy_factory import GeneratedStrategy
from .models import BacktestResult, WalkForwardResult


class AutomatedBacktester:
    """Automated backtesting engine with advanced analysis capabilities"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_dir = Path(config.backtest_results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize benchmark data
        self.benchmark_data = self._load_benchmark_data()

    def run_backtest(self, strategy: GeneratedStrategy, data_config: Dict[str, Any]) -> Optional[BacktestResult]:
        """
        Run a comprehensive backtest for a generated strategy

        Args:
            strategy: The generated strategy to backtest
            data_config: Configuration for data feeds, periods, etc.

        Returns:
            BacktestResult object or None if backtest failed
        """
        try:
            # Create backtrader cerebro instance
            cerebro = self._setup_cerebro(strategy, data_config)

            # Add analyzers for comprehensive metrics
            self._add_analyzers(cerebro)

            # Run the backtest
            self.logger.info(f"Running backtest for strategy: {strategy.strategy_name}")
            results = cerebro.run()

            # Extract and process results
            backtest_result = self._process_results(strategy, results, data_config)

            if backtest_result:
                # Save detailed results
                self._save_backtest_results(backtest_result)

            return backtest_result

        except Exception as e:
            self.logger.error(f"Backtest failed for {strategy.strategy_name}: {e}")
            return None

    def run_walk_forward_analysis(self, strategy: GeneratedStrategy,
                                data_config: Dict[str, Any],
                                window_size: int = 252,
                                step_size: int = 21) -> Optional[WalkForwardResult]:
        """
        Perform walk-forward analysis to detect overfitting

        Args:
            strategy: Strategy to analyze
            data_config: Data configuration
            window_size: Size of in-sample window (trading days)
            step_size: Step size for moving window (trading days)

        Returns:
            WalkForwardResult or None if analysis failed
        """
        try:
            in_sample_results = []
            out_of_sample_results = []

            # Get total data range
            start_date = data_config.get('start_date')
            end_date = data_config.get('end_date')

            if not start_date or not end_date:
                self.logger.error("Start and end dates required for walk-forward analysis")
                return None

            current_date = start_date

            while current_date + timedelta(days=window_size*2) <= end_date:
                # Define in-sample period
                is_end = current_date + timedelta(days=window_size)
                oos_end = min(is_end + timedelta(days=window_size), end_date)

                # In-sample backtest
                is_config = data_config.copy()
                is_config['start_date'] = current_date
                is_config['end_date'] = is_end

                is_result = self.run_backtest(strategy, is_config)
                if is_result:
                    in_sample_results.append(is_result)

                # Out-of-sample backtest
                oos_config = data_config.copy()
                oos_config['start_date'] = is_end
                oos_config['end_date'] = oos_end

                oos_result = self.run_backtest(strategy, oos_config)
                if oos_result:
                    out_of_sample_results.append(oos_result)

                # Move window
                current_date += timedelta(days=step_size)

            if not in_sample_results or not out_of_sample_results:
                return None

            # Calculate walk-forward metrics
            wf_result = self._calculate_walk_forward_metrics(
                strategy.strategy_name, in_sample_results, out_of_sample_results
            )

            return wf_result

        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed for {strategy.strategy_name}: {e}")
            return None

    def run_multi_asset_backtest(self, strategy: GeneratedStrategy,
                               assets: List[str],
                               data_config: Dict[str, Any]) -> Dict[str, BacktestResult]:
        """
        Run backtest across multiple assets

        Args:
            strategy: Strategy to test
            assets: List of asset symbols
            data_config: Base data configuration

        Returns:
            Dictionary of asset -> BacktestResult
        """
        results = {}

        for asset in assets:
            try:
                asset_config = data_config.copy()
                asset_config['symbol'] = asset

                result = self.run_backtest(strategy, asset_config)
                if result:
                    results[asset] = result

            except Exception as e:
                self.logger.error(f"Multi-asset backtest failed for {asset}: {e}")

        return results

    def run_regime_analysis(self, strategy: GeneratedStrategy,
                          data_config: Dict[str, Any]) -> Dict[str, BacktestResult]:
        """
        Analyze strategy performance across different market regimes

        Args:
            strategy: Strategy to analyze
            data_config: Data configuration

        Returns:
            Dictionary of regime -> BacktestResult
        """
        regimes = {
            'bull': {'volatility': 'low', 'trend': 'up'},
            'bear': {'volatility': 'high', 'trend': 'down'},
            'sideways': {'volatility': 'low', 'trend': 'flat'},
            'volatile': {'volatility': 'high', 'trend': 'mixed'}
        }

        results = {}

        for regime_name, regime_config in regimes.items():
            try:
                regime_data_config = data_config.copy()
                regime_data_config['regime_filter'] = regime_config

                result = self.run_backtest(strategy, regime_data_config)
                if result:
                    results[regime_name] = result

            except Exception as e:
                self.logger.error(f"Regime analysis failed for {regime_name}: {e}")

        return results

    def _setup_cerebro(self, strategy: GeneratedStrategy, data_config: Dict[str, Any]) -> bt.Cerebro:
        """Set up backtrader cerebro instance"""

        cerebro = bt.Cerebro()

        # Set initial cash
        cerebro.broker.setcash(data_config.get('initial_cash', 100000))

        # Set commission
        commission = data_config.get('commission', 0.001)
        cerebro.broker.setcommission(commission=commission)

        # Load and add data feeds
        data_feeds = self._load_data_feeds(data_config)
        for data_feed in data_feeds:
            cerebro.adddata(data_feed)

        # Load and add strategy
        strategy_class = self._load_strategy_class(strategy)
        if strategy_class:
            cerebro.addstrategy(strategy_class, **strategy.parameters)
        else:
            raise ValueError(f"Could not load strategy class for {strategy.strategy_name}")

        return cerebro

    def _load_data_feeds(self, data_config: Dict[str, Any]) -> List[bt.DataBase]:
        """Load data feeds based on configuration"""

        feeds = []

        # Support multiple data sources
        data_sources = data_config.get('data_sources', ['csv'])

        for source in data_sources:
            if source == 'csv':
                feed = self._load_csv_data(data_config)
            elif source == 'mssql':
                feed = self._load_mssql_data(data_config)
            elif source == 'ccxt':
                feed = self._load_ccxt_data(data_config)
            else:
                self.logger.warning(f"Unsupported data source: {source}")
                continue

            if feed:
                feeds.append(feed)

        return feeds

    def _load_csv_data(self, data_config: Dict[str, Any]) -> Optional[bt.DataBase]:
        """Load CSV data feed"""

        csv_path = data_config.get('csv_path')
        if not csv_path or not Path(csv_path).exists():
            return None

        return bt.feeds.YahooFinanceCSVData(
            dataname=csv_path,
            fromdate=data_config.get('start_date'),
            todate=data_config.get('end_date')
        )

    def _load_mssql_data(self, data_config: Dict[str, Any]) -> Optional[bt.DataBase]:
        """Load MSSQL data feed"""

        # Use existing BTQuant MSSQL integration
        try:
            from backtrader.feeds import MSSQLData

            return MSSQLData(
                dbhost=data_config.get('db_host'),
                dbname=data_config.get('db_name'),
                symbol=data_config.get('symbol'),
                fromdate=data_config.get('start_date'),
                todate=data_config.get('end_date')
            )
        except ImportError:
            self.logger.error("MSSQL data feed not available")
            return None

    def _load_ccxt_data(self, data_config: Dict[str, Any]) -> Optional[bt.DataBase]:
        """Load CCXT data feed"""

        try:
            from backtrader.feeds import CCXTData

            return CCXTData(
                exchange=data_config.get('exchange', 'binance'),
                symbol=data_config.get('symbol'),
                timeframe=data_config.get('timeframe', bt.TimeFrame.Minutes),
                fromdate=data_config.get('start_date'),
                todate=data_config.get('end_date')
            )
        except ImportError:
            self.logger.error("CCXT data feed not available")
            return None

    def _load_strategy_class(self, strategy: GeneratedStrategy):
        """Dynamically load strategy class from generated code"""

        try:
            # Import the generated strategy module
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                strategy.strategy_name, strategy.code_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the strategy class
            strategy_class = getattr(module, strategy.class_name)
            return strategy_class

        except Exception as e:
            self.logger.error(f"Failed to load strategy class: {e}")
            return None

    def _add_analyzers(self, cerebro: bt.Cerebro):
        """Add comprehensive analyzers to cerebro"""

        # Performance analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        # Risk analyzers
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')

        # Custom analyzers if available
        try:
            from backtrader.analyzers import QuantStatsAnalyzer
            cerebro.addanalyzer(QuantStatsAnalyzer, _name='quantstats')
        except ImportError:
            pass

    def _process_results(self, strategy: GeneratedStrategy,
                        results: List, data_config: Dict[str, Any]) -> BacktestResult:
        """Process backtrader results into BacktestResult object"""

        try:
            result = results[0]

            # Extract basic metrics
            total_return = result.analyzers.returns.get_analysis()['rtot']
            sharpe_ratio = result.analyzers.sharpe.get_analysis()['sharperatio']
            max_drawdown = result.analyzers.drawdown.get_analysis()['max']['drawdown']

            # Extract trade analysis
            trade_analysis = result.analyzers.trades.get_analysis()
            num_trades = trade_analysis.get('total', {}).get('total', 0)
            win_rate = trade_analysis.get('won', {}).get('total', 0) / max(num_trades, 1)
            profit_factor = self._calculate_profit_factor(trade_analysis)

            # Calculate additional metrics
            calmar_ratio = total_return / max(max_drawdown, 0.001) if max_drawdown > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(result)

            # Calculate alpha/beta vs benchmark
            alpha, beta = self._calculate_alpha_beta(result, data_config)

            # Get benchmark return
            benchmark_return = self._get_benchmark_return(data_config)

            # Extract equity and drawdown curves
            equity_curve = [x for x in result.observers.portfolio.value]
            drawdown_curve = [x.drawdown for x in result.analyzers.drawdown.get_analysis()['drawdown']]

            # Calculate monthly returns
            monthly_returns = self._calculate_monthly_returns(result)

            # Calculate average trade duration
            avg_trade_duration = self._calculate_avg_trade_duration(trade_analysis)

            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(result)

            # Regime performance (placeholder)
            regime_performance = {}

            return BacktestResult(
                strategy_name=strategy.strategy_name,
                hypothesis_id=strategy.hypothesis_id,
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                calmar_ratio=calmar_ratio,
                sortino_ratio=sortino_ratio,
                alpha=alpha,
                beta=beta,
                benchmark_return=benchmark_return,
                start_date=data_config.get('start_date'),
                end_date=data_config.get('end_date'),
                num_trades=num_trades,
                avg_trade_duration=avg_trade_duration,
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                monthly_returns=monthly_returns,
                regime_performance=regime_performance,
                risk_metrics=risk_metrics
            )

        except Exception as e:
            self.logger.error(f"Failed to process backtest results: {e}")
            return None

    def _calculate_profit_factor(self, trade_analysis: Dict) -> float:
        """Calculate profit factor from trade analysis"""

        total_won = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        total_lost = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))

        return total_won / max(total_lost, 0.001) if total_lost > 0 else float('inf')

    def _calculate_sortino_ratio(self, result) -> float:
        """Calculate Sortino ratio"""

        try:
            returns = pd.Series([x for x in result.observers.portfolio.value])
            daily_returns = returns.pct_change().dropna()

            # Downside deviation
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std()

            # Annualized Sortino
            avg_return = daily_returns.mean()
            sortino = (avg_return * 252) / max(downside_deviation * np.sqrt(252), 0.001)

            return sortino

        except:
            return 0.0

    def _calculate_alpha_beta(self, result, data_config: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark"""

        try:
            if not self.benchmark_data:
                return 0.0, 1.0

            # Get strategy returns
            strategy_returns = pd.Series([x for x in result.observers.portfolio.value])
            strategy_returns = strategy_returns.pct_change().dropna()

            # Get benchmark returns for same period
            benchmark_returns = self._get_benchmark_returns_for_period(
                data_config.get('start_date'), data_config.get('end_date')
            )

            if len(strategy_returns) != len(benchmark_returns):
                return 0.0, 1.0

            # Calculate beta
            covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / max(benchmark_variance, 0.001)

            # Calculate alpha (annualized)
            strategy_avg_return = strategy_returns.mean() * 252
            benchmark_avg_return = benchmark_returns.mean() * 252
            alpha = strategy_avg_return - beta * benchmark_avg_return

            return alpha, beta

        except:
            return 0.0, 1.0

    def _get_benchmark_return(self, data_config: Dict[str, Any]) -> float:
        """Get benchmark return for the period"""

        try:
            if not self.benchmark_data:
                return 0.0

            benchmark_returns = self._get_benchmark_returns_for_period(
                data_config.get('start_date'), data_config.get('end_date')
            )

            return (benchmark_returns + 1).prod() - 1

        except:
            return 0.0

    def _load_benchmark_data(self) -> Optional[pd.DataFrame]:
        """Load benchmark data (e.g., S&P 500 or BTC)"""

        try:
            # Try to load from config or default location
            benchmark_path = getattr(config, 'benchmark_data_path', None)
            if benchmark_path and Path(benchmark_path).exists():
                return pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
            else:
                # Use a simple proxy or skip
                self.logger.warning("Benchmark data not available")
                return None
        except:
            return None

    def _get_benchmark_returns_for_period(self, start_date: datetime, end_date: datetime) -> pd.Series:
        """Get benchmark returns for specific period"""

        if not self.benchmark_data:
            return pd.Series()

        try:
            period_data = self.benchmark_data.loc[start_date:end_date]
            return period_data['Close'].pct_change().dropna()
        except:
            return pd.Series()

    def _calculate_monthly_returns(self, result) -> Dict[str, float]:
        """Calculate monthly returns"""

        try:
            portfolio_values = pd.Series([x for x in result.observers.portfolio.value])
            portfolio_values.index = pd.date_range(
                start=result.datas[0].datetime.datetime(0),
                periods=len(portfolio_values),
                freq='D'
            )

            monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()
            return monthly_returns.to_dict()

        except:
            return {}

    def _calculate_avg_trade_duration(self, trade_analysis: Dict) -> timedelta:
        """Calculate average trade duration"""

        try:
            durations = []
            for trade_type in ['won', 'lost']:
                if trade_type in trade_analysis:
                    for trade in trade_analysis[trade_type].get('trades', []):
                        if 'duration' in trade:
                            durations.append(trade['duration'])

            if durations:
                avg_duration_td = sum(durations, timedelta()) / len(durations)
                return avg_duration_td
            else:
                return timedelta()

        except:
            return timedelta()

    def _calculate_risk_metrics(self, result) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""

        try:
            returns = pd.Series([x for x in result.observers.portfolio.value])
            daily_returns = returns.pct_change().dropna()

            return {
                'volatility': daily_returns.std() * np.sqrt(252),
                'var_95': np.percentile(daily_returns, 5),
                'cvar_95': daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean(),
                'skewness': daily_returns.skew(),
                'kurtosis': daily_returns.kurtosis(),
                'tail_ratio': abs(np.percentile(daily_returns, 95) / np.percentile(daily_returns, 5))
            }

        except:
            return {}

    def _calculate_walk_forward_metrics(self, strategy_name: str,
                                     in_sample_results: List[BacktestResult],
                                     out_of_sample_results: List[BacktestResult]) -> WalkForwardResult:
        """Calculate walk-forward analysis metrics"""

        try:
            # Calculate average metrics
            is_sharpe = np.mean([r.sharpe_ratio for r in in_sample_results])
            oos_sharpe = np.mean([r.sharpe_ratio for r in out_of_sample_results])

            is_return = np.mean([r.total_return for r in in_sample_results])
            oos_return = np.mean([r.total_return for r in out_of_sample_results])

            # Walk-forward efficiency
            wf_sharpe = oos_sharpe / max(is_sharpe, 0.001)
            wf_return = oos_return / max(is_return, 0.001)

            # Overfitting probability (simplified)
            overfitting_prob = max(0, 1 - min(wf_sharpe, wf_return))

            # Stability score
            stability_score = 1 - overfitting_prob

            return WalkForwardResult(
                strategy_name=strategy_name,
                in_sample_results=in_sample_results,
                out_of_sample_results=out_of_sample_results,
                walk_forward_sharpe=wf_sharpe,
                walk_forward_return=wf_return,
                overfitting_probability=overfitting_prob,
                stability_score=stability_score
            )

        except Exception as e:
            self.logger.error(f"Failed to calculate walk-forward metrics: {e}")
            return None

    def _save_backtest_results(self, result: BacktestResult):
        """Save detailed backtest results to disk"""

        try:
            result_file = self.results_dir / f"{result.strategy_name}_backtest_results.json"

            # Convert to serializable format
            result_dict = {
                'strategy_name': result.strategy_name,
                'hypothesis_id': result.hypothesis_id,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'calmar_ratio': result.calmar_ratio,
                'sortino_ratio': result.sortino_ratio,
                'alpha': result.alpha,
                'beta': result.beta,
                'benchmark_return': result.benchmark_return,
                'start_date': result.start_date.isoformat() if result.start_date else None,
                'end_date': result.end_date.isoformat() if result.end_date else None,
                'num_trades': result.num_trades,
                'avg_trade_duration_days': result.avg_trade_duration.days if result.avg_trade_duration else 0,
                'monthly_returns': result.monthly_returns,
                'risk_metrics': result.risk_metrics,
                'validation_status': result.validation_status
            }

            with open(result_file, 'w') as f:
                import json
                json.dump(result_dict, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save backtest results: {e}")

    def run_parallel_backtests(self, strategies: List[GeneratedStrategy],
                             data_configs: List[Dict[str, Any]],
                             max_workers: int = 4) -> Dict[str, BacktestResult]:
        """
        Run multiple backtests in parallel

        Args:
            strategies: List of strategies to backtest
            data_configs: List of data configurations (one per strategy)
            max_workers: Maximum number of parallel workers

        Returns:
            Dictionary of strategy_name -> BacktestResult
        """
        results = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all backtest jobs
            future_to_strategy = {
                executor.submit(self.run_backtest, strategy, config): strategy
                for strategy, config in zip(strategies, data_configs)
            }

            # Collect results as they complete
            for future in as_completed(future_to_strategy):
                strategy = future_to_strategy[future]
                try:
                    result = future.result()
                    if result:
                        results[strategy.strategy_name] = result
                except Exception as e:
                    self.logger.error(f"Parallel backtest failed for {strategy.strategy_name}: {e}")

        return results