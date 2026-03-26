"""
Live Deployment Module for Autonomous Quantitative Research Agency

This module handles the deployment of validated strategies to live trading environments.
It manages risk controls, performance monitoring, and automatic shutdown mechanisms
for strategies that fail to meet performance thresholds.
"""

import os
import logging
import asyncio
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np

import backtrader as bt
from backtrader import Strategy
import ccxt

from .config import AgencyConfig
from .evaluator import EvaluationMetrics
from .strategy_factory import GeneratedStrategy as StrategyMetadata


@dataclass
class LiveDeployment:
    """Represents a live deployment of a strategy."""
    strategy_id: str
    strategy_metadata: StrategyMetadata
    evaluation: EvaluationMetrics
    deployment_time: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, paused, stopped, failed
    capital_allocated: float = 0.0
    current_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    live_cerebro: Optional[bt.Cerebro] = None
    exchange_client: Optional[Any] = None
    risk_limits: Dict[str, float] = field(default_factory=dict)
    monitoring_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Remove non-serializable objects
        data.pop('live_cerebro', None)
        data.pop('exchange_client', None)
        return data


class LiveDeploymentManager:
    """
    Manages live deployments of trading strategies with automated risk controls
    and performance monitoring.
    """

    def __init__(self, config: AgencyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.deployments: Dict[str, LiveDeployment] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_deployments)
        self.monitoring_interval = config.live_monitoring_interval_minutes
        self.risk_limits = {
            'max_drawdown': config.max_live_drawdown,
            'max_daily_loss': config.max_daily_loss,
            'min_sharpe_ratio': config.min_live_sharpe_ratio,
            'max_consecutive_losses': config.max_consecutive_losses
        }

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for live deployment operations."""
        log_dir = Path(self.config.log_dir) / "live_deployments"
        log_dir.mkdir(exist_ok=True)

        handler = logging.FileHandler(log_dir / "live_deployment.log")
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def deploy_strategy(self, strategy_metadata: StrategyMetadata,
                             evaluation: EvaluationMetrics) -> Optional[str]:
        """
        Deploy a validated strategy to live trading.

        Args:
            strategy_metadata: Metadata for the strategy to deploy
            evaluation: Evaluation results from backtesting

        Returns:
            Deployment ID if successful, None otherwise
        """
        try:
            # Check if we have capacity for new deployments
            active_deployments = sum(1 for d in self.deployments.values()
                                   if d.status == "active")
            if active_deployments >= self.config.max_concurrent_deployments:
                self.logger.warning("Maximum concurrent deployments reached")
                return None

            # Generate deployment ID
            deployment_id = f"{strategy_metadata.strategy_id}_live_{int(time.time())}"

            # Calculate capital allocation based on evaluation metrics
            capital_allocation = self._calculate_capital_allocation(evaluation)

            # Create live deployment instance
            deployment = LiveDeployment(
                strategy_id=strategy_metadata.strategy_id,
                strategy_metadata=strategy_metadata,
                evaluation=evaluation,
                capital_allocated=capital_allocation,
                risk_limits=self.risk_limits.copy()
            )

            # Initialize live trading setup
            success = await self._initialize_live_trading(deployment)
            if not success:
                self.logger.error(f"Failed to initialize live trading for {deployment_id}")
                return None

            # Start the deployment
            deployment.status = "active"
            self.deployments[deployment_id] = deployment

            # Start monitoring in background
            asyncio.create_task(self._monitor_deployment(deployment_id))

            self.logger.info(f"Successfully deployed strategy {strategy_metadata.strategy_id} "
                           f"with ID {deployment_id}")
            return deployment_id

        except Exception as e:
            self.logger.error(f"Error deploying strategy {strategy_metadata.strategy_id}: {e}")
            return None

    def _calculate_capital_allocation(self, evaluation: EvaluationMetrics) -> float:
        """Calculate capital allocation based on strategy performance metrics."""
        base_allocation = self.config.base_live_capital_allocation

        # Scale based on Sharpe ratio (higher Sharpe = more capital)
        sharpe_multiplier = min(2.0, max(0.1, evaluation.sharpe_ratio / 2.0))

        # Scale based on win rate
        win_rate_multiplier = min(2.0, max(0.1, evaluation.win_rate / 0.6))

        # Scale based on max drawdown (lower drawdown = more capital)
        drawdown_multiplier = min(2.0, max(0.1, (1 - evaluation.max_drawdown) * 2))

        allocation = base_allocation * sharpe_multiplier * win_rate_multiplier * drawdown_multiplier

        # Cap at maximum allocation
        return min(allocation, self.config.max_live_capital_allocation)

    async def _initialize_live_trading(self, deployment: LiveDeployment) -> bool:
        """Initialize live trading setup for a deployment."""
        try:
            # Import the strategy class dynamically
            strategy_module = __import__(f"generated_strategies.{deployment.strategy_metadata.strategy_id}",
                                       fromlist=[deployment.strategy_metadata.class_name])
            strategy_class = getattr(strategy_module, deployment.strategy_metadata.class_name)

            # Create Cerebro instance for live trading
            cerebro = bt.Cerebro()
            cerebro.addstrategy(strategy_class)

            # Configure broker (using CCXT for live trading)
            broker_config = self._get_broker_config()
            cerebro.setbroker(bt.brokers.CCXTBroker(**broker_config))

            # Add data feeds for live trading
            data_feeds = self._setup_live_data_feeds(deployment)
            for data in data_feeds:
                cerebro.adddata(data)

            # Set initial capital
            cerebro.broker.setcash(deployment.capital_allocated)

            # Store references
            deployment.live_cerebro = cerebro

            # Initialize exchange client for direct API access if needed
            deployment.exchange_client = self._create_exchange_client()

            return True

        except Exception as e:
            self.logger.error(f"Error initializing live trading: {e}")
            return False

    def _get_broker_config(self) -> Dict[str, Any]:
        """Get broker configuration for live trading."""
        return {
            'exchange': self.config.live_exchange,
            'apiKey': os.getenv('CCXT_API_KEY'),
            'secret': os.getenv('CCXT_SECRET'),
            'sandbox': self.config.live_sandbox_mode,
            'currency': self.config.live_base_currency
        }

    def _setup_live_data_feeds(self, deployment: LiveDeployment) -> List[bt.DataBase]:
        """Setup live data feeds for the deployment."""
        feeds = []

        # Use CCXT data feeds for live trading
        for symbol in deployment.strategy_metadata.symbols:
            data = bt.feeds.CCXT(
                exchange=self.config.live_exchange,
                symbol=symbol,
                timeframe=bt.TimeFrame.Minutes,
                compression=1,
                ohlcv_limit=100,
                currency=self.config.live_base_currency,
                apiKey=os.getenv('CCXT_API_KEY'),
                secret=os.getenv('CCXT_SECRET'),
                sandbox=self.config.live_sandbox_mode
            )
            feeds.append(data)

        return feeds

    def _create_exchange_client(self) -> Any:
        """Create exchange client for direct API interactions."""
        exchange_class = getattr(ccxt, self.config.live_exchange)
        return exchange_class({
            'apiKey': os.getenv('CCXT_API_KEY'),
            'secret': os.getenv('CCXT_SECRET'),
            'sandbox': self.config.live_sandbox_mode
        })

    async def _monitor_deployment(self, deployment_id: str):
        """Monitor a live deployment for performance and risk metrics."""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return

        while deployment.status == "active":
            try:
                # Update performance metrics
                await self._update_performance_metrics(deployment)

                # Check risk limits
                if self._check_risk_limits(deployment):
                    await self._stop_deployment(deployment_id, "risk_limit_breached")
                    break

                # Check performance thresholds
                if self._check_performance_thresholds(deployment):
                    await self._stop_deployment(deployment_id, "performance_threshold")
                    break

                # Log monitoring data
                self._log_monitoring_data(deployment)

                # Wait for next monitoring interval
                await asyncio.sleep(self.monitoring_interval * 60)

            except Exception as e:
                self.logger.error(f"Error monitoring deployment {deployment_id}: {e}")
                await self._stop_deployment(deployment_id, "monitoring_error")
                break

    async def _update_performance_metrics(self, deployment: LiveDeployment):
        """Update performance metrics for a deployment."""
        if not deployment.live_cerebro:
            return

        try:
            # Get current portfolio value
            current_value = deployment.live_cerebro.broker.getvalue()
            initial_value = deployment.capital_allocated

            # Calculate P&L
            deployment.current_pnl = current_value - initial_value

            # Calculate drawdown
            peak_value = max(initial_value, max((d.get('portfolio_value', initial_value)
                                               for d in deployment.monitoring_data), default=initial_value))
            deployment.max_drawdown = max(deployment.max_drawdown,
                                        (peak_value - current_value) / peak_value)

            # Get trading statistics from cerebro
            if hasattr(deployment.live_cerebro, 'runstrats'):
                strat = deployment.live_cerebro.runstrats[0][0]
                if hasattr(strat, 'analyzers'):
                    # Extract metrics from analyzers
                    deployment.sharpe_ratio = getattr(strat.analyzers.sharpe, 'sharpe', 0)
                    deployment.win_rate = getattr(strat.analyzers.tradeanalyzer, 'win_rate', 0)
                    deployment.total_trades = getattr(strat.analyzers.tradeanalyzer, 'total_trades', 0)

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def _check_risk_limits(self, deployment: LiveDeployment) -> bool:
        """Check if deployment violates risk limits."""
        if deployment.max_drawdown > self.risk_limits['max_drawdown']:
            self.logger.warning(f"Max drawdown limit breached for {deployment.strategy_id}")
            return True

        if deployment.current_pnl < -self.risk_limits['max_daily_loss']:
            self.logger.warning(f"Daily loss limit breached for {deployment.strategy_id}")
            return True

        if deployment.sharpe_ratio < self.risk_limits['min_sharpe_ratio']:
            self.logger.warning(f"Sharpe ratio below minimum for {deployment.strategy_id}")
            return True

        return False

    def _check_performance_thresholds(self, deployment: LiveDeployment) -> bool:
        """Check if deployment meets performance thresholds for continuation."""
        # Stop if strategy has been running too long without profit
        days_running = (datetime.now() - deployment.deployment_time).days
        if days_running > self.config.max_deployment_days and deployment.current_pnl <= 0:
            self.logger.info(f"Stopping deployment {deployment.strategy_id} - no profit after {days_running} days")
            return True

        # Stop if too many consecutive losses
        # This would require tracking consecutive losses in the strategy
        # For now, we'll use a simple heuristic
        if deployment.total_trades > 10 and deployment.win_rate < 0.3:
            self.logger.info(f"Stopping deployment {deployment.strategy_id} - low win rate")
            return True

        return False

    async def _stop_deployment(self, deployment_id: str, reason: str):
        """Stop a live deployment."""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return

        try:
            self.logger.info(f"Stopping deployment {deployment_id} - reason: {reason}")

            # Stop the live trading
            if deployment.live_cerebro:
                # Close all positions
                deployment.live_cerebro.broker.cancel_all()
                # Note: In a real implementation, you'd need to implement proper position closing

            # Update status
            deployment.status = "stopped"

            # Archive the deployment
            await self._archive_deployment(deployment, reason)

        except Exception as e:
            self.logger.error(f"Error stopping deployment {deployment_id}: {e}")
            deployment.status = "failed"

    async def _archive_deployment(self, deployment: LiveDeployment, reason: str):
        """Archive deployment data for analysis."""
        archive_dir = Path(self.config.archive_dir) / "live_deployments"
        archive_dir.mkdir(exist_ok=True)

        archive_data = {
            'deployment': deployment.to_dict(),
            'stop_reason': reason,
            'final_metrics': {
                'pnl': deployment.current_pnl,
                'max_drawdown': deployment.max_drawdown,
                'sharpe_ratio': deployment.sharpe_ratio,
                'win_rate': deployment.win_rate,
                'total_trades': deployment.total_trades
            },
            'monitoring_history': deployment.monitoring_data
        }

        archive_file = archive_dir / f"{deployment.strategy_id}_{deployment.deployment_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(archive_file, 'w') as f:
            json.dump(archive_data, f, indent=2, default=str)

    def _log_monitoring_data(self, deployment: LiveDeployment):
        """Log current monitoring data."""
        monitoring_entry = {
            'timestamp': datetime.now().isoformat(),
            'pnl': deployment.current_pnl,
            'max_drawdown': deployment.max_drawdown,
            'sharpe_ratio': deployment.sharpe_ratio,
            'win_rate': deployment.win_rate,
            'total_trades': deployment.total_trades,
            'portfolio_value': deployment.live_cerebro.broker.getvalue() if deployment.live_cerebro else 0
        }

        deployment.monitoring_data.append(monitoring_entry)

        # Keep only last N entries to prevent memory issues
        if len(deployment.monitoring_data) > 1000:
            deployment.monitoring_data = deployment.monitoring_data[-1000:]

    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a deployment."""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return None

        return {
            'deployment_id': deployment_id,
            'strategy_id': deployment.strategy_id,
            'status': deployment.status,
            'deployment_time': deployment.deployment_time.isoformat(),
            'capital_allocated': deployment.capital_allocated,
            'current_pnl': deployment.current_pnl,
            'max_drawdown': deployment.max_drawdown,
            'sharpe_ratio': deployment.sharpe_ratio,
            'win_rate': deployment.win_rate,
            'total_trades': deployment.total_trades
        }

    async def get_all_deployments(self) -> List[Dict[str, Any]]:
        """Get status of all deployments."""
        return [await self.get_deployment_status(did) for did in self.deployments.keys()
                if await self.get_deployment_status(did) is not None]

    async def emergency_stop_all(self):
        """Emergency stop all active deployments."""
        self.logger.warning("Emergency stop initiated for all deployments")

        tasks = []
        for deployment_id in self.deployments:
            if self.deployments[deployment_id].status == "active":
                tasks.append(self._stop_deployment(deployment_id, "emergency_stop"))

        await asyncio.gather(*tasks)

    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)