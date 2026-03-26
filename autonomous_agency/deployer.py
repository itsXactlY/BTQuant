"""
Live trading deployer for top-performing strategies
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

from .config import DeployerConfig
from .evolution_engine import StrategyGenome


@dataclass
class Deployment:
    """Represents a live trading deployment"""
    strategy_id: str
    strategy_name: str
    deployed_at: datetime
    capital_allocated: float
    status: str  # 'active', 'stopped', 'liquidated'
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]


class LiveDeployer:
    """Manages live deployment of validated strategies"""

    def __init__(self, config: DeployerConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.active_deployments: Dict[str, Deployment] = {}
        self.deployment_history: List[Deployment] = []

    async def deploy_strategy(self, genome: StrategyGenome) -> bool:
        """
        Deploy a strategy to live trading

        Args:
            genome: Strategy genome to deploy

        Returns:
            Success status
        """

        if len(self.active_deployments) >= self.config.max_live_strategies:
            self.logger.warning("Maximum live strategies reached, cannot deploy")
            return False

        if genome.evaluation_metrics.overall_score < self.config.min_confidence_score:
            self.logger.warning(f"Strategy {genome.strategy_id} confidence too low for deployment")
            return False

        try:
            # Calculate position size
            capital = self._calculate_position_size(genome)

            # Create deployment record
            deployment = Deployment(
                strategy_id=genome.strategy_id,
                strategy_name=genome.hypothesis.name,
                deployed_at=datetime.now(),
                capital_allocated=capital,
                status='active',
                performance={'initial_capital': capital, 'current_pnl': 0.0},
                risk_metrics={
                    'max_drawdown_limit': genome.evaluation_metrics.max_drawdown * 1.2,  # Slightly more conservative
                    'stop_loss_level': -abs(genome.evaluation_metrics.max_drawdown) * 0.8
                }
            )

            # In a real implementation, this would:
            # 1. Connect to live trading API (HotSpine, CCXT, etc.)
            # 2. Initialize strategy with live data feeds
            # 3. Start trading with allocated capital
            # 4. Set up monitoring and risk controls

            self.active_deployments[genome.strategy_id] = deployment
            self.deployment_history.append(deployment)

            self.logger.info(f"🚀 Deployed strategy {genome.strategy_id} with ${capital:.2f} capital")
            return True

        except Exception as e:
            self.logger.error(f"Failed to deploy strategy {genome.strategy_id}: {e}")
            return False

    def _calculate_position_size(self, genome: StrategyGenome) -> float:
        """Calculate position size based on risk metrics"""
        # Kelly criterion approximation
        win_rate = genome.evaluation_metrics.win_rate
        avg_win = genome.evaluation_metrics.total_return * 0.1  # Estimate
        avg_loss = abs(genome.evaluation_metrics.max_drawdown) * 0.05  # Estimate

        if avg_loss == 0:
            kelly_fraction = 0.02  # Conservative default
        else:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        # Conservative Kelly
        position_size = min(kelly_fraction * 0.5, self.config.position_size_limit)

        # Apply risk limits
        max_allocation = self.config.max_total_exposure / max(1, self.config.max_live_strategies)
        position_size = min(position_size, max_allocation)

        return max(position_size, 0.001)  # Minimum allocation

    async def monitor_deployments(self):
        """Monitor active deployments and manage risk"""
        while True:
            try:
                for strategy_id, deployment in list(self.active_deployments.items()):
                    await self._check_deployment_health(deployment)

                # Check total exposure
                total_exposure = sum(d.capital_allocated for d in self.active_deployments.values())
                if total_exposure > self.config.max_total_exposure:
                    await self._reduce_exposure(total_exposure - self.config.max_total_exposure)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                self.logger.error(f"Error monitoring deployments: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _check_deployment_health(self, deployment: Deployment):
        """Check health of individual deployment"""
        # In real implementation, this would:
        # 1. Get current P&L from live trading
        # 2. Check drawdown against limits
        # 3. Monitor for liquidation events
        # 4. Update performance metrics

        # Placeholder logic
        current_pnl = deployment.performance.get('current_pnl', 0.0)
        max_drawdown = deployment.risk_metrics['max_drawdown_limit']

        # Simulate some performance tracking
        # In reality, this would come from live trading data

        if current_pnl < -max_drawdown:
            await self._stop_deployment(deployment.strategy_id, "stop_loss_triggered")

    async def _stop_deployment(self, strategy_id: str, reason: str):
        """Stop a deployment"""
        if strategy_id in self.active_deployments:
            deployment = self.active_deployments[strategy_id]
            deployment.status = 'stopped'

            # In real implementation: close all positions, log final P&L

            self.logger.info(f"🛑 Stopped deployment {strategy_id}: {reason}")

            # Move to history
            del self.active_deployments[strategy_id]

    async def _reduce_exposure(self, excess_amount: float):
        """Reduce total exposure by stopping worst performers"""
        if not self.active_deployments:
            return

        # Sort by performance (worst first)
        sorted_deployments = sorted(
            self.active_deployments.items(),
            key=lambda x: x[1].performance.get('current_pnl', 0)
        )

        amount_reduced = 0.0
        for strategy_id, deployment in sorted_deployments:
            if amount_reduced >= excess_amount:
                break

            await self._stop_deployment(strategy_id, "exposure_reduction")
            amount_reduced += deployment.capital_allocated

        self.logger.info(f"Reduced exposure by ${amount_reduced:.2f}")

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        total_exposure = sum(d.capital_allocated for d in self.active_deployments.values())
        total_pnl = sum(d.performance.get('current_pnl', 0.0) for d in self.active_deployments.values())

        return {
            'active_deployments': len(self.active_deployments),
            'total_exposure': total_exposure,
            'total_pnl': total_pnl,
            'max_live_strategies': self.config.max_live_strategies,
            'deployments': [
                {
                    'strategy_id': d.strategy_id,
                    'strategy_name': d.strategy_name,
                    'capital_allocated': d.capital_allocated,
                    'status': d.status,
                    'performance': d.performance,
                    'deployed_at': d.deployed_at.isoformat()
                } for d in self.active_deployments.values()
            ]
        }

    async def emergency_stop_all(self):
        """Emergency stop all deployments"""
        self.logger.warning("🚨 Emergency stopping all deployments")

        stop_tasks = []
        for strategy_id in list(self.active_deployments.keys()):
            stop_tasks.append(self._stop_deployment(strategy_id, "emergency_stop"))

        await asyncio.gather(*stop_tasks)

        self.logger.info("✅ All deployments stopped")