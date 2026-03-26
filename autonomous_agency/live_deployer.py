#!/usr/bin/env python3
"""
Live Deployment Module for Autonomous Quantitative Research Agency

This module handles the safe deployment of top-performing strategies to live trading environments.
It includes comprehensive risk controls, performance monitoring, and emergency shutdown capabilities.
"""

import os
import sys
import time
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import json
import uuid

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_agency.config import LiveDeploymentConfig
from autonomous_agency.models import StrategyPerformance, LiveStrategyInstance
from autonomous_agency.archiver import Archiver

class LiveDeployer:
    """
    Handles live deployment of validated strategies with comprehensive risk management.
    """
    
    def __init__(self, config: LiveDeploymentConfig, archiver: Archiver):
        self.config = config
        self.archiver = archiver
        self.logger = logging.getLogger(f"{__name__}.LiveDeployer")
        self.logger.setLevel(logging.INFO)
        
        # Active strategy instances
        self.active_strategies: Dict[str, LiveStrategyInstance] = {}
        self.lock = threading.Lock()
        
        # Performance monitoring
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        
        # Risk management flags
        self.emergency_stop = False
        self.system_health = "healthy"
        
        self.logger.info("LiveDeployer initialized with safety-first deployment protocols")
    
    def deploy_strategy(self, strategy_id: str, strategy_code: str, 
                       performance_metrics: StrategyPerformance) -> bool:
        """
        Deploy a validated strategy to live trading environment.
        
        Args:
            strategy_id: Unique identifier for the strategy
            strategy_code: Executable strategy code
            performance_metrics: Backtested performance data
            
        Returns:
            bool: True if deployment successful, False otherwise
        """
        try:
            self.logger.info(f"Initiating deployment for strategy {strategy_id}")
            
            # Validate deployment conditions
            if not self._validate_deployment_conditions(strategy_id, performance_metrics):
                self.logger.warning(f"Deployment conditions not met for {strategy_id}")
                return False
            
            # Create live strategy instance
            live_instance = self._create_live_instance(strategy_id, strategy_code, performance_metrics)
            
            # Add to active strategies
            with self.lock:
                self.active_strategies[strategy_id] = live_instance
                self.performance_history[strategy_id] = []
            
            # Start monitoring thread
            monitoring_thread = threading.Thread(
                target=self._monitor_strategy_performance,
                args=(strategy_id,),
                daemon=True
            )
            monitoring_thread.start()
            
            self.logger.info(f"Strategy {strategy_id} deployed successfully")
            self.archiver.log_event("deployment", {
                "strategy_id": strategy_id,
                "status": "deployed",
                "timestamp": datetime.utcnow().isoformat(),
                "performance": performance_metrics.dict()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Deployment failed for {strategy_id}: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _validate_deployment_conditions(self, strategy_id: str, 
                                      performance: StrategyPerformance) -> bool:
        """
        Validate that strategy meets all deployment criteria.
        """
        # Check minimum performance thresholds
        if performance.sharpe_ratio < self.config.min_sharpe_ratio:
            self.logger.warning(f"Strategy {strategy_id} Sharpe ratio {performance.sharpe_ratio:.2f} "
                             f"below threshold {self.config.min_sharpe_ratio:.2f}")
            return False
            
        if performance.max_drawdown > self.config.max_allowed_drawdown:
            self.logger.warning(f"Strategy {strategy_id} max drawdown {performance.max_drawdown:.2%} "
                             f"exceeds threshold {self.config.max_allowed_drawdown:.2%}")
            return False
            
        if performance.win_rate < self.config.min_win_rate:
            self.logger.warning(f"Strategy {strategy_id} win rate {performance.win_rate:.2%} "
                             f"below threshold {self.config.min_win_rate:.2%}")
            return False
            
        # Check system capacity
        if len(self.active_strategies) >= self.config.max_concurrent_strategies:
            self.logger.warning("Maximum concurrent strategies reached")
            return False
            
        return True
    
    def _create_live_instance(self, strategy_id: str, strategy_code: str, 
                            performance: StrategyPerformance) -> LiveStrategyInstance:
        """
        Create a live strategy instance with risk controls.
        """
        # Generate unique deployment ID
        deployment_id = f"{strategy_id}-{uuid.uuid4().hex[:8]}"
        
        # Create instance with conservative risk parameters
        instance = LiveStrategyInstance(
            deployment_id=deployment_id,
            strategy_id=strategy_id,
            strategy_code=strategy_code,
            initial_capital=self.config.initial_capital,
            max_position_size=self.config.max_position_size,
            max_daily_loss=self.config.max_daily_loss,
            max_drawdown=self.config.max_live_drawdown,
            performance_targets=performance,
            status="initializing"
        )
        
        # Apply additional risk controls
        instance.risk_parameters = {
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "position_sizing_model": "conservative",
            "max_leverage": self.config.max_leverage,
            "volatility_scaling": True
        }
        
        return instance
    
    def _monitor_strategy_performance(self, strategy_id: str):
        """
        Continuous monitoring of live strategy performance.
        """
        try:
            self.logger.info(f"Starting performance monitoring for {strategy_id}")
            
            while not self.emergency_stop:
                # Simulate performance data collection
                time.sleep(self.config.monitoring_interval)
                
                with self.lock:
                    if strategy_id not in self.active_strategies:
                        break
                    
                    instance = self.active_strategies[strategy_id]
                    
                    # Simulate performance update (in real implementation, this would come from broker)
                    current_performance = self._simulate_performance_update(instance)
                    
                    # Store performance history
                    self.performance_history[strategy_id].append(current_performance)
                    
                    # Check risk thresholds
                    if self._check_risk_thresholds(instance, current_performance):
                        self._trigger_emergency_shutdown(strategy_id, 
                                                       "Risk threshold violation")
                        break
                    
                    # Log performance
                    self.logger.info(f"Strategy {strategy_id} performance: "
                                   f"PnL: {current_performance.pnl:.2%}, "
                                   f"Drawdown: {current_performance.max_drawdown:.2%}")
                    
        except Exception as e:
            self.logger.error(f"Monitoring failed for {strategy_id}: {str(e)}")
            self._trigger_emergency_shutdown(strategy_id, f"Monitoring failure: {str(e)}")
    
    def _simulate_performance_update(self, instance: LiveStrategyInstance) -> StrategyPerformance:
        """
        Simulate performance update (replace with real broker integration).
        """
        # This is a simulation - in production, this would connect to broker API
        import random
        
        # Simulate realistic performance with some volatility
        base_pnl = random.uniform(-0.02, 0.03)  # -2% to +3% range
        volatility = random.uniform(0.8, 1.2)
        
        return StrategyPerformance(
            strategy_id=instance.strategy_id,
            pnl=base_pnl * volatility,
            sharpe_ratio=instance.performance_targets.sharpe_ratio * random.uniform(0.9, 1.1),
            max_drawdown=min(instance.performance_targets.max_drawdown * random.uniform(0.8, 1.3), 0.2),
            win_rate=instance.performance_targets.win_rate * random.uniform(0.95, 1.05),
            sortino_ratio=instance.performance_targets.sortino_ratio * random.uniform(0.9, 1.1),
            calmar_ratio=instance.performance_targets.calmar_ratio * random.uniform(0.9, 1.1),
            omega_ratio=instance.performance_targets.omega_ratio * random.uniform(0.9, 1.1),
            beta=instance.performance_targets.beta * random.uniform(0.9, 1.1),
            alpha=instance.performance_targets.alpha * random.uniform(0.9, 1.1),
            r_squared=instance.performance_targets.r_squared * random.uniform(0.9, 1.1)
        )
    
    def _check_risk_thresholds(self, instance: LiveStrategyInstance, 
                              current_performance: StrategyPerformance) -> bool:
        """
        Check if any risk thresholds have been violated.
        """
        # Check drawdown limits
        if current_performance.max_drawdown > instance.max_drawdown:
            self.logger.error(f"Drawdown limit violated: {current_performance.max_drawdown:.2%} > {instance.max_drawdown:.2%}")
            return True
            
        # Check daily loss limits (simplified)
        if current_performance.pnl < -instance.max_daily_loss:
            self.logger.error(f"Daily loss limit violated: {current_performance.pnl:.2%} < {-instance.max_daily_loss:.2%}")
            return True
            
        # Check performance degradation
        if (current_performance.sharpe_ratio < instance.performance_targets.sharpe_ratio * 0.7 or
            current_performance.win_rate < instance.performance_targets.win_rate * 0.8):
            self.logger.error("Significant performance degradation detected")
            return True
            
        return False
    
    def _trigger_emergency_shutdown(self, strategy_id: str, reason: str):
        """
        Emergency shutdown of a live strategy.
        """
        with self.lock:
            if strategy_id in self.active_strategies:
                instance = self.active_strategies[strategy_id]
                instance.status = "emergency_stop"
                instance.shutdown_reason = reason
                instance.shutdown_timestamp = datetime.utcnow().isoformat()
                
                self.logger.error(f"EMERGENCY SHUTDOWN: {strategy_id} - {reason}")
                
                # Log to archiver
                self.archiver.log_event("emergency_shutdown", {
                    "strategy_id": strategy_id,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat(),
                    "performance": instance.performance_targets.dict()
                })
                
                # In real implementation, this would also:
                # 1. Close all open positions
                # 2. Cancel all pending orders
                # 3. Disable strategy execution
                # 4. Notify risk management team
    
    def undeploy_strategy(self, strategy_id: str, reason: str = "manual_undeployment") -> bool:
        """
        Gracefully undeploy a live strategy.
        """
        try:
            with self.lock:
                if strategy_id not in self.active_strategies:
                    self.logger.warning(f"Strategy {strategy_id} not found in active deployments")
                    return False
                    
                instance = self.active_strategies[strategy_id]
                instance.status = "undeployed"
                instance.shutdown_reason = reason
                instance.shutdown_timestamp = datetime.utcnow().isoformat()
                
                # Remove from active strategies
                del self.active_strategies[strategy_id]
                
                self.logger.info(f"Strategy {strategy_id} undeployed: {reason}")
                
                # Log to archiver
                self.archiver.log_event("undeployment", {
                    "strategy_id": strategy_id,
                    "reason": reason,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                return True
                
        except Exception as e:
            self.logger.error(f"Undeployment failed for {strategy_id}: {str(e)}")
            return False
    
    def get_deployment_status(self) -> Dict[str, dict]:
        """
        Get current deployment status for all active strategies.
        """
        status = {}
        with self.lock:
            for strategy_id, instance in self.active_strategies.items():
                status[strategy_id] = {
                    "status": instance.status,
                    "deployment_id": instance.deployment_id,
                    "start_time": instance.shutdown_timestamp if hasattr(instance, 'shutdown_timestamp') else "active",
                    "current_pnl": self.performance_history[strategy_id][-1].pnl if self.performance_history[strategy_id] else 0,
                    "risk_level": self._assess_risk_level(strategy_id)
                }
        return status
    
    def _assess_risk_level(self, strategy_id: str) -> str:
        """
        Assess current risk level of a deployed strategy.
        """
        if strategy_id not in self.performance_history or not self.performance_history[strategy_id]:
            return "unknown"
            
        latest_perf = self.performance_history[strategy_id][-1]
        
        # Simple risk assessment
        if latest_perf.max_drawdown > 0.15:  # 15% drawdown
            return "high"
        elif latest_perf.max_drawdown > 0.10:  # 10% drawdown
            return "medium"
        elif latest_perf.pnl > 0.05:  # 5% profit
            return "low"
        else:
            return "moderate"
    
    def emergency_system_shutdown(self):
        """
        Emergency shutdown of entire live deployment system.
        """
        self.emergency_stop = True
        self.system_health = "emergency"
        
        self.logger.critical("EMERGENCY SYSTEM SHUTDOWN INITIATED")
        
        # Shutdown all active strategies
        for strategy_id in list(self.active_strategies.keys()):
            self._trigger_emergency_shutdown(strategy_id, "System-wide emergency shutdown")
        
        # Log system event
        self.archiver.log_event("system_emergency", {
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Complete system shutdown initiated",
            "active_strategies": len(self.active_strategies)
        })
    
    def get_system_health(self) -> dict:
        """
        Get overall system health status.
        """
        return {
            "system_status": self.system_health,
            "active_strategies": len(self.active_strategies),
            "emergency_stop_active": self.emergency_stop,
            "capacity_usage": len(self.active_strategies) / self.config.max_concurrent_strategies,
            "last_check": datetime.utcnow().isoformat()
        }
