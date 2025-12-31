"""
Deployment Manager Module

Handles the deployment of selected strategies to live trading environments.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import deployment components
from .brokers.broker_interface import BrokerFactory, Order, Position, AccountBalance
from .brokers.broker_interface import SimulatedBroker
from .monitoring.monitoring_system import MonitoringSystem, Alert, PerformanceMetric
from .risk_management import RiskManager, PerformanceTracker, RiskParameters
from .lifecycle_manager import StrategyLifecycleManager, StrategyDeployment

class DeploymentManager:
    """Main class for deploying trading strategies to live environments"""
     
    def __init__(self):
        self.logger = logging.getLogger('DeploymentManager')
        self.logger.info("DeploymentManager initialized")
        
        # Initialize deployment components
        self.broker_factory = BrokerFactory()
        self.monitoring_system = MonitoringSystem()
        self.risk_manager = RiskManager()
        self.performance_tracker = PerformanceTracker()
        self.lifecycle_manager = StrategyLifecycleManager()
        
        # Deployment state
        self.active_deployments = {}  # deployment_id -> deployment_info
        self.broker_connections = {}  # deployment_id -> broker_instance
        
        # Start monitoring system
        self.monitoring_system.start()
         
    def deploy_strategy(self, strategy: Dict[str, Any], deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deploy a strategy to a live trading environment
        
        Args:
            strategy: Strategy dictionary to deploy
            deployment_config: Configuration for the deployment
             
        Returns:
            Dictionary containing deployment results and status
        """
        self.logger.info(f"Deploying strategy: {strategy.get('template', 'unknown')}")
        
        try:
            # Step 1: Create deployment lifecycle record
            deployment = self.lifecycle_manager.create_deployment(strategy, deployment_config)
            
            # Step 2: Create broker connection
            broker_type = deployment_config.get('broker_type', 'simulated')
            broker_config = deployment_config.get('broker_config', {})
            
            broker = self.broker_factory.create_broker(broker_type, broker_config)
            broker.connect()
            
            # Store broker connection
            self.broker_connections[deployment.deployment_id] = broker
            
            # Step 3: Register with monitoring system
            self.monitoring_system.register_deployment(
                deployment.deployment_id,
                deployment.strategy_id
            )
            
            # Step 4: Activate the strategy
            self.lifecycle_manager.deploy_strategy(deployment.deployment_id)
            self.lifecycle_manager.activate_strategy(deployment.deployment_id)
            
            # Step 5: Set baseline metrics for drift detection
            baseline_metrics = strategy.get('backtest_results', {})
            self.performance_tracker.set_baseline_metrics(
                deployment.deployment_id,
                deployment.strategy_id,
                baseline_metrics
            )
            
            # Store deployment information
            self.active_deployments[deployment.deployment_id] = {
                'strategy': strategy,
                'deployment_config': deployment_config,
                'deployment': deployment,
                'broker': broker,
                'status': 'active'
            }
            
            # Enhanced deployment result structure
            deployment_result = {
                'deployment_id': deployment.deployment_id,
                'strategy_id': strategy.get('id', 'unknown'),
                'template': strategy.get('template', 'unknown'),
                'status': 'deployed',
                'broker_type': broker_type,
                'deployment_timestamp': deployment.deployment_timestamp,
                'initial_balance': deployment.initial_balance,
                'risk_parameters': deployment.risk_parameters,
                'metadata': {
                    'deployed_by': 'DeploymentManager',
                    'version': '2.0',
                    'components': {
                        'broker': broker_type,
                        'monitoring': 'active',
                        'risk_management': 'active',
                        'lifecycle_management': 'active'
                    }
                }
            }
            
            self.logger.info(f"Strategy deployed successfully: {deployment_result}")
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Failed to deploy strategy: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'strategy_id': strategy.get('id', 'unknown'),
                'template': strategy.get('template', 'unknown')
            }
       
    def monitor_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Monitor a deployed strategy
        
        Args:
            deployment_id: ID of the deployment to monitor
             
        Returns:
            Dictionary containing monitoring results and status
        """
        self.logger.info(f"Monitoring deployment: {deployment_id}")
        
        if deployment_id not in self.active_deployments:
            return {
                'deployment_id': deployment_id,
                'status': 'not_found',
                'error': 'Deployment not found'
            }
            
        deployment_info = self.active_deployments[deployment_id]
        broker = self.broker_connections.get(deployment_id)
        
        try:
            # Get current account balance
            account_balance = broker.get_account_balance()
            
            # Get open positions
            open_positions = broker.get_open_positions()
            
            # Get performance metrics from monitoring system
            performance_metrics = self.monitoring_system.get_performance_metrics(deployment_id)
            
            # Get alerts
            alerts = self.monitoring_system.get_alerts(deployment_id)
            
            # Calculate portfolio risk
            position_dicts = [{
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl
            } for pos in open_positions]
            
            portfolio_risk = self.risk_manager.calculate_portfolio_risk(
                position_dicts,
                account_balance.total_balance
            )
            
            # Check risk limits
            risk_check = self.risk_manager.check_risk_limits(
                account_balance.total_balance,
                portfolio_risk.get('unrealized_pnl_percentage', 0) * -1  # Convert to positive drawdown
            )
            
            # Enhanced monitoring result structure
            monitoring_result = {
                'deployment_id': deployment_id,
                'status': 'active',
                'account_balance': {
                    'total_balance': account_balance.total_balance,
                    'available_balance': account_balance.available_balance,
                    'margin_used': account_balance.margin_used,
                    'margin_available': account_balance.margin_available,
                    'currency': account_balance.currency
                },
                'open_positions': [{
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'strategy_id': pos.strategy_id
                } for pos in open_positions],
                'performance_metrics': [{
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'timestamp': metric.timestamp
                } for metric in performance_metrics],
                'portfolio_risk': portfolio_risk,
                'risk_check': risk_check,
                'alerts': [{
                    'alert_id': alert.alert_id,
                    'alert_level': alert.alert_level.value,
                    'alert_type': alert.alert_type.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp
                } for alert in alerts],
                'metadata': {
                    'monitored_by': 'DeploymentManager',
                    'version': '2.0',
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }
            }
            
            self.logger.debug(f"Monitoring result: {monitoring_result}")
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"Error monitoring deployment {deployment_id}: {e}")
            return {
                'deployment_id': deployment_id,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    def execute_trade(self, deployment_id: str, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade for a deployed strategy
        
        Args:
            deployment_id: ID of the deployment
            order: Order dictionary containing trade details
             
        Returns:
            Dictionary containing execution results
        """
        self.logger.info(f"Executing trade for deployment {deployment_id}")
        
        if deployment_id not in self.active_deployments:
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': 'Deployment not found'
            }
            
        try:
            deployment_info = self.active_deployments[deployment_id]
            broker = self.broker_connections.get(deployment_id)
            
            # Get current account balance for risk management
            account_balance = broker.get_account_balance()
            
            # Calculate position size using risk manager
            position_size = self.risk_manager.calculate_position_size(
                account_balance.total_balance,
                order['symbol'],
                order['price']
            )
            
            # Create order object
            order_obj = Order(
                order_id=f"order_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                symbol=order['symbol'],
                order_type=order.get('order_type', 'market'),
                quantity=position_size.quantity,
                price=order.get('price'),
                side=order['side'],
                status='pending',
                strategy_id=deployment_info['strategy'].get('id', 'unknown')
            )
            
            # Execute order through broker
            execution_result = broker.place_order(order_obj)
            
            # Log trade execution
            self.logger.info(f"Trade executed: {execution_result}")
            
            return {
                'deployment_id': deployment_id,
                'order_id': order_obj.order_id,
                'status': execution_result.get('status', 'unknown'),
                'execution_price': execution_result.get('execution_price'),
                'quantity': position_size.quantity,
                'risk_amount': position_size.risk_amount,
                'stop_loss_price': position_size.stop_loss_price,
                'take_profit_price': position_size.take_profit_price,
                'risk_reward_ratio': position_size.risk_reward_ratio,
                'metadata': {
                    'executed_by': 'DeploymentManager',
                    'version': '2.0',
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute trade: {e}")
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get status of a deployment
        
        Args:
            deployment_id: ID of the deployment
             
        Returns:
            Dictionary containing deployment status
        """
        if deployment_id not in self.active_deployments:
            return {
                'deployment_id': deployment_id,
                'status': 'not_found',
                'error': 'Deployment not found'
            }
            
        deployment_info = self.active_deployments[deployment_id]
        deployment = deployment_info['deployment']
        
        return {
            'deployment_id': deployment_id,
            'strategy_id': deployment.strategy_id,
            'strategy_name': deployment.strategy_name,
            'status': deployment.status.value,
            'broker_type': deployment.broker_type,
            'deployment_timestamp': deployment.deployment_timestamp,
            'initial_balance': deployment.initial_balance,
            'risk_parameters': deployment.risk_parameters,
            'metadata': {
                'version': '2.0',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        }
    
    def get_all_deployments(self) -> List[Dict[str, Any]]:
        """
        Get information about all active deployments
        
        Returns:
            List of deployment information dictionaries
        """
        return [self.get_deployment_status(deployment_id)
               for deployment_id in self.active_deployments.keys()]
    
    def stop_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Stop a deployment
        
        Args:
            deployment_id: ID of the deployment to stop
             
        Returns:
            Dictionary containing stop results
        """
        self.logger.info(f"Stopping deployment: {deployment_id}")
        
        if deployment_id not in self.active_deployments:
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': 'Deployment not found'
            }
            
        try:
            # Stop lifecycle
            self.lifecycle_manager.stop_strategy(deployment_id)
            
            # Unregister from monitoring
            self.monitoring_system.unregister_deployment(deployment_id)
            
            # Disconnect broker
            if deployment_id in self.broker_connections:
                broker = self.broker_connections[deployment_id]
                broker.disconnect()
                del self.broker_connections[deployment_id]
                
            # Remove from active deployments
            del self.active_deployments[deployment_id]
            
            return {
                'deployment_id': deployment_id,
                'status': 'stopped',
                'message': 'Deployment stopped successfully',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop deployment: {e}")
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    def retire_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """
        Retire a deployment
        
        Args:
            deployment_id: ID of the deployment to retire
             
        Returns:
            Dictionary containing retirement results
        """
        self.logger.info(f"Retiring deployment: {deployment_id}")
        
        if deployment_id not in self.active_deployments:
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': 'Deployment not found'
            }
            
        try:
            # Stop deployment first
            stop_result = self.stop_deployment(deployment_id)
            
            if stop_result['status'] != 'stopped':
                return stop_result
                
            # Retire lifecycle
            self.lifecycle_manager.retire_strategy(deployment_id)
            
            return {
                'deployment_id': deployment_id,
                'status': 'retired',
                'message': 'Deployment retired successfully',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retire deployment: {e}")
            return {
                'deployment_id': deployment_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
    
    def get_performance_report(self, deployment_id: str) -> Dict[str, Any]:
        """
        Get performance report for a deployment
        
        Args:
            deployment_id: ID of the deployment
             
        Returns:
            Dictionary containing performance report
        """
        if deployment_id not in self.active_deployments:
            return {
                'deployment_id': deployment_id,
                'status': 'not_found',
                'error': 'Deployment not found'
            }
            
        deployment_info = self.active_deployments[deployment_id]
        
        # Get performance history
        performance_history = self.performance_tracker.get_performance_history(deployment_id)
        
        # Get rolling performance
        rolling_performance = self.performance_tracker.calculate_rolling_performance(deployment_id)
        
        # Get current monitoring data
        monitoring_data = self.monitor_deployment(deployment_id)
        
        return {
            'deployment_id': deployment_id,
            'strategy_id': deployment_info['strategy'].get('id', 'unknown'),
            'strategy_name': deployment_info['strategy'].get('name', 'unknown'),
            'performance_history': performance_history,
            'rolling_performance': rolling_performance,
            'current_metrics': monitoring_data.get('performance_metrics', []),
            'risk_metrics': monitoring_data.get('portfolio_risk', {}),
            'alerts': monitoring_data.get('alerts', []),
            'metadata': {
                'generated_by': 'DeploymentManager',
                'version': '2.0',
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        }
    
    def shutdown(self):
        """
        Shutdown the deployment manager and clean up resources
        """
        self.logger.info("Shutting down DeploymentManager")
        
        # Stop all active deployments
        for deployment_id in list(self.active_deployments.keys()):
            self.stop_deployment(deployment_id)
            
        # Stop monitoring system
        self.monitoring_system.stop()
        
        self.logger.info("DeploymentManager shutdown complete")