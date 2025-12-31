"""
Monitoring System Module

Real-time monitoring, alerting, and performance tracking for deployed strategies.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import threading
import queue
from enum import Enum

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Alert types"""
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    STRATEGY = "strategy"

@dataclass
class Alert:
    """Data class representing an alert"""
    alert_id: str
    deployment_id: str
    strategy_id: str
    alert_level: AlertLevel
    alert_type: AlertType
    message: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetric:
    """Data class representing a performance metric"""
    deployment_id: str
    strategy_id: str
    metric_name: str
    value: float
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class MonitoringSystem:
    """Real-time monitoring and alerting system"""
    
    def __init__(self):
        self.logger = logging.getLogger('MonitoringSystem')
        self.alerts = []
        self.performance_metrics = []
        self.active_deployments = set()
        self.monitoring_threads = {}
        self.alert_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.running = False
        
    def start(self):
        """Start the monitoring system"""
        if self.running:
            self.logger.warning("Monitoring system is already running")
            return
            
        self.running = True
        self.logger.info("Starting monitoring system")
        
        # Start alert processing thread
        self.alert_processor_thread = threading.Thread(
            target=self._process_alerts,
            daemon=True
        )
        self.alert_processor_thread.start()
        
        # Start metrics processing thread
        self.metrics_processor_thread = threading.Thread(
            target=self._process_metrics,
            daemon=True
        )
        self.metrics_processor_thread.start()
        
    def stop(self):
        """Stop the monitoring system"""
        if not self.running:
            self.logger.warning("Monitoring system is not running")
            return
            
        self.running = False
        self.logger.info("Stopping monitoring system")
        
        # Stop all deployment monitoring threads
        for deployment_id, thread in self.monitoring_threads.items():
            thread.stop()
            thread.join()
            
        self.monitoring_threads.clear()
        
    def register_deployment(self, deployment_id: str, strategy_id: str):
        """Register a new deployment for monitoring"""
        if deployment_id in self.active_deployments:
            self.logger.warning(f"Deployment {deployment_id} already registered")
            return
            
        self.active_deployments.add(deployment_id)
        self.logger.info(f"Registered deployment {deployment_id} for monitoring")
        
        # Start monitoring thread for this deployment
        monitoring_thread = DeploymentMonitorThread(
            deployment_id=deployment_id,
            strategy_id=strategy_id,
            alert_queue=self.alert_queue,
            metrics_queue=self.metrics_queue
        )
        monitoring_thread.start()
        self.monitoring_threads[deployment_id] = monitoring_thread
        
    def unregister_deployment(self, deployment_id: str):
        """Unregister a deployment from monitoring"""
        if deployment_id not in self.active_deployments:
            self.logger.warning(f"Deployment {deployment_id} not found")
            return
            
        self.active_deployments.remove(deployment_id)
        
        # Stop monitoring thread for this deployment
        if deployment_id in self.monitoring_threads:
            monitoring_thread = self.monitoring_threads[deployment_id]
            monitoring_thread.stop()
            monitoring_thread.join()
            del self.monitoring_threads[deployment_id]
            
        self.logger.info(f"Unregistered deployment {deployment_id} from monitoring")
        
    def get_alerts(self, deployment_id: Optional[str] = None) -> List[Alert]:
        """Get alerts for a specific deployment or all deployments"""
        if deployment_id:
            return [alert for alert in self.alerts if alert.deployment_id == deployment_id]
        return self.alerts
        
    def get_performance_metrics(self, deployment_id: Optional[str] = None) -> List[PerformanceMetric]:
        """Get performance metrics for a specific deployment or all deployments"""
        if deployment_id:
            return [metric for metric in self.performance_metrics if metric.deployment_id == deployment_id]
        return self.performance_metrics
        
    def create_alert(self, alert: Alert):
        """Create a new alert"""
        self.alert_queue.put(alert)
        
    def add_performance_metric(self, metric: PerformanceMetric):
        """Add a new performance metric"""
        self.metrics_queue.put(metric)
        
    def _process_alerts(self):
        """Process alerts from the queue"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                self.alerts.append(alert)
                
                # Log alert based on severity
                if alert.alert_level == AlertLevel.CRITICAL or alert.alert_level == AlertLevel.EMERGENCY:
                    self.logger.error(f"ALERT: {alert.message}")
                elif alert.alert_level == AlertLevel.WARNING:
                    self.logger.warning(f"ALERT: {alert.message}")
                else:
                    self.logger.info(f"ALERT: {alert.message}")
                    
                self.alert_queue.task_done()
            except queue.Empty:
                continue
                
    def _process_metrics(self):
        """Process performance metrics from the queue"""
        while self.running:
            try:
                metric = self.metrics_queue.get(timeout=1.0)
                self.performance_metrics.append(metric)
                self.logger.debug(f"METRIC: {metric.metric_name} = {metric.value}")
                self.metrics_queue.task_done()
            except queue.Empty:
                continue

class DeploymentMonitorThread(threading.Thread):
    """Thread for monitoring a specific deployment"""
    
    def __init__(self, deployment_id: str, strategy_id: str, 
                 alert_queue: queue.Queue, metrics_queue: queue.Queue):
        super().__init__()
        self.deployment_id = deployment_id
        self.strategy_id = strategy_id
        self.alert_queue = alert_queue
        self.metrics_queue = metrics_queue
        self.running = False
        self.logger = logging.getLogger(f'DeploymentMonitor.{deployment_id}')
        
    def run(self):
        """Main monitoring loop"""
        self.running = True
        self.logger.info(f"Starting monitoring for deployment {self.deployment_id}")
        
        while self.running:
            try:
                # Simulate monitoring cycle
                self._monitor_deployment()
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring thread: {e}")
                time.sleep(10)  # Wait before retrying
                
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        
    def _monitor_deployment(self):
        """Perform monitoring for the deployment"""
        # Simulate performance metrics
        timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Generate simulated metrics
        metrics = [
            PerformanceMetric(
                deployment_id=self.deployment_id,
                strategy_id=self.strategy_id,
                metric_name='pnl',
                value=self._simulate_pnl(),
                timestamp=timestamp
            ),
            PerformanceMetric(
                deployment_id=self.deployment_id,
                strategy_id=self.strategy_id,
                metric_name='drawdown',
                value=self._simulate_drawdown(),
                timestamp=timestamp
            ),
            PerformanceMetric(
                deployment_id=self.deployment_id,
                strategy_id=self.strategy_id,
                metric_name='win_rate',
                value=self._simulate_win_rate(),
                timestamp=timestamp
            )
        ]
        
        # Add metrics to queue
        for metric in metrics:
            self.metrics_queue.put(metric)
            
        # Check for alert conditions
        self._check_alert_conditions(metrics)
        
    def _simulate_pnl(self) -> float:
        """Simulate PnL for the deployment"""
        # Simulate PnL based on deployment ID for variety
        base_pnl = hash(self.deployment_id) % 1000 - 500
        return base_pnl + (hash(str(time.time())) % 100 - 50)
        
    def _simulate_drawdown(self) -> float:
        """Simulate drawdown for the deployment"""
        # Simulate drawdown
        return abs(hash(self.deployment_id + str(time.time())) % 200) / 10.0
        
    def _simulate_win_rate(self) -> float:
        """Simulate win rate for the deployment"""
        # Simulate win rate between 0.4 and 0.8
        return 0.4 + (hash(self.deployment_id) % 40) / 100.0
        
    def _check_alert_conditions(self, metrics: List[PerformanceMetric]):
        """Check for alert conditions based on metrics"""
        metric_dict = {metric.metric_name: metric.value for metric in metrics}
        
        # Check for critical drawdown
        if metric_dict.get('drawdown', 0) > 15.0:
            alert = Alert(
                alert_id=f"alert_{len(self.alert_queue.queue) + 1}",
                deployment_id=self.deployment_id,
                strategy_id=self.strategy_id,
                alert_level=AlertLevel.CRITICAL,
                alert_type=AlertType.RISK,
                message=f"High drawdown detected: {metric_dict['drawdown']:.2f}%",
                timestamp=datetime.utcnow().isoformat() + 'Z',
                metadata={'drawdown': metric_dict['drawdown']}
            )
            self.alert_queue.put(alert)
            
        # Check for negative PnL
        elif metric_dict.get('pnl', 0) < -200:
            alert = Alert(
                alert_id=f"alert_{len(self.alert_queue.queue) + 1}",
                deployment_id=self.deployment_id,
                strategy_id=self.strategy_id,
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.PERFORMANCE,
                message=f"Negative PnL detected: {metric_dict['pnl']:.2f}",
                timestamp=datetime.utcnow().isoformat() + 'Z',
                metadata={'pnl': metric_dict['pnl']}
            )
            self.alert_queue.put(alert)
            
        # Check for low win rate
        elif metric_dict.get('win_rate', 0) < 0.45:
            alert = Alert(
                alert_id=f"alert_{len(self.alert_queue.queue) + 1}",
                deployment_id=self.deployment_id,
                strategy_id=self.strategy_id,
                alert_level=AlertLevel.WARNING,
                alert_type=AlertType.STRATEGY,
                message=f"Low win rate detected: {metric_dict['win_rate']:.2f}",
                timestamp=datetime.utcnow().isoformat() + 'Z',
                metadata={'win_rate': metric_dict['win_rate']}
            )
            self.alert_queue.put(alert)