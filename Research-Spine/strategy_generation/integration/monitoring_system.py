"""
LLM Integration Monitoring System

Comprehensive monitoring and metrics tracking for LLM integration,
including performance metrics, health checks, and alerting.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics for LLM operations"""
    
    operation: str
    component: str
    timestamp: datetime
    execution_time: float
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'operation': self.operation,
            'component': self.component,
            'timestamp': self.timestamp.isoformat(),
            'execution_time': self.execution_time,
            'success': self.success,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'cost': self.cost,
            'error_message': self.error_message,
            'retry_count': self.retry_count
        }


@dataclass
class HealthStatus:
    """Health status snapshot"""
    
    timestamp: datetime
    llm_operational: bool
    health_score: float
    success_rate: float
    avg_latency: float
    failure_count: int
    total_requests: int
    circuit_breaker_state: str
    active_components: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'llm_operational': self.llm_operational,
            'health_score': self.health_score,
            'success_rate': self.success_rate,
            'avg_latency': self.avg_latency,
            'failure_count': self.failure_count,
            'total_requests': self.total_requests,
            'circuit_breaker_state': self.circuit_breaker_state,
            'active_components': self.active_components
        }


class MetricsCollector:
    """Collects and manages performance metrics"""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to keep metrics in memory
        """
        self.logger = logging.getLogger('MetricsCollector')
        self.retention_hours = retention_hours
        self.metrics: List[PerformanceMetrics] = []
        self.lock = threading.Lock()
        
        # Aggregated metrics
        self.operation_stats = defaultdict(lambda: {
            'total': 0,
            'success': 0,
            'failed': 0,
            'avg_latency': 0.0,
            'total_latency': 0.0,
            'total_tokens': 0,
            'total_cost': 0.0
        })
        
        self.logger.info(f"Metrics collector initialized with {retention_hours}h retention")
    
    def record_metric(self, metric: PerformanceMetrics) -> None:
        """Record a performance metric"""
        with self.lock:
            self.metrics.append(metric)
            
            # Update aggregated stats
            stats = self.operation_stats[metric.operation]
            stats['total'] += 1
            stats['total_latency'] += metric.execution_time
            
            if metric.success:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            stats['total_tokens'] += metric.total_tokens
            stats['total_cost'] += metric.cost
            
            # Calculate average latency
            if stats['total'] > 0:
                stats['avg_latency'] = stats['total_latency'] / stats['total']
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def get_metrics(self, operation: Optional[str] = None, 
                   last_n: Optional[int] = None) -> List[PerformanceMetrics]:
        """Get metrics filtered by operation and/or last N entries"""
        with self.lock:
            filtered = self.metrics
            
            if operation:
                filtered = [m for m in filtered if m.operation == operation]
            
            if last_n:
                filtered = filtered[-last_n:]
            
            return filtered.copy()
    
    def get_operation_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated statistics for operations"""
        with self.lock:
            if operation:
                return dict(self.operation_stats[operation])
            else:
                return dict(self.operation_stats)
    
    def get_success_rate(self, operation: Optional[str] = None) -> float:
        """Get success rate for operations"""
        stats = self.get_operation_stats(operation)
        
        if operation:
            if stats['total'] == 0:
                return 0.0
            return stats['success'] / stats['total']
        else:
            # Overall success rate
            total_success = sum(s['success'] for s in stats.values())
            total_total = sum(s['total'] for s in stats.values())
            return total_success / total_total if total_total > 0 else 0.0
    
    def get_average_latency(self, operation: Optional[str] = None) -> float:
        """Get average latency for operations"""
        stats = self.get_operation_stats(operation)
        
        if operation:
            return stats['avg_latency']
        else:
            # Weighted average
            total_latency = sum(s['total_latency'] for s in stats.values())
            total_total = sum(s['total'] for s in stats.values())
            return total_latency / total_total if total_total > 0 else 0.0
    
    def get_total_cost(self, operation: Optional[str] = None) -> float:
        """Get total cost for operations"""
        stats = self.get_operation_stats(operation)
        
        if operation:
            return stats['total_cost']
        else:
            return sum(s['total_cost'] for s in stats.values())
    
    def get_throughput(self, operation: Optional[str] = None, 
                      window_minutes: int = 60) -> float:
        """Get throughput (requests per minute)"""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        with self.lock:
            recent_metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            
            if operation:
                recent_metrics = [m for m in recent_metrics if m.operation == operation]
            
            return len(recent_metrics) / window_minutes
    
    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            initial_count = len(self.metrics)
            self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]
            removed_count = initial_count - len(self.metrics)
            
            if removed_count > 0:
                self.logger.debug(f"Cleaned up {removed_count} old metrics")
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to JSON file"""
        try:
            with self.lock:
                metrics_data = [m.to_dict() for m in self.metrics]
            
            with open(filepath, 'w') as f:
                json.dump({
                    'export_time': datetime.now().isoformat(),
                    'metrics': metrics_data,
                    'aggregated_stats': dict(self.operation_stats)
                }, f, indent=2)
            
            self.logger.info(f"Exported {len(metrics_data)} metrics to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            return False


class HealthMonitor:
    """Monitors LLM system health and provides alerts"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize health monitor
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.logger = logging.getLogger('HealthMonitor')
        self.metrics_collector = metrics_collector
        
        # Health thresholds
        self.min_success_rate = 0.7
        self.max_avg_latency = 5.0
        self.max_consecutive_failures = 3
        
        # Circuit breaker state
        self.circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.circuit_breaker_open_time: Optional[datetime] = None
        self.consecutive_failures = 0
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        self.logger.info("Health monitor initialized")
    
    def check_health(self) -> HealthStatus:
        """Check current health status"""
        success_rate = self.metrics_collector.get_success_rate()
        avg_latency = self.metrics_collector.get_average_latency()
        
        # Calculate health score
        health_score = 1.0
        
        # Penalize for low success rate
        if success_rate < self.min_success_rate:
            health_score -= (self.min_success_rate - success_rate) * 2
        
        # Penalize for high latency
        if avg_latency > self.max_avg_latency:
            health_score -= (avg_latency - self.max_avg_latency) * 0.2
        
        # Penalize for circuit breaker issues
        if self.circuit_breaker_state == "OPEN":
            health_score -= 0.5
        elif self.circuit_breaker_state == "HALF_OPEN":
            health_score -= 0.2
        
        health_score = max(0.0, min(1.0, health_score))
        
        # Determine operational status
        llm_operational = (health_score > 0.6 and 
                          success_rate > self.min_success_rate and
                          self.circuit_breaker_state != "OPEN")
        
        # Get failure count from recent metrics
        recent_metrics = self.metrics_collector.get_metrics(last_n=10)
        failure_count = sum(1 for m in recent_metrics if not m.success)
        
        # Get total requests
        total_requests = sum(s['total'] for s in self.metrics_collector.get_operation_stats().values())
        
        # Get active components
        active_components = []
        if llm_operational:
            active_components.extend(['llm_client', 'strategy_agent', 'validation_agent'])
            if self.metrics_collector.get_total_cost() > 0:
                active_components.append('feedback_agent')
        
        return HealthStatus(
            timestamp=datetime.now(),
            llm_operational=llm_operational,
            health_score=health_score,
            success_rate=success_rate,
            avg_latency=avg_latency,
            failure_count=failure_count,
            total_requests=total_requests,
            circuit_breaker_state=self.circuit_breaker_state,
            active_components=active_components
        )
    
    def record_operation_result(self, success: bool, latency: float, 
                              operation: str = "unknown") -> None:
        """Record operation result for health monitoring"""
        # Update circuit breaker logic
        if not success:
            self.consecutive_failures += 1
            self.logger.warning(f"Operation failed: {operation} (consecutive: {self.consecutive_failures})")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                if self.circuit_breaker_state == "CLOSED":
                    self._open_circuit_breaker()
                elif self.circuit_breaker_state == "HALF_OPEN":
                    # Failed in half-open state, reopen circuit
                    self._open_circuit_breaker()
        else:
            if self.circuit_breaker_state == "HALF_OPEN":
                # Success in half-open state, close circuit
                self._close_circuit_breaker()
            elif self.circuit_breaker_state == "CLOSED":
                self.consecutive_failures = 0
        
        # Check for alerts
        self._check_alerts()
    
    def _open_circuit_breaker(self) -> None:
        """Open circuit breaker"""
        self.circuit_breaker_state = "OPEN"
        self.circuit_breaker_open_time = datetime.now()
        self.logger.error(f"🚨 CIRCUIT BREAKER OPENED - Too many failures ({self.consecutive_failures})")
        self._trigger_alert("circuit_breaker_opened", {
            "consecutive_failures": self.consecutive_failures,
            "timestamp": datetime.now().isoformat()
        })
    
    def _close_circuit_breaker(self) -> None:
        """Close circuit breaker"""
        self.circuit_breaker_state = "CLOSED"
        self.consecutive_failures = 0
        self.circuit_breaker_open_time = None
        self.logger.info("✅ Circuit breaker closed - system recovered")
        self._trigger_alert("circuit_breaker_closed", {
            "timestamp": datetime.now().isoformat()
        })
    
    def attempt_reset(self) -> bool:
        """Attempt to reset circuit breaker (half-open state)"""
        if self.circuit_breaker_state == "OPEN":
            if self.circuit_breaker_open_time:
                time_since_open = (datetime.now() - self.circuit_breaker_open_time).total_seconds()
                if time_since_open > 60:  # 1 minute timeout
                    self.circuit_breaker_state = "HALF_OPEN"
                    self.logger.info("🔄 Circuit breaker entering HALF_OPEN state")
                    return True
        return False
    
    def register_alert_callback(self, callback: Callable) -> None:
        """Register a callback for alerts"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self) -> None:
        """Check for conditions that trigger alerts"""
        health = self.check_health()
        
        # Alert on low health score
        if health.health_score < 0.5:
            self._trigger_alert("low_health_score", {
                "health_score": health.health_score,
                "timestamp": health.timestamp.isoformat()
            })
        
        # Alert on high latency
        if health.avg_latency > self.max_avg_latency:
            self._trigger_alert("high_latency", {
                "avg_latency": health.avg_latency,
                "threshold": self.max_avg_latency
            })
        
        # Alert on low success rate
        if health.success_rate < self.min_success_rate:
            self._trigger_alert("low_success_rate", {
                "success_rate": health.success_rate,
                "threshold": self.min_success_rate
            })
    
    def _trigger_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """Trigger alert callbacks"""
        alert_data = {
            'alert_type': alert_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {str(e)}")


class MonitoringSystem:
    """Complete monitoring system combining metrics and health monitoring"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize monitoring system
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger('MonitoringSystem')
        
        # Configuration
        self.config = config or {}
        self.retention_hours = self.config.get('retention_hours', 24)
        self.export_interval = self.config.get('export_interval', 3600)  # 1 hour
        self.export_dir = Path(self.config.get('export_dir', 'monitoring_exports'))
        
        # Create export directory
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.metrics_collector = MetricsCollector(retention_hours=self.retention_hours)
        self.health_monitor = HealthMonitor(self.metrics_collector)
        
        # Export timer
        self.last_export_time = time.time()
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        self.logger.info("Monitoring system initialized")
    
    def record_llm_operation(self, operation: str, component: str, 
                           success: bool, execution_time: float,
                           input_tokens: int = 0, output_tokens: int = 0,
                           cost: float = 0.0, error_message: Optional[str] = None,
                           retry_count: int = 0) -> None:
        """Record an LLM operation"""
        metric = PerformanceMetrics(
            operation=operation,
            component=component,
            timestamp=datetime.now(),
            execution_time=execution_time,
            success=success,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            error_message=error_message,
            retry_count=retry_count
        )
        
        self.metrics_collector.record_metric(metric)
        self.health_monitor.record_operation_result(success, execution_time, operation)
        
        # Check if we should auto-export
        if time.time() - self.last_export_time > self.export_interval:
            self.auto_export()
    
    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        return self.health_monitor.check_health()
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance summary"""
        stats = self.metrics_collector.get_operation_stats(operation)
        success_rate = self.metrics_collector.get_success_rate(operation)
        avg_latency = self.metrics_collector.get_average_latency(operation)
        throughput = self.metrics_collector.get_throughput(operation)
        total_cost = self.metrics_collector.get_total_cost(operation)
        
        summary = {
            'operation': operation or 'all',
            'success_rate': success_rate,
            'avg_latency': avg_latency,
            'throughput_rpm': throughput,
            'total_cost': total_cost,
            'stats': stats
        }
        
        return summary
    
    def register_alert_handler(self, handler: Callable) -> None:
        """Register an alert handler"""
        self.alert_handlers.append(handler)
        self.health_monitor.register_alert_callback(handler)
    
    def export_metrics(self, filepath: Optional[str] = None) -> bool:
        """Export metrics to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.export_dir / f"metrics_{timestamp}.json"
        
        success = self.metrics_collector.export_metrics(str(filepath))
        if success:
            self.last_export_time = time.time()
        return success
    
    def auto_export(self) -> None:
        """Automatically export metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.export_dir / f"metrics_auto_{timestamp}.json"
        
        if self.export_metrics(str(filepath)):
            self.logger.info(f"Auto-exported metrics to {filepath}")
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        health = self.get_health_status()
        return {
            'state': health.circuit_breaker_state,
            'consecutive_failures': health.failure_count,
            'can_attempt_reset': self.health_monitor.attempt_reset()
        }
    
    def attempt_circuit_breaker_reset(self) -> bool:
        """Attempt to reset circuit breaker"""
        return self.health_monitor.attempt_reset()
    
    def get_recent_metrics(self, last_n: int = 10, 
                          operation: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent metrics"""
        metrics = self.metrics_collector.get_metrics(operation=operation, last_n=last_n)
        return [m.to_dict() for m in metrics]
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        health = self.get_health_status()
        performance = self.get_performance_summary()
        circuit_breaker = self.get_circuit_breaker_status()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_status': health.to_dict(),
            'performance_summary': performance,
            'circuit_breaker': circuit_breaker,
            'recommendations': self._generate_recommendations(health, performance)
        }
    
    def _generate_recommendations(self, health: HealthStatus, 
                                 performance: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on current state"""
        recommendations = []
        
        if health.health_score < 0.7:
            recommendations.append("Consider increasing LLM timeout or reducing request rate")
        
        if performance['success_rate'] < 0.8:
            recommendations.append("Review LLM model selection and prompt engineering")
        
        if performance['avg_latency'] > 3.0:
            recommendations.append("Optimize prompts or consider model with lower latency")
        
        if health.circuit_breaker_state == "OPEN":
            recommendations.append("Circuit breaker is open - wait for recovery period")
        
        if performance['total_cost'] > 100.0:
            recommendations.append("High costs detected - review usage patterns")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations
    
    def register_health_check_callback(self, callback: Callable, 
                                     interval_seconds: int = 60) -> None:
        """
        Register a health check callback that runs periodically
        
        Args:
            callback: Function to call with health status
            interval_seconds: How often to run the check
        """
        def health_check_worker():
            while True:
                try:
                    health = self.get_health_status()
                    callback(health)
                    time.sleep(interval_seconds)
                except Exception as e:
                    self.logger.error(f"Health check callback failed: {str(e)}")
                    break
        
        thread = threading.Thread(target=health_check_worker, daemon=True)
        thread.start()
        self.logger.info(f"Registered health check callback with {interval_seconds}s interval")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Get all metrics from the period
        all_metrics = self.metrics_collector.get_metrics()
        period_metrics = [m for m in all_metrics if m.timestamp >= cutoff_time]
        
        if not period_metrics:
            return {'error': 'No metrics in specified period'}
        
        # Calculate summary statistics
        total_operations = len(period_metrics)
        successful_operations = sum(1 for m in period_metrics if m.success)
        total_execution_time = sum(m.execution_time for m in period_metrics)
        total_tokens = sum(m.total_tokens for m in period_metrics)
        total_cost = sum(m.cost for m in period_metrics)
        
        # Group by operation type
        by_operation = defaultdict(lambda: {'count': 0, 'success': 0, 'latency': []})
        for metric in period_metrics:
            op_stats = by_operation[metric.operation]
            op_stats['count'] += 1
            if metric.success:
                op_stats['success'] += 1
            op_stats['latency'].append(metric.execution_time)
        
        # Calculate per-operation stats
        operation_stats = {}
        for op, stats in by_operation.items():
            latencies = stats['latency']
            operation_stats[op] = {
                'count': stats['count'],
                'success_rate': stats['success'] / stats['count'],
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
                'min_latency': min(latencies) if latencies else 0,
                'max_latency': max(latencies) if latencies else 0
            }
        
        return {
            'period_hours': hours,
            'total_operations': total_operations,
            'success_rate': successful_operations / total_operations,
            'avg_execution_time': total_execution_time / total_operations,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'operations_by_type': operation_stats,
            'throughput_per_hour': total_operations / hours
        }