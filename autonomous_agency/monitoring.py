#!/usr/bin/env python3
"""
Monitoring Module for Autonomous Quantitative Research Agency

This module provides comprehensive monitoring, alerting, and system health tracking
for the autonomous agency. It ensures operational reliability and provides
real-time insights into the agency's performance.
"""

import os
import sys
import time
import logging
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import psutil
import socket

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autonomous_agency.config import MonitoringConfig
from autonomous_agency.archiver import StrategyArchiver as Archiver

class SystemMonitor:
    """
    Comprehensive system monitoring for the autonomous agency.
    """
    
    def __init__(self, config: MonitoringConfig, archiver: Archiver):
        self.config = config
        self.archiver = archiver
        self.logger = logging.getLogger(f"{__name__}.SystemMonitor")
        self.logger.setLevel(logging.INFO)
        
        # System metrics
        self.system_metrics: Dict[str, Any] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Alerts and issues
        self.active_alerts: Dict[str, Dict[str, Any]] = {}
        self.alert_counter = 0
        
        # System health
        self.system_health = "healthy"
        self.last_health_check = datetime.utcnow()
        
        # Monitoring threads
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        self.logger.info("SystemMonitor initialized with comprehensive monitoring capabilities")
    
    def start_monitoring(self):
        """
        Start all monitoring processes.
        """
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.logger.info("Starting comprehensive system monitoring")
        
        # Start main monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._run_monitoring_loop,
            daemon=True,
            name="SystemMonitorMain"
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """
        Stop all monitoring processes.
        """
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("System monitoring stopped")
    
    def _run_monitoring_loop(self):
        """
        Main monitoring loop that runs continuously.
        """
        try:
            while self.monitoring_active:
                start_time = time.time()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check system health
                self._check_system_health()
                
                # Store performance history
                self._store_performance_history()
                
                # Clean up old alerts
                self._cleanup_old_alerts()
                
                # Calculate sleep time to maintain monitoring interval
                processing_time = time.time() - start_time
                sleep_time = max(0, self.config.monitoring_interval - processing_time)
                time.sleep(sleep_time)
                
        except Exception as e:
            self.logger.error(f"Monitoring loop failed: {str(e)}")
            self.logger.debug(traceback.format_exc())
            self._create_alert("monitoring_failure", f"Monitoring loop crashed: {str(e)}", "critical")
    
    def _collect_system_metrics(self):
        """
        Collect comprehensive system metrics.
        """
        try:
            # System resource metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            
            # Network metrics
            net_io = psutil.net_io_counters()
            
            # Process metrics
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent(interval=0.1)
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # System info
            hostname = socket.gethostname()
            
            # Store metrics
            self.system_metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "system": {
                    "hostname": hostname,
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "disk_usage": disk_usage,
                    "network_bytes_sent": net_io.bytes_sent,
                    "network_bytes_recv": net_io.bytes_recv,
                },
                "process": {
                    "cpu_usage": process_cpu,
                    "memory_mb": process_memory,
                    "thread_count": process.num_threads(),
                },
                "agency": {
                    "active_strategies": 0,  # Would be populated by orchestrator
                    "backtest_queue": 0,    # Would be populated by orchestrator
                    "evaluation_queue": 0,  # Would be populated by orchestrator
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {str(e)}")
            self._create_alert("metric_collection_failure", str(e), "warning")
    
    def _check_system_health(self):
        """
        Perform comprehensive system health checks.
        """
        try:
            issues = []
            
            # Check CPU usage
            if self.system_metrics["system"]["cpu_usage"] > self.config.max_cpu_usage:
                issues.append(f"High CPU usage: {self.system_metrics['system']['cpu_usage']:.1f}%")
                
            # Check memory usage
            if self.system_metrics["system"]["memory_usage"] > self.config.max_memory_usage:
                issues.append(f"High memory usage: {self.system_metrics['system']['memory_usage']:.1f}%")
                
            # Check disk usage
            if self.system_metrics["system"]["disk_usage"] > self.config.max_disk_usage:
                issues.append(f"High disk usage: {self.system_metrics['system']['disk_usage']:.1f}%")
                
            # Check process memory
            if self.system_metrics["process"]["memory_mb"] > self.config.max_process_memory_mb:
                issues.append(f"High process memory: {self.system_metrics['process']['memory_mb']:.1f} MB")
                
            # Determine health status
            if issues:
                self.system_health = "degraded"
                for issue in issues:
                    self._create_alert("system_health_degraded", issue, "warning")
            else:
                self.system_health = "healthy"
                
            self.last_health_check = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"System health check failed: {str(e)}")
            self._create_alert("health_check_failure", str(e), "error")
    
    def _store_performance_history(self):
        """
        Store system performance history with retention policy.
        """
        try:
            # Add current metrics to history
            self.performance_history.append(self.system_metrics.copy())
            
            # Apply retention policy
            if len(self.performance_history) > self.config.metrics_retention_count:
                self.performance_history = self.performance_history[-self.config.metrics_retention_count:]
                
        except Exception as e:
            self.logger.error(f"Failed to store performance history: {str(e)}")
    
    def _cleanup_old_alerts(self):
        """
        Clean up resolved or old alerts.
        """
        try:
            current_time = datetime.utcnow()
            alerts_to_remove = []
            
            for alert_id, alert in self.active_alerts.items():
                alert_time = datetime.fromisoformat(alert["timestamp"])
                
                # Remove alerts older than retention period
                if (current_time - alert_time) > timedelta(days=self.config.alert_retention_days):
                    alerts_to_remove.append(alert_id)
                
                # Remove resolved alerts after some time
                if alert.get("status") == "resolved" and (current_time - alert_time) > timedelta(hours=1):
                    alerts_to_remove.append(alert_id)
                    
            # Remove identified alerts
            for alert_id in alerts_to_remove:
                del self.active_alerts[alert_id]
                
        except Exception as e:
            self.logger.error(f"Alert cleanup failed: {str(e)}")
    
    def _create_alert(self, alert_type: str, message: str, severity: str = "info"):
        """
        Create a new alert in the system.
        """
        try:
            self.alert_counter += 1
            alert_id = f"alert-{self.alert_counter}"
            
            alert = {
                "alert_id": alert_id,
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "active",
                "acknowledged": False,
                "acknowledged_by": None,
                "acknowledged_time": None
            }
            
            self.active_alerts[alert_id] = alert
            
            # Log alert
            log_method = getattr(self.logger, severity.lower(), self.logger.info)
            log_method(f"ALERT: {alert_type} - {message}")
            
            # Log to archiver
            self.archiver.log_event("alert_created", {
                "alert_id": alert_id,
                "alert_type": alert_type,
                "message": message,
                "severity": severity,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {str(e)}")
            return None
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system"):
        """
        Acknowledge an active alert.
        """
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert["acknowledged"] = True
                alert["acknowledged_by"] = acknowledged_by
                alert["acknowledged_time"] = datetime.utcnow().isoformat()
                alert["status"] = "acknowledged"
                
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
            return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system", resolution_notes: str = ""):
        """
        Resolve an active alert.
        """
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert["status"] = "resolved"
                alert["resolved_by"] = resolved_by
                alert["resolved_time"] = datetime.utcnow().isoformat()
                alert["resolution_notes"] = resolution_notes
                
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
            return False
    
    def get_active_alerts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active alerts.
        """
        return {k: v for k, v in self.active_alerts.items() if v["status"] == "active"}
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report.
        """
        return {
            "system_health": self.system_health,
            "last_health_check": self.last_health_check.isoformat(),
            "current_metrics": self.system_metrics,
            "active_alerts_count": len(self.get_active_alerts()),
            "total_alerts_count": len(self.active_alerts),
            "monitoring_status": "active" if self.monitoring_active else "inactive",
            "uptime": str(datetime.utcnow() - self.last_health_check) if hasattr(self, 'last_health_check') else "unknown"
        }
    
    def get_performance_trends(self, metric: str, hours: int = 24) -> List[Tuple[str, float]]:
        """
        Get performance trends for a specific metric.
        """
        try:
            trends = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            for entry in reversed(self.performance_history):
                entry_time = datetime.fromisoformat(entry["timestamp"])
                if entry_time < cutoff_time:
                    break
                    
                # Navigate through nested metrics
                metric_parts = metric.split('.')
                value = entry
                
                try:
                    for part in metric_parts:
                        value = value[part]
                    
                    trends.append((entry["timestamp"], float(value)))
                    
                except (KeyError, ValueError, TypeError):
                    continue
                    
            return trends
            
        except Exception as e:
            self.logger.error(f"Failed to get performance trends: {str(e)}")
            return []
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """
        Get current resource utilization percentages.
        """
        if not self.system_metrics:
            return {"cpu": 0.0, "memory": 0.0, "disk": 0.0}
            
        return {
            "cpu": self.system_metrics["system"]["cpu_usage"],
            "memory": self.system_metrics["system"]["memory_usage"],
            "disk": self.system_metrics["system"]["disk_usage"]
        }
    
    def check_agency_health(self, agency_stats: Dict[str, Any]) -> bool:
        """
        Check the health of the agency components.
        """
        try:
            issues = []
            
            # Check backtest queue
            if agency_stats.get("backtest_queue", 0) > self.config.max_backtest_queue:
                issues.append(f"Backtest queue too large: {agency_stats['backtest_queue']}")
                
            # Check evaluation queue
            if agency_stats.get("evaluation_queue", 0) > self.config.max_evaluation_queue:
                issues.append(f"Evaluation queue too large: {agency_stats['evaluation_queue']}")
                
            # Check active strategies
            if agency_stats.get("active_strategies", 0) > self.config.max_active_strategies:
                issues.append(f"Too many active strategies: {agency_stats['active_strategies']}")
                
            # Create alerts for any issues
            for issue in issues:
                self._create_alert("agency_health_issue", issue, "warning")
                
            return len(issues) == 0
            
        except Exception as e:
            self.logger.error(f"Agency health check failed: {str(e)}")
            return False

class PerformanceAnalyzer:
    """
    Analyzes performance trends and provides insights.
    """
    
    def __init__(self, monitor: SystemMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(f"{__name__}.PerformanceAnalyzer")
        self.logger.setLevel(logging.INFO)
    
    def analyze_system_trends(self) -> Dict[str, Any]:
        """
        Analyze system performance trends and provide insights.
        """
        analysis = {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "trends": {},
            "insights": [],
            "recommendations": []
        }
        
        try:
            # Analyze CPU trends
            cpu_trends = self.monitor.get_performance_trends("system.cpu_usage", 24)
            if cpu_trends:
                avg_cpu = sum(val for _, val in cpu_trends) / len(cpu_trends)
                max_cpu = max(val for _, val in cpu_trends)
                
                analysis["trends"]["cpu"] = {
                    "average": avg_cpu,
                    "maximum": max_cpu,
                    "current": cpu_trends[-1][1] if cpu_trends else 0
                }
                
                if avg_cpu > 80:
                    analysis["insights"].append("High average CPU usage detected")
                    analysis["recommendations"].append("Consider optimizing CPU-intensive operations")
            
            # Analyze memory trends
            memory_trends = self.monitor.get_performance_trends("system.memory_usage", 24)
            if memory_trends:
                avg_memory = sum(val for _, val in memory_trends) / len(memory_trends)
                max_memory = max(val for _, val in memory_trends)
                
                analysis["trends"]["memory"] = {
                    "average": avg_memory,
                    "maximum": max_memory,
                    "current": memory_trends[-1][1] if memory_trends else 0
                }
                
                if avg_memory > 75:
                    analysis["insights"].append("High average memory usage detected")
                    analysis["recommendations"].append("Check for memory leaks or inefficient data structures")
            
            # Analyze alert patterns
            active_alerts = self.monitor.get_active_alerts()
            analysis["alerts"] = {
                "active_count": len(active_alerts),
                "severity_distribution": self._analyze_alert_severity(active_alerts)
            }
            
            if len(active_alerts) > 5:
                analysis["insights"].append("High number of active alerts")
                analysis["recommendations"].append("Investigate recurring issues and improve system stability")
                
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}")
            analysis["error"] = str(e)
            
        return analysis
    
    def _analyze_alert_severity(self, alerts: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze severity distribution of alerts.
        """
        severity_counts = {"critical": 0, "error": 0, "warning": 0, "info": 0}
        
        for alert in alerts.values():
            severity = alert.get("severity", "info").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
            
        return severity_counts
    
    def generate_system_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system report.
        """
        report = {
            "report_timestamp": datetime.utcnow().isoformat(),
            "system_info": self.monitor.get_system_health_report(),
            "performance_analysis": self.analyze_system_trends(),
            "active_alerts": self.monitor.get_active_alerts(),
            "resource_utilization": self.monitor.get_resource_utilization()
        }
        
        return report