"""
Configuration Types - Python equivalent of config structures

This module defines configuration structures used throughout the
market data collector, equivalent to the C++ config structures.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DebugConfig:
    """Debug configuration options"""
    enabled: bool = False
    verbose_logging: bool = False
    websocket_debug: bool = False
    database_debug: bool = False
    buffer_debug: bool = False
    flush_debug: bool = False
    performance_monitoring: bool = False
    connection_health_checks: bool = False
    data_flow_validation: bool = False
    error_tracing: bool = False
    timestamp_precision: str = "milliseconds"
    log_level: str = "INFO"

    # Database debug configuration
    @dataclass
    class DatabaseDebugConfig:
        verify_connection_on_start: bool = False
        test_table_creation: bool = False
        log_all_queries: bool = False
        log_connection_status: bool = False
        enable_detailed_error_logging: bool = False
        connection_timeout_ms: int = 30000
        command_timeout_ms: int = 60000

    database_debug_config: DatabaseDebugConfig = None

    # WebSocket debug configuration
    @dataclass
    class WebSocketDebugConfig:
        log_connection_attempts: bool = False
        log_subscription_status: bool = False
        log_data_reception: bool = False
        ping_pong_monitoring: bool = False
        connection_health_monitoring: bool = False
        timeout_debugging: bool = False
        protocol_error_logging: bool = False
        data_rate_monitoring: bool = False

    websocket_debug_config: WebSocketDebugConfig = None

    # Buffer debug configuration
    @dataclass
    class BufferDebugConfig:
        log_buffer_operations: bool = False
        log_threshold_crossing: bool = False
        log_flush_decisions: bool = False
        monitor_memory_usage: bool = False
        track_buffer_sizes: bool = False
        log_batch_operations: bool = False

    buffer_debug_config: BufferDebugConfig = None

    # Performance monitoring configuration
    @dataclass
    class PerformanceConfig:
        measure_processing_latency: bool = False
        track_throughput: bool = False
        monitor_resource_usage: bool = False
        log_performance_metrics: bool = False
        alert_on_slow_operations: bool = False

    performance_config: PerformanceConfig = None

    def __post_init__(self):
        if self.database_debug_config is None:
            self.database_debug_config = self.DatabaseDebugConfig()
        if self.websocket_debug_config is None:
            self.websocket_debug_config = self.WebSocketDebugConfig()
        if self.buffer_debug_config is None:
            self.buffer_debug_config = self.BufferDebugConfig()
        if self.performance_config is None:
            self.performance_config = self.PerformanceConfig()


@dataclass
class WarmupConfig:
    """Warmup configuration for initial connections"""
    batch_size: int = 20
    startup_delay_ms: int = 5000
    wait_for_connection: bool = True


@dataclass
class ParallelSettings:
    """Parallel processing settings"""
    num_worker_threads: int = 16


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange"""
    exchange_name: str
    symbols: List[str]
    channels: List[str]  # "TRADE", "MARKET_DEPTH"
    market_type: str = "spot"  # "spot", "perpetual", etc.


@dataclass
class MarketDataCollectorConfig:
    """Main configuration for the market data collector"""
    # Database configuration
    db_connection_string: str = ""

    # Exchange configurations
    exchanges: List[ExchangeConfig] = None

    # Timeframes for candle aggregation
    timeframes: List[str] = None

    # Buffer sizes
    trade_buffer_size: int = 1000
    candle_buffer_size: int = 200
    orderbook_buffer_size: int = 100

    # Timing intervals
    flush_interval_ms: int = 1000
    stats_report_interval_s: int = 10

    # Feature toggles
    enable_mssql: bool = True
    enable_exclusive_hotspine: bool = False

    # Debug configuration
    debug_config: DebugConfig = None

    # Warmup configuration
    warmup_config: WarmupConfig = None

    # Parallel processing
    num_worker_threads: int = 16

    def __post_init__(self):
        if self.exchanges is None:
            self.exchanges = []
        if self.timeframes is None:
            self.timeframes = ["1m", "5m", "15m", "1h"]
        if self.debug_config is None:
            self.debug_config = DebugConfig()
        if self.warmup_config is None:
            self.warmup_config = WarmupConfig()