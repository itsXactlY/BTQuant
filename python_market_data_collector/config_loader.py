"""
Config Loader - Python equivalent of config_loader.cpp

This module provides configuration loading from JSON files,
equivalent to the C++ config_loader.cpp.
"""

import json
import os
from typing import Optional

from .config_types import (
    MarketDataCollectorConfig,
    ExchangeConfig,
    DebugConfig,
    WarmupConfig
)
from .utilities import get_current_timestamp, create_connection_string


def load_config(path: str) -> MarketDataCollectorConfig:
    """
    Load configuration from JSON file or create default configuration

    Args:
        path: Path to JSON configuration file

    Returns:
        MarketDataCollectorConfig object
    """
    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Starting configuration loading from {path}")

    config = MarketDataCollectorConfig()

    if not os.path.exists(path):
        # Create default configuration
        print(f"[{get_current_timestamp()}][WARNING] ConfigLoader: Config file not found: {path}, using default configuration.")

        # Default database connection
        config.db_connection_string = create_connection_string(
            "localhost",
            "market_data",
            "sa",
            "your_password"
        )

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default database connection string created")

        # Default timeframes
        config.timeframes = ["1m", "5m", "15m", "1h"]
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default timeframes set: {', '.join(config.timeframes)}")

        # Default exchange configuration
        ec = ExchangeConfig(
            exchange_name="binance",
            symbols=["BTC-USDT", "ETH-USDT"],
            channels=["TRADE", "MARKET_DEPTH"],
            market_type="spot"
        )
        config.exchanges.append(ec)

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default exchange configuration added: {ec.exchange_name} with {ec.symbols}")

        # Default buffer sizes and intervals
        config.trade_buffer_size = 1000
        config.candle_buffer_size = 200
        config.orderbook_buffer_size = 100
        config.flush_interval_ms = 1000
        config.stats_report_interval_s = 10

        # Default parallel settings
        config.num_worker_threads = 16
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default parallel settings set - num_worker_threads: {config.num_worker_threads}")

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default buffer sizes set:")
        print(f"[{get_current_timestamp()}][INFO]   Trade buffer: {config.trade_buffer_size}")
        print(f"[{get_current_timestamp()}][INFO]   Candle buffer: {config.candle_buffer_size}")
        print(f"[{get_current_timestamp()}][INFO]   Orderbook buffer: {config.orderbook_buffer_size}")
        print(f"[{get_current_timestamp()}][INFO]   Flush interval: {config.flush_interval_ms}ms")
        print(f"[{get_current_timestamp()}][INFO]   Stats interval: {config.stats_report_interval_s}s")

        # Default toggle settings
        config.enable_mssql = True
        config.enable_exclusive_hotspine = False

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default toggle settings:")
        print(f"[{get_current_timestamp()}][INFO]   MS SQL enabled: {config.enable_mssql}")
        print(f"[{get_current_timestamp()}][INFO]   Exclusive HotSpine: {config.enable_exclusive_hotspine}")

        # Default warmup configuration
        config.warmup_config = WarmupConfig(
            batch_size=20,
            startup_delay_ms=5000,
            wait_for_connection=True
        )

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default warmup configuration:")
        print(f"[{get_current_timestamp()}][INFO]   Batch size: {config.warmup_config.batch_size}")
        print(f"[{get_current_timestamp()}][INFO]   Startup delay: {config.warmup_config.startup_delay_ms}ms")
        print(f"[{get_current_timestamp()}][INFO]   Wait for connection: {config.warmup_config.wait_for_connection}")

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default configuration loading completed")

        return config

    # Load configuration from file
    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Loading JSON configuration from file")
    with open(path, 'r') as f:
        j = json.load(f)

    # Load database configuration
    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Loading database configuration")
    db = j.get("db", {})
    server = db.get("server", "localhost")
    database = db.get("database", "market_data")
    user = db.get("user", "sa")
    password = db.get("password", "")

    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Database connection details:")
    print(f"[{get_current_timestamp()}][INFO]   Server: {server}")
    print(f"[{get_current_timestamp()}][INFO]   Database: {database}")
    print(f"[{get_current_timestamp()}][INFO]   User: {user}")
    print(f"[{get_current_timestamp()}][INFO]   Password: [REDACTED]")

    config.db_connection_string = create_connection_string(server, database, user, password)
    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Database connection string created")

    # Load timeframes
    config.timeframes = j.get("timeframes", ["1m", "5m", "15m", "1h"])
    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Timeframes loaded: {', '.join(config.timeframes)}")

    # Load exchange configurations
    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Loading exchange configurations")
    for ex in j.get("exchanges", []):
        ec = ExchangeConfig(
            exchange_name=ex.get("name", ""),
            symbols=ex.get("symbols", []),
            channels=ex.get("channels", ["TRADE", "MARKET_DEPTH"]),
            market_type=ex.get("market_type", "spot")
        )

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Exchange configuration loaded:")
        print(f"[{get_current_timestamp()}][INFO]   Exchange: {ec.exchange_name}")
        print(f"[{get_current_timestamp()}][INFO]   Symbols: {', '.join(ec.symbols)}")
        print(f"[{get_current_timestamp()}][INFO]   Channels: {', '.join(ec.channels)}")
        print(f"[{get_current_timestamp()}][INFO]   Market type: {ec.market_type}")

        config.exchanges.append(ec)

    # Load buffer sizes and intervals
    config.trade_buffer_size = j.get("trade_buffer_size", 500)
    config.candle_buffer_size = j.get("candle_buffer_size", 200)
    config.orderbook_buffer_size = j.get("orderbook_buffer_size", 100)
    config.flush_interval_ms = j.get("flush_interval_ms", 1000)
    config.stats_report_interval_s = j.get("stats_report_interval_s", 10)

    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Buffer and interval settings loaded:")
    print(f"[{get_current_timestamp()}][INFO]   Trade buffer size: {config.trade_buffer_size}")
    print(f"[{get_current_timestamp()}][INFO]   Candle buffer size: {config.candle_buffer_size}")
    print(f"[{get_current_timestamp()}][INFO]   Orderbook buffer size: {config.orderbook_buffer_size}")
    print(f"[{get_current_timestamp()}][INFO]   Flush interval: {config.flush_interval_ms}ms")
    print(f"[{get_current_timestamp()}][INFO]   Stats report interval: {config.stats_report_interval_s}s")

    # Load feature toggles
    config.enable_mssql = j.get("enable_mssql", True)
    config.enable_exclusive_hotspine = j.get("enable_exclusive_hotspine", False)

    # Load parallel settings
    if "parallel_settings" in j:
        parallel = j["parallel_settings"]
        config.num_worker_threads = parallel.get("num_worker_threads", 16)
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Parallel settings loaded - num_worker_threads: {config.num_worker_threads}")
    else:
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: No parallel_settings found, using default (16 threads)")

    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Feature toggle settings loaded:")
    print(f"[{get_current_timestamp()}][INFO]   Enable MS SQL: {config.enable_mssql}")
    print(f"[{get_current_timestamp()}][INFO]   Enable exclusive HotSpine: {config.enable_exclusive_hotspine}")

    # Load debug mode configuration
    if "debug_mode" in j:
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Loading debug mode configuration")
        debug = j["debug_mode"]

        config.debug_config = DebugConfig(
            enabled=debug.get("enabled", False),
            verbose_logging=debug.get("verbose_logging", False),
            websocket_debug=debug.get("websocket_debug", False),
            database_debug=debug.get("database_debug", False),
            buffer_debug=debug.get("buffer_debug", False),
            flush_debug=debug.get("flush_debug", False),
            performance_monitoring=debug.get("performance_monitoring", False),
            connection_health_checks=debug.get("connection_health_checks", False),
            data_flow_validation=debug.get("data_flow_validation", False),
            error_tracing=debug.get("error_tracing", False),
            timestamp_precision=debug.get("timestamp_precision", "milliseconds"),
            log_level=debug.get("log_level", "INFO")
        )

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Debug mode settings loaded:")
        print(f"[{get_current_timestamp()}][INFO]   Debug enabled: {config.debug_config.enabled}")
        print(f"[{get_current_timestamp()}][INFO]   Verbose logging: {config.debug_config.verbose_logging}")
        print(f"[{get_current_timestamp()}][INFO]   WebSocket debug: {config.debug_config.websocket_debug}")
        print(f"[{get_current_timestamp()}][INFO]   Database debug: {config.debug_config.database_debug}")
        print(f"[{get_current_timestamp()}][INFO]   Buffer debug: {config.debug_config.buffer_debug}")
        print(f"[{get_current_timestamp()}][INFO]   Log level: {config.debug_config.log_level}")

        # Load database debug configuration
        if "database_debug" in debug and isinstance(debug["database_debug"], dict):
            db_debug = debug["database_debug"]
            config.debug_config.database_debug_config = DebugConfig.DatabaseDebugConfig(
                verify_connection_on_start=db_debug.get("verify_connection_on_start", False),
                test_table_creation=db_debug.get("test_table_creation", False),
                log_all_queries=db_debug.get("log_all_queries", False),
                log_connection_status=db_debug.get("log_connection_status", False),
                enable_detailed_error_logging=db_debug.get("enable_detailed_error_logging", False),
                connection_timeout_ms=db_debug.get("connection_timeout_ms", 30000),
                command_timeout_ms=db_debug.get("command_timeout_ms", 60000)
            )

        # Load WebSocket debug configuration
        if "websocket_debug" in debug and isinstance(debug["websocket_debug"], dict):
            ws_debug = debug["websocket_debug"]
            config.debug_config.websocket_debug_config = DebugConfig.WebSocketDebugConfig(
                log_connection_attempts=ws_debug.get("log_connection_attempts", False),
                log_subscription_status=ws_debug.get("log_subscription_status", False),
                log_data_reception=ws_debug.get("log_data_reception", False),
                ping_pong_monitoring=ws_debug.get("ping_pong_monitoring", False),
                connection_health_monitoring=ws_debug.get("connection_health_monitoring", False),
                timeout_debugging=ws_debug.get("timeout_debugging", False),
                protocol_error_logging=ws_debug.get("protocol_error_logging", False),
                data_rate_monitoring=ws_debug.get("data_rate_monitoring", False)
            )

        # Load buffer debug configuration
        if "buffer_debug" in debug and isinstance(debug["buffer_debug"], dict):
            buf_debug = debug["buffer_debug"]
            config.debug_config.buffer_debug_config = DebugConfig.BufferDebugConfig(
                log_buffer_operations=buf_debug.get("log_buffer_operations", False),
                log_threshold_crossing=buf_debug.get("log_threshold_crossing", False),
                log_flush_decisions=buf_debug.get("log_flush_decisions", False),
                monitor_memory_usage=buf_debug.get("monitor_memory_usage", False),
                track_buffer_sizes=buf_debug.get("track_buffer_sizes", False),
                log_batch_operations=buf_debug.get("log_batch_operations", False)
            )

        # Load performance monitoring configuration
        if "performance_monitoring" in debug and isinstance(debug["performance_monitoring"], dict):
            perf_debug = debug["performance_monitoring"]
            config.debug_config.performance_config = DebugConfig.PerformanceConfig(
                measure_processing_latency=perf_debug.get("measure_processing_latency", False),
                track_throughput=perf_debug.get("track_throughput", False),
                monitor_resource_usage=perf_debug.get("monitor_resource_usage", False),
                log_performance_metrics=perf_debug.get("log_performance_metrics", False),
                alert_on_slow_operations=perf_debug.get("alert_on_slow_operations", False)
            )

        # If debug mode is disabled, override all sub-flags to false
        if not config.debug_config.enabled:
            config.debug_config.verbose_logging = False
            config.debug_config.websocket_debug = False
            config.debug_config.database_debug = False
            config.debug_config.buffer_debug = False
            config.debug_config.flush_debug = False
            config.debug_config.performance_monitoring = False
            config.debug_config.connection_health_checks = False
            config.debug_config.data_flow_validation = False
            config.debug_config.error_tracing = False

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Debug mode configuration completed successfully")
    else:
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: No debug mode configuration found, using defaults")

    # Load warmup configuration
    if "warmup_config" in j:
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Loading warmup configuration")
        warmup = j["warmup_config"]

        config.warmup_config = WarmupConfig(
            batch_size=warmup.get("batch_size", 20),
            startup_delay_ms=warmup.get("startup_delay_ms", 5000),
            wait_for_connection=warmup.get("wait_for_connection", True)
        )

        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Warmup configuration loaded:")
        print(f"[{get_current_timestamp()}][INFO]   Batch size: {config.warmup_config.batch_size}")
        print(f"[{get_current_timestamp()}][INFO]   Startup delay: {config.warmup_config.startup_delay_ms}ms")
        print(f"[{get_current_timestamp()}][INFO]   Wait for connection: {config.warmup_config.wait_for_connection}")
    else:
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: No warmup_config found, using defaults")
        print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Default warmup configuration:")
        print(f"[{get_current_timestamp()}][INFO]   Batch size: {config.warmup_config.batch_size}")
        print(f"[{get_current_timestamp()}][INFO]   Startup delay: {config.warmup_config.startup_delay_ms}ms")
        print(f"[{get_current_timestamp()}][INFO]   Wait for connection: {config.warmup_config.wait_for_connection}")

    print(f"[{get_current_timestamp()}][INFO] ConfigLoader: Configuration loading completed successfully")

    return config