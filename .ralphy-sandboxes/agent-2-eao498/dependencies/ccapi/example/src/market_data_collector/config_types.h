#pragma once
#include <cstdint>
#include <string>

namespace ConfigTypes {

struct DatabaseDebugConfig {
    bool verify_connection_on_start{false};
    bool test_table_creation{false};
    bool log_all_queries{false};
    bool log_connection_status{false};
    bool enable_detailed_error_logging{false};
    int connection_timeout_ms{30000};
    int command_timeout_ms{60000};
};

struct WebSocketDebugConfig {
    bool log_connection_attempts{false};
    bool log_subscription_status{false};
    bool log_data_reception{false};
    bool ping_pong_monitoring{false};
    bool connection_health_monitoring{false};
    bool timeout_debugging{false};
    bool protocol_error_logging{false};
    bool data_rate_monitoring{false};
};

struct BufferDebugConfig {
    bool log_buffer_operations{false};
    bool log_threshold_crossing{false};
    bool log_flush_decisions{false};
    bool monitor_memory_usage{false};
    bool track_buffer_sizes{false};
    bool log_batch_operations{false};
};

struct PerformanceConfig {
    bool measure_processing_latency{false};
    bool track_throughput{false};
    bool monitor_resource_usage{false};
    bool log_performance_metrics{false};
    bool alert_on_slow_operations{false};
};

struct DebugConfig {
    bool enabled{false};
    bool verbose_logging{false};
    bool websocket_debug{false};
    bool database_debug{false};
    bool buffer_debug{false};
    bool flush_debug{false};
    bool performance_monitoring{false};
    bool connection_health_checks{false};
    bool data_flow_validation{false};
    bool error_tracing{false};
    std::string timestamp_precision{"milliseconds"};
    std::string log_level{"INFO"};
    
    DatabaseDebugConfig database_debug_config;
    WebSocketDebugConfig websocket_debug_config;
    BufferDebugConfig buffer_debug_config;
    PerformanceConfig performance_config;
};

struct WarmupConfig {
    std::size_t batch_size{20};
    int startup_delay_ms{5000};
    bool wait_for_connection{true};
};

struct ParallelSettings {
    std::size_t num_worker_threads{16};
};

} // namespace ConfigTypes
