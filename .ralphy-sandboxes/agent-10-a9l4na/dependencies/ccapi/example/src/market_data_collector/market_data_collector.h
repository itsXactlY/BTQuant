#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <optional>

#include "candle_aggregator.h"
#include "exchange_connection_manager.h"
#include "market_data_processor.h"
#include "mssql_bulk_inserter.h"
#include "config_types.h"

// Forward declarations for ConfigLoader and SymbolRegistry
namespace BTQuant {
namespace Config {
class ConfigLoader;
}
class SymbolRegistry;
}

// HotSpine service information structure
struct HotSpineServiceInfo {
    bool is_available{false};
    std::string shared_memory_path;
    size_t buffer_capacity{0};
    uint64_t write_position{0};
    std::string service_version;
    uint32_t version_major{0};
    uint32_t version_minor{0};
    uint32_t version_patch{0};
    bool is_healthy{false};
    std::string last_error;
};

// HotSpine statistics structure
struct HotSpineStatistics {
    uint64_t trades_written{0};
    uint64_t write_errors{0};
    uint64_t buffer_used{0};
    uint64_t buffer_capacity{0};
    double buffer_usage_percent{0.0};
    bool is_attached{false};
    std::string shared_memory_path;
    uint64_t last_write_timestamp{0};
    uint64_t last_health_check{0};
};

// Health status enum
enum class HotSpineHealthStatus {
    HEALTHY,
    DEGRADED,
    UNHEALTHY,
    UNAVAILABLE
};

// ============================================================================
// Validation Result Types
// ============================================================================

// Result structure for individual validation operations
struct ValidationResult {
    bool is_valid{true};
    std::string exchange_name;  // Name of the exchange being validated
    std::vector<std::string> valid_items;
    std::vector<std::string> invalid_items;
    std::vector<std::string> warnings;
    std::string message;
    int total_exchanges{0};
    int valid_exchanges{0};
    int total_symbols{0};
    int valid_symbols{0};
    int auto_registered{0};
};

// Summary structure for overall validation status
struct ValidationSummary {
    int total_exchanges{0};
    int valid_exchanges{0};
    int total_symbols{0};
    int valid_symbols{0};
    int auto_registered{0};
    std::vector<ValidationResult> exchange_results;
};

// Symbol information structure
struct SymbolInfo {
    uint32_t id{0};
    std::string symbol;
    std::string exchange;
    std::string market_type;
    bool is_registered{false};
};

class MarketDataCollector {
public:
    struct Config {
        std::string db_connection_string;
        std::size_t bulk_insert_batch_size{500};

        std::vector<ExchangeConnectionManager::ExchangeConfig> exchanges;

        std::vector<std::string> timeframes{"1m", "5m", "15m", "1h"};

        std::size_t trade_buffer_size{1000};
        std::size_t candle_buffer_size{200};
        std::size_t orderbook_buffer_size{100};

        int flush_interval_ms{1000};
        int stats_report_interval_s{10};

        // New configuration options
        bool enable_mssql{true};
        bool enable_exclusive_hotspine{false};
        
        // Debug mode configuration
        ConfigTypes::DebugConfig debug_config;
        
        // Warmup configuration
        ConfigTypes::WarmupConfig warmup_config;
        
        // Parallel processing configuration
        std::size_t num_worker_threads{ConfigTypes::ParallelSettings{}.num_worker_threads};
        
        // HotSpine configuration
        std::string hotspine_shm_name{"/btquant_hotspine"};
        
        // Health monitoring configuration
        int health_check_interval_ms{5000};
        int reconnect_interval_ms{30000};
        int max_reconnect_attempts{10};
        bool enable_health_monitoring{true};
    };

    // Legacy constructor accepting explicit config
    explicit MarketDataCollector(const Config& cfg);
    
    // Dynamic constructor - loads all configuration from ConfigLoader
    // Accepts only essential runtime parameters (like hotspine service connection)
    explicit MarketDataCollector(const std::string& hotspine_shm_name);
    
    ~MarketDataCollector();

    void start();
    void stop();
    void waitForShutdown();

    void printStats() const;
    
    // ============================================================================
    // HotSpine Service Discovery Methods
    // ============================================================================
    
    // Discover HotSpine service information from shared memory
    HotSpineServiceInfo discoverHotSpineService();
    
    // Wait for HotSpine service to become available
    bool waitForHotSpineService(int timeout_ms = 30000);
    
    // ============================================================================
    // Health Monitoring Methods
    // ============================================================================
    
    // Check HotSpine service health and return detailed status
    HotSpineHealthStatus checkHotSpineHealth();
    
    // Get detailed HotSpine statistics
    HotSpineStatistics getHotSpineStatistics();
    
    // Simple boolean health check
    bool isHotSpineHealthy();
    
    // Log current health status
    void logHotSpineHealthStatus();
    
    // ============================================================================
    // Dynamic Reconnection Methods
    // ============================================================================
    
    // Attempt to reconnect to HotSpine service
    bool reconnectHotSpine();
    
    // Start background health monitoring thread
    void startHealthMonitorThread();
    
    // Stop background health monitoring thread
    void stopHealthMonitorThread();
    
    // Register callback for HotSpine availability changes
    void setHotSpineAvailabilityCallback(std::function<void(bool)> callback);
    
    // ============================================================================
    // Exchange Validation Methods
    // ============================================================================
    
    // Validate all configured exchanges
    ValidationResult validateAllExchanges();
    
    // Check if exchange is supported by CCAPI
    bool isExchangeSupported(const std::string& exchange_name);
    
    // Get list of supported exchanges
    std::vector<std::string> getSupportedExchanges();
    
    // Log unsupported exchanges with alternatives
    void logUnsupportedExchanges(const std::vector<std::string>& unsupported);
    
    // ============================================================================
    // Symbol Validation Methods
    // ============================================================================
    
    // Validate all symbols for an exchange
    ValidationResult validateExchangeSymbols(const std::string& exchange, 
                                             const std::vector<std::string>& symbols);
    
    // Try to resolve symbol to ID, return 0 if not found
    uint32_t resolveSymbolId(const std::string& exchange, const std::string& symbol);
    
    // Auto-register unknown symbols
    uint32_t registerUnknownSymbol(const std::string& exchange, const std::string& symbol);
    
    // Get validation summary for all exchanges
    ValidationSummary getValidationSummary();
    
    // ============================================================================
    // Runtime Symbol Addition Methods
    // ============================================================================
    
    // Add a new symbol at runtime
    bool addSymbolAtRuntime(const std::string& exchange, const std::string& symbol);
    
    // Add multiple symbols at runtime
    bool addSymbolsAtRuntime(const std::string& exchange, 
                             const std::vector<std::string>& symbols);
    
    // Check if symbol exists
    bool symbolExists(const std::string& exchange, const std::string& symbol);
    
    // Get symbol info
    std::optional<SymbolInfo> getSymbolInfo(const std::string& exchange, 
                                             const std::string& symbol);
    
    // ============================================================================
    // Dynamic Validation Loop Methods
    // ============================================================================
    
    // Start periodic validation
    void startValidationLoop(int interval_sec = 60);
    
    // Stop validation loop
    void stopValidationLoop();
    
    // Force re-validation
    void forceRevalidation();
    
    // Get last validation timestamp
    std::chrono::steady_clock::time_point getLastValidationTime();

private:
    Config config_;
    std::shared_ptr<MSSQLBulkInserter> db_;
    std::shared_ptr<CandleAggregator> candle_agg_;
    std::shared_ptr<HotSpine::HotSpineWriter> hotspine_writer_;
    std::shared_ptr<MarketDataProcessor> processor_;
    std::unique_ptr<ExchangeConnectionManager> conn_mgr_;

    std::thread flush_thread_;
    std::thread stats_thread_;
    std::thread health_monitor_thread_;
    std::atomic<bool> running_{false};
    
    // Health monitoring state
    std::atomic<bool> hotspine_available_{false};
    std::atomic<HotSpineHealthStatus> health_status_{HotSpineHealthStatus::UNAVAILABLE};
    std::atomic<int> reconnect_attempts_{0};
    std::mutex health_mutex_;
    std::function<void(bool)> availability_callback_;
    
    // Validation loop state
    std::thread validation_thread_;
    std::atomic<bool> validation_running_{false};
    std::atomic<bool> validation_dirty_{false};
    std::chrono::steady_clock::time_point last_validation_time_;
    std::mutex validation_mutex_;
    
    // Configuration discovery from environment and config file
    void discoverHotSpineConfiguration();
    
    void flushLoop();
    void statsLoop() const;
    void healthMonitorLoop();
    
    // Dynamic configuration methods
    void loadDynamicConfiguration();
    void discoverExchangesFromConfig();
    void discoverSymbolsFromConfig();
    void registerSymbolsWithRegistry();
    void logDiscoveredConfiguration();
    void initializeComponents();
    
    // Helper to check hotspine service health
    bool checkHotSpineServiceHealth();
    
    // Validation loop
    void validationLoop();
    
    // Perform validation and update state
    ValidationSummary performValidation();
};
