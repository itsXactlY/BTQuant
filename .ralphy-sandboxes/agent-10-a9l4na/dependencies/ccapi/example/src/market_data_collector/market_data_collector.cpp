#include "market_data_collector.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "../hotspine/hotspine_writer.hpp"

// Include configuration and symbol registry for dynamic discovery
// These headers are in the tests/new directory - adjust paths as needed
#include "../../../tests/new/include/config/config_loader.hpp"
#include "../../../tests/new/include/symbol_registry.hpp"

// Helper function for timestamped logging
static std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

  std::tm tm = *std::localtime(&now_time);
  char buffer[64];
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);

  char ms_buffer[10];
  snprintf(ms_buffer, sizeof(ms_buffer), "%03d", static_cast<int>(now_ms.count()));

  return std::string(buffer) + "." + ms_buffer;
}

// Helper to get health status string
static std::string healthStatusToString(HotSpineHealthStatus status) {
    switch (status) {
        case HotSpineHealthStatus::HEALTHY: return "HEALTHY";
        case HotSpineHealthStatus::DEGRADED: return "DEGRADED";
        case HotSpineHealthStatus::UNHEALTHY: return "UNHEALTHY";
        case HotSpineHealthStatus::UNAVAILABLE: return "UNAVAILABLE";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Legacy constructor - kept for backward compatibility
// ============================================================================
MarketDataCollector::MarketDataCollector(const Config& cfg) : config_(cfg) {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing with explicit configuration..." << std::endl;

  // Log configuration details
  logDiscoveredConfiguration();
  
  // Initialize components (same as dynamic constructor)
  initializeComponents();
}

// ============================================================================
// Dynamic constructor - loads all configuration from ConfigLoader
// ============================================================================
MarketDataCollector::MarketDataCollector(const std::string& hotspine_shm_name) {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing with dynamic configuration..." << std::endl;
  
  // Initialize configuration system
  auto& config = BTQuant::Config::ConfigLoader::instance();
  config.initialize();
  config.load();
  
  // Discover HotSpine configuration from environment and config file
  discoverHotSpineConfiguration();
  
  // Load all configuration dynamically
  loadDynamicConfiguration();
  
  // Override HotSpine name if provided
  if (!hotspine_shm_name.empty()) {
    config_.hotspine_shm_name = hotspine_shm_name;
  }
  
  // Log discovered configuration
  logDiscoveredConfiguration();
  
  // Initialize components
  initializeComponents();
}

// ============================================================================
// Shared initialization logic for both constructors
// ============================================================================
void MarketDataCollector::initializeComponents() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing components..." << std::endl;

  // Conditionally initialize MS SQL database
  if (config_.enable_mssql) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing MS SQL database connection" << std::endl;
    db_ = std::make_shared<MSSQLBulkInserter>(config_.db_connection_string, config_.debug_config.database_debug_config.verify_connection_on_start);

    // Enable debug mode if configured
    if (config_.debug_config.enabled && config_.debug_config.database_debug_config.log_connection_status) {
      std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataCollector: Enabling database debug mode" << std::endl;
      db_->setDebugMode(true);
      db_->enableDetailedLogging(config_.debug_config.verbose_logging);
    }

    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: MS SQL database connection initialized" << std::endl;
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: MS SQL database disabled" << std::endl;
  }

  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing candle aggregator with timeframes" << std::endl;
  candle_agg_ = std::make_shared<CandleAggregator>(config_.timeframes);
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Candle aggregator initialized" << std::endl;

  // Create HotSpine writer with configurable shared memory name
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing HotSpine writer with shared memory: " 
            << config_.hotspine_shm_name << std::endl;
  
  hotspine_writer_ = std::make_shared<HotSpine::HotSpineWriter>(config_.hotspine_shm_name);

  // Load symbol mappings for the writer
  auto& config = BTQuant::Config::ConfigLoader::instance();
  std::string symbol_mapping_path = config.get_as<std::string>("symbols", "fallback_file")
      .value_or("config/symbol_mapping.json");

  if (!hotspine_writer_->loadSymbolMappings(symbol_mapping_path)) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Failed to load symbol mappings for HotSpine writer" << std::endl;
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Symbol mappings loaded for HotSpine writer" << std::endl;
  }

  // Check HotSpine service health
  if (!checkHotSpineServiceHealth()) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: HotSpine service not available, continuing anyway..." << std::endl;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine writer initialized" << std::endl;
  hotspine_writer_->setBatchingEnabled(false);
  hotspine_writer_->setBatchSize(1);  // Immediate write
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine batching disabled for immediate data flow" << std::endl;

  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing market data processor" << std::endl;
  processor_ = std::make_shared<MarketDataProcessor>(db_, candle_agg_, hotspine_writer_, config_.enable_exclusive_hotspine, config_.debug_config);
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Market data processor initialized" << std::endl;

  // Log HotSpine integration status
  if (hotspine_writer_) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine integration ENABLED" << std::endl;
    if (config_.enable_exclusive_hotspine) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Running in EXCLUSIVE HotSpine mode (database disabled)" << std::endl;
    } else {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Running in DUAL mode (HotSpine + database)" << std::endl;
    }
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine integration DISABLED" << std::endl;
  }

  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Setting buffer limits" << std::endl;
  processor_->setBufferLimits(config_.trade_buffer_size, config_.candle_buffer_size, config_.orderbook_buffer_size);
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Buffer limits set successfully" << std::endl;

  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initializing exchange connection manager" << std::endl;
  conn_mgr_ = std::make_unique<ExchangeConnectionManager>(processor_);
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Exchange connection manager initialized" << std::endl;

  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Initialization completed successfully" << std::endl;
}

// ============================================================================
// Discover HotSpine configuration from environment and config file
// ============================================================================
void MarketDataCollector::discoverHotSpineConfiguration() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Discovering HotSpine configuration..." << std::endl;
  
  auto& config = BTQuant::Config::ConfigLoader::instance();
  
  // Priority 1: Environment variable
  const char* env_shm_name = std::getenv("BTQ_HOTSPINE_SHM_NAME");
  if (env_shm_name && std::strlen(env_shm_name) > 0) {
    config_.hotspine_shm_name = env_shm_name;
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine SHM name from environment: " 
              << config_.hotspine_shm_name << std::endl;
    return;
  }
  
  // Priority 2: Config file - shared_memory.name
  if (auto shm_name = config.get_as<std::string>("shared_memory", "name")) {
    config_.hotspine_shm_name = *shm_name;
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine SHM name from config: " 
              << config_.hotspine_shm_name << std::endl;
  }
  
  // Priority 3: Config file - hotspine.shared_memory_name (alternative key)
  if (config_.hotspine_shm_name.empty()) {
    if (auto shm_name = config.get_as<std::string>("hotspine", "shared_memory_name")) {
      config_.hotspine_shm_name = *shm_name;
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine SHM name from hotspine config: " 
                << config_.hotspine_shm_name << std::endl;
    }
  }
  
  // Load health monitoring configuration
  config_.health_check_interval_ms = config.get_as<int64_t>("hotspine", "health_check_interval_ms")
      .value_or(config_.health_check_interval_ms);
  config_.reconnect_interval_ms = config.get_as<int64_t>("hotspine", "reconnect_interval_ms")
      .value_or(config_.reconnect_interval_ms);
  config_.max_reconnect_attempts = config.get_as<int64_t>("hotspine", "max_reconnect_attempts")
      .value_or(config_.max_reconnect_attempts);
  config_.enable_health_monitoring = config.get_as<bool>("hotspine", "enable_health_monitoring")
      .value_or(config_.enable_health_monitoring);
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Health check interval: " 
            << config_.health_check_interval_ms << "ms" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Reconnect interval: " 
            << config_.reconnect_interval_ms << "ms" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Health monitoring enabled: " 
            << (config_.enable_health_monitoring ? "true" : "false") << std::endl;
}

// ============================================================================
// Dynamic configuration loading from ConfigLoader
// ============================================================================
void MarketDataCollector::loadDynamicConfiguration() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Loading dynamic configuration..." << std::endl;
  
  auto& config = BTQuant::Config::ConfigLoader::instance();
  
  // Load database configuration
  config_.enable_mssql = config.get_as<bool>("database", "enabled").value_or(config_.enable_mssql);
  config_.db_connection_string = config.get_as<std::string>("database", "connection_string")
      .value_or(config_.db_connection_string);
  
  // Load timeframes from config with fallback to defaults
  std::vector<std::string> default_timeframes{"1m", "5m", "15m", "1h", "4h", "1d"};
  if (auto tf_config = config.get_as<std::vector<std::string>>("collector", "timeframes")) {
    config_.timeframes = *tf_config;
  } else {
    config_.timeframes = default_timeframes;
  }
  
  // Load buffer sizes from config with defaults
  config_.trade_buffer_size = config.get_as<int64_t>("collector", "buffer.trade")
      .value_or(static_cast<int64_t>(config_.trade_buffer_size));
  config_.candle_buffer_size = config.get_as<int64_t>("collector", "buffer.candle")
      .value_or(static_cast<int64_t>(config_.candle_buffer_size));
  config_.orderbook_buffer_size = config.get_as<int64_t>("collector", "buffer.orderbook")
      .value_or(static_cast<int64_t>(config_.orderbook_buffer_size));
  
  // Load intervals from config with defaults
  config_.flush_interval_ms = config.get_as<int64_t>("collector", "interval.flush_ms")
      .value_or(config_.flush_interval_ms);
  config_.stats_report_interval_s = config.get_as<int64_t>("collector", "interval.stats_sec")
      .value_or(config_.stats_report_interval_s);
  
  // Load exclusive HotSpine mode
  config_.enable_exclusive_hotspine = config.get_as<bool>("hotspine", "exclusive_mode")
      .value_or(config_.enable_exclusive_hotspine);
  
  // Discover exchanges and symbols from config
  discoverExchangesFromConfig();
  discoverSymbolsFromConfig();
  
  // Register symbols with SymbolRegistry
  registerSymbolsWithRegistry();
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Dynamic configuration loaded successfully" << std::endl;
}

// ============================================================================
// Discover symbols from configuration
// ============================================================================
void MarketDataCollector::discoverSymbolsFromConfig() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Discovering symbols from configuration..." << std::endl;
  
  auto& config = BTQuant::Config::ConfigLoader::instance();
  
  // Get monitored symbols from config
  std::vector<std::string> symbols;
  if (auto syms = config.get_as<std::vector<std::string>>("monitoring", "symbols")) {
    symbols = *syms;
  } else {
    // Default symbols if not configured
    symbols = {"BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "ADA-USDT"};
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: No symbols configured, using defaults" << std::endl;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Found " 
            << symbols.size() << " symbols to monitor" << std::endl;
  
  // Build exchange configurations from symbols
  // This is a simplified mapping - real implementation would parse exchange/symbol pairs
  ExchangeConnectionManager::ExchangeConfig binance_config;
  binance_config.exchange_name = "binance";
  binance_config.channels = {"TRADE", "MARKET_DEPTH"};
  binance_config.market_type = "spot";
  
  for (const auto& symbol : symbols) {
    // Convert format (BTC-USDT -> BTCUSDT) for exchange compatibility
    std::string exchange_symbol = symbol;
    exchange_symbol.erase(remove(exchange_symbol.begin(), exchange_symbol.end(), '-'), exchange_symbol.end());
    binance_config.symbols.push_back(exchange_symbol);
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector:   - Symbol: " << exchange_symbol << std::endl;
  }
  
  // Only add if we have symbols
  if (!binance_config.symbols.empty()) {
    config_.exchanges.push_back(binance_config);
  }
}

// ============================================================================
// Discover exchanges from configuration
// ============================================================================
void MarketDataCollector::discoverExchangesFromConfig() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Discovering exchanges from configuration..." << std::endl;
  
  auto& config = BTQuant::Config::ConfigLoader::instance();
  
  // Get enabled exchanges list
  std::vector<std::string> enabled_exchanges;
  if (auto exchanges = config.get_as<std::vector<std::string>>("exchanges", "enabled")) {
    enabled_exchanges = *exchanges;
  } else {
    // Default exchanges if not configured
    enabled_exchanges = {"binance", "coinbase", "kraken"};
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: No exchanges configured, using defaults" << std::endl;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Found " 
            << enabled_exchanges.size() << " enabled exchanges" << std::endl;
  
  // Note: Full exchange/symbol discovery would require API integration
  // For now, we keep the exchange list for reference
  for (const auto& exchange : enabled_exchanges) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector:   - Exchange: " << exchange << std::endl;
  }
}

// ============================================================================
// Register discovered symbols with SymbolRegistry
// ============================================================================
void MarketDataCollector::registerSymbolsWithRegistry() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Registering symbols with SymbolRegistry..." << std::endl;
  
  auto& registry = BTQuant::SymbolRegistry::instance();
  
  // Load symbol mappings from config file
  auto& config = BTQuant::Config::ConfigLoader::instance();
  std::string symbol_mapping_path = config.get_as<std::string>("symbols", "fallback_file")
      .value_or("config/symbol_mapping.json");
  
  if (registry.load_from_file(symbol_mapping_path)) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Loaded symbol mappings from " 
              << symbol_mapping_path << std::endl;
  } else {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Could not load symbol mappings from " 
              << symbol_mapping_path << ", using auto-registration" << std::endl;
  }
  
  // Register each discovered symbol
  for (const auto& exchange_config : config_.exchanges) {
    for (const auto& symbol : exchange_config.symbols) {
      // Check if symbol already registered
      if (!registry.has_symbol(exchange_config.exchange_name, symbol)) {
        // Auto-assign ID for new symbol
        uint32_t symbol_id = registry.register_symbol(exchange_config.exchange_name, symbol);
        std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Registered symbol "
                  << exchange_config.exchange_name << "/" << symbol << " with ID " << symbol_id << std::endl;
      } else {
        // Get existing symbol ID
        auto symbol_id = registry.get_symbol_id(exchange_config.exchange_name, symbol);
        if (symbol_id) {
          std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Symbol "
                    << exchange_config.exchange_name << "/" << symbol << " already registered with ID " 
                    << *symbol_id << std::endl;
        }
      }
    }
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Symbol registration complete" << std::endl;
}

// ============================================================================
// Check HotSpine service health
// ============================================================================
bool MarketDataCollector::checkHotSpineServiceHealth() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Checking HotSpine service health..." << std::endl;
  
  if (!hotspine_writer_) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: HotSpine writer is null" << std::endl;
    hotspine_available_ = false;
    health_status_ = HotSpineHealthStatus::UNAVAILABLE;
    return false;
  }
  
  bool is_healthy = hotspine_writer_->isHealthy();
  hotspine_available_ = is_healthy;
  
  if (is_healthy) {
    health_status_ = HotSpineHealthStatus::HEALTHY;
    reconnect_attempts_ = 0;
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine service is HEALTHY" << std::endl;
  } else {
    health_status_ = HotSpineHealthStatus::UNAVAILABLE;
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: HotSpine service is UNAVAILABLE" << std::endl;
  }
  
  return is_healthy;
}

// ============================================================================
// Discover HotSpine service information
// ============================================================================
HotSpineServiceInfo MarketDataCollector::discoverHotSpineService() {
  HotSpineServiceInfo info;
  info.shared_memory_path = config_.hotspine_shm_name;
  
  if (!hotspine_writer_) {
    info.is_available = false;
    info.last_error = "HotSpine writer not initialized";
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine service discovery - writer not initialized" << std::endl;
    return info;
  }
  
  // Check if attached to shared memory
  bool is_attached = hotspine_writer_->isHealthy();
  
  if (!is_attached) {
    info.is_available = false;
    info.last_error = "Failed to attach to shared memory";
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine service discovery - not attached to shared memory" << std::endl;
    return info;
  }
  
  info.is_available = true;
  info.is_healthy = true;
  
  // Get statistics
  auto stats = hotspine_writer_->getDetailedStats();
  
  // Parse version from stats (simplified parsing)
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine service discovered successfully" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector:   Shared Memory: " << info.shared_memory_path << std::endl;
  
  return info;
}

// ============================================================================
// Wait for HotSpine service to become available
// ============================================================================
bool MarketDataCollector::waitForHotSpineService(int timeout_ms) {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Waiting for HotSpine service (timeout: " 
            << timeout_ms << "ms)..." << std::endl;
  
  auto start_time = std::chrono::steady_clock::now();
  int check_interval_ms = std::min(1000, config_.health_check_interval_ms);
  
  while (true) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
    
    if (elapsed_ms >= timeout_ms) {
      std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Timeout waiting for HotSpine service" << std::endl;
      return false;
    }
    
    // Check if service is available
    if (checkHotSpineServiceHealth() && hotspine_writer_ && hotspine_writer_->isHealthy()) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: HotSpine service became available after " 
                << elapsed_ms << "ms" << std::endl;
      
      // Notify callback if registered
      if (availability_callback_) {
        availability_callback_(true);
      }
      
      return true;
    }
    
    // Wait before next check
    std::this_thread::sleep_for(std::chrono::milliseconds(check_interval_ms));
  }
}

// ============================================================================
// Check HotSpine health and return status
// ============================================================================
HotSpineHealthStatus MarketDataCollector::checkHotSpineHealth() {
  if (!hotspine_writer_) {
    health_status_ = HotSpineHealthStatus::UNAVAILABLE;
    return health_status_;
  }
  
  if (hotspine_writer_->isHealthy()) {
    // Check for degraded conditions (high error rate, buffer nearly full)
    uint64_t errors = hotspine_writer_->getWriteErrors();
    
    if (errors > 0) {
      // If we have errors but writer is still responsive, consider it degraded
      health_status_ = HotSpineHealthStatus::DEGRADED;
    } else {
      health_status_ = HotSpineHealthStatus::HEALTHY;
    }
  } else {
    health_status_ = HotSpineHealthStatus::UNHEALTHY;
  }
  
  hotspine_available_ = (health_status_ == HotSpineHealthStatus::HEALTHY || 
                         health_status_ == HotSpineHealthStatus::DEGRADED);
  
  return health_status_;
}

// ============================================================================
// Get detailed HotSpine statistics
// ============================================================================
HotSpineStatistics MarketDataCollector::getHotSpineStatistics() {
  HotSpineStatistics stats;
  
  stats.shared_memory_path = config_.hotspine_shm_name;
  
  if (!hotspine_writer_) {
    stats.is_attached = false;
    return stats;
  }
  
  stats.is_attached = hotspine_writer_->isHealthy();
  stats.trades_written = hotspine_writer_->getTradesWritten();
  stats.write_errors = hotspine_writer_->getWriteErrors();
  stats.last_health_check = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  
  return stats;
}

// ============================================================================
// Simple boolean health check
// ============================================================================
bool MarketDataCollector::isHotSpineHealthy() {
  return checkHotSpineHealth() == HotSpineHealthStatus::HEALTHY;
}

// ============================================================================
// Log current health status
// ============================================================================
void MarketDataCollector::logHotSpineHealthStatus() {
  auto status = checkHotSpineHealth();
  auto stats = getHotSpineStatistics();
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] === HotSpine Health Status ===" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Status: " << healthStatusToString(status) << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Attached: " << (stats.is_attached ? "YES" : "NO") << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Shared Memory: " << stats.shared_memory_path << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Trades Written: " << stats.trades_written << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Write Errors: " << stats.write_errors << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO] ============================" << std::endl;
}

// ============================================================================
// Attempt to reconnect to HotSpine service
// ============================================================================
bool MarketDataCollector::reconnectHotSpine() {
  std::lock_guard<std::mutex> lock(health_mutex_);
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Attempting to reconnect to HotSpine..." << std::endl;
  
  if (reconnect_attempts_ >= static_cast<uint32_t>(config_.max_reconnect_attempts)) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Max reconnect attempts reached (" 
              << config_.max_reconnect_attempts << ")" << std::endl;
    return false;
  }
  
  reconnect_attempts_++;
  
  try {
    // Create new HotSpine writer
    auto new_writer = std::make_shared<HotSpine::HotSpineWriter>(config_.hotspine_shm_name);
    
    // Check if new writer is healthy
    if (new_writer && new_writer->isHealthy()) {
      hotspine_writer_ = new_writer;
      reconnect_attempts_ = 0;
      
      // Update processor with new writer
      if (processor_) {
        processor_->updateHotSpineWriter(hotspine_writer_);
      }
      
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Successfully reconnected to HotSpine" << std::endl;
      
      // Notify callback
      if (availability_callback_) {
        availability_callback_(true);
      }
      
      return true;
    }
    
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Reconnect attempt " 
              << reconnect_attempts_ << " failed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Reconnect exception: " 
              << e.what() << std::endl;
  }
  
  return false;
}

// ============================================================================
// Health monitoring loop (background thread)
// ============================================================================
void MarketDataCollector::healthMonitorLoop() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Health monitor thread started" << std::endl;
  
  while (running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(config_.health_check_interval_ms));
    
    if (!running_) break;
    
    auto status = checkHotSpineHealth();
    
    // If unhealthy, attempt reconnection
    if (status == HotSpineHealthStatus::UNHEALTHY || status == HotSpineHealthStatus::UNAVAILABLE) {
      std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: HotSpine unhealthy (status: " 
                << healthStatusToString(status) << "), attempting reconnection..." << std::endl;
      
      if (!reconnectHotSpine()) {
        std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Reconnection failed, will retry in " 
                  << config_.reconnect_interval_ms << "ms" << std::endl;
        
        // Wait before next health check
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_interval_ms));
      }
    }
    
    // Periodically log health status
    static int log_counter = 0;
    log_counter++;
    if (log_counter >= 6) {  // Every ~30 seconds with 5s interval
      log_counter = 0;
      logHotSpineHealthStatus();
    }
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Health monitor thread stopped" << std::endl;
}

// ============================================================================
// Start health monitoring thread
// ============================================================================
void MarketDataCollector::startHealthMonitorThread() {
  if (!config_.enable_health_monitoring) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Health monitoring disabled" << std::endl;
    return;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Starting health monitor thread" << std::endl;
  health_monitor_thread_ = std::thread(&MarketDataCollector::healthMonitorLoop, this);
}

// ============================================================================
// Stop health monitoring thread
// ============================================================================
void MarketDataCollector::stopHealthMonitorThread() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Stopping health monitor thread" << std::endl;
  
  if (health_monitor_thread_.joinable()) {
    health_monitor_thread_.join();
  }
}

// ============================================================================
// Set availability callback
// ============================================================================
void MarketDataCollector::setHotSpineAvailabilityCallback(std::function<void(bool)> callback) {
  availability_callback_ = callback;
}

// ============================================================================
// Log all discovered configuration
// ============================================================================
void MarketDataCollector::logDiscoveredConfiguration() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: === Discovered Configuration ===" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   HotSpine Shared Memory: " << config_.hotspine_shm_name << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   MS SQL enabled: " << (config_.enable_mssql ? "true" : "false") << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Exclusive HotSpine: " << (config_.enable_exclusive_hotspine ? "true" : "false") << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Timeframes: ";
  for (const auto& tf : config_.timeframes) {
    std::cout << tf << " ";
  }
  std::cout << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Exchanges: " << config_.exchanges.size() << std::endl;
  for (const auto& ex : config_.exchanges) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]     - " << ex.exchange_name << " (symbols: ";
    for (const auto& sym : ex.symbols) {
      std::cout << sym << " ";
    }
    std::cout << ", channels: ";
    for (const auto& ch : ex.channels) {
      std::cout << ch << " ";
    }
    std::cout << ", market_type: " << ex.market_type << ")" << std::endl;
  }
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Buffer sizes - Trades: " << config_.trade_buffer_size 
            << ", Candles: " << config_.candle_buffer_size
            << ", Orderbooks: " << config_.orderbook_buffer_size << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Intervals - Flush: " << config_.flush_interval_ms 
            << "ms, Stats: " << config_.stats_report_interval_s << "s" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO] === End Configuration ===" << std::endl;
}

MarketDataCollector::~MarketDataCollector() { stop(); }

void MarketDataCollector::start() {
  running_ = true;

  // Add comprehensive WebSocket debugging
  conn_mgr_->addWebSocketDebugging();

  // Log WebSocket debug information
  conn_mgr_->logWebSocketDebugInfo();

  // Log WebSocket status before starting
  conn_mgr_->logWebSocketStatus();

  // Check WebSocket connection health
  conn_mgr_->checkWebSocketConnection();

  // Monitor WebSocket data flow
  conn_mgr_->monitorWebSocketDataFlow();

  conn_mgr_->subscribe(config_.exchanges);
  conn_mgr_->start();

  // Add WebSocket debugging
  processor_->addWebSocketDebugging();

  // Run comprehensive WebSocket diagnostics
  conn_mgr_->diagnoseWebSocketIssues();

  // Log session status after starting
  conn_mgr_->logSessionStatus();

  // Check WebSocket connection health after starting
  conn_mgr_->checkWebSocketConnection();

  // Monitor WebSocket data flow after starting
  conn_mgr_->monitorWebSocketDataFlow();

  flush_thread_ = std::thread(&MarketDataCollector::flushLoop, this);
  stats_thread_ = std::thread(&MarketDataCollector::statsLoop, this);
  
  // Start health monitoring thread
  startHealthMonitorThread();
}

void MarketDataCollector::stop() {
  if (!running_) return;
  running_ = false;
  conn_mgr_->stop();

  if (flush_thread_.joinable()) flush_thread_.join();
  if (stats_thread_.joinable()) stats_thread_.join();
  
  // Stop health monitoring thread
  stopHealthMonitorThread();

  // final flush
  processor_->flushBuffers();

  // Final HotSpine flush
  if (hotspine_writer_) {
    hotspine_writer_->flushBatch();
  }

  auto final_candles = candle_agg_->flushAll();
  if (!final_candles.empty() && config_.enable_mssql) {
    // group per table
    std::unordered_map<std::string, std::vector<MarketData::OHLCV>> per_table;
    for (const auto& c : final_candles) {
      per_table[c.getTableName()].push_back(c);
    }
    for (auto& kv : per_table) {
      db_->bulkInsertOHLCV(kv.first, kv.second);
    }
  }
}

void MarketDataCollector::waitForShutdown() {
  while (running_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

void MarketDataCollector::printStats() const {
  auto st = processor_->getStats();
  std::cout << "=== Market Data Stats ===\n";
  std::cout << "Trades: received=" << st.trades_received << ", inserted=" << st.trades_inserted << ", rate=" << st.trades_per_sec << " /s\n";
  std::cout << "Candles: generated=" << st.candles_generated << ", inserted=" << st.candles_inserted << "\n";
  std::cout << "Orderbooks: received=" << st.orderbooks_received << ", inserted=" << st.orderbooks_inserted << ", rate=" << st.orderbooks_per_sec << " /s\n";
  std::cout << "Avg latency (ms): " << st.avg_latency_ms << "\n";
  std::cout << "Errors: " << st.errors << "\n";

  // scrape-friendly JSON line for Prom/Grafana → Loki/Tempo/etc.
  std::cout << "STATS_JSON " << processor_->getStatsJson() << "\n";
}

void MarketDataCollector::flushLoop() {
  while (running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(config_.flush_interval_ms));
    processor_->flushBuffers();

    // Also flush HotSpine batch if writer is available
    if (hotspine_writer_) {
      hotspine_writer_->flushBatch();
    }
  }
}

void MarketDataCollector::statsLoop() const {
  int health_check_counter = 0;
  int data_flow_counter = 0;
  int candle_validation_counter = 0;
  while (const_cast<std::atomic<bool>&>(running_)) {
    std::this_thread::sleep_for(std::chrono::seconds(config_.stats_report_interval_s));
    if (!const_cast<std::atomic<bool>&>(running_)) break;

    printStats();

    // Log HotSpine statistics if enabled
    if (hotspine_writer_) {
      std::string hotspine_stats = hotspine_writer_->getDetailedStats();
      std::cout << "[" << getCurrentTimestamp() << "][INFO] HotSpine Stats: " << hotspine_stats << std::endl;
    }

    // Log WebSocket data flow statistics every 3 stats intervals
    data_flow_counter++;
    if (data_flow_counter >= 3) {
      data_flow_counter = 0;
      const_cast<MarketDataProcessor*>(processor_.get())->logWebSocketDataFlowStats();
    }

    // Validate WebSocket data flow every 10 stats intervals
    static int validation_counter = 0;
    validation_counter++;
    if (validation_counter >= 10) {
      validation_counter = 0;
      const_cast<MarketDataProcessor*>(processor_.get())->validateWebSocketDataFlow();
    }

    // Validate candle aggregation every 7 stats intervals
    candle_validation_counter++;
    if (candle_validation_counter >= 7) {
      candle_validation_counter = 0;
      const_cast<CandleAggregator*>(candle_agg_.get())->validateCandleAggregation();
    }

    // Perform WebSocket health check every 5 stats intervals
    health_check_counter++;
    if (health_check_counter >= 5) {
      health_check_counter = 0;
      const_cast<ExchangeConnectionManager*>(conn_mgr_.get())->checkWebSocketConnection();
    }
  }
}

// ============================================================================
// Exchange Validation Methods
// ============================================================================

ValidationResult MarketDataCollector::validateAllExchanges() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validating all exchanges..." << std::endl;
  
  ValidationResult result;
  auto& registry = BTQuant::SymbolRegistry::instance();
  auto exchanges = registry.get_exchanges();
  
  result.total_exchanges = static_cast<int>(exchanges.size());
  
  for (const auto& exchange : exchanges) {
    bool is_valid = conn_mgr_->validateExchange(exchange);
    
    if (is_valid) {
      result.valid_items.push_back(exchange);
      result.valid_exchanges++;
      std::cout << "[" << getCurrentTimestamp() << "][DEBUG] MarketDataCollector: Exchange " 
                << exchange << " validated successfully" << std::endl;
    } else {
      result.invalid_items.push_back(exchange);
      result.warnings.push_back("Exchange " + exchange + " failed validation");
      std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Exchange " 
                << exchange << " failed validation" << std::endl;
    }
  }
  
  result.is_valid = result.invalid_items.empty();
  result.message = result.is_valid 
      ? "All " + std::to_string(static_cast<int>(result.valid_items.size())) + " exchanges validated successfully"
      : std::to_string(static_cast<int>(result.invalid_items.size())) + " exchanges failed validation";
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Exchange validation complete. "
            << "Valid: " << result.valid_items.size() << ", Invalid: " << result.invalid_items.size() << std::endl;
  
  return result;
}

bool MarketDataCollector::isExchangeSupported(const std::string& exchange_name) {
  return conn_mgr_->validateExchange(exchange_name);
}

std::vector<std::string> MarketDataCollector::getSupportedExchanges() {
  return conn_mgr_->getSupportedExchanges();
}

void MarketDataCollector::logUnsupportedExchanges(const std::vector<std::string>& unsupported) {
  std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Unsupported exchanges detected: " 
            << unsupported.size() << std::endl;
  
  auto supported = getSupportedExchanges();
  
  for (const auto& exchange : unsupported) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector:   - " << exchange 
              << " (not supported by CCAPI)" << std::endl;
    
    // Suggest similar supported exchanges
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector:   Alternatives: ";
    bool first = true;
    for (const auto& sup : supported) {
      // Simple similarity check - same first letter or containing similar characters
      if (!sup.empty() && !exchange.empty() && (sup[0] == exchange[0] || 
          sup.find(exchange.substr(0, std::min<size_t>(3, exchange.length()))) != std::string::npos)) {
        if (!first) std::cout << ", ";
        std::cout << sup;
        first = false;
      }
    }
    if (first) {
      std::cout << "(no close matches found)";
    }
    std::cout << std::endl;
  }
}

// ============================================================================
// Symbol Validation Methods
// ============================================================================

ValidationResult MarketDataCollector::validateExchangeSymbols(
    const std::string& exchange, 
    const std::vector<std::string>& symbols) {
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validating symbols for exchange: " 
            << exchange << std::endl;
  
  ValidationResult result;
  result.exchange_name = exchange;
  auto& registry = BTQuant::SymbolRegistry::instance();
  
  result.total_symbols = static_cast<int>(symbols.size());
  
  for (const auto& symbol : symbols) {
    bool is_valid = conn_mgr_->validateSymbol(exchange, symbol);
    
    if (is_valid) {
      result.valid_items.push_back(symbol);
      result.valid_symbols++;
      
      // Check if symbol is registered
      if (!registry.has_symbol(exchange, symbol)) {
        result.warnings.push_back("Symbol " + exchange + "/" + symbol + " validated but not in registry");
      }
    } else {
      result.invalid_items.push_back(symbol);
      result.warnings.push_back("Symbol " + exchange + "/" + symbol + " failed validation");
    }
  }
  
  result.is_valid = result.invalid_items.empty();
  result.message = result.is_valid 
      ? "All " + std::to_string(result.valid_symbols) + " symbols validated successfully"
      : std::to_string(result.invalid_items.size()) + " symbols failed validation";
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Symbol validation complete for " 
            << exchange << ". Valid: " << result.valid_symbols << ", Invalid: " << result.invalid_items.size() << std::endl;
  
  return result;
}

uint32_t MarketDataCollector::resolveSymbolId(const std::string& exchange, const std::string& symbol) {
  auto symbol_id = conn_mgr_->resolveSymbolId(exchange, symbol);
  return symbol_id ? *symbol_id : 0;
}

uint32_t MarketDataCollector::registerUnknownSymbol(const std::string& exchange, const std::string& symbol) {
  auto& registry = BTQuant::SymbolRegistry::instance();
  
  if (registry.has_symbol(exchange, symbol)) {
    auto existing_id = registry.get_symbol_id(exchange, symbol);
    if (existing_id) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Symbol " 
                << exchange << "/" << symbol << " already registered with ID " << *existing_id << std::endl;
      return *existing_id;
    }
  }
  
  uint32_t new_id = registry.register_symbol(exchange, symbol);
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Auto-registered new symbol " 
            << exchange << "/" << symbol << " with ID " << new_id << std::endl;
  
  validation_dirty_ = true;
  return new_id;
}

ValidationSummary MarketDataCollector::getValidationSummary() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Generating validation summary..." << std::endl;
  
  ValidationSummary summary;
  auto& registry = BTQuant::SymbolRegistry::instance();
  auto exchanges = registry.get_exchanges();
  
  summary.total_exchanges = static_cast<int>(exchanges.size());
  
  for (const auto& exchange : exchanges) {
    ValidationResult exchange_result;
    exchange_result.exchange_name = exchange;
    
    auto exchange_symbols = registry.get_exchange_symbols(exchange);
    summary.total_symbols += static_cast<int>(exchange_symbols.size());
    
    bool exchange_valid = conn_mgr_->validateExchange(exchange);
    if (exchange_valid) {
      summary.valid_exchanges++;
    }
    
    for (const auto& symbol_info : exchange_symbols) {
      bool symbol_valid = conn_mgr_->validateSymbol(exchange, symbol_info.symbol);
      
      if (symbol_valid) {
        summary.valid_symbols++;
        exchange_result.valid_items.push_back(symbol_info.symbol);
      } else {
        exchange_result.invalid_items.push_back(symbol_info.symbol);
        exchange_result.warnings.push_back("Symbol " + symbol_info.symbol + " validation failed");
      }
    }
    
    exchange_result.is_valid = exchange_result.invalid_items.empty();
    exchange_result.message = exchange_result.is_valid 
        ? "All " + std::to_string(exchange_result.valid_items.size()) + " symbols valid"
        : std::to_string(exchange_result.invalid_items.size()) + " symbols invalid";
    
    summary.exchange_results.push_back(exchange_result);
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validation Summary:"
            << " Exchanges: " << summary.valid_exchanges << "/" << summary.total_exchanges
            << ", Symbols: " << summary.valid_symbols << "/" << summary.total_symbols
            << ", Auto-registered: " << summary.auto_registered << std::endl;
  
  return summary;
}

// ============================================================================
// Runtime Symbol Addition Methods
// ============================================================================

bool MarketDataCollector::addSymbolAtRuntime(const std::string& exchange, const std::string& symbol) {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Adding symbol at runtime: " 
            << exchange << "/" << symbol << std::endl;
  
  // Validate exchange is supported
  if (!isExchangeSupported(exchange)) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Cannot add symbol - exchange " 
              << exchange << " not supported" << std::endl;
    return false;
  }
  
  auto& registry = BTQuant::SymbolRegistry::instance();
  
  // Register symbol if not exists
  uint32_t symbol_id = registry.register_symbol(exchange, symbol);
  
  // Create exchange config for the symbol
  ExchangeConnectionManager::ExchangeConfig config;
  config.exchange_name = exchange;
  config.symbols.push_back(symbol);
  config.channels = {"TRADE", "MARKET_DEPTH"};
  config.market_type = "spot";
  
  // Add to current exchanges
  {
    std::lock_guard<std::mutex> lock(conn_mgr_->getExchangesMutex());
    conn_mgr_->getCurrentExchangesRef().push_back(config);
  }
  
  // Create subscriptions for the new symbol
  auto subs = conn_mgr_->createExchangeSubscriptions(config);
  
  if (!subs.empty()) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Created " 
              << subs.size() << " subscriptions for " << exchange << "/" << symbol << std::endl;
    validation_dirty_ = true;
    return true;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Failed to create subscriptions for " 
            << exchange << "/" << symbol << std::endl;
  return false;
}

bool MarketDataCollector::addSymbolsAtRuntime(const std::string& exchange, 
                                               const std::vector<std::string>& symbols) {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Adding " << symbols.size() 
            << " symbols at runtime for exchange: " << exchange << std::endl;
  
  bool all_success = true;
  
  for (const auto& symbol : symbols) {
    if (!addSymbolAtRuntime(exchange, symbol)) {
      all_success = false;
    }
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Runtime symbol addition complete. "
            << "All successful: " << (all_success ? "YES" : "NO") << std::endl;
  
  return all_success;
}

bool MarketDataCollector::symbolExists(const std::string& exchange, const std::string& symbol) {
  auto& registry = BTQuant::SymbolRegistry::instance();
  return registry.has_symbol(exchange, symbol);
}

std::optional<SymbolInfo> MarketDataCollector::getSymbolInfo(const std::string& exchange, 
                                                               const std::string& symbol) {
  auto& registry = BTQuant::SymbolRegistry::instance();
  
  if (!registry.has_symbol(exchange, symbol)) {
    return std::nullopt;
  }
  
  auto symbol_id = registry.get_symbol_id(exchange, symbol);
  if (!symbol_id) {
    return std::nullopt;
  }
  
  SymbolInfo info;
  info.id = *symbol_id;
  info.symbol = symbol;
  info.exchange = exchange;
  info.market_type = "spot";
  info.is_registered = true;
  
  return info;
}

// ============================================================================
// Dynamic Validation Loop Methods
// ============================================================================

void MarketDataCollector::startValidationLoop(int interval_sec) {
  if (validation_running_) {
    std::cout << "[" << getCurrentTimestamp() << "][WARN] MarketDataCollector: Validation loop already running" << std::endl;
    return;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Starting validation loop with interval: " 
            << interval_sec << "s" << std::endl;
  
  validation_running_ = true;
  validation_thread_ = std::thread(&MarketDataCollector::validationLoop, this);
}

void MarketDataCollector::stopValidationLoop() {
  if (!validation_running_) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validation loop not running" << std::endl;
    return;
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Stopping validation loop" << std::endl;
  
  validation_running_ = false;
  
  if (validation_thread_.joinable()) {
    validation_thread_.join();
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validation loop stopped" << std::endl;
}

void MarketDataCollector::forceRevalidation() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Forcing re-validation..." << std::endl;
  
  validation_dirty_ = true;
  
  // Perform immediate validation
  auto summary = performValidation();
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Forced re-validation complete" << std::endl;
}

std::chrono::steady_clock::time_point MarketDataCollector::getLastValidationTime() {
  std::lock_guard<std::mutex> lock(validation_mutex_);
  return last_validation_time_;
}

// ============================================================================
// Validation Loop Implementation
// ============================================================================

void MarketDataCollector::validationLoop() {
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validation loop started" << std::endl;
  
  while (validation_running_) {
    std::this_thread::sleep_for(std::chrono::seconds(60));
    
    if (!validation_running_) break;
    
    // Check if validation is needed
    if (validation_dirty_) {
      performValidation();
    }
  }
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Validation loop exited" << std::endl;
}

ValidationSummary MarketDataCollector::performValidation() {
  std::lock_guard<std::mutex> lock(validation_mutex_);
  
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: Performing validation..." << std::endl;
  
  auto summary = getValidationSummary();
  
  // Log summary
  std::cout << "[" << getCurrentTimestamp() << "][INFO] MarketDataCollector: === Validation Summary ===" << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Exchanges: " << summary.valid_exchanges << "/" 
            << summary.total_exchanges << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Symbols: " << summary.valid_symbols << "/" 
            << summary.total_symbols << std::endl;
  std::cout << "[" << getCurrentTimestamp() << "][INFO]   Auto-registered: " << summary.auto_registered << std::endl;
  
  // Check for issues
  if (!summary.exchange_results.empty()) {
    std::cout << "[" << getCurrentTimestamp() << "][INFO]   Exchange Details:" << std::endl;
    for (const auto& result : summary.exchange_results) {
      std::cout << "[" << getCurrentTimestamp() << "][INFO]     " << result.exchange_name << ": " 
                << result.valid_items.size() << " valid, " << result.invalid_items.size() << " invalid" << std::endl;
      
      if (!result.warnings.empty()) {
        for (const auto& warning : result.warnings) {
          std::cout << "[" << getCurrentTimestamp() << "][WARN]       " << warning << std::endl;
        }
      }
    }
  }
  
  last_validation_time_ = std::chrono::steady_clock::now();
  validation_dirty_ = false;
  
  return summary;
}
