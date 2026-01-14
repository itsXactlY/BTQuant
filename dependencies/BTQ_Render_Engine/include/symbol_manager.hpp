#pragma once

#include "../../tests/new/include/symbol_registry.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>

namespace BTQuant {
namespace RenderEngine {

// Symbol metadata for enhanced information
struct SymbolMetadata {
    std::string base_currency;
    std::string quote_currency;
    std::string market_type;  // "spot", "futures", "options"
    double tick_size = 0.0;
    double min_quantity = 0.0;
    double max_quantity = 0.0;
    bool is_active = true;
    std::chrono::high_resolution_clock::time_point last_seen;
    std::string description;
};

// Symbol filtering criteria
enum class SymbolSortCriteria {
    ALPHABETICAL,
    EXCHANGE,
    VOLUME,
    ACTIVITY
};

struct SymbolFilter {
    std::vector<std::string> exchanges;  // Filter by specific exchanges
    std::string symbol_pattern;          // Pattern matching for symbol names
    std::string base_currency;           // Filter by base currency (e.g., "BTC")
    std::string quote_currency;          // Filter by quote currency (e.g., "USDT")
    bool active_only = false;            // Only show active symbols
    SymbolSortCriteria sort_by = SymbolSortCriteria::ALPHABETICAL;
    size_t limit = 0;                    // Limit number of results (0 = no limit)
};

// Exchange-specific filtering and configuration
struct ExchangeFilter {
    bool enabled = true;
    size_t max_symbols = 1000;
    std::vector<std::string> priority_symbols;  // High-priority symbols to always include
    std::vector<std::string> excluded_symbols;  // Symbols to exclude
    std::string market_type_filter;             // Filter by market type
};

// Exchange statistics
struct ExchangeStatistics {
    std::string exchange_name;
    size_t total_symbols = 0;
    size_t active_symbols = 0;
    std::chrono::high_resolution_clock::time_point last_update;
};

// Symbol manager statistics
struct SymbolManagerStatistics {
    size_t total_symbols = 0;
    size_t total_exchanges = 0;
    size_t symbols_with_metadata = 0;
    std::unordered_map<std::string, size_t> symbols_per_exchange;
};

/**
 * SymbolManager - Advanced symbol management and discovery system
 * 
 * This class provides comprehensive symbol management capabilities including:
 * - Integration with existing symbol registry
 * - Dynamic symbol discovery and registration
 * - Symbol filtering and categorization
 * - Exchange-specific configuration
 * - Metadata management and caching
 * - Auto-discovery from data streams
 */
class SymbolManager {
public:
    SymbolManager();
    ~SymbolManager();
    
    // Non-copyable, non-movable
    SymbolManager(const SymbolManager&) = delete;
    SymbolManager& operator=(const SymbolManager&) = delete;
    SymbolManager(SymbolManager&&) = delete;
    SymbolManager& operator=(SymbolManager&&) = delete;
    
    /**
     * Initialize the symbol manager
     * @param symbols_file Path to symbol mappings file
     * @param config_file Path to configuration file
     * @return true if initialization successful
     */
    bool initialize(const std::string& symbols_file = "/dev/shm/btquant_symbols.json",
                   const std::string& config_file = "");
    
    /**
     * Start/stop auto-discovery of new symbols
     */
    void startAutoDiscovery();
    void stopAutoDiscovery();
    
    /**
     * Get all registered symbols
     * @return Vector of all symbol information
     */
    std::vector<SymbolInfo> getAllSymbols() const;
    
    /**
     * Get symbols for a specific exchange
     * @param exchange Exchange name
     * @return Vector of symbols for the exchange
     */
    std::vector<SymbolInfo> getExchangeSymbols(const std::string& exchange) const;
    
    /**
     * Get filtered symbols based on criteria
     * @param filter Filtering criteria
     * @return Vector of filtered symbols
     */
    std::vector<SymbolInfo> getFilteredSymbols(const SymbolFilter& filter) const;
    
    /**
     * Get list of available exchanges
     * @return Vector of exchange names
     */
    std::vector<std::string> getAvailableExchanges() const;
    
    /**
     * Get symbol information by ID
     * @param symbol_id Symbol ID
     * @return Symbol information if found
     */
    std::optional<SymbolInfo> getSymbolInfo(uint32_t symbol_id) const;
    
    /**
     * Get symbol ID by exchange and symbol name
     * @param exchange Exchange name
     * @param symbol Symbol name
     * @return Symbol ID if found
     */
    std::optional<uint32_t> getSymbolId(const std::string& exchange, const std::string& symbol) const;
    
    /**
     * Register a new symbol with metadata
     * @param exchange Exchange name
     * @param symbol Symbol name
     * @param metadata Symbol metadata
     * @return Assigned symbol ID
     */
    uint32_t registerSymbol(const std::string& exchange, const std::string& symbol, 
                           const SymbolMetadata& metadata = {});
    
    /**
     * Update symbol metadata
     * @param symbol_id Symbol ID
     * @param metadata Updated metadata
     * @return true if successful
     */
    bool updateSymbolMetadata(uint32_t symbol_id, const SymbolMetadata& metadata);
    
    /**
     * Get symbol metadata
     * @param symbol_id Symbol ID
     * @return Symbol metadata if found
     */
    std::optional<SymbolMetadata> getSymbolMetadata(uint32_t symbol_id) const;
    
    /**
     * Get exchange statistics
     * @param exchange Exchange name
     * @return Exchange statistics
     */
    ExchangeStatistics getExchangeStatistics(const std::string& exchange) const;
    
    /**
     * Get statistics for all exchanges
     * @return Vector of exchange statistics
     */
    std::vector<ExchangeStatistics> getAllExchangeStatistics() const;
    
    /**
     * Save symbol mappings to file
     * @return true if successful
     */
    bool saveSymbolMappings() const;
    
    /**
     * Reload symbol mappings from file
     * @return true if successful
     */
    bool reloadSymbolMappings();
    
    /**
     * Set exchange-specific filter
     * @param exchange Exchange name
     * @param filter Exchange filter configuration
     */
    void setExchangeFilter(const std::string& exchange, const ExchangeFilter& filter);
    
    /**
     * Get exchange filter
     * @param exchange Exchange name
     * @return Exchange filter if configured
     */
    std::optional<ExchangeFilter> getExchangeFilter(const std::string& exchange) const;
    
    /**
     * Configuration methods
     */
    void enableAutoDiscovery(bool enabled);
    void setUpdateInterval(uint32_t interval_ms);
    void setMaxSymbolsPerExchange(size_t max_symbols) { max_symbols_per_exchange_ = max_symbols; }
    
    /**
     * Get symbol manager statistics
     * @return Current statistics
     */
    SymbolManagerStatistics getStatistics() const;

private:
    // Core components
    BTQuant::SymbolRegistry* symbol_registry_;
    
    // Configuration
    std::string symbols_file_;
    std::string config_file_;
    std::atomic<bool> auto_discovery_enabled_;
    size_t max_symbols_per_exchange_;
    uint32_t symbol_update_interval_ms_;
    
    // Auto-discovery thread
    std::atomic<bool> discovery_running_{false};
    std::thread discovery_thread_;
    
    // Symbol metadata storage
    mutable std::mutex symbols_mutex_;
    std::unordered_map<uint32_t, SymbolMetadata> symbol_metadata_;
    
    // Exchange filters
    mutable std::mutex filters_mutex_;
    std::unordered_map<std::string, ExchangeFilter> exchange_filters_;
    
    // Exchange statistics
    mutable std::mutex stats_mutex_;
    std::unordered_map<std::string, ExchangeStatistics> exchange_stats_;
    
    // Private methods
    void autoDiscoveryLoop();
    void discoverNewSymbols();
    void updateExchangeStatistics(const std::string& exchange);
    void updateAllExchangeStatistics();
    bool matchesFilter(const SymbolInfo& symbol, const SymbolFilter& filter) const;
    std::string extractBaseCurrency(const std::string& symbol) const;
    std::string extractQuoteCurrency(const std::string& symbol) const;
    void initializeExchangeFilters();
    bool loadConfiguration(const std::string& config_file);
};

} // namespace RenderEngine
} // namespace BTQuant