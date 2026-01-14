#pragma once

#include "stubs/symbol_registry.hpp"
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

class SymbolManager {
public:
    SymbolManager();
    ~SymbolManager();
    
    SymbolManager(const SymbolManager&) = delete;
    SymbolManager& operator=(const SymbolManager&) = delete;
    SymbolManager(SymbolManager&&) = delete;
    SymbolManager& operator=(SymbolManager&&) = delete;
    
    bool initialize(const std::string& symbols_file = "/dev/shm/btquant_symbols.json",
                   const std::string& config_file = "");
    
    void startAutoDiscovery();
    void stopAutoDiscovery();
    
    std::vector<SymbolInfo> getAllSymbols() const;
    std::vector<SymbolInfo> getExchangeSymbols(const std::string& exchange) const;
    std::vector<SymbolInfo> getFilteredSymbols(const SymbolFilter& filter) const;
    std::vector<std::string> getAvailableExchanges() const;
    
    std::optional<SymbolInfo> getSymbolInfo(uint32_t symbol_id) const;
    std::optional<uint32_t> getSymbolId(const std::string& exchange, const std::string& symbol) const;
    
    uint32_t registerSymbol(const std::string& exchange, const std::string& symbol, 
                           const SymbolMetadata& metadata = {});
    
    bool updateSymbolMetadata(uint32_t symbol_id, const SymbolMetadata& metadata);
    std::optional<SymbolMetadata> getSymbolMetadata(uint32_t symbol_id) const;
    
    ExchangeStatistics getExchangeStatistics(const std::string& exchange) const;
    std::vector<ExchangeStatistics> getAllExchangeStatistics() const;
    
    bool saveSymbolMappings() const;
    bool reloadSymbolMappings();
    
    void setExchangeFilter(const std::string& exchange, const ExchangeFilter& filter);
    std::optional<ExchangeFilter> getExchangeFilter(const std::string& exchange) const;
    
    void enableAutoDiscovery(bool enabled);
    void setUpdateInterval(uint32_t interval_ms);
    void setMaxSymbolsPerExchange(size_t max_symbols) { max_symbols_per_exchange_ = max_symbols; }
    
    SymbolManagerStatistics getStatistics() const;

private:
    BTQuant::SymbolRegistry* symbol_registry_;
    
    std::string symbols_file_;
    std::string config_file_;
    std::atomic<bool> auto_discovery_enabled_;
    size_t max_symbols_per_exchange_;
    uint32_t symbol_update_interval_ms_;
    
    std::atomic<bool> discovery_running_{false};
    std::thread discovery_thread_;
    
    mutable std::mutex symbols_mutex_;
    std::unordered_map<uint32_t, SymbolMetadata> symbol_metadata_;
    
    mutable std::mutex filters_mutex_;
    std::unordered_map<std::string, ExchangeFilter> exchange_filters_;
    
    mutable std::mutex stats_mutex_;
    std::unordered_map<std::string, ExchangeStatistics> exchange_stats_;
    
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