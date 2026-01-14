#include "symbol_manager.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>

namespace BTQuant {
namespace RenderEngine {

SymbolManager::SymbolManager()
    : symbol_registry_(&BTQuant::SymbolRegistry::instance())
    , auto_discovery_enabled_(true)
    , max_symbols_per_exchange_(500)
    , symbol_update_interval_ms_(1000)
{
    std::cout << "[SymbolManager] Initialized" << std::endl;
}

SymbolManager::~SymbolManager() {
    stopAutoDiscovery();
}

bool SymbolManager::initialize(const std::string& symbols_file, const std::string& config_file) {
    symbols_file_ = symbols_file;
    config_file_ = config_file;
    
    // Load existing symbol mappings
    if (!symbols_file_.empty()) {
        if (!symbol_registry_->load_from_file(symbols_file_)) {
            std::cout << "[SymbolManager] Warning: Could not load symbols from " << symbols_file_ << std::endl;
        }
    }
    
    // Load configuration
    if (!config_file_.empty()) {
        loadConfiguration(config_file_);
    }
    
    // Initialize exchange filters
    initializeExchangeFilters();
    
    std::cout << "[SymbolManager] Initialized with " << symbol_registry_->get_all_symbols().size() 
              << " symbols across " << symbol_registry_->get_exchanges().size() << " exchanges" << std::endl;
    
    return true;
}

void SymbolManager::startAutoDiscovery() {
    if (auto_discovery_enabled_ && !discovery_thread_.joinable()) {
        discovery_running_ = true;
        discovery_thread_ = std::thread(&SymbolManager::autoDiscoveryLoop, this);
        std::cout << "[SymbolManager] Started auto-discovery thread" << std::endl;
    }
}

void SymbolManager::stopAutoDiscovery() {
    if (discovery_thread_.joinable()) {
        discovery_running_ = false;
        discovery_thread_.join();
        std::cout << "[SymbolManager] Stopped auto-discovery thread" << std::endl;
    }
}

std::vector<SymbolInfo> SymbolManager::getAllSymbols() const {
    return symbol_registry_->get_all_symbols();
}

std::vector<SymbolInfo> SymbolManager::getExchangeSymbols(const std::string& exchange) const {
    return symbol_registry_->get_exchange_symbols(exchange);
}

std::vector<SymbolInfo> SymbolManager::getFilteredSymbols(const SymbolFilter& filter) const {
    auto all_symbols = symbol_registry_->get_all_symbols();
    std::vector<SymbolInfo> filtered_symbols;
    
    for (const auto& symbol : all_symbols) {
        if (matchesFilter(symbol, filter)) {
            filtered_symbols.push_back(symbol);
        }
    }
    
    // Apply sorting
    if (filter.sort_by == SymbolSortCriteria::ALPHABETICAL) {
        std::sort(filtered_symbols.begin(), filtered_symbols.end(),
                 [](const SymbolInfo& a, const SymbolInfo& b) {
                     return a.symbol < b.symbol;
                 });
    } else if (filter.sort_by == SymbolSortCriteria::EXCHANGE) {
        std::sort(filtered_symbols.begin(), filtered_symbols.end(),
                 [](const SymbolInfo& a, const SymbolInfo& b) {
                     if (a.exchange != b.exchange) {
                         return a.exchange < b.exchange;
                     }
                     return a.symbol < b.symbol;
                 });
    }
    
    // Apply limit
    if (filter.limit > 0 && filtered_symbols.size() > filter.limit) {
        filtered_symbols.resize(filter.limit);
    }
    
    return filtered_symbols;
}

std::vector<std::string> SymbolManager::getAvailableExchanges() const {
    return symbol_registry_->get_exchanges();
}

std::optional<SymbolInfo> SymbolManager::getSymbolInfo(uint32_t symbol_id) const {
    return symbol_registry_->get_symbol_info(symbol_id);
}

std::optional<uint32_t> SymbolManager::getSymbolId(const std::string& exchange, const std::string& symbol) const {
    return symbol_registry_->get_symbol_id(exchange, symbol);
}

uint32_t SymbolManager::registerSymbol(const std::string& exchange, const std::string& symbol, 
                                      const SymbolMetadata& metadata) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    
    // Register with symbol registry
    uint32_t symbol_id = symbol_registry_->register_symbol(exchange, symbol);
    
    // Store metadata
    symbol_metadata_[symbol_id] = metadata;
    
    // Update statistics
    updateExchangeStatistics(exchange);
    
    std::cout << "[SymbolManager] Registered symbol " << exchange << ":" << symbol 
              << " with ID " << symbol_id << std::endl;
    
    return symbol_id;
}

bool SymbolManager::updateSymbolMetadata(uint32_t symbol_id, const SymbolMetadata& metadata) {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    
    auto it = symbol_metadata_.find(symbol_id);
    if (it != symbol_metadata_.end()) {
        it->second = metadata;
        return true;
    }
    
    return false;
}

std::optional<SymbolMetadata> SymbolManager::getSymbolMetadata(uint32_t symbol_id) const {
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    
    auto it = symbol_metadata_.find(symbol_id);
    if (it != symbol_metadata_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

ExchangeStatistics SymbolManager::getExchangeStatistics(const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto it = exchange_stats_.find(exchange);
    if (it != exchange_stats_.end()) {
        return it->second;
    }
    
    return ExchangeStatistics{};
}

std::vector<ExchangeStatistics> SymbolManager::getAllExchangeStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    std::vector<ExchangeStatistics> stats;
    stats.reserve(exchange_stats_.size());
    
    for (const auto& [exchange, stat] : exchange_stats_) {
        stats.push_back(stat);
    }
    
    return stats;
}

bool SymbolManager::saveSymbolMappings() const {
    if (symbols_file_.empty()) {
        std::cerr << "[SymbolManager] No symbols file configured for saving" << std::endl;
        return false;
    }
    
    return symbol_registry_->save_to_file(symbols_file_);
}

bool SymbolManager::reloadSymbolMappings() {
    if (symbols_file_.empty()) {
        std::cerr << "[SymbolManager] No symbols file configured for reloading" << std::endl;
        return false;
    }
    
    std::cout << "[SymbolManager] Reloading symbol mappings from " << symbols_file_ << std::endl;
    return symbol_registry_->load_from_file(symbols_file_);
}

void SymbolManager::setExchangeFilter(const std::string& exchange, const ExchangeFilter& filter) {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    exchange_filters_[exchange] = filter;
    
    std::cout << "[SymbolManager] Set filter for exchange " << exchange 
              << " (enabled: " << filter.enabled << ", max_symbols: " << filter.max_symbols << ")" << std::endl;
}

std::optional<ExchangeFilter> SymbolManager::getExchangeFilter(const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(filters_mutex_);
    
    auto it = exchange_filters_.find(exchange);
    if (it != exchange_filters_.end()) {
        return it->second;
    }
    
    return std::nullopt;
}

void SymbolManager::enableAutoDiscovery(bool enabled) {
    auto_discovery_enabled_ = enabled;
    
    if (enabled) {
        startAutoDiscovery();
    } else {
        stopAutoDiscovery();
    }
    
    std::cout << "[SymbolManager] Auto-discovery " << (enabled ? "enabled" : "disabled") << std::endl;
}

void SymbolManager::setUpdateInterval(uint32_t interval_ms) {
    symbol_update_interval_ms_ = std::max(100u, interval_ms);  // Minimum 100ms
    std::cout << "[SymbolManager] Set update interval to " << symbol_update_interval_ms_ << "ms" << std::endl;
}

SymbolManagerStatistics SymbolManager::getStatistics() const {
    SymbolManagerStatistics stats;
    
    auto all_symbols = symbol_registry_->get_all_symbols();
    stats.total_symbols = all_symbols.size();
    stats.total_exchanges = symbol_registry_->get_exchanges().size();
    
    // Count symbols by exchange
    std::unordered_map<std::string, size_t> exchange_counts;
    for (const auto& symbol : all_symbols) {
        exchange_counts[symbol.exchange]++;
    }
    
    stats.symbols_per_exchange = exchange_counts;
    
    // Get metadata statistics
    std::lock_guard<std::mutex> lock(symbols_mutex_);
    stats.symbols_with_metadata = symbol_metadata_.size();
    
    return stats;
}

void SymbolManager::autoDiscoveryLoop() {
    std::cout << "[SymbolManager] Auto-discovery loop started" << std::endl;
    
    while (discovery_running_) {
        try {
            // Discover new symbols from active data sources
            discoverNewSymbols();
            
            // Update exchange statistics
            updateAllExchangeStatistics();
            
            // Save updated mappings periodically
            static int save_counter = 0;
            if (++save_counter % 60 == 0) {  // Save every 60 iterations
                saveSymbolMappings();
            }
            
        } catch (const std::exception& e) {
            std::cerr << "[SymbolManager] Auto-discovery error: " << e.what() << std::endl;
        }
        
        // Sleep for the configured interval
        std::this_thread::sleep_for(std::chrono::milliseconds(symbol_update_interval_ms_));
    }
    
    std::cout << "[SymbolManager] Auto-discovery loop stopped" << std::endl;
}

void SymbolManager::discoverNewSymbols() {
    // This would typically integrate with exchange APIs or data feeds
    // For now, we'll implement a placeholder that monitors for new symbols
    // in the HotSpine data stream
    
    // In a real implementation, this would:
    // 1. Check for new symbol IDs in the HotSpine stream
    // 2. Query exchange APIs for symbol information
    // 3. Register new symbols with appropriate metadata
    
    // Placeholder: Check if we have any unregistered symbol IDs
    // This would be integrated with the HotSpine data bridge
}

void SymbolManager::updateExchangeStatistics(const std::string& exchange) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto& stats = exchange_stats_[exchange];
    stats.exchange_name = exchange;
    
    auto exchange_symbols = symbol_registry_->get_exchange_symbols(exchange);
    stats.total_symbols = exchange_symbols.size();
    stats.last_update = std::chrono::high_resolution_clock::now();
    
    // Count active symbols (those with recent metadata updates)
    stats.active_symbols = 0;
    auto now = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> symbols_lock(symbols_mutex_);
    for (const auto& symbol : exchange_symbols) {
        auto metadata_it = symbol_metadata_.find(symbol.id);
        if (metadata_it != symbol_metadata_.end()) {
            auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                now - metadata_it->second.last_seen);
            if (time_diff.count() < 5) {  // Active if seen within 5 minutes
                stats.active_symbols++;
            }
        }
    }
}

void SymbolManager::updateAllExchangeStatistics() {
    auto exchanges = symbol_registry_->get_exchanges();
    for (const auto& exchange : exchanges) {
        updateExchangeStatistics(exchange);
    }
}

bool SymbolManager::matchesFilter(const SymbolInfo& symbol, const SymbolFilter& filter) const {
    // Check exchange filter
    if (!filter.exchanges.empty()) {
        if (std::find(filter.exchanges.begin(), filter.exchanges.end(), symbol.exchange) == filter.exchanges.end()) {
            return false;
        }
    }
    
    // Check symbol pattern
    if (!filter.symbol_pattern.empty()) {
        if (symbol.symbol.find(filter.symbol_pattern) == std::string::npos) {
            return false;
        }
    }
    
    // Check base currency
    if (!filter.base_currency.empty()) {
        // Extract base currency from symbol (assumes format like "BTCUSDT")
        std::string base = extractBaseCurrency(symbol.symbol);
        if (base != filter.base_currency) {
            return false;
        }
    }
    
    // Check quote currency
    if (!filter.quote_currency.empty()) {
        std::string quote = extractQuoteCurrency(symbol.symbol);
        if (quote != filter.quote_currency) {
            return false;
        }
    }
    
    // Check if symbol is active (has recent metadata)
    if (filter.active_only) {
        std::lock_guard<std::mutex> lock(symbols_mutex_);
        auto metadata_it = symbol_metadata_.find(symbol.id);
        if (metadata_it != symbol_metadata_.end()) {
            auto now = std::chrono::high_resolution_clock::now();
            auto time_diff = std::chrono::duration_cast<std::chrono::minutes>(
                now - metadata_it->second.last_seen);
            if (time_diff.count() >= 5) {  // Not active if not seen within 5 minutes
                return false;
            }
        } else if (filter.active_only) {
            return false;  // No metadata means not active
        }
    }
    
    return true;
}

std::string SymbolManager::extractBaseCurrency(const std::string& symbol) const {
    // Simple heuristic: assume common quote currencies and extract base
    std::vector<std::string> common_quotes = {"USDT", "USDC", "BTC", "ETH", "BNB", "USD", "EUR"};
    
    for (const auto& quote : common_quotes) {
        if (symbol.length() > quote.length() && 
            symbol.substr(symbol.length() - quote.length()) == quote) {
            return symbol.substr(0, symbol.length() - quote.length());
        }
    }
    
    // If no common quote found, assume first 3-4 characters are base
    return symbol.substr(0, std::min(4ul, symbol.length()));
}

std::string SymbolManager::extractQuoteCurrency(const std::string& symbol) const {
    // Simple heuristic: assume common quote currencies
    std::vector<std::string> common_quotes = {"USDT", "USDC", "BTC", "ETH", "BNB", "USD", "EUR"};
    
    for (const auto& quote : common_quotes) {
        if (symbol.length() > quote.length() && 
            symbol.substr(symbol.length() - quote.length()) == quote) {
            return quote;
        }
    }
    
    // If no common quote found, assume last 3-4 characters are quote
    return symbol.substr(std::max(0ul, symbol.length() - 4));
}

void SymbolManager::initializeExchangeFilters() {
    // Initialize default filters for known exchanges
    std::vector<std::string> known_exchanges = {"binance", "okx", "coinbase", "kraken", "bybit"};
    
    for (const auto& exchange : known_exchanges) {
        ExchangeFilter filter;
        filter.enabled = true;
        filter.max_symbols = max_symbols_per_exchange_;
        filter.priority_symbols = {"BTC", "ETH", "USDT", "BNB"};  // High priority symbols
        
        std::lock_guard<std::mutex> lock(filters_mutex_);
        exchange_filters_[exchange] = filter;
    }
}

bool SymbolManager::loadConfiguration(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[SymbolManager] Could not open config file: " << config_file << std::endl;
        return false;
    }
    
    // Simple configuration parsing (in a real implementation, use YAML/JSON parser)
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string key, value;
        if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            if (key == "auto_discovery") {
                auto_discovery_enabled_ = (value == "true");
            } else if (key == "max_symbols_per_exchange") {
                max_symbols_per_exchange_ = std::stoul(value);
            } else if (key == "update_interval_ms") {
                symbol_update_interval_ms_ = std::stoul(value);
            }
        }
    }
    
    std::cout << "[SymbolManager] Loaded configuration from " << config_file << std::endl;
    return true;
}

} // namespace RenderEngine
} // namespace BTQuant