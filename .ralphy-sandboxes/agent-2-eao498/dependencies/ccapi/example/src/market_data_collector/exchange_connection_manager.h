#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <mutex>

#include "ccapi_cpp/ccapi_session.h"
#include "market_data_processor.h"

// Forward declaration for SymbolRegistry
namespace BTQuant {
class SymbolRegistry;
}

class ExchangeConnectionManager {
public:
    struct ExchangeConfig {
        std::string exchange_name;
        std::vector<std::string> symbols;
        std::vector<std::string> channels;  // "TRADE", "MARKET_DEPTH"
        std::string market_type;            // "spot", "perpetual"
    };

    explicit ExchangeConnectionManager(
        std::shared_ptr<MarketDataProcessor> processor,
        const ConfigTypes::DebugConfig& debug_config = {});

    // =========================================================================
    // Dynamic Subscription Methods
    // =========================================================================
    
    // Subscribe to exchanges with symbol validation and ID resolution
    void subscribe(const std::vector<ExchangeConfig>& exchanges);
    
    // Refresh subscriptions by re-reading from config
    void refreshSubscriptions();
    
    // Add an exchange at runtime
    void addExchange(const ExchangeConfig& exchange);
    
    // Remove an exchange at runtime
    void removeExchange(const std::string& exchange_name);
    
    // =========================================================================
    // Validation Methods
    // =========================================================================
    
    // Validate that an exchange is supported
    bool validateExchange(const std::string& exchange_name) const;
    
    // Validate that a symbol exists for an exchange
    bool validateSymbol(const std::string& exchange, const std::string& symbol) const;
    
    // =========================================================================
    // Discovery Helper Methods
    // =========================================================================
    
    // Get list of supported exchanges
    std::vector<std::string> getSupportedExchanges() const;
    
    // Get all symbols for a specific exchange
    std::vector<std::string> getExchangeSymbols(const std::string& exchange) const;
    
    // =========================================================================
    // Lifecycle Methods
    // =========================================================================
    void start();
    void stop();
    
    // =========================================================================
    // Debug and Diagnostic Methods
    // =========================================================================
    void logSessionStatus() const;
    void logWebSocketStatus() const;
    void checkWebSocketConnection() const;
    void logWebSocketDebugInfo() const;
    void monitorWebSocketDataFlow() const;
    void addWebSocketDebugging() const;
    void diagnoseWebSocketIssues() const;

    bool isRunning() const { return running_; }
    
    // =========================================================================
    // Public Accessor Methods for Runtime Management
    // =========================================================================
    
    // Get current exchanges (thread-safe)
    std::vector<ExchangeConfig> getCurrentExchanges() const {
        std::lock_guard<std::mutex> lock(exchanges_mutex_);
        return current_exchanges_;
    }
    
    // Get reference to exchanges mutex for external locking
    std::mutex& getExchangesMutex() {
        return exchanges_mutex_;
    }
    
    // Get reference to current exchanges for modification (must hold mutex)
    std::vector<ExchangeConfig>& getCurrentExchangesRef() {
        return current_exchanges_;
    }
    
    // Create subscriptions for an exchange (public for runtime symbol addition)
    std::vector<ccapi::Subscription> createExchangeSubscriptions(
        const ExchangeConfig& exchange_config);
    
    // Resolve symbol to ID (public for validation)
    std::optional<uint32_t> resolveSymbolId(const std::string& exchange, 
                                             const std::string& symbol);
    
    // Create subscription with resolved symbol ID (public for runtime symbol addition)
    ccapi::Subscription createSubscription(const ExchangeConfig& cfg,
                                           const std::string& symbol,
                                           const std::string& channel);

private:
    // =========================================================================
    // Internal Helpers
    // =========================================================================
    
    // Check if an exchange is supported by CCAPI
    bool isExchangeSupported(const std::string& exchange_name) const;
    
    // Log a warning for unknown exchange/symbol
    void logUnknownExchangeWarning(const std::string& exchange_name) const;
    void logUnknownSymbolWarning(const std::string& exchange, 
                                  const std::string& symbol) const;

    // =========================================================================
    // Member Variables
    // =========================================================================
    std::unique_ptr<ccapi::SessionOptions> session_options_;
    std::unique_ptr<ccapi::SessionConfigs> session_configs_;
    std::unique_ptr<ccapi::Session> session_;
    std::shared_ptr<MarketDataProcessor> processor_;

    // Debug configuration
    ConfigTypes::DebugConfig debug_config_;

    // Cached resolved symbol IDs for performance: key = "exchange:symbol", value = symbol_id
    std::unordered_map<std::string, uint32_t> symbol_id_cache_;
    mutable std::mutex cache_mutex_;

    // Current subscriptions for runtime management
    std::vector<ExchangeConfig> current_exchanges_;
    mutable std::mutex exchanges_mutex_;

    std::atomic<bool> running_{false};
};
