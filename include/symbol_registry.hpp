#ifndef SYMBOL_REGISTRY_HPP
#define SYMBOL_REGISTRY_HPP

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

namespace PubBT {

/**
 * @brief A static symbol registry that maps symbol strings to unique identifiers
 * and provides metadata about financial instruments.
 */
class SymbolRegistry {
public:
    struct SymbolInfo {
        int id;
        std::string name;
        std::string exchange;
        double tick_size;
        double min_notional;
        
        SymbolInfo(int _id, const std::string& _name, const std::string& _exchange, 
                   double _tick_size = 0.01, double _min_notional = 0.0) 
            : id(_id), name(_name), exchange(_exchange), 
              tick_size(_tick_size), min_notional(_min_notional) {}
    };

private:
    static inline std::unordered_map<std::string, std::unique_ptr<SymbolInfo>> symbol_map_;
    static inline std::unordered_map<int, std::string> id_to_symbol_map_;
    static inline int next_id_ = 1;

public:
    /**
     * @brief Register a new symbol with the registry
     * @param symbol The symbol string (e.g., "BTCUSDT")
     * @param name The full name of the instrument (e.g., "Bitcoin/US Dollar Tether")
     * @param exchange The exchange where the symbol trades
     * @param tick_size Minimum price increment
     * @param min_notional Minimum trading amount
     * @return Unique ID for the symbol
     */
    static int registerSymbol(const std::string& symbol, 
                             const std::string& name,
                             const std::string& exchange,
                             double tick_size = 0.01,
                             double min_notional = 0.0) {
        if (symbol_map_.find(symbol) != symbol_map_.end()) {
            return symbol_map_[symbol]->id; // Return existing ID if symbol already registered
        }

        auto symbol_info = std::make_unique<SymbolInfo>(next_id_, name, exchange, tick_size, min_notional);
        int id = next_id_++;
        
        symbol_map_[symbol] = std::move(symbol_info);
        id_to_symbol_map_[id] = symbol;
        
        return id;
    }

    /**
     * @brief Get the unique ID for a symbol
     * @param symbol The symbol string
     * @return The unique ID, or -1 if symbol not found
     */
    static int getSymbolId(const std::string& symbol) {
        auto it = symbol_map_.find(symbol);
        return (it != symbol_map_.end()) ? it->second->id : -1;
    }

    /**
     * @brief Get the symbol string from its ID
     * @param id The unique ID
     * @return The symbol string, or empty string if ID not found
     */
    static std::string getSymbolFromId(int id) {
        auto it = id_to_symbol_map_.find(id);
        return (it != id_to_symbol_map_.end()) ? it->second : "";
    }

    /**
     * @brief Get symbol information
     * @param symbol The symbol string
     * @return Pointer to SymbolInfo, or nullptr if symbol not found
     */
    static const SymbolInfo* getSymbolInfo(const std::string& symbol) {
        auto it = symbol_map_.find(symbol);
        return (it != symbol_map_.end()) ? it->second.get() : nullptr;
    }

    /**
     * @brief Get symbol information by ID
     * @param id The unique ID
     * @return Pointer to SymbolInfo, or nullptr if ID not found
     */
    static const SymbolInfo* getSymbolInfoById(int id) {
        auto symbol = getSymbolFromId(id);
        return symbol.empty() ? nullptr : getSymbolInfo(symbol);
    }

    /**
     * @brief Check if a symbol exists in the registry
     * @param symbol The symbol string
     * @return True if symbol exists, false otherwise
     */
    static bool hasSymbol(const std::string& symbol) {
        return symbol_map_.find(symbol) != symbol_map_.end();
    }

    /**
     * @brief Get all registered symbols
     * @return Vector of all registered symbol strings
     */
    static std::vector<std::string> getAllSymbols() {
        std::vector<std::string> symbols;
        for (const auto& pair : symbol_map_) {
            symbols.push_back(pair.first);
        }
        return symbols;
    }

    /**
     * @brief Clear all registered symbols
     */
    static void clear() {
        symbol_map_.clear();
        id_to_symbol_map_.clear();
        next_id_ = 1;
    }
};

// Convenience functions for common operations
inline int registerSymbol(const std::string& symbol, 
                         const std::string& name = "",
                         const std::string& exchange = "",
                         double tick_size = 0.01,
                         double min_notional = 0.0) {
    return SymbolRegistry::registerSymbol(symbol, name, exchange, tick_size, min_notional);
}

inline int getSymbolId(const std::string& symbol) {
    return SymbolRegistry::getSymbolId(symbol);
}

inline std::string getSymbolFromId(int id) {
    return SymbolRegistry::getSymbolFromId(id);
}

inline const typename SymbolRegistry::SymbolInfo* getSymbolInfo(const std::string& symbol) {
    return SymbolRegistry::getSymbolInfo(symbol);
}

} // namespace PubBT

#endif // SYMBOL_REGISTRY_HPP