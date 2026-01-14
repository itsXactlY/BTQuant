#pragma once

#include <string>
#include <vector>
#include <optional>
#include <unordered_map>
#include <cstdint>
#include <algorithm>

namespace BTQuant {

struct SymbolInfo {
    uint32_t symbol_id;
    std::string exchange;
    std::string symbol;
    std::string full_name;
};

class SymbolRegistry {
public:
    static SymbolRegistry& getInstance() {
        static SymbolRegistry instance;
        return instance;
    }
    
    bool load_from_file(const std::string& filename) {
        (void)filename;
        return true;
    }
    
    bool save_to_file(const std::string& filename) const {
        (void)filename;
        return true;
    }
    
    uint32_t register_symbol(const std::string& exchange, const std::string& symbol) {
        static uint32_t next_id = 1;
        std::string key = exchange + ":" + symbol;
        if (symbol_map_.find(key) != symbol_map_.end()) {
            return symbol_map_[key];
        }
        uint32_t id = next_id++;
        symbol_map_[key] = id;
        SymbolInfo info{id, exchange, symbol, exchange + ":" + symbol};
        symbols_[id] = info;
        return id;
    }
    
    std::optional<uint32_t> get_symbol_id(const std::string& exchange, const std::string& symbol) const {
        std::string key = exchange + ":" + symbol;
        auto it = symbol_map_.find(key);
        if (it != symbol_map_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    std::optional<SymbolInfo> get_symbol_info(uint32_t symbol_id) const {
        auto it = symbols_.find(symbol_id);
        if (it != symbols_.end()) {
            return it->second;
        }
        return std::nullopt;
    }
    
    std::vector<SymbolInfo> get_all_symbols() const {
        std::vector<SymbolInfo> result;
        for (const auto& [id, info] : symbols_) {
            result.push_back(info);
        }
        return result;
    }
    
    std::vector<std::string> get_exchanges() const {
        std::vector<std::string> exchanges;
        for (const auto& [id, info] : symbols_) {
            if (std::find(exchanges.begin(), exchanges.end(), info.exchange) == exchanges.end()) {
                exchanges.push_back(info.exchange);
            }
        }
        return exchanges;
    }
    
private:
    SymbolRegistry() = default;
    ~SymbolRegistry() = default;
    SymbolRegistry(const SymbolRegistry&) = delete;
    SymbolRegistry& operator=(const SymbolRegistry&) = delete;
    
    std::unordered_map<std::string, uint32_t> symbol_map_;
    std::unordered_map<uint32_t, SymbolInfo> symbols_;
};

} // namespace BTQuant
