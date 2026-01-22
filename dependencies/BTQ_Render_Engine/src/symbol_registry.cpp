#include "symbol_registry.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace BTQuant {

SymbolRegistry& SymbolRegistry::instance() {
    static SymbolRegistry instance;
    return instance;
}

bool SymbolRegistry::load_from_file(const std::string& filepath) {
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open symbol registry file: " << filepath << std::endl;
            return false;
        }

        nlohmann::json json;
        file >> json;

        for (const auto& item : json) {
            SymbolInfo info;
            info.symbol = item["symbol"];
            info.exchange = item["exchange"];
            info.id = item["id"];

            symbols_[info.id] = info;
            if (info.id >= next_id_) {
                next_id_ = info.id + 1;
            }
        }

        std::cout << "Loaded " << symbols_.size() << " symbols from registry" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading symbol registry: " << e.what() << std::endl;
        return false;
    }
}

std::optional<SymbolInfo> SymbolRegistry::get_symbol_info(uint32_t symbol_id) const {
    auto it = symbols_.find(symbol_id);
    if (it != symbols_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<SymbolInfo> SymbolRegistry::get_all_symbols() const {
    std::vector<SymbolInfo> result;
    result.reserve(symbols_.size());
    for (const auto& pair : symbols_) {
        result.push_back(pair.second);
    }
    return result;
}

std::vector<std::string> SymbolRegistry::get_exchanges() const {
    std::vector<std::string> exchanges;
    for (const auto& pair : symbols_) {
        if (std::find(exchanges.begin(), exchanges.end(), pair.second.exchange) == exchanges.end()) {
            exchanges.push_back(pair.second.exchange);
        }
    }
    return exchanges;
}

std::vector<SymbolInfo> SymbolRegistry::get_exchange_symbols(const std::string& exchange) const {
    std::vector<SymbolInfo> result;
    for (const auto& pair : symbols_) {
        if (pair.second.exchange == exchange) {
            result.push_back(pair.second);
        }
    }
    return result;
}

std::optional<uint32_t> SymbolRegistry::get_symbol_id(const std::string& exchange, const std::string& symbol) const {
    for (const auto& pair : symbols_) {
        if (pair.second.exchange == exchange && pair.second.symbol == symbol) {
            return pair.second.id;
        }
    }
    return std::nullopt;
}

uint32_t SymbolRegistry::register_symbol(const std::string& exchange, const std::string& symbol) {
    // Check if already exists
    auto existing = get_symbol_id(exchange, symbol);
    if (existing) {
        return *existing;
    }

    // Create new symbol
    SymbolInfo info;
    info.symbol = symbol;
    info.exchange = exchange;
    info.id = next_id_++;

    symbols_[info.id] = info;
    return info.id;
}

bool SymbolRegistry::save_to_file(const std::string& filepath) const {
    try {
        nlohmann::json json = nlohmann::json::array();

        for (const auto& pair : symbols_) {
            nlohmann::json item;
            item["symbol"] = pair.second.symbol;
            item["exchange"] = pair.second.exchange;
            item["id"] = pair.second.id;
            json.push_back(item);
        }

        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
            return false;
        }

        file << json.dump(4); // Pretty print with 4 spaces
        std::cout << "Saved " << symbols_.size() << " symbols to registry" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving symbol registry: " << e.what() << std::endl;
        return false;
    }
}

} // namespace BTQuant