#include "symbol_registry.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// Simple JSON parser (no dependencies)
#include <regex>

namespace BTQuant {

SymbolRegistry& SymbolRegistry::instance() {
    static SymbolRegistry instance;
    return instance;
}

std::string SymbolRegistry::make_key(const std::string& exchange, 
                                     const std::string& symbol) const {
    return exchange + ":" + symbol;
}

bool SymbolRegistry::load_from_file(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open symbol mapping file: " << filepath << std::endl;
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        
        // Regex-based JSON parsing (good enough for our use case)
        // Using raw string with custom delimiter to handle quotes inside
        // TODO :: rework later for productive with nlohmann
        std::regex exchange_pattern(R"json("name"\s*:\s*"([^"]+)")json");
        std::regex symbol_pattern(R"json(\{"symbol"\s*:\s*"([^"]+)"\s*,\s*"id"\s*:\s*(\d+)\})json");
        
        std::sregex_iterator exchanges_begin(content.begin(), content.end(), exchange_pattern);
        std::sregex_iterator exchanges_end;
        
        for (auto it = exchanges_begin; it != exchanges_end; ++it) {
            std::string exchange = (*it)[1].str();
            
            // Find symbols for this exchange
            size_t exchange_pos = it->position();
            size_t next_exchange_pos = content.find(R"json("name")json", exchange_pos + 1);
            if (next_exchange_pos == std::string::npos) {
                next_exchange_pos = content.length();
            }
            
            std::string exchange_section = content.substr(exchange_pos, 
                                                         next_exchange_pos - exchange_pos);
            
            std::sregex_iterator symbols_begin(exchange_section.begin(), 
                                               exchange_section.end(), 
                                               symbol_pattern);
            std::sregex_iterator symbols_end;
            
            for (auto sym_it = symbols_begin; sym_it != symbols_end; ++sym_it) {
                std::string symbol = (*sym_it)[1].str();
                uint32_t id = std::stoul((*sym_it)[2].str());
                
                SymbolInfo info{id, exchange, symbol};
                id_to_info_[id] = info;
                key_to_id_[make_key(exchange, symbol)] = id;
            }
        }
        
        std::cout << "Loaded " << id_to_info_.size() 
                  << " symbol mappings from " << filepath << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading symbol mappings: " << e.what() << std::endl;
        return false;
    }
}

uint32_t SymbolRegistry::register_symbol(const std::string& exchange,
                                        const std::string& symbol,
                                        uint32_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = make_key(exchange, symbol);
    
    // Check if already registered
    auto it = key_to_id_.find(key);
    if (it != key_to_id_.end()) {
        return it->second;
    }
    
    // Use provided ID or auto-assign
    if (id == 0) {
        id = next_auto_id_++;
    }
    
    SymbolInfo info{id, exchange, symbol};
    id_to_info_[id] = info;
    key_to_id_[key] = id;
    
    return id;
}

std::optional<SymbolInfo> SymbolRegistry::get_symbol_info(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = id_to_info_.find(id);
    if (it != id_to_info_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::optional<uint32_t> SymbolRegistry::get_symbol_id(const std::string& exchange,
                                                       const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string key = make_key(exchange, symbol);
    auto it = key_to_id_.find(key);
    if (it != key_to_id_.end()) {
        return it->second;
    }
    return std::nullopt;
}

std::vector<SymbolInfo> SymbolRegistry::get_exchange_symbols(const std::string& exchange) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<SymbolInfo> result;
    for (const auto& [id, info] : id_to_info_) {
        if (info.exchange == exchange) {
            result.push_back(info);
        }
    }
    return result;
}

std::vector<std::string> SymbolRegistry::get_exchanges() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> result;
    for (const auto& [id, info] : id_to_info_) {
        if (std::find(result.begin(), result.end(), info.exchange) == result.end()) {
            result.push_back(info.exchange);
        }
    }
    return result;
}

bool SymbolRegistry::has_symbol(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return id_to_info_.find(id) != id_to_info_.end();
}

bool SymbolRegistry::has_symbol(const std::string& exchange, 
                               const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_key(exchange, symbol);
    return key_to_id_.find(key) != key_to_id_.end();
}

} // namespace BTQuant
