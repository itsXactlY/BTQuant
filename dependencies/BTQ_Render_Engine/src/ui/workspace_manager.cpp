#include "../../../include/ui/workspace_manager.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "../../../include/components/quant_workspace_component.hpp"
#include "../../../include/symbol_registry.hpp"

using json = nlohmann::json;

namespace BTQuant {

WorkspaceManager::WorkspaceManager(QuantWorkspaceComponent* workspace)
    : workspace_(workspace) {}

bool WorkspaceManager::export_workspace(const std::string& filepath) {
    try {
        json workspace_json;
        
        // Export panel layouts
        if (workspace_ && workspace_->getPanelManager()) {
            workspace_json["layout"] = workspace_->getPanelManager()->serialize_layout();
        }
        
        // Export symbol registry data
        auto& symbol_registry = SymbolRegistry::instance();
        auto symbols = symbol_registry.get_all_symbols();
        json symbols_json = json::array();
        
        for (const auto& symbol : symbols) {
            json symbol_json;
            symbol_json["id"] = symbol.id;
            symbol_json["name"] = symbol.name;
            symbol_json["exchange"] = symbol.exchange;
            symbol_json["base_currency"] = symbol.base_currency;
            symbol_json["quote_currency"] = symbol.quote_currency;
            symbol_json["min_notional"] = symbol.min_notional;
            symbol_json["min_qty"] = symbol.min_qty;
            symbol_json["max_qty"] = symbol.max_qty;
            symbol_json["step_size"] = symbol.step_size;
            symbol_json["tick_size"] = symbol.tick_size;
            symbol_json["status"] = symbol.status;
            symbols_json.push_back(symbol_json);
        }
        workspace_json["symbols"] = symbols_json;
        
        // Export settings (if any exist)
        json settings_json;
        // Add any global settings here
        workspace_json["settings"] = settings_json;
        
        // Write to file
        std::ofstream file(filepath);
        if (file.is_open()) {
            file << workspace_json.dump(4);
            file.close();
            std::cout << "Workspace exported successfully to " << filepath << std::endl;
            return true;
        } else {
            std::cerr << "Failed to open file for writing: " << filepath << std::endl;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error exporting workspace: " << e.what() << std::endl;
        return false;
    }
}

bool WorkspaceManager::import_workspace(const std::string& filepath) {
    try {
        // Read from file
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open file for reading: " << filepath << std::endl;
            return false;
        }
        
        std::string json_str((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        file.close();
        
        json workspace_json = json::parse(json_str);
        
        // Import panel layouts
        if (workspace_ && workspace_->getPanelManager() && workspace_json.contains("layout")) {
            workspace_->getPanelManager()->deserialize_layout(workspace_json["layout"]);
        }
        
        // Import symbol registry data
        if (workspace_json.contains("symbols")) {
            auto& symbol_registry = SymbolRegistry::instance();
            
            // Clear existing symbols
            symbol_registry.clear();
            
            // Add symbols from the imported data
            for (const auto& symbol_json : workspace_json["symbols"]) {
                SymbolInfo symbol_info;
                symbol_info.id = symbol_json["id"].get<uint32_t>();
                symbol_info.name = symbol_json["name"].get<std::string>();
                symbol_info.exchange = symbol_json["exchange"].get<std::string>();
                symbol_info.base_currency = symbol_json["base_currency"].get<std::string>();
                symbol_info.quote_currency = symbol_json["quote_currency"].get<std::string>();
                symbol_info.min_notional = symbol_json["min_notional"].get<double>();
                symbol_info.min_qty = symbol_json["min_qty"].get<double>();
                symbol_info.max_qty = symbol_json["max_qty"].get<double>();
                symbol_info.step_size = symbol_json["step_size"].get<double>();
                symbol_info.tick_size = symbol_json["tick_size"].get<double>();
                symbol_info.status = symbol_json["status"].get<std::string>();
                
                symbol_registry.add_symbol(symbol_info);
            }
            
            // Refresh hierarchical selector if it exists
            if (workspace_) {
                workspace_->refresh_hierarchical_selector();
            }
        }
        
        // Import settings (if any exist)
        if (workspace_json.contains("settings")) {
            // Apply settings from the imported data
            // This could include UI themes, default timeframes, etc.
        }
        
        std::cout << "Workspace imported successfully from " << filepath << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error importing workspace: " << e.what() << std::endl;
        return false;
    }
}

}  // namespace BTQuant