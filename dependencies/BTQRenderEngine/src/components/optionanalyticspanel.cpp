#include <iostream>
#include <string>
#include <vector>

// Simple Option Analytics Panel implementation with three tabs
class OptionAnalyticsPanel {
private:
    std::vector<std::string> tabs;
    int activeTab;

public:
    OptionAnalyticsPanel() : activeTab(0) {
        // Initialize the three tabs: Desk, Analyzer, Smile
        tabs.push_back("Desk");
        tabs.push_back("Analyzer");
        tabs.push_back("Smile");
    }

    void renderNavigationHeader() {
        std::cout << "=== Option Analytics Panel ===" << std::endl;
        
        // Render tab navigation header
        std::cout << "[ ";
        for (size_t i = 0; i < tabs.size(); ++i) {
            if (i == activeTab) {
                std::cout << "[" << tabs[i] << "]";  // Active tab in brackets
            } else {
                std::cout << tabs[i];               // Inactive tab
            }
            
            if (i < tabs.size() - 1) {
                std::cout << " | ";                 // Separator between tabs
            }
        }
        std::cout << " ]" << std::endl;
        std::cout << "=============================" << std::endl;
    }

    void switchTab(int tabIndex) {
        if (tabIndex >= 0 && tabIndex < static_cast<int>(tabs.size())) {
            activeTab = tabIndex;
        }
    }

    std::string getActiveTabName() const {
        if (activeTab >= 0 && activeTab < static_cast<int>(tabs.size())) {
            return tabs[activeTab];
        }
        return "";
    }

    void renderContent() {
        std::cout << "Content for tab: " << getActiveTabName() << std::endl;
        // Placeholder content for each tab
        if (getActiveTabName() == "Desk") {
            std::cout << "// Desk tab content would go here" << std::endl;
        } else if (getActiveTabName() == "Analyzer") {
            std::cout << "// Analyzer tab content would go here" << std::endl;
        } else if (getActiveTabName() == "Smile") {
            std::cout << "// Smile tab content would go here" << std::endl;
        }
    }
};

// Example usage
int main() {
    OptionAnalyticsPanel panel;
    
    // Render the navigation header with three tabs
    panel.renderNavigationHeader();
    
    // Example: Switch between tabs
    panel.switchTab(1);  // Switch to Analyzer tab
    std::cout << "Switched to: " << panel.getActiveTabName() << std::endl;
    
    panel.switchTab(2);  // Switch to Smile tab
    std::cout << "Switched to: " << panel.getActiveTabName() << std::endl;
    
    panel.switchTab(0);  // Switch back to Desk tab
    std::cout << "Switched to: " << panel.getActiveTabName() << std::endl;
    
    return 0;
}