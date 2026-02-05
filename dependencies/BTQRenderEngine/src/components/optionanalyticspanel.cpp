#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

#ifdef IMGUI_INCLUDE
#include "imgui.h"
#endif

// Structure to hold option data
struct OptionData {
    double strike;
    // Call data
    double call_bid;
    double call_ask;
    double call_delta;
    double call_gamma;
    // Put data
    double put_bid;
    double put_ask;
    double put_delta;
    double put_gamma;
    
    OptionData(double s = 0.0) : strike(s), 
        call_bid(0.0), call_ask(0.0), call_delta(0.0), call_gamma(0.0),
        put_bid(0.0), put_ask(0.0), put_delta(0.0), put_gamma(0.0) {}
};

// Simple Option Analytics Panel implementation with three tabs
class OptionAnalyticsPanel {
private:
    std::vector<std::string> tabs;
    int activeTab;
    std::vector<OptionData> optionsGrid;

public:
    OptionAnalyticsPanel() : activeTab(0) {
        // Initialize the three tabs: Desk, Analyzer, Smile
        tabs.push_back("Desk");
        tabs.push_back("Analyzer");
        tabs.push_back("Smile");
        
        // Initialize sample option data for demonstration
        initializeSampleData();
    }

    void initializeSampleData() {
        // Generate sample option data for strikes from 100 to 200 in increments of 5
        for (double strike = 100.0; strike <= 200.0; strike += 5.0) {
            OptionData opt(strike);
            
            // Generate sample values for calls and puts
            opt.call_bid = 15.0 + (strike - 150.0) * 0.1;  // Sample bid price
            opt.call_ask = opt.call_bid + 0.1;              // Ask is slightly higher than bid
            opt.call_delta = 0.1 + (strike - 100.0) * 0.01; // Delta increases with strike for calls
            opt.call_gamma = 0.02 - abs(strike - 150.0) * 0.0002; // Gamma peaks near ATM
            
            opt.put_bid = 15.0 - (strike - 150.0) * 0.1;   // Sample bid price for puts
            opt.put_ask = opt.put_bid + 0.1;                // Ask is slightly higher than bid
            opt.put_delta = -0.9 + (strike - 100.0) * 0.01; // Delta decreases with strike for puts
            opt.put_gamma = 0.02 - abs(strike - 150.0) * 0.0002; // Same gamma for puts
            
            optionsGrid.push_back(opt);
        }
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
            renderDeskTab();
        } else if (getActiveTabName() == "Analyzer") {
            std::cout << "// Analyzer tab content would go here" << std::endl;
        } else if (getActiveTabName() == "Smile") {
            std::cout << "// Smile tab content would go here" << std::endl;
        }
    }

    void renderDeskTab() {
        std::cout << "\n=== OPTIONS DESK ===" << std::endl;
        
        // Print header
        std::cout << std::setw(10) << std::left << "Calls" 
                  << std::setw(30) << std::left << " "
                  << std::setw(10) << std::left << "Strike" 
                  << std::setw(30) << std::left << " "
                  << std::setw(10) << std::right << "Puts" << std::endl;
                  
        std::cout << std::setw(10) << std::left << "Bid/Ask" 
                  << std::setw(8) << std::left << "Delta" 
                  << std::setw(8) << std::left << "Gamma" 
                  << std::setw(6) << std::left << "" 
                  << std::setw(10) << std::left << "" 
                  << std::setw(6) << std::left << "" 
                  << std::setw(10) << std::right << "Bid/Ask" 
                  << std::setw(8) << std::right << "Delta" 
                  << std::setw(8) << std::right << "Gamma" << std::endl;
                  
        std::cout << std::string(80, '-') << std::endl;
        
        // Print option data
        for (const auto& opt : optionsGrid) {
            std::cout << std::fixed << std::setprecision(2);
            
            // Left side - Calls (Bid/Ask/Delta/Gamma)
            std::cout << std::setw(6) << std::left << opt.call_bid 
                      << "/" << std::setw(3) << opt.call_ask 
                      << std::setw(8) << std::left << opt.call_delta 
                      << std::setw(8) << std::left << opt.call_gamma 
                      << std::setw(6) << std::left << "";
                      
            // Center - Strike
            std::cout << std::setw(10) << std::left << opt.strike 
                      << std::setw(6) << std::left << "";
                      
            // Right side - Puts (Bid/Ask/Delta/Gamma)
            std::cout << std::setw(6) << std::right << opt.put_bid 
                      << "/" << std::setw(3) << opt.put_ask 
                      << std::setw(8) << std::right << opt.put_delta 
                      << std::setw(8) << std::right << opt.put_gamma 
                      << std::endl;
        }
    }
};

// Example usage
int main() {
    OptionAnalyticsPanel panel;

    // Render the navigation header with three tabs
    panel.renderNavigationHeader();

    // Switch to Desk tab to see the options grid
    panel.switchTab(0);  // Switch to Desk tab
    std::cout << "Switched to: " << panel.getActiveTabName() << std::endl;
    
    // Render the content of the current tab
    panel.renderContent();

    return 0;
}