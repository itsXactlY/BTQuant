#pragma once

// ============================================================================
// MMT GENESIS - TPO Engine (Time Price Opportunity)
// Market Profile Analysis with ASCII Time Brackets
// ============================================================================

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

namespace BTQuant {
namespace Analytics {

// ============================================================================
// TPO Configuration
// ============================================================================

struct TPOConfig {
    uint64_t session_start_us = 0;          // Session start timestamp in microseconds
    uint64_t bracket_duration_us = 180000000; // 30 minutes in microseconds (default)
    double tick_size = 0.01;                 // Minimum price increment
    bool use_extended_hours = false;         // Include pre/post market
    size_t max_price_levels = 1000;          // Maximum price levels to track
};

// ============================================================================
// TPO Bracket - Single Time Bracket Character
// ============================================================================

struct TPOBracket {
    char character;           // ASCII character (A-Z, a-z)
    uint64_t start_time_us;   // Bracket start timestamp
    uint64_t end_time_us;     // Bracket end timestamp
    uint32_t bracket_index;   // Numerical index (0-51)
    
    TPOBracket() : character('?'), start_time_us(0), end_time_us(0), bracket_index(0) {}
    
    TPOBracket(uint32_t index, uint64_t start, uint64_t duration)
        : start_time_us(start), end_time_us(start + duration), bracket_index(index) {
        character = indexToChar(index);
    }
    
    // Convert bracket index to ASCII character
    static char indexToChar(uint32_t index) {
        if (index < 26) return 'A' + index;      // A-Z for first 26 brackets
        if (index < 52) return 'a' + (index - 26); // a-z for next 26 brackets
        return '?';  // Beyond 52 brackets
    }
    
    // Convert ASCII character to bracket index
    static uint32_t charToIndex(char c) {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return 26 + (c - 'a');
        return 0;
    }
};

// ============================================================================
// TPO Level - Price Level with TPO Characters
// ============================================================================

struct TPOLevel {
    double price;
    std::vector<char> tpo_chars;      // TPO characters at this level
    uint32_t tpo_count;               // Number of TPOs at this level
    double total_volume;              // Total volume at this level
    double buy_volume;                // Buy volume at this level
    double sell_volume;               // Sell volume at this level
    bool is_single_print;             // Single print flag
    bool is_in_value_area;            // Value area membership
    
    TPOLevel() : price(0.0), tpo_count(0), total_volume(0.0), 
                 buy_volume(0.0), sell_volume(0.0),
                 is_single_print(false), is_in_value_area(false) {}
    
    explicit TPOLevel(double p) : price(p), tpo_count(0), total_volume(0.0),
                                   buy_volume(0.0), sell_volume(0.0),
                                   is_single_print(false), is_in_value_area(false) {}
    
    // Add TPO character to this level
    void addTPO(char c) {
        // Check if character already exists
        if (std::find(tpo_chars.begin(), tpo_chars.end(), c) == tpo_chars.end()) {
            tpo_chars.push_back(c);
            tpo_count = static_cast<uint32_t>(tpo_chars.size());
        }
    }
    
    // Sort TPO characters for display
    void sortChars() {
        std::sort(tpo_chars.begin(), tpo_chars.end(), [](char a, char b) {
            return TPOBracket::charToIndex(a) < TPOBracket::charToIndex(b);
        });
    }
};

// ============================================================================
// TPO Engine - Market Profile Generator
// ============================================================================

class TPOEngine {
public:
    explicit TPOEngine(const TPOConfig& config = TPOConfig());
    ~TPOEngine() = default;
    
    // --- Core Processing ---
    
    // Process a trade and update TPO profile
    void processTrade(double price, double volume, uint64_t timestamp_us, bool is_buy);
    
    // Process OHLCV bar for TPO
    void processOHLCV(double open, double high, double low, double close, 
                      double volume, uint64_t timestamp_us);
    
    // --- Bracket Management ---
    
    // Get current time bracket
    TPOBracket getCurrentBracket(uint64_t timestamp_us) const;
    
    // Get bracket for a specific timestamp
    TPOBracket getBracketAt(uint64_t timestamp_us) const;
    
    // Get all brackets for the session
    std::vector<TPOBracket> getSessionBrackets() const;
    
    // --- Profile Analysis ---
    
    // Calculate Point of Control (POC) - price with most TPOs
    double calculatePOC() const;
    
    // Calculate Value Area (70% of volume around POC)
    std::pair<double, double> calculateValueArea(double percentage = 0.68) const;
    
    // Identify single prints (price levels with only 1 TPO)
    std::vector<double> identifySinglePrints() const;
    
    // Get initial balance (first 2 brackets range)
    std::pair<double, double> getInitialBalance() const;
    
    // --- Data Access ---
    
    // Get all TPO levels
    const std::map<double, TPOLevel>& getLevels() const { return tpo_levels_; }
    
    // Get TPO level at specific price
    const TPOLevel* getLevelAt(double price) const;
    
    // Get session statistics
    uint32_t getTotalTPOCount() const { return total_tpo_count_; }
    double getTotalVolume() const { return total_volume_; }
    double getSessionHigh() const { return session_high_; }
    double getSessionLow() const { return session_low_; }
    uint32_t getBracketCount() const { return current_bracket_index_ + 1; }
    
    // --- Reset & Configuration ---
    
    // Reset for new session
    void reset();
    
    // Update configuration
    void setConfig(const TPOConfig& config) { config_ = config; }
    const TPOConfig& getConfig() const { return config_; }
    
    // --- Rendering Data ---
    
    // Get TPO matrix for rendering (price -> TPO string)
    std::vector<std::pair<double, std::string>> getTPOMatrix() const;
    
    // Get sorted price levels (high to low)
    std::vector<TPOLevel> getSortedLevels() const;

private:
    // Price to bin index conversion (O(1))
    int64_t priceToBin(double price) const;
    double binToPrice(int64_t bin) const;
    
    // Update value area flags
    void updateValueArea();
    
    // Check and mark single prints
    void updateSinglePrints();
    
    // Configuration
    TPOConfig config_;
    
    // TPO Levels (price -> level)
    std::map<double, TPOLevel> tpo_levels_;
    mutable std::mutex levels_mutex_;
    
    // Session tracking
    std::atomic<uint32_t> current_bracket_index_{0};
    std::atomic<uint64_t> last_bracket_start_{0};
    std::atomic<uint32_t> total_tpo_count_{0};
    std::atomic<double> total_volume_{0.0};
    std::atomic<double> session_high_{0.0};
    std::atomic<double> session_low_{std::numeric_limits<double>::max()};
    
    // Value Area cache
    std::atomic<double> poc_price_{0.0};
    std::atomic<double> va_high_{0.0};
    std::atomic<double> va_low_{0.0};
};

// ============================================================================
// Inline Implementations
// ============================================================================

inline int64_t TPOEngine::priceToBin(double price) const {
    return static_cast<int64_t>(std::round(price / config_.tick_size));
}

inline double TPOEngine::binToPrice(int64_t bin) const {
    return bin * config_.tick_size;
}

inline TPOBracket TPOEngine::getCurrentBracket(uint64_t timestamp_us) const {
    return getBracketAt(timestamp_us);
}

inline TPOBracket TPOEngine::getBracketAt(uint64_t timestamp_us) const {
    if (config_.session_start_us == 0) {
        return TPOBracket();
    }
    
    uint64_t elapsed = timestamp_us - config_.session_start_us;
    uint32_t bracket_idx = static_cast<uint32_t>(elapsed / config_.bracket_duration_us);
    
    return TPOBracket(bracket_idx, 
                      config_.session_start_us + bracket_idx * config_.bracket_duration_us,
                      config_.bracket_duration_us);
}

inline const TPOLevel* TPOEngine::getLevelAt(double price) const {
    auto it = tpo_levels_.find(price);
    return it != tpo_levels_.end() ? &it->second : nullptr;
}

} // namespace Analytics
} // namespace BTQuant
