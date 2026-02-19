#include "../../include/analytics/tpo_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace BTQuant {
namespace Analytics {

TPOEngine::TPOEngine(const TPOConfig& config)
    : config_(config) {
}

void TPOEngine::processTrade(double price, double volume, uint64_t timestamp_us, bool is_buy) {
    // Initialize session start if not set
    if (config_.session_start_us == 0) {
        config_.session_start_us = timestamp_us;
        last_bracket_start_.store(timestamp_us, std::memory_order_release);
    }
    
    // Get current bracket
    TPOBracket bracket = getBracketAt(timestamp_us);
    
    // Update current bracket index
    uint32_t prev_index = current_bracket_index_.load(std::memory_order_acquire);
    if (bracket.bracket_index > prev_index) {
        current_bracket_index_.store(bracket.bracket_index, std::memory_order_release);
        last_bracket_start_.store(bracket.start_time_us, std::memory_order_release);
    }
    
    // Update session high/low
    double current_high = session_high_.load(std::memory_order_acquire);
    while (price > current_high) {
        if (session_high_.compare_exchange_weak(current_high, price, 
                                                 std::memory_order_release, 
                                                 std::memory_order_acquire)) {
            break;
        }
    }
    
    double current_low = session_low_.load(std::memory_order_acquire);
    while (price < current_low) {
        if (session_low_.compare_exchange_weak(current_low, price,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {
            break;
        }
    }
    
    // Update TPO level
    {
        std::lock_guard<std::mutex> lock(levels_mutex_);
        
        // Find or create level
        auto& level = tpo_levels_[price];
        level.price = price;
        
        // Add TPO character if not already present
        level.addTPO(bracket.character);
        
        // Update volume
        level.total_volume += volume;
        if (is_buy) {
            level.buy_volume += volume;
        } else {
            level.sell_volume += volume;
        }
        
        // Update totals
        total_volume_.fetch_add(volume, std::memory_order_relaxed);
        total_tpo_count_.store(static_cast<uint32_t>(tpo_levels_.size()), std::memory_order_relaxed);
    }
    
    // Update value area and single prints periodically
    if (bracket.bracket_index > prev_index) {
        updateValueArea();
        updateSinglePrints();
    }
}

void TPOEngine::processOHLCV(double open, double high, double low, double close,
                              double volume, uint64_t timestamp_us) {
    // Process the range of prices touched by this bar
    // This ensures all price levels in the bar get the TPO
    
    TPOBracket bracket = getBracketAt(timestamp_us);
    
    // Initialize session start if not set
    if (config_.session_start_us == 0) {
        config_.session_start_us = timestamp_us;
        last_bracket_start_.store(timestamp_us, std::memory_order_release);
    }
    
    // Update current bracket index
    uint32_t prev_index = current_bracket_index_.load(std::memory_order_acquire);
    if (bracket.bracket_index > prev_index) {
        current_bracket_index_.store(bracket.bracket_index, std::memory_order_release);
        last_bracket_start_.store(bracket.start_time_us, std::memory_order_release);
    }
    
    // Update session high/low
    double current_high = session_high_.load(std::memory_order_acquire);
    while (high > current_high) {
        if (session_high_.compare_exchange_weak(current_high, high,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
            break;
        }
    }
    
    double current_low = session_low_.load(std::memory_order_acquire);
    while (low < current_low) {
        if (session_low_.compare_exchange_weak(current_low, low,
                                                std::memory_order_release,
                                                std::memory_order_acquire)) {
            break;
        }
    }
    
    // Add TPO to all price levels in the bar range
    {
        std::lock_guard<std::mutex> lock(levels_mutex_);
        
        int64_t low_bin = priceToBin(low);
        int64_t high_bin = priceToBin(high);
        
        for (int64_t bin = low_bin; bin <= high_bin; ++bin) {
            double price = binToPrice(bin);
            auto& level = tpo_levels_[price];
            level.price = price;
            level.addTPO(bracket.character);
            
            // Distribute volume across levels (simplified)
            double volume_per_level = volume / static_cast<double>(high_bin - low_bin + 1);
            level.total_volume += volume_per_level;
        }
        
        total_volume_.fetch_add(volume, std::memory_order_relaxed);
        total_tpo_count_.store(static_cast<uint32_t>(tpo_levels_.size()), std::memory_order_relaxed);
    }
    
    // Update value area and single prints
    if (bracket.bracket_index > prev_index) {
        updateValueArea();
        updateSinglePrints();
    }
}

std::vector<TPOBracket> TPOEngine::getSessionBrackets() const {
    std::vector<TPOBracket> brackets;
    uint32_t count = current_bracket_index_.load(std::memory_order_acquire) + 1;
    
    for (uint32_t i = 0; i < count && i < 52; ++i) {
        brackets.emplace_back(i, 
                              config_.session_start_us + i * config_.bracket_duration_us,
                              config_.bracket_duration_us);
    }
    
    return brackets;
}

double TPOEngine::calculatePOC() const {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    if (tpo_levels_.empty()) {
        return 0.0;
    }
    
    double poc_price = 0.0;
    uint32_t max_tpos = 0;
    
    for (const auto& [price, level] : tpo_levels_) {
        if (level.tpo_count > max_tpos) {
            max_tpos = level.tpo_count;
            poc_price = price;
        }
    }
    
    return poc_price;
}

std::pair<double, double> TPOEngine::calculateValueArea(double percentage) const {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    if (tpo_levels_.empty()) {
        return {0.0, 0.0};
    }
    
    // Calculate total TPOs
    uint32_t total_tpos = 0;
    for (const auto& [price, level] : tpo_levels_) {
        total_tpos += level.tpo_count;
    }
    
    // Target TPOs for value area
    uint32_t target_tpos = static_cast<uint32_t>(total_tpos * percentage);
    
    // Find POC
    double poc_price = 0.0;
    uint32_t max_tpos = 0;
    for (const auto& [price, level] : tpo_levels_) {
        if (level.tpo_count > max_tpos) {
            max_tpos = level.tpo_count;
            poc_price = price;
        }
    }
    
    // Expand outward from POC until we reach target
    auto poc_it = tpo_levels_.find(poc_price);
    if (poc_it == tpo_levels_.end()) {
        return {0.0, 0.0};
    }
    
    double va_high = poc_price;
    double va_low = poc_price;
    uint32_t accumulated_tpos = poc_it->second.tpo_count;
    
    auto upper_it = poc_it;
    auto lower_it = poc_it;
    
    while (accumulated_tpos < target_tpos) {
        // Move upper iterator up
        ++upper_it;
        // Move lower iterator down
        bool has_upper = (upper_it != tpo_levels_.end());
        bool has_lower = (lower_it != tpo_levels_.begin());
        
        if (!has_upper && !has_lower) {
            break;
        }
        
        // Choose direction with more TPOs
        uint32_t upper_tpos = has_upper ? upper_it->second.tpo_count : 0;
        uint32_t lower_tpos = has_lower ? (--lower_it)->second.tpo_count : 0;
        
        if (has_upper && (!has_lower || upper_tpos >= lower_tpos)) {
            accumulated_tpos += upper_tpos;
            va_high = upper_it->first;
        } else if (has_lower) {
            accumulated_tpos += lower_tpos;
            va_low = lower_it->first;
        }
    }
    
    return {va_low, va_high};
}

std::vector<double> TPOEngine::identifySinglePrints() const {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    std::vector<double> single_prints;
    
    for (const auto& [price, level] : tpo_levels_) {
        if (level.tpo_count == 1) {
            single_prints.push_back(price);
        }
    }
    
    return single_prints;
}

std::pair<double, double> TPOEngine::getInitialBalance() const {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    if (tpo_levels_.empty()) {
        return {0.0, 0.0};
    }
    
    // Find prices with TPO 'A' or 'B' (first two brackets)
    double ib_high = 0.0;
    double ib_low = std::numeric_limits<double>::max();
    
    for (const auto& [price, level] : tpo_levels_) {
        for (char c : level.tpo_chars) {
            if (c == 'A' || c == 'B') {
                ib_high = std::max(ib_high, price);
                ib_low = std::min(ib_low, price);
                break;
            }
        }
    }
    
    if (ib_low == std::numeric_limits<double>::max()) {
        return {0.0, 0.0};
    }
    
    return {ib_low, ib_high};
}

void TPOEngine::reset() {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    tpo_levels_.clear();
    config_.session_start_us = 0;
    current_bracket_index_.store(0, std::memory_order_release);
    last_bracket_start_.store(0, std::memory_order_release);
    total_tpo_count_.store(0, std::memory_order_release);
    total_volume_.store(0.0, std::memory_order_release);
    session_high_.store(0.0, std::memory_order_release);
    session_low_.store(std::numeric_limits<double>::max(), std::memory_order_release);
    poc_price_.store(0.0, std::memory_order_release);
    va_high_.store(0.0, std::memory_order_release);
    va_low_.store(0.0, std::memory_order_release);
}

std::vector<std::pair<double, std::string>> TPOEngine::getTPOMatrix() const {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    std::vector<std::pair<double, std::string>> matrix;
    matrix.reserve(tpo_levels_.size());
    
    for (const auto& [price, level] : tpo_levels_) {
        std::string tpo_str(level.tpo_chars.begin(), level.tpo_chars.end());
        matrix.emplace_back(price, std::move(tpo_str));
    }
    
    // Sort by price descending (high to low)
    std::sort(matrix.begin(), matrix.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    return matrix;
}

std::vector<TPOLevel> TPOEngine::getSortedLevels() const {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    std::vector<TPOLevel> levels;
    levels.reserve(tpo_levels_.size());
    
    for (const auto& [price, level] : tpo_levels_) {
        levels.push_back(level);
    }
    
    // Sort by price descending (high to low)
    std::sort(levels.begin(), levels.end(),
              [](const TPOLevel& a, const TPOLevel& b) { return a.price > b.price; });
    
    return levels;
}

void TPOEngine::updateValueArea() {
    auto [va_low, va_high] = calculateValueArea(0.68);
    va_low_.store(va_low, std::memory_order_release);
    va_high_.store(va_high, std::memory_order_release);
    
    double poc = calculatePOC();
    poc_price_.store(poc, std::memory_order_release);
    
    // Update value area flags
    std::lock_guard<std::mutex> lock(levels_mutex_);
    for (auto& [price, level] : tpo_levels_) {
        level.is_in_value_area = (price >= va_low && price <= va_high);
    }
}

void TPOEngine::updateSinglePrints() {
    std::lock_guard<std::mutex> lock(levels_mutex_);
    
    for (auto& [price, level] : tpo_levels_) {
        level.is_single_print = (level.tpo_count == 1);
    }
}

} // namespace Analytics
} // namespace BTQuant
