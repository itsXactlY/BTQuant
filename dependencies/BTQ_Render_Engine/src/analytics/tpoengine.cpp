/// @file tpoengine.cpp
/// @brief Implements the TPO (Time Price Opportunity) engine for market profile analysis.

#include "analytics/tpoengine.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

#include "hotspine_layout_v3.hpp"

namespace BTQuant {

TPOEngine::TPOEngine()
    : session_open_us_(0)
    , bins_(nullptr)
    , bin_count_(0)
    , day_low_(0.0)
    , tick_size_(0.0) {}

void TPOEngine::initialize(uint64_t session_open_time_us, double day_low, double tick_size, size_t max_bins) {
    session_open_us_ = session_open_time_us;
    day_low_ = day_low;
    tick_size_ = tick_size;
    bin_count_ = max_bins;
}

void TPOEngine::set_cluster_data(HotSpine::V3::VolumeNode* bins, size_t count) {
    bins_ = bins;
    bin_count_ = count;
}

uint8_t TPOEngine::timestamp_to_bracket(uint64_t ts_us) {
    // Minutes since session open (session_open_us_ set at day start)
    uint64_t minutes = (ts_us - session_open_us_) / (60ULL * 1000000ULL);
    return static_cast<uint8_t>(std::min(minutes / 30, uint64_t(15))); // Max 16 brackets (0-15)
}

char TPOEngine::bracket_to_char(uint8_t bracket) {
    return bracket < 26 ? ('A' + bracket) : ('a' + bracket - 26);
}

void TPOEngine::update_tpo_bit(uint64_t timestamp_us, double price) {
    if (!bins_) return;
    
    size_t bin = static_cast<size_t>((price - day_low_) / tick_size_);
    if (bin >= bin_count_) return;

    uint8_t bracket = timestamp_to_bracket(timestamp_us);
    
    // Set bit in VolumeNode::tpo_bits atomically
    auto* raw = reinterpret_cast<std::atomic<uint16_t>*>(&bins_[bin].tpo_bits);
    raw->fetch_or(uint16_t(1 << bracket), std::memory_order_relaxed);
}

void TPOEngine::calculate_value_area(size_t& va_low_bin, size_t& va_high_bin) {
    if (!bins_ || bin_count_ == 0) {
        va_low_bin = 0;
        va_high_bin = 0;
        return;
    }

    // Count total TPO characters across all bins
    size_t total_chars = 0;
    for (size_t i = 0; i < bin_count_; ++i) {
        total_chars += std::popcount(bins_[i].tpo_bits);
    }

    size_t va_target = static_cast<size_t>(total_chars * 0.68); // 68% target
    
    // Start from POC and expand outward
    size_t poc = find_poc_bin();
    size_t lo = poc, hi = poc;
    size_t enclosed = std::popcount(bins_[poc].tpo_bits);

    while (enclosed < va_target && (lo > 0 || hi < bin_count_ - 1)) {
        size_t add_lo = (lo > 0) ? std::popcount(bins_[lo-1].tpo_bits) : 0;
        size_t add_hi = (hi < bin_count_-1) ? std::popcount(bins_[hi+1].tpo_bits) : 0;
        
        if (add_lo >= add_hi && lo > 0) {
            lo--;
            enclosed += add_lo;
        } else if (hi < bin_count_ - 1) {
            hi++;
            enclosed += add_hi;
        } else {
            break;
        }
    }

    va_low_bin = lo;
    va_high_bin = hi;
}

size_t TPOEngine::find_poc_bin() {
    if (!bins_ || bin_count_ == 0) return 0;

    size_t poc = 0;
    size_t max_chars = std::popcount(bins_[0].tpo_bits);

    for (size_t i = 1; i < bin_count_; ++i) {
        size_t chars = std::popcount(bins_[i].tpo_bits);
        if (chars > max_chars) {
            max_chars = chars;
            poc = i;
        }
    }

    return poc;
}

std::vector<TPOBar> TPOEngine::get_tpo_profile() {
    std::vector<TPOBar> profile;
    
    if (!bins_ || bin_count_ == 0) return profile;

    for (size_t i = 0; i < bin_count_; ++i) {
        TPOBar bar;
        bar.bin_index = i;
        bar.price = day_low_ + i * tick_size_;
        bar.tpo_bits = bins_[i].tpo_bits;
        bar.char_count = std::popcount(bins_[i].tpo_bits);
        
        // Check for single print
        bool is_single_print = (bar.char_count == 1);
        bool adjacent_above = (i+1 < bin_count_ && std::popcount(bins_[i+1].tpo_bits) > 1);
        bool adjacent_below = (i > 0 && std::popcount(bins_[i-1].tpo_bits) > 1);
        bar.is_single_print = is_single_print && adjacent_above && adjacent_below;
        
        profile.push_back(bar);
    }

    return profile;
}

}  // namespace BTQuant