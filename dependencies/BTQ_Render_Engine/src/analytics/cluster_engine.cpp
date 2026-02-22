/// @file cluster_engine.cpp
/// @brief Implements the ClusterEngine for O(1) cluster binning and CVD calculations.

#include "analytics/cluster_engine.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include "hotspine_layout_v3.hpp"
#include "memory/memory_arena.hpp"

namespace BTQuant {

ClusterEngine::ClusterEngine() 
    : bins_(nullptr)
    , bin_count_(0)
    , day_low_(0.0)
    , tick_size_(0.0)
    , cvd_(0)
    , poc_bin_(0) {}

void ClusterEngine::initialize(double day_low, double tick_size, size_t max_bins) {
    day_low_ = day_low;
    tick_size_ = tick_size;
    bin_count_ = max_bins;

    // Allocate from g_arena
    bins_ = reinterpret_cast<HotSpine::V3::VolumeNode*>(
        g_arena.acquire(bin_count_ * sizeof(HotSpine::V3::VolumeNode), 64));
    
    if (bins_) {
        // Initialize all bins to zero
        std::memset(bins_, 0, bin_count_ * sizeof(HotSpine::V3::VolumeNode));
    }
}

void ClusterEngine::ingest(const TradeData& t) noexcept {
    if (!bins_) return;
    
    size_t bin = static_cast<size_t>((t.price - day_low_) / tick_size_);
    if (bin >= bin_count_) return;

    auto* node = &bins_[bin];
    auto& target = t.side == TradeSide::BUY ? node->buy_vol : node->sell_vol;

    // Use atomic operations for thread-safe float accumulation
    // Use bit_cast for atomic float operations (safe conversion between float and uint32_t)
    std::atomic<uint32_t>* atomic_target = reinterpret_cast<std::atomic<uint32_t>*>(&target);
    
    uint32_t current_bits = atomic_target->load(std::memory_order_relaxed);
    uint32_t new_bits;
    float current_val, new_val;
    
    do {
        current_val = std::bit_cast<float>(current_bits);
        new_val = current_val + static_cast<float>(t.volume);
        new_bits = std::bit_cast<uint32_t>(new_val);
    } while (!atomic_target->compare_exchange_weak(current_bits, new_bits,
                 std::memory_order_release, std::memory_order_acquire));

    // CVD: atomic add (Phase 5.5)
    int64_t delta = static_cast<int64_t>(t.volume * 100); // Scale to avoid float precision issues
    if (t.side == TradeSide::BUY) {
        cvd_.fetch_add( delta, std::memory_order_relaxed);
    } else {
        cvd_.fetch_add(-delta, std::memory_order_relaxed);
    }

    // Update POC (Point of Control) - track the bin with highest total volume
    // Use atomic compare-and-swap to ensure thread safety
    float total = bins_[bin].buy_vol + bins_[bin].sell_vol;
    size_t current_poc = poc_bin_.load(std::memory_order_acquire);
    float poc_total = bins_[current_poc].buy_vol + bins_[current_poc].sell_vol;
    
    // Keep trying to update POC until successful or a higher POC is found by another thread
    while (total > poc_total) {
        size_t expected_poc = current_poc;
        if (poc_bin_.compare_exchange_weak(expected_poc, bin, 
                                          std::memory_order_release, 
                                          std::memory_order_acquire)) {
            break; // Successfully updated POC
        }
        // If CAS failed, re-read values and try again
        current_poc = poc_bin_.load(std::memory_order_acquire);
        poc_total = bins_[current_poc].buy_vol + bins_[current_poc].sell_vol;
    }
}

const HotSpine::V3::VolumeNode* ClusterEngine::get_bins() const noexcept {
    return bins_;
}

size_t ClusterEngine::get_bin_count() const noexcept {
    return bin_count_;
}

double ClusterEngine::get_day_low() const noexcept {
    return day_low_;
}

double ClusterEngine::get_tick_size() const noexcept {
    return tick_size_;
}

int64_t ClusterEngine::get_cvd() const noexcept {
    return cvd_.load(std::memory_order_acquire);
}

size_t ClusterEngine::get_poc_bin() const noexcept {
    return poc_bin_.load(std::memory_order_acquire);
}

}  // namespace BTQuant