#pragma once

#include <atomic>
#include <cstddef>

#include "data/core_types.hpp"
#include "hotspine_layout_v3.hpp"

namespace BTQuant {

/**
 * @brief ClusterEngine - O(1) cluster binning and CVD calculations
 * 
 * This engine provides ultra-fast O(1) cluster binning for trade data,
 * maintaining buy/sell volume separately using atomic operations.
 * It also tracks Cumulative Volume Delta (CVD) and Point of Control (POC).
 */
class ClusterEngine {
public:
    ClusterEngine();

    /**
     * @brief Initialize the cluster engine with price range and bin count
     * @param day_low Starting price for binning
     * @param tick_size Size of each price bin
     * @param max_bins Maximum number of bins to allocate
     */
    void initialize(double day_low, double tick_size, size_t max_bins = 1000);

    /**
     * @brief Ingest a trade for clustering (O(1) operation)
     * @param t Trade data to process
     */
    void ingest(const TradeData& t) noexcept;

    /**
     * @brief Get the cluster bins array
     */
    const HotSpine::V3::VolumeNode* get_bins() const noexcept;

    /**
     * @brief Get the number of bins
     */
    size_t get_bin_count() const noexcept;

    /**
     * @brief Get the day low price
     */
    double get_day_low() const noexcept;

    /**
     * @brief Get the tick size
     */
    double get_tick_size() const noexcept;

    /**
     * @brief Get the Cumulative Volume Delta (CVD)
     */
    int64_t get_cvd() const noexcept;

    /**
     * @brief Get the Point of Control bin index
     */
    size_t get_poc_bin() const noexcept;

private:
    HotSpine::V3::VolumeNode* bins_ = nullptr;  // From g_arena
    size_t bin_count_ = 0;
    double day_low_ = 0.0;
    double tick_size_ = 0.0;
    std::atomic<int64_t> cvd_{0};      // Phase 5.5 - CVD
    std::atomic<size_t> poc_bin_{0};   // Point of Control bin
};

}  // namespace BTQuant