#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace HotSpine::V3 {
    struct VolumeNode;
}

namespace BTQuant {

/**
 * @brief Structure representing a single TPO bar for market profile analysis
 */
struct TPOBar {
    size_t bin_index = 0;           ///< Index of the bin
    double price = 0.0;             ///< Price level of the bin
    uint16_t tpo_bits = 0;          ///< Bitmask for 30-minute brackets (A-P)
    size_t char_count = 0;          ///< Number of TPO characters at this level
    bool is_single_print = false;   ///< True if this is a single print level
};

/**
 * @brief TPO (Time Price Opportunity) Engine for market profile analysis
 * 
 * This engine tracks time-based brackets (30-minute intervals) for each price level
 * and provides market profile functionality including value area calculation.
 */
class TPOEngine {
public:
    TPOEngine();

    /**
     * @brief Initialize the TPO engine
     * @param session_open_time_us Unix timestamp (in microseconds) when the session opened
     * @param day_low Starting price for binning
     * @param tick_size Size of each price bin
     * @param max_bins Maximum number of bins to handle
     */
    void initialize(uint64_t session_open_time_us, double day_low, double tick_size, size_t max_bins = 1000);

    /**
     * @brief Set the cluster data to operate on
     * @param bins Pointer to the VolumeNode array from ClusterEngine
     * @param count Number of bins
     */
    void set_cluster_data(HotSpine::V3::VolumeNode* bins, size_t count);

    /**
     * @brief Update TPO bits for a trade at a given price and time
     * @param timestamp_us Microsecond timestamp of the trade
     * @param price Price level of the trade
     */
    void update_tpo_bit(uint64_t timestamp_us, double price);

    /**
     * @brief Calculate the value area (68% of total TPO characters)
     * @param[out] va_low_bin Lower bin index of value area
     * @param[out] va_high_bin Upper bin index of value area
     */
    void calculate_value_area(size_t& va_low_bin, size_t& va_high_bin);

    /**
     * @brief Get the complete TPO profile for rendering
     * @return Vector of TPOBar structures
     */
    std::vector<TPOBar> get_tpo_profile();

    /**
     * @brief Find the Point of Control (highest TPO character count) bin
     * @return Bin index with highest TPO character count
     */
    size_t find_poc_bin();

private:
    /**
     * @brief Convert timestamp to bracket index (0-15 for 16 30-min periods)
     * @param ts_us Microsecond timestamp
     * @return Bracket index (0-15)
     */
    uint8_t timestamp_to_bracket(uint64_t ts_us);

    /**
     * @brief Convert bracket index to character (A-P)
     * @param bracket Bracket index (0-15)
     * @return Character representation (A-P)
     */
    char bracket_to_char(uint8_t bracket);

private:
    uint64_t session_open_us_ = 0;
    HotSpine::V3::VolumeNode* bins_ = nullptr;
    size_t bin_count_ = 0;
    double day_low_ = 0.0;
    double tick_size_ = 0.0;
};

}  // namespace BTQuant