/**
 * @file HotspineData.h
 * @brief C++20 Hotspine Data Packet Definition for GPU SSBO Alignment
 * 
 * Defines the data structures for the Shared Data Hotspine - high-throughput
 * ring buffer in shared memory providing Level 3 Market Data. These structures
 * are designed to match GPU SSBO (Shader Storage Buffer Object) layouts exactly
 * using C++20 features and GPU alignment requirements.
 * 
 * @author Market Microstructure Renderer Team
 * @version 1.0.0
 */

#ifndef HOTSPINE_DATA_H
#define HOTSPINE_DATA_H

#include <cstdint>
#include <array>
#include <span>
#include <concepts>

namespace trading {

// ============================================
// Type Aliases for GPU Compatibility
// ============================================

using float32_t = float;
using uint32_t = std::uint32_t;
using uint64_t = std::uint64_t;

/**
 * @brief Concept for GPU-alignable POD types
 * 
 * Ensures types used for GPU communication are standard-layout, trivially-copyable,
 * and have no padding issues that would cause alignment mismatches between CPU and GPU.
 */
template<typename T>
concept GpuAlignable = std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>;

// ============================================
// Order Book Level Structure
// ============================================

/**
 * @struct OrderBookLevel
 * @brief Represents a single price level in the limit order book
 * 
 * Memory layout is explicitly controlled for std430 SSBO compatibility.
 * This structure matches the GLSL OrderBookLevel definition.
 */
struct alignas(16) OrderBookLevel {
    float32_t price;           ///< Price of this level
    uint32_t askQuantity;      ///< Total ask quantity at this price
    uint32_t bidQuantity;      ///< Total bid quantity at this price
    uint32_t numOrders;        ///< Number of individual orders at this price
    
    // Default constructor for POD compatibility
    constexpr OrderBookLevel() noexcept = default;
    
    /**
     * @brief Construct a new OrderBookLevel object
     * @param price Price level
     * @param askQuantity Ask quantity
     * @param bidQuantity Bid quantity
     * @param numOrders Number of orders
     */
    constexpr OrderBookLevel(float32_t price_, uint32_t askQuantity_,
                           uint32_t bidQuantity_, uint32_t numOrders_) noexcept
        : price(price_)
        , askQuantity(askQuantity_)
        , bidQuantity(bidQuantity_)
        , numOrders(numOrders_)
    {}
    
    /**
     * @brief Calculate total liquidity at this price level
     * @return Total quantity available (bid + ask)
     */
    [[nodiscard]] constexpr uint32_t totalLiquidity() const noexcept {
        return bidQuantity + askQuantity;
    }
};

static_assert(GpuAlignable<OrderBookLevel>, "OrderBookLevel must be GPU alignable");
static_assert(sizeof(OrderBookLevel) == 16, "OrderBookLevel size mismatch with GPU");

// ============================================
// Hotspine Order Book Snapshot
// ============================================

/**
 * @struct HotspineOrderBookSnapshot
 * @brief Complete order book snapshot for GPU transfer
 * 
 * This structure defines the memory layout of the SSBO used by the LOB Heatmap
 * compute shader. It contains metadata followed by a variable-length array of
 * price levels.
 */
struct alignas(16) HotspineOrderBookSnapshot {
    uint32_t currentTimeIndex;  ///< Current cyclic time index for heatmap
    uint32_t priceLevelsCount;  ///< Number of valid price levels in this snapshot
    float32_t basePrice;        ///< Base price for rendering (bottom of heatmap)
    float32_t priceRange;       ///< Total price range to display
    
    // Variable-length price levels (must be last member)
    OrderBookLevel levels[];    ///< Array of price levels
    
    /**
     * @brief Get span of valid price levels
     * @return Span view of price levels array
     */
    [[nodiscard]] std::span<const OrderBookLevel> getPriceLevels() const noexcept {
        return std::span<const OrderBookLevel>(levels, priceLevelsCount);
    }
    
    /**
     * @brief Get mutable span of valid price levels
     * @return Mutable span view of price levels array
     */
    [[nodiscard]] std::span<OrderBookLevel> getPriceLevels() noexcept {
        return std::span<OrderBookLevel>(levels, priceLevelsCount);
    }
    
    /**
     * @brief Calculate maximum liquidity in this snapshot
     * @return Maximum liquidity value for normalization
     */
    [[nodiscard]] uint32_t getMaxLiquidity() const noexcept {
        uint32_t max = 0;
        for (const auto& level : getPriceLevels()) {
            max = std::max(max, level.totalLiquidity());
        }
        return max;
    }
};

static_assert(GpuAlignable<HotspineOrderBookSnapshot>, "HotspineOrderBookSnapshot must be GPU alignable");

// ============================================
// Trade Tick Structure
// ============================================

/**
 * @struct TradeTick
 * @brief Represents a single trade tick for TPO profile calculation
 * 
 * Memory layout matches the GLSL TradeTick definition with std430 alignment.
 */
struct alignas(16) TradeTick {
    float32_t price;        ///< Price at which the trade occurred
    uint32_t size;          ///< Trade size in contracts/shares
    uint64_t timestamp;     ///< Nanosecond timestamp of the trade
    bool isBuy;             ///< True if this was a buy order, false for sell
    
    // Default constructor for POD compatibility
    constexpr TradeTick() noexcept = default;
    
    /**
     * @brief Construct a new TradeTick object
     * @param price Trade price
     * @param size Trade size
     * @param timestamp Nanosecond timestamp
     * @param isBuy Buy/sell indicator
     */
    constexpr TradeTick(float32_t price_, uint32_t size_,
                      uint64_t timestamp_, bool isBuy_) noexcept
        : price(price_)
        , size(size_)
        , timestamp(timestamp_)
        , isBuy(isBuy_)
    {}
};

static_assert(GpuAlignable<TradeTick>, "TradeTick must be GPU alignable");
static_assert(sizeof(TradeTick) == 16, "TradeTick size mismatch with GPU");

// ============================================
// Trade Ticks SSBO Structure
// ============================================

/**
 * @struct HotspineTradeTicks
 * @brief Structure for streaming trade ticks to the GPU
 * 
 * Contains a batch of trade ticks with metadata for TPO profile computation.
 */
struct alignas(16) HotspineTradeTicks {
    uint32_t tickCount;     ///< Number of valid trade ticks in this batch
    uint64_t startTime;     ///< Start time of the current window (nanoseconds)
    uint64_t endTime;       ///< End time of the current window (nanoseconds)
    float32_t minPrice;     ///< Minimum price in this batch
    float32_t maxPrice;     ///< Maximum price in this batch
    
    // Variable-length trade ticks (must be last member)
    TradeTick ticks[];      ///< Array of trade ticks
    
    /**
     * @brief Get span of valid trade ticks
     * @return Span view of trade ticks array
     */
    [[nodiscard]] std::span<const TradeTick> getTicks() const noexcept {
        return std::span<const TradeTick>(ticks, tickCount);
    }
    
    /**
     * @brief Get mutable span of valid trade ticks
     * @return Mutable span view of trade ticks array
     */
    [[nodiscard]] std::span<TradeTick> getTicks() noexcept {
        return std::span<TradeTick>(ticks, tickCount);
    }
};

static_assert(GpuAlignable<HotspineTradeTicks>, "HotspineTradeTicks must be GPU alignable");

// ============================================
// Candle Cluster Structure
// ============================================

/**
 * @struct CandleCluster
 * @brief Represents a single candle cluster for volumetric footprint visualization
 * 
 * Memory layout matches the GLSL CandleCluster definition with std430 alignment.
 * Used for instanced rendering of footprint cells.
 */
struct alignas(16) CandleCluster {
    float32_t centerX;          ///< Center X coordinate (time)
    float32_t centerY;          ///< Center Y coordinate (price)
    float32_t width;            ///< Cluster width (time duration)
    float32_t height;           ///< Cluster height (price range)
    uint32_t bidVolume;         ///< Total bid volume in this cluster
    uint32_t askVolume;         ///< Total ask volume in this cluster
    uint32_t tradeCount;        ///< Number of trades in this cluster
    float32_t vwap;             ///< Volume-weighted average price
    bool hasTrades;             ///< Trade activity indicator
    
    // Default constructor for POD compatibility
    constexpr CandleCluster() noexcept = default;
    
    /**
     * @brief Construct a new CandleCluster object
     * @param centerX Center X coordinate (time)
     * @param centerY Center Y coordinate (price)
     * @param width Cluster width (time duration)
     * @param height Cluster height (price range)
     * @param bidVolume Total bid volume
     * @param askVolume Total ask volume
     * @param tradeCount Number of trades
     * @param vwap Volume-weighted average price
     * @param hasTrades Trade activity indicator
     */
    constexpr CandleCluster(float32_t centerX_, float32_t centerY_,
                          float32_t width_, float32_t height_,
                          uint32_t bidVolume_, uint32_t askVolume_,
                          uint32_t tradeCount_, float32_t vwap_,
                          bool hasTrades_) noexcept
        : centerX(centerX_)
        , centerY(centerY_)
        , width(width_)
        , height(height_)
        , bidVolume(bidVolume_)
        , askVolume(askVolume_)
        , tradeCount(tradeCount_)
        , vwap(vwap_)
        , hasTrades(hasTrades_)
    {}
    
    /**
     * @brief Calculate volume delta (ask - bid)
     * @return Volume delta for coloring
     */
    [[nodiscard]] int32_t volumeDelta() const noexcept {
        return static_cast<int32_t>(askVolume) - static_cast<int32_t>(bidVolume);
    }
    
    /**
     * @brief Calculate total volume in this cluster
     * @return Total volume
     */
    [[nodiscard]] uint32_t totalVolume() const noexcept {
        return bidVolume + askVolume;
    }
    
    /**
     * @brief Calculate delta ratio for coloring
     * @return Normalized delta ratio in [-1, 1] range
     */
    [[nodiscard]] float32_t deltaRatio() const noexcept {
        const uint32_t total = totalVolume();
        return total > 0 ? static_cast<float32_t>(volumeDelta()) / static_cast<float32_t>(total) : 0.0f;
    }
};

static_assert(GpuAlignable<CandleCluster>, "CandleCluster must be GPU alignable");
static_assert(sizeof(CandleCluster) == 40, "CandleCluster size mismatch with GPU");

// ============================================
// TPO Histogram Structure
// ============================================

/**
 * @struct TPOHistogram
 * @brief Structure for TPO (Time Price Opportunity) histogram data
 * 
 * Contains histogram buckets for real-time TPO profile calculation.
 */
struct alignas(16) TPOHistogram {
    uint32_t bucketCount;    ///< Number of histogram buckets
    float32_t bucketSize;    ///< Price range per bucket
    float32_t basePrice;     ///< Base price for histogram (bottom of Y-axis)
    uint32_t buckets[];      ///< Histogram data (count per bucket)
    
    /**
     * @brief Get span of histogram buckets
     * @return Span view of histogram buckets
     */
    [[nodiscard]] std::span<const uint32_t> getBuckets() const noexcept {
        return std::span<const uint32_t>(buckets, bucketCount);
    }
    
    /**
     * @brief Get mutable span of histogram buckets
     * @return Mutable span view of histogram buckets
     */
    [[nodiscard]] std::span<uint32_t> getBuckets() noexcept {
        return std::span<uint32_t>(buckets, bucketCount);
    }
    
    /**
     * @brief Calculate bucket index for a given price
     * @param price Price to find bucket for
     * @return Bucket index (or -1 if out of range)
     */
    [[nodiscard]] int32_t priceToBucketIndex(float32_t price) const noexcept {
        const float32_t offset = price - basePrice;
        if (offset < 0.0f || offset >= bucketCount * bucketSize) {
            return -1;
        }
        return static_cast<int32_t>(offset / bucketSize);
    }
    
    /**
     * @brief Get price range for a specific bucket
     * @param bucketIndex Index of the bucket
     * @return Tuple of (minPrice, maxPrice) for the bucket
     */
    [[nodiscard]] std::pair<float32_t, float32_t> bucketIndexToPriceRange(uint32_t bucketIndex) const noexcept {
        const float32_t minPrice = basePrice + bucketIndex * bucketSize;
        return {minPrice, minPrice + bucketSize};
    }
};

static_assert(GpuAlignable<TPOHistogram>, "TPOHistogram must be GPU alignable");

// ============================================
// Vulkan Buffer Allocation Helper
// ============================================

/**
 * @brief Helper function to calculate required buffer size for Hotspine data
 * @tparam T Hotspine structure type
 * @tparam U Array element type
 * @param elementCount Number of array elements
 * @return Total required buffer size in bytes
 */
template<typename T, typename U>
[[nodiscard]] constexpr size_t calculateBufferSize(uint32_t elementCount) noexcept {
    return sizeof(T) + elementCount * sizeof(U);
}

/**
 * @brief Helper for order book snapshot buffer size
 */
[[nodiscard]] constexpr size_t calculateOrderBookBufferSize(uint32_t priceLevelCount) noexcept {
    return calculateBufferSize<HotspineOrderBookSnapshot, OrderBookLevel>(priceLevelCount);
}

/**
 * @brief Helper for trade ticks buffer size
 */
[[nodiscard]] constexpr size_t calculateTradeTicksBufferSize(uint32_t tickCount) noexcept {
    return calculateBufferSize<HotspineTradeTicks, TradeTick>(tickCount);
}

/**
 * @brief Helper for TPO histogram buffer size
 */
[[nodiscard]] constexpr size_t calculateTPOHistogramBufferSize(uint32_t bucketCount) noexcept {
    return calculateBufferSize<TPOHistogram, uint32_t>(bucketCount);
}

} // namespace trading

#endif // HOTSPINE_DATA_H