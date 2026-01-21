#pragma once

#include <cstdint>
#include <type_traits>
#include <vector>
#include <cstring>

namespace BTQuant {
namespace RenderEngine {

// ============================================================================
// GPU-CPU Memory Alignment Concepts
// ============================================================================

// Concept for types that can be directly copied to GPU memory
template<typename T>
concept GpuAlignable = std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>;

// ============================================================================
// Core Data Structures
// ============================================================================

// 16-byte aligned order book level for GPU processing
struct alignas(16) OrderBookLevel {
    float price;
    uint32_t askQuantity;
    uint32_t bidQuantity;
    uint32_t numOrders;
};

static_assert(GpuAlignable<OrderBookLevel>, "OrderBookLevel must be GPU-alignable");

// Variable-length order book snapshot with GPU-friendly layout
struct alignas(16) HotspineOrderBookSnapshot {
    uint32_t currentTimeIndex;
    uint32_t priceLevelsCount;
    float basePrice;
    float priceRange;
    OrderBookLevel levels[];  // Must be last member - variable length array

    // Calculate required buffer size for a given number of price levels
    static constexpr size_t calculateBufferSize(uint32_t priceLevelCount) {
        return sizeof(HotspineOrderBookSnapshot) + priceLevelCount * sizeof(OrderBookLevel);
    }
};

static_assert(GpuAlignable<HotspineOrderBookSnapshot>, "HotspineOrderBookSnapshot must be GPU-alignable");

// Candle cluster structure for footprint chart rendering
struct alignas(16) CandleCluster {
    float centerX;          // X coordinate (time)
    float centerY;          // Y coordinate (price)
    float width;            // Time duration
    float height;           // Price range
    uint32_t bidVolume;     // Total bid volume
    uint32_t askVolume;     // Total ask volume
    uint32_t tradeCount;    // Number of trades
    float vwap;             // Volume-weighted average price
    bool hasTrades;         // Trade activity indicator

    CandleCluster() = default;
    
    CandleCluster(float x, float y, float w, float h,
                 uint32_t bidVol, uint32_t askVol,
                 uint32_t tradeCnt, float vw, bool hasTrades)
        : centerX(x), centerY(y), width(w), height(h),
          bidVolume(bidVol), askVolume(askVol),
          tradeCount(tradeCnt), vwap(vw), hasTrades(hasTrades) {}
};

static_assert(GpuAlignable<CandleCluster>, "CandleCluster must be GPU-alignable");

// Trade tick structure for TPO profile calculation
struct alignas(16) HotspineTradeTick {
    uint64_t timestamp;     // Exchange timestamp
    float price;            // Trade price
    float size;             // Trade size
    uint32_t symbolId;      // Symbol identifier
    bool isBuy;             // true = buy, false = sell

    HotspineTradeTick() = default;
    
    HotspineTradeTick(uint64_t ts, float p, float s, uint32_t id, bool buy)
        : timestamp(ts), price(p), size(s), symbolId(id), isBuy(buy) {}
};

static_assert(GpuAlignable<HotspineTradeTick>, "HotspineTradeTick must be GPU-alignable");

// ============================================================================
// Configuration Structures
// ============================================================================

struct LOBHeatmapConfig {
    uint32_t width = 1024;
    uint32_t height = 512;
    float maxLiquidity = 100000.0f;
    bool invertYAxis = true;
};

struct FootprintChartConfig {
    uint32_t maxClusters = 4096;
    float cellMinSize = 2.0f;
    float cellMaxSize = 20.0f;
    bool showLabels = true;
};

struct TPOProfileConfig {
    uint32_t bucketCount = 256;
    float priceResolution = 0.1f;
    uint32_t timeWindowMs = 30000;
    bool resetOnUpdate = true;
};

struct RendererConfig {
    LOBHeatmapConfig lobHeatmap;
    FootprintChartConfig footprintChart;
    TPOProfileConfig tpoProfile;
};

// ============================================================================
// Renderer Statistics
// ============================================================================

struct RendererStats {
    uint32_t framesRendered = 0;
    uint32_t lobUpdates = 0;
    uint32_t tradeUpdates = 0;
    uint32_t footprintCellsRendered = 0;
    double averageFrameTimeMs = 0.0;
    uint64_t lastUpdateTimeNs = 0;
};

// ============================================================================
// Helper Functions
// ============================================================================

// Convert Hotspine data to GPU-friendly format
inline std::vector<uint8_t> serializeOrderBookSnapshot(const HotspineOrderBookSnapshot& snapshot) {
    const size_t bufferSize = HotspineOrderBookSnapshot::calculateBufferSize(snapshot.priceLevelsCount);
    std::vector<uint8_t> buffer(bufferSize);
    std::memcpy(buffer.data(), &snapshot, bufferSize);
    return buffer;
}

// Calculate price level index for histogram bucketing
inline uint32_t getPriceLevelIndex(float price, float basePrice, float priceRange, uint32_t bucketCount) {
    const float normalizedPrice = (price - basePrice) / priceRange;
    const uint32_t index = static_cast<uint32_t>(normalizedPrice * bucketCount);
    return std::clamp(index, 0U, bucketCount - 1);
}

} // namespace RenderEngine
} // namespace BTQuant
