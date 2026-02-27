#pragma once

#include <vector>
#include <cstdint>

#include "market_data_processor.hpp"  // For OHLCVCandle
#include "TradeData.h"                // For TradeData

namespace BTQuant {
namespace RenderEngine {
    // Forward declaration
    struct OHLCVCandle;
}

namespace Data {

// Quantization strategies for different data types
enum class QuantizationStrategy {
    FIXED_PRECISION,      // Fixed decimal precision (e.g., 0.01)
    RELATIVE_PERCENTAGE,  // Percentage-based quantization (e.g., 0.001% of value)
    LOGARITHMIC,          // Logarithmic quantization for wide range values
    ADAPTIVE              // Adaptive quantization based on data characteristics
};

// Compressed representations of market data
struct CompressedCandle {
    uint64_t timestamp_delta;    // Delta from previous timestamp
    double open_delta;           // Delta from previous open price
    double high_delta;           // Delta from previous high price
    double low_delta;            // Delta from previous low price
    double close_delta;          // Delta from previous close price
    float volume_delta;          // Delta from previous volume
    uint64_t trade_count_delta;  // Delta from previous trade count
};

struct CompressedTrade {
    uint64_t timestamp_delta;    // Delta from previous timestamp
    double price_delta;          // Delta from previous price
    float volume_delta;          // Delta from previous volume
    TradeSide side;              // Side of the trade (no compression for enum)
    uint8_t exchange_id;         // Exchange identifier (no compression)
    uint8_t flags;               // Trade flags (no compression)
};

// Delta compression class for market data
class DeltaCompressor {
public:
    // Constructors with configurable precision for quantization
    explicit DeltaCompressor(double price_precision = 0.0, float volume_precision = 0.0f);
    explicit DeltaCompressor(double price_precision, float volume_precision,
                            QuantizationStrategy strategy);

    // Single candle compression/decompression
    void compressCandle(const BTQuant::RenderEngine::OHLCVCandle& current, const BTQuant::RenderEngine::OHLCVCandle& previous, CompressedCandle& compressed);
    void decompressCandle(const CompressedCandle& compressed, const BTQuant::RenderEngine::OHLCVCandle& previous, BTQuant::RenderEngine::OHLCVCandle& decompressed);

    // Single trade compression/decompression
    void compressTrade(const TradeData& current, const TradeData& previous, CompressedTrade& compressed);
    void decompressTrade(const CompressedTrade& compressed, const TradeData& previous, TradeData& decompressed);

    // Batch compression/decompression methods
    std::vector<CompressedCandle> compressCandles(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& candles);
    std::vector<BTQuant::RenderEngine::OHLCVCandle> decompressCandles(const std::vector<CompressedCandle>& compressed);

    std::vector<CompressedTrade> compressTrades(const std::vector<TradeData>& trades);
    std::vector<TradeData> decompressTrades(const std::vector<CompressedTrade>& compressed);

    // Utility methods for arrays of prices/volumes
    std::vector<double> compressPrices(const std::vector<double>& prices);
    std::vector<double> decompressPrices(const std::vector<double>& compressed_prices);

    std::vector<float> compressVolumes(const std::vector<float>& volumes);
    std::vector<float> decompressVolumes(const std::vector<float>& compressed_volumes);

    // Set quantization parameters
    void setPricePrecision(double precision) { price_precision_ = precision; }
    void setVolumePrecision(float precision) { volume_precision_ = precision; }
    void setQuantizationStrategy(QuantizationStrategy strategy) { strategy_ = strategy; }

    // Get compression ratio estimate
    double estimateCompressionRatio(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& original) const;

private:
    double price_precision_;           // Quantization precision for prices
    float volume_precision_;           // Quantization precision for volumes
    QuantizationStrategy strategy_;    // Quantization strategy to use

    // Helper functions for quantization
    double quantizePrice(double price) const;
    float quantizeVolume(float volume) const;
    double quantizeValue(double value, double precision) const;
};

} // namespace Data
} // namespace BTQuant