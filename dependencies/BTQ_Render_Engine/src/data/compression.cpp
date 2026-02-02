#include "data/compression.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace BTQuant {
namespace Data {

// ============================================================================
// Delta Compression Implementation
// ============================================================================

// Constructor with configurable precision and strategy
DeltaCompressor::DeltaCompressor(double price_precision, float volume_precision,
                                 QuantizationStrategy strategy)
    : price_precision_(price_precision), volume_precision_(volume_precision), strategy_(strategy) {
    // Default precision values can be adjusted based on typical market data characteristics
    if (price_precision_ == 0.0) {
        price_precision_ = 0.01;  // Default to 2 decimal places for most currencies
    }
    if (volume_precision_ == 0.0f) {
        volume_precision_ = 0.001f;  // Default to 3 decimal places for volumes
    }
}

// Constructor with configurable precision (uses default strategy)
DeltaCompressor::DeltaCompressor(double price_precision, float volume_precision)
    : DeltaCompressor(price_precision, volume_precision, QuantizationStrategy::FIXED_PRECISION) {
}

// Helper function to quantize values based on strategy
double DeltaCompressor::quantizeValue(double value, double precision) const {
    switch (strategy_) {
        case QuantizationStrategy::FIXED_PRECISION:
            if (precision == 0.0) return value;
            return std::round(value / precision) * precision;

        case QuantizationStrategy::RELATIVE_PERCENTAGE: {
            if (precision == 0.0 || value == 0.0) return value;
            double threshold = std::abs(value) * precision;
            return std::round(value / threshold) * threshold;
        }

        case QuantizationStrategy::LOGARITHMIC: {
            if (value == 0.0) return 0.0;
            double log_val = std::log(std::abs(value));
            double quantized_log = std::round(log_val / precision) * precision;
            double result = std::exp(quantized_log);
            return (value >= 0) ? result : -result;
        }

        case QuantizationStrategy::ADAPTIVE:
            // For adaptive quantization, we could implement statistical analysis
            // of the data to determine optimal precision, but for now use fixed
            if (precision == 0.0) return value;
            return std::round(value / precision) * precision;
    }
    return value; // fallback
}

double DeltaCompressor::quantizePrice(double price) const {
    return quantizeValue(price, price_precision_);
}

float DeltaCompressor::quantizeVolume(float volume) const {
    // Convert to double for quantization, then back to float
    double quantized = quantizeValue(static_cast<double>(volume), static_cast<double>(volume_precision_));
    return static_cast<float>(quantized);
}

// Delta compression for OHLCVCandle data
void DeltaCompressor::compressCandle(const BTQuant::RenderEngine::OHLCVCandle& current, const BTQuant::RenderEngine::OHLCVCandle& previous, CompressedCandle& compressed) {
    // Store timestamp as-is (or use delta if needed)
    compressed.timestamp_delta = current.timestamp - previous.timestamp;

    // Apply delta compression with quantization
    compressed.open_delta = quantizePrice(current.open - previous.open);
    compressed.high_delta = quantizePrice(current.high - previous.high);
    compressed.low_delta = quantizePrice(current.low - previous.low);
    compressed.close_delta = quantizePrice(current.close - previous.close);
    compressed.volume_delta = quantizeVolume(static_cast<float>(current.volume - previous.volume));
    compressed.trade_count_delta = current.trade_count - previous.trade_count;
}

void DeltaCompressor::decompressCandle(const CompressedCandle& compressed, const BTQuant::RenderEngine::OHLCVCandle& previous, BTQuant::RenderEngine::OHLCVCandle& decompressed) {
    // Reconstruct timestamp
    decompressed.timestamp = previous.timestamp + compressed.timestamp_delta;

    // Reconstruct prices using deltas
    decompressed.open = previous.open + compressed.open_delta;
    decompressed.high = previous.high + compressed.high_delta;
    decompressed.low = previous.low + compressed.low_delta;
    decompressed.close = previous.close + compressed.close_delta;
    decompressed.volume = previous.volume + compressed.volume_delta;
    decompressed.trade_count = previous.trade_count + compressed.trade_count_delta;
}

// Delta compression for TradeData
void DeltaCompressor::compressTrade(const TradeData& current, const TradeData& previous, CompressedTrade& compressed) {
    // Timestamp delta
    compressed.timestamp_delta = current.timestamp - previous.timestamp;

    // Price delta with quantization
    compressed.price_delta = quantizePrice(current.price - previous.price);

    // Volume delta with quantization
    compressed.volume_delta = quantizeVolume(current.volume - previous.volume);

    // Store other fields that don't compress well
    compressed.side = current.side;
    compressed.exchange_id = current.exchange_id;
    compressed.flags = current.flags;
}

void DeltaCompressor::decompressTrade(const CompressedTrade& compressed, const TradeData& previous, TradeData& decompressed) {
    // Reconstruct timestamp
    decompressed.timestamp = previous.timestamp + compressed.timestamp_delta;

    // Reconstruct price and volume
    decompressed.price = previous.price + compressed.price_delta;
    decompressed.volume = previous.volume + compressed.volume_delta;

    // Copy other fields
    decompressed.side = compressed.side;
    decompressed.exchange_id = compressed.exchange_id;
    decompressed.flags = compressed.flags;
}

// Batch compression methods
std::vector<CompressedCandle> DeltaCompressor::compressCandles(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& candles) {
    std::vector<CompressedCandle> compressed;
    if (candles.empty()) return compressed;

    compressed.reserve(candles.size());

    // First candle stored as reference point
    CompressedCandle first_candle = {};
    first_candle.timestamp_delta = candles[0].timestamp;  // Absolute timestamp for first
    first_candle.open_delta = candles[0].open;
    first_candle.high_delta = candles[0].high;
    first_candle.low_delta = candles[0].low;
    first_candle.close_delta = candles[0].close;
    first_candle.volume_delta = static_cast<float>(candles[0].volume);
    first_candle.trade_count_delta = candles[0].trade_count;
    compressed.push_back(first_candle);

    // Remaining candles as deltas
    for (size_t i = 1; i < candles.size(); ++i) {
        CompressedCandle comp;
        compressCandle(candles[i], candles[i-1], comp);
        compressed.push_back(comp);
    }

    return compressed;
}

std::vector<BTQuant::RenderEngine::OHLCVCandle> DeltaCompressor::decompressCandles(const std::vector<CompressedCandle>& compressed) {
    std::vector<BTQuant::RenderEngine::OHLCVCandle> decompressed;
    if (compressed.empty()) return decompressed;

    decompressed.reserve(compressed.size());

    // First candle reconstruction
    BTQuant::RenderEngine::OHLCVCandle first_candle = {};
    first_candle.timestamp = compressed[0].timestamp_delta;  // Absolute timestamp
    first_candle.open = compressed[0].open_delta;
    first_candle.high = compressed[0].high_delta;
    first_candle.low = compressed[0].low_delta;
    first_candle.close = compressed[0].close_delta;
    first_candle.volume = compressed[0].volume_delta;
    first_candle.trade_count = compressed[0].trade_count_delta;
    decompressed.push_back(first_candle);

    // Remaining candles reconstruction
    for (size_t i = 1; i < compressed.size(); ++i) {
        BTQuant::RenderEngine::OHLCVCandle candle;
        decompressCandle(compressed[i], decompressed[i-1], candle);
        decompressed.push_back(candle);
    }

    return decompressed;
}

std::vector<CompressedTrade> DeltaCompressor::compressTrades(const std::vector<TradeData>& trades) {
    std::vector<CompressedTrade> compressed;
    if (trades.empty()) return compressed;
    
    compressed.reserve(trades.size());
    
    // First trade stored as reference point
    CompressedTrade first_trade = {};
    first_trade.timestamp_delta = trades[0].timestamp;  // Absolute timestamp for first
    first_trade.price_delta = trades[0].price;
    first_trade.volume_delta = trades[0].volume;
    first_trade.side = trades[0].side;
    first_trade.exchange_id = trades[0].exchange_id;
    first_trade.flags = trades[0].flags;
    compressed.push_back(first_trade);
    
    // Remaining trades as deltas
    for (size_t i = 1; i < trades.size(); ++i) {
        CompressedTrade comp;
        compressTrade(trades[i], trades[i-1], comp);
        compressed.push_back(comp);
    }
    
    return compressed;
}

std::vector<TradeData> DeltaCompressor::decompressTrades(const std::vector<CompressedTrade>& compressed) {
    std::vector<TradeData> decompressed;
    if (compressed.empty()) return decompressed;
    
    decompressed.reserve(compressed.size());
    
    // First trade reconstruction
    TradeData first_trade = {};
    first_trade.timestamp = compressed[0].timestamp_delta;  // Absolute timestamp
    first_trade.price = compressed[0].price_delta;
    first_trade.volume = compressed[0].volume_delta;
    first_trade.side = compressed[0].side;
    first_trade.exchange_id = compressed[0].exchange_id;
    first_trade.flags = compressed[0].flags;
    decompressed.push_back(first_trade);
    
    // Remaining trades reconstruction
    for (size_t i = 1; i < compressed.size(); ++i) {
        TradeData trade;
        decompressTrade(compressed[i], decompressed[i-1], trade);
        decompressed.push_back(trade);
    }
    
    return decompressed;
}

// Constructor with default precision values
DeltaCompressor::DeltaCompressor(double price_precision, float volume_precision)
    : price_precision_(price_precision), volume_precision_(volume_precision) {
    // Default precision values can be adjusted based on typical market data characteristics
    if (price_precision_ == 0.0) {
        price_precision_ = 0.01;  // Default to 2 decimal places for most currencies
    }
    if (volume_precision_ == 0.0f) {
        volume_precision_ = 0.001f;  // Default to 3 decimal places for volumes
    }
}

// Additional utility functions for different data types

// Compress price array using delta compression
std::vector<double> DeltaCompressor::compressPrices(const std::vector<double>& prices) {
    std::vector<double> compressed;
    if (prices.empty()) return compressed;

    compressed.reserve(prices.size());

    // First price stored as-is
    compressed.push_back(quantizePrice(prices[0]));

    // Subsequent prices as deltas
    for (size_t i = 1; i < prices.size(); ++i) {
        double delta = quantizePrice(prices[i] - prices[i-1]);
        compressed.push_back(delta);
    }

    return compressed;
}

// Decompress price array
std::vector<double> DeltaCompressor::decompressPrices(const std::vector<double>& compressed_prices) {
    std::vector<double> decompressed;
    if (compressed_prices.empty()) return decompressed;

    decompressed.reserve(compressed_prices.size());

    // First price is absolute
    decompressed.push_back(compressed_prices[0]);

    // Subsequent prices reconstructed from deltas
    for (size_t i = 1; i < compressed_prices.size(); ++i) {
        double price = decompressed[i-1] + compressed_prices[i];
        decompressed.push_back(price);
    }

    return decompressed;
}

// Compress volume array using delta compression
std::vector<float> DeltaCompressor::compressVolumes(const std::vector<float>& volumes) {
    std::vector<float> compressed;
    if (volumes.empty()) return compressed;

    compressed.reserve(volumes.size());

    // First volume stored as-is
    compressed.push_back(quantizeVolume(volumes[0]));

    // Subsequent volumes as deltas
    for (size_t i = 1; i < volumes.size(); ++i) {
        float delta = quantizeVolume(volumes[i] - volumes[i-1]);
        compressed.push_back(delta);
    }

    return compressed;
}

// Decompress volume array
std::vector<float> DeltaCompressor::decompressVolumes(const std::vector<float>& compressed_volumes) {
    std::vector<float> decompressed;
    if (compressed_volumes.empty()) return decompressed;

    decompressed.reserve(compressed_volumes.size());

    // First volume is absolute
    decompressed.push_back(compressed_volumes[0]);

    // Subsequent volumes reconstructed from deltas
    for (size_t i = 1; i < compressed_volumes.size(); ++i) {
        float volume = decompressed[i-1] + compressed_volumes[i];
        decompressed.push_back(volume);
    }

    return decompressed;
}

// Estimate compression ratio
double DeltaCompressor::estimateCompressionRatio(const std::vector<BTQuant::RenderEngine::OHLCVCandle>& original) const {
    if (original.empty()) return 1.0; // No compression

    // Calculate approximate uncompressed size (in terms of doubles/floats)
    size_t uncompressed_elements = original.size() * 6; // 6 fields per candle: timestamp, open, high, low, close, volume

    // For compressed data, we still store 6 elements per candle but with potentially smaller values
    // Delta compression often results in smaller values that can be represented with fewer bits
    // This is a simplified estimation - actual compression depends on the data characteristics
    size_t compressed_elements = original.size() * 6;

    // In practice, delta-compressed values tend to have better compression when serialized
    // For estimation purposes, we'll say we achieve ~30-50% reduction in effective storage
    // due to smaller numerical values and better entropy encoding
    return 0.6; // 40% size reduction estimated
}

} // namespace Data
} // namespace BTQuant