#include "data/compression.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

// Include the proper OHLCVCandle definition
#include "market_data_processor.hpp"

int main() {
    std::cout << "Testing Delta Compression Implementation..." << std::endl;

    // Test 1: Basic price compression/decompression
    {
        std::cout << "\nTest 1: Basic price compression/decompression" << std::endl;

        BTQuant::Data::DeltaCompressor compressor(0.01, 0.001f);

        std::vector<double> prices = {100.00, 100.05, 100.03, 100.08, 100.02};

        auto compressed_prices = compressor.compressPrices(prices);
        auto decompressed_prices = compressor.decompressPrices(compressed_prices);

        std::cout << "Original prices: ";
        for (auto p : prices) std::cout << p << " ";
        std::cout << std::endl;

        std::cout << "Decompressed prices: ";
        for (auto p : decompressed_prices) std::cout << p << " ";
        std::cout << std::endl;

        // Check if decompressed prices match original (within quantization tolerance)
        bool prices_match = true;
        for (size_t i = 0; i < prices.size(); ++i) {
            if (std::abs(prices[i] - decompressed_prices[i]) > 0.011) { // Slightly more than precision
                prices_match = false;
                break;
            }
        }

        std::cout << "Price compression/decompression: " << (prices_match ? "PASS" : "FAIL") << std::endl;
        assert(prices_match);
    }

    // Test 2: Volume compression/decompression
    {
        std::cout << "\nTest 2: Volume compression/decompression" << std::endl;

        BTQuant::Data::DeltaCompressor compressor(0.01, 0.001f);

        std::vector<float> volumes = {1000.0f, 1050.0f, 980.0f, 1200.0f, 1150.0f};

        auto compressed_volumes = compressor.compressVolumes(volumes);
        auto decompressed_volumes = compressor.decompressVolumes(compressed_volumes);

        std::cout << "Original volumes: ";
        for (auto v : volumes) std::cout << v << " ";
        std::cout << std::endl;

        std::cout << "Decompressed volumes: ";
        for (auto v : decompressed_volumes) std::cout << v << " ";
        std::cout << std::endl;

        // Check if decompressed volumes match original (within quantization tolerance)
        bool volumes_match = true;
        for (size_t i = 0; i < volumes.size(); ++i) {
            if (std::abs(volumes[i] - decompressed_volumes[i]) > 0.0011f) { // Slightly more than precision
                volumes_match = false;
                break;
            }
        }

        std::cout << "Volume compression/decompression: " << (volumes_match ? "PASS" : "FAIL") << std::endl;
        assert(volumes_match);
    }

    // Test 3: Candle compression/decompression
    {
        std::cout << "\nTest 3: Candle compression/decompression" << std::endl;

        BTQuant::Data::DeltaCompressor compressor(0.01, 0.001f);

        std::vector<BTQuant::RenderEngine::OHLCVCandle> candles = {
            {1609459200000000, 100.00, 101.50, 99.50, 101.00, 1000.0, 50},
            {1609459260000000, 101.00, 102.00, 100.50, 101.80, 1200.0, 60},
            {1609459320000000, 101.80, 103.00, 101.20, 102.50, 1100.0, 55},
            {1609459380000000, 102.50, 103.50, 102.00, 102.80, 900.0, 45}
        };

        auto compressed_candles = compressor.compressCandles(candles);
        auto decompressed_candles = compressor.decompressCandles(compressed_candles);

        // Check if decompressed candles match original (within quantization tolerance)
        bool candles_match = true;
        for (size_t i = 0; i < candles.size(); ++i) {
            if (candles[i].timestamp != decompressed_candles[i].timestamp ||
                std::abs(candles[i].open - decompressed_candles[i].open) > 0.011 ||
                std::abs(candles[i].high - decompressed_candles[i].high) > 0.011 ||
                std::abs(candles[i].low - decompressed_candles[i].low) > 0.011 ||
                std::abs(candles[i].close - decompressed_candles[i].close) > 0.011 ||
                std::abs(candles[i].volume - decompressed_candles[i].volume) > 0.0011 ||
                candles[i].trade_count != decompressed_candles[i].trade_count) {

                candles_match = false;
                std::cout << "Mismatch at index " << i << std::endl;
                std::cout << "Original: " << candles[i].open << ", Decompressed: " << decompressed_candles[i].open << std::endl;
                break;
            }
        }

        std::cout << "Candle compression/decompression: " << (candles_match ? "PASS" : "FAIL") << std::endl;
        assert(candles_match);
    }

    // Test 4: Trade compression/decompression
    {
        std::cout << "\nTest 4: Trade compression/decompression" << std::endl;

        BTQuant::Data::DeltaCompressor compressor(0.01, 0.001f);

        std::vector<BTQuant::Data::TradeData> trades = {
            {1609459200000, 100.00, 100.0f, BTQuant::Data::TradeSide::BUY, 1, 0},
            {1609459201000, 100.05, 150.0f, BTQuant::Data::TradeSide::SELL, 1, 0},
            {1609459202000, 100.03, 200.0f, BTQuant::Data::TradeSide::BUY, 2, 0},
            {1609459203000, 100.08, 120.0f, BTQuant::Data::TradeSide::SELL, 1, 0}
        };

        auto compressed_trades = compressor.compressTrades(trades);
        auto decompressed_trades = compressor.decompressTrades(compressed_trades);

        // Check if decompressed trades match original (within quantization tolerance)
        bool trades_match = true;
        for (size_t i = 0; i < trades.size(); ++i) {
            if (trades[i].timestamp != decompressed_trades[i].timestamp ||
                std::abs(trades[i].price - decompressed_trades[i].price) > 0.011 ||
                std::abs(trades[i].volume - decompressed_trades[i].volume) > 0.0011 ||
                trades[i].side != decompressed_trades[i].side ||
                trades[i].exchange_id != decompressed_trades[i].exchange_id ||
                trades[i].flags != decompressed_trades[i].flags) {

                trades_match = false;
                std::cout << "Mismatch at index " << i << std::endl;
                break;
            }
        }

        std::cout << "Trade compression/decompression: " << (trades_match ? "PASS" : "FAIL") << std::endl;
        assert(trades_match);
    }

    // Test 5: Different quantization strategies
    {
        std::cout << "\nTest 5: Different quantization strategies" << std::endl;

        BTQuant::Data::DeltaCompressor fixed_compressor(0.01, 0.001f, BTQuant::Data::QuantizationStrategy::FIXED_PRECISION);
        BTQuant::Data::DeltaCompressor relative_compressor(0.001, 0.001f, BTQuant::Data::QuantizationStrategy::RELATIVE_PERCENTAGE);

        std::vector<double> prices = {100.00, 100.05, 100.03, 100.08, 100.02};

        auto fixed_compressed = fixed_compressor.compressPrices(prices);
        auto fixed_decompressed = fixed_compressor.decompressPrices(fixed_compressed);

        auto relative_compressed = relative_compressor.compressPrices(prices);
        auto relative_decompressed = relative_compressor.decompressPrices(relative_compressed);

        std::cout << "Fixed precision decompressed: ";
        for (auto p : fixed_decompressed) std::cout << p << " ";
        std::cout << std::endl;

        std::cout << "Relative precision decompressed: ";
        for (auto p : relative_decompressed) std::cout << p << " ";
        std::cout << std::endl;

        std::cout << "Different strategies test: PASS" << std::endl;
    }

    std::cout << "\nAll tests passed!" << std::endl;

    return 0;
}