#include "../include/data/persistence.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <thread>

namespace fs = std::filesystem;

namespace BTQuant {
namespace Data {

DataPersistence::DataPersistence(const std::string& data_directory)
    : data_directory_(data_directory) {
    // Create data directory if it doesn't exist
    if (!fs::exists(data_directory_)) {
        fs::create_directories(data_directory_);
    }
}

DataPersistence::~DataPersistence() {
    stopPeriodicSave();
}

bool DataPersistence::saveAggregatedData(
    const std::shared_ptr<RenderEngine::CacheManager>& cache_manager,
    const std::string& filename) {
    if (!cache_manager) {
        std::cerr << "[DataPersistence] Error: Cache manager is null" << std::endl;
        return false;
    }

    std::string output_filename = filename.empty() ? generateFilename("aggregated_data") : filename;
    std::string filepath = data_directory_ + "/" + output_filename;

    // Wait for any ongoing save operation to complete
    while (save_in_progress_.test_and_set()) {
        std::this_thread::yield();
    }

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DataPersistence] Error: Could not open file for writing: " << filepath << std::endl;
        save_in_progress_.clear();
        return false;
    }

    // Write file header
    std::string header = "BTQ_DATA_V1";
    file.write(header.c_str(), header.length());
    file.put('\0'); // Null terminator

    // Use the new serialization method
    bool success = cache_manager->serializeToStream(file);

    file.close();

    if (success && file.good()) {
        last_saved_filename_ = output_filename;
        std::cout << "[DataPersistence] Successfully saved aggregated data to: " << filepath << std::endl;
        save_in_progress_.clear();
        return true;
    } else {
        std::cerr << "[DataPersistence] Error: Failed to save aggregated data to: " << filepath << std::endl;
        save_in_progress_.clear();
        return false;
    }
}

bool DataPersistence::loadAggregatedData(
    const std::shared_ptr<RenderEngine::CacheManager>& cache_manager,
    const std::string& filename) {
    if (!cache_manager) {
        std::cerr << "[DataPersistence] Error: Cache manager is null" << std::endl;
        return false;
    }

    std::string input_filename = filename.empty() ? findLatestFile("aggregated_data") : filename;
    if (input_filename.empty()) {
        std::cerr << "[DataPersistence] Error: No file found to load" << std::endl;
        return false;
    }

    std::string filepath = data_directory_ + "/" + input_filename;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DataPersistence] Error: Could not open file for reading: " << filepath << std::endl;
        return false;
    }

    // Read and validate header
    char header[11]; // "BTQ_DATA_V1" + '\0'
    file.read(header, 10);
    header[10] = '\0';

    if (std::string(header) != "BTQ_DATA_V1") {
        std::cerr << "[DataPersistence] Error: Invalid file format" << std::endl;
        file.close();
        return false;
    }

    // Use the new deserialization method
    bool success = cache_manager->deserializeFromStream(file);

    file.close();

    if (success && file.good()) {
        std::cout << "[DataPersistence] Successfully loaded aggregated data from: " << filepath << std::endl;
        return true;
    } else {
        std::cerr << "[DataPersistence] Error: Failed to load aggregated data from: " << filepath << std::endl;
        return false;
    }
}

bool DataPersistence::saveMarketAnalytics(
    const std::shared_ptr<RenderEngine::MarketDataProcessor>& processor,
    const std::string& filename) {
    if (!processor) {
        std::cerr << "[DataPersistence] Error: Market data processor is null" << std::endl;
        return false;
    }

    std::string output_filename = filename.empty() ? generateFilename("market_analytics") : filename;
    std::string filepath = data_directory_ + "/" + output_filename;

    // Wait for any ongoing save operation to complete
    while (save_in_progress_.test_and_set()) {
        std::this_thread::yield();
    }

    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DataPersistence] Error: Could not open file for writing: " << filepath << std::endl;
        save_in_progress_.clear();
        return false;
    }

    // Write file header
    std::string header = "BTQ_ANALYTICS_V1";
    file.write(header.c_str(), header.length());
    file.put('\0'); // Null terminator

    // Get active symbols to save their analytics
    auto active_symbols = processor->getActiveSymbols();
    size_t symbol_count = active_symbols.size();
    file.write(reinterpret_cast<const char*>(&symbol_count), sizeof(symbol_count));

    // For each active symbol, save its analytics
    for (uint32_t symbol_id : active_symbols) {
        auto analytics = processor->getSymbolAnalytics(symbol_id);
        
        // Write symbol ID
        file.write(reinterpret_cast<const char*>(&symbol_id), sizeof(symbol_id));
        
        // Write basic analytics data
        file.write(reinterpret_cast<const char*>(&analytics.last_update_time), sizeof(analytics.last_update_time));
        file.write(reinterpret_cast<const char*>(&analytics.trade_count), sizeof(analytics.trade_count));
        file.write(reinterpret_cast<const char*>(&analytics.last_trade_price), sizeof(analytics.last_trade_price));
        file.write(reinterpret_cast<const char*>(&analytics.last_trade_size), sizeof(analytics.last_trade_size));
        file.write(reinterpret_cast<const char*>(&analytics.last_trade_time), sizeof(analytics.last_trade_time));
        
        // Write volume analytics
        file.write(reinterpret_cast<const char*>(&analytics.volume_1m), sizeof(analytics.volume_1m));
        file.write(reinterpret_cast<const char*>(&analytics.volume_5m), sizeof(analytics.volume_5m));
        file.write(reinterpret_cast<const char*>(&analytics.volume_15m), sizeof(analytics.volume_15m));
        file.write(reinterpret_cast<const char*>(&analytics.buy_volume), sizeof(analytics.buy_volume));
        file.write(reinterpret_cast<const char*>(&analytics.sell_volume), sizeof(analytics.sell_volume));
        file.write(reinterpret_cast<const char*>(&analytics.buy_count), sizeof(analytics.buy_count));
        file.write(reinterpret_cast<const char*>(&analytics.sell_count), sizeof(analytics.sell_count));
        file.write(reinterpret_cast<const char*>(&analytics.buy_sell_ratio), sizeof(analytics.buy_sell_ratio));
        
        // Write price analytics
        file.write(reinterpret_cast<const char*>(&analytics.vwap), sizeof(analytics.vwap));
        file.write(reinterpret_cast<const char*>(&analytics.vwap_deviation), sizeof(analytics.vwap_deviation));
        file.write(reinterpret_cast<const char*>(&analytics.momentum), sizeof(analytics.momentum));
        file.write(reinterpret_cast<const char*>(&analytics.momentum_strength), sizeof(analytics.momentum_strength));
        file.write(reinterpret_cast<const char*>(&analytics.price_min), sizeof(analytics.price_min));
        file.write(reinterpret_cast<const char*>(&analytics.price_max), sizeof(analytics.price_max));
        file.write(reinterpret_cast<const char*>(&analytics.price_position), sizeof(analytics.price_position));
        
        // Write volatility analytics
        file.write(reinterpret_cast<const char*>(&analytics.volatility), sizeof(analytics.volatility));
        file.write(reinterpret_cast<const char*>(&analytics.sharpe_ratio), sizeof(analytics.sharpe_ratio));
        
        // Write trade size analytics
        file.write(reinterpret_cast<const char*>(&analytics.avg_trade_size), sizeof(analytics.avg_trade_size));
        file.write(reinterpret_cast<const char*>(&analytics.large_trade_count), sizeof(analytics.large_trade_count));
        
        // Write orderbook analytics
        file.write(reinterpret_cast<const char*>(&analytics.current_spread), sizeof(analytics.current_spread));
        file.write(reinterpret_cast<const char*>(&analytics.current_spread_percent), sizeof(analytics.current_spread_percent));
        file.write(reinterpret_cast<const char*>(&analytics.avg_spread), sizeof(analytics.avg_spread));
        file.write(reinterpret_cast<const char*>(&analytics.avg_spread_percent), sizeof(analytics.avg_spread_percent));
        file.write(reinterpret_cast<const char*>(&analytics.current_imbalance), sizeof(analytics.current_imbalance));
        file.write(reinterpret_cast<const char*>(&analytics.avg_imbalance), sizeof(analytics.avg_imbalance));
        file.write(reinterpret_cast<const char*>(&analytics.market_depth), sizeof(analytics.market_depth));
        
        // Write OHLCV candles count for each timeframe
        for (int tf = 0; tf < static_cast<int>(RenderEngine::TimeFrame::TF_1WEEK) + 1; ++tf) {
            RenderEngine::TimeFrame timeframe = static_cast<RenderEngine::TimeFrame>(tf);
            
            auto candles_it = analytics.candles.find(timeframe);
            size_t candle_count = (candles_it != analytics.candles.end()) ? candles_it->second.size() : 0;
            file.write(reinterpret_cast<const char*>(&candle_count), sizeof(candle_count));
            
            // Write each candle if they exist
            if (candles_it != analytics.candles.end()) {
                for (const auto& candle : candles_it->second) {
                    file.write(reinterpret_cast<const char*>(&candle.timestamp), sizeof(candle.timestamp));
                    file.write(reinterpret_cast<const char*>(&candle.open), sizeof(candle.open));
                    file.write(reinterpret_cast<const char*>(&candle.high), sizeof(candle.high));
                    file.write(reinterpret_cast<const char*>(&candle.low), sizeof(candle.low));
                    file.write(reinterpret_cast<const char*>(&candle.close), sizeof(candle.close));
                    file.write(reinterpret_cast<const char*>(&candle.volume), sizeof(candle.volume));
                    file.write(reinterpret_cast<const char*>(&candle.trade_count), sizeof(candle.trade_count));
                }
            }
            
            // Write current candle if it exists
            auto current_candle_it = analytics.current_candles.find(timeframe);
            bool has_current_candle = (current_candle_it != analytics.current_candles.end());
            file.write(reinterpret_cast<const char*>(&has_current_candle), sizeof(has_current_candle));
            
            if (has_current_candle) {
                const auto& current_candle = current_candle_it->second;
                file.write(reinterpret_cast<const char*>(&current_candle.timestamp), sizeof(current_candle.timestamp));
                file.write(reinterpret_cast<const char*>(&current_candle.open), sizeof(current_candle.open));
                file.write(reinterpret_cast<const char*>(&current_candle.high), sizeof(current_candle.high));
                file.write(reinterpret_cast<const char*>(&current_candle.low), sizeof(current_candle.low));
                file.write(reinterpret_cast<const char*>(&current_candle.close), sizeof(current_candle.close));
                file.write(reinterpret_cast<const char*>(&current_candle.volume), sizeof(current_candle.volume));
                file.write(reinterpret_cast<const char*>(&current_candle.trade_count), sizeof(current_candle.trade_count));
            }
        }
        
        // Write volume profile data
        size_t vp_count = analytics.session_volume_profile.size();
        file.write(reinterpret_cast<const char*>(&vp_count), sizeof(vp_count));
        
        for (const auto& [price, level] : analytics.session_volume_profile) {
            file.write(reinterpret_cast<const char*>(&level.price), sizeof(level.price));
            file.write(reinterpret_cast<const char*>(&level.total_volume), sizeof(level.total_volume));
            file.write(reinterpret_cast<const char*>(&level.buy_volume), sizeof(level.buy_volume));
            file.write(reinterpret_cast<const char*>(&level.sell_volume), sizeof(level.sell_volume));
        }
    }

    file.close();
    
    if (file.good()) {
        last_saved_filename_ = output_filename;
        std::cout << "[DataPersistence] Successfully saved market analytics to: " << filepath
                  << " for " << symbol_count << " symbols" << std::endl;
        save_in_progress_.clear();
        return true;
    } else {
        std::cerr << "[DataPersistence] Error: Failed to write to file: " << filepath << std::endl;
        save_in_progress_.clear();
        return false;
    }
}

bool DataPersistence::loadMarketAnalytics(
    const std::shared_ptr<RenderEngine::MarketDataProcessor>& processor,
    const std::string& filename) {
    if (!processor) {
        std::cerr << "[DataPersistence] Error: Market data processor is null" << std::endl;
        return false;
    }

    std::string input_filename = filename.empty() ? findLatestFile("market_analytics") : filename;
    if (input_filename.empty()) {
        std::cerr << "[DataPersistence] Error: No file found to load" << std::endl;
        return false;
    }

    std::string filepath = data_directory_ + "/" + input_filename;

    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DataPersistence] Error: Could not open file for reading: " << filepath << std::endl;
        return false;
    }

    // Read and validate header
    char header[16]; // "BTQ_ANALYTICS_V1" + '\0'
    file.read(header, 15);
    header[15] = '\0';
    
    if (std::string(header) != "BTQ_ANALYTICS_V1") {
        std::cerr << "[DataPersistence] Error: Invalid file format" << std::endl;
        file.close();
        return false;
    }

    // Read number of symbols
    size_t symbol_count;
    file.read(reinterpret_cast<char*>(&symbol_count), sizeof(symbol_count));

    // For each symbol, read its analytics
    for (size_t i = 0; i < symbol_count; ++i) {
        uint32_t symbol_id;
        file.read(reinterpret_cast<char*>(&symbol_id), sizeof(symbol_id));
        
        RenderEngine::SymbolAnalytics analytics;
        analytics.symbol_id = symbol_id;
        
        // Read basic analytics data
        file.read(reinterpret_cast<char*>(&analytics.last_update_time), sizeof(analytics.last_update_time));
        file.read(reinterpret_cast<char*>(&analytics.trade_count), sizeof(analytics.trade_count));
        file.read(reinterpret_cast<char*>(&analytics.last_trade_price), sizeof(analytics.last_trade_price));
        file.read(reinterpret_cast<char*>(&analytics.last_trade_size), sizeof(analytics.last_trade_size));
        file.read(reinterpret_cast<char*>(&analytics.last_trade_time), sizeof(analytics.last_trade_time));
        
        // Read volume analytics
        file.read(reinterpret_cast<char*>(&analytics.volume_1m), sizeof(analytics.volume_1m));
        file.read(reinterpret_cast<char*>(&analytics.volume_5m), sizeof(analytics.volume_5m));
        file.read(reinterpret_cast<char*>(&analytics.volume_15m), sizeof(analytics.volume_15m));
        file.read(reinterpret_cast<char*>(&analytics.buy_volume), sizeof(analytics.buy_volume));
        file.read(reinterpret_cast<char*>(&analytics.sell_volume), sizeof(analytics.sell_volume));
        file.read(reinterpret_cast<char*>(&analytics.buy_count), sizeof(analytics.buy_count));
        file.read(reinterpret_cast<char*>(&analytics.sell_count), sizeof(analytics.sell_count));
        file.read(reinterpret_cast<char*>(&analytics.buy_sell_ratio), sizeof(analytics.buy_sell_ratio));
        
        // Read price analytics
        file.read(reinterpret_cast<char*>(&analytics.vwap), sizeof(analytics.vwap));
        file.read(reinterpret_cast<char*>(&analytics.vwap_deviation), sizeof(analytics.vwap_deviation));
        file.read(reinterpret_cast<char*>(&analytics.momentum), sizeof(analytics.momentum));
        file.read(reinterpret_cast<char*>(&analytics.momentum_strength), sizeof(analytics.momentum_strength));
        file.read(reinterpret_cast<char*>(&analytics.price_min), sizeof(analytics.price_min));
        file.read(reinterpret_cast<char*>(&analytics.price_max), sizeof(analytics.price_max));
        file.read(reinterpret_cast<char*>(&analytics.price_position), sizeof(analytics.price_position));
        
        // Read volatility analytics
        file.read(reinterpret_cast<char*>(&analytics.volatility), sizeof(analytics.volatility));
        file.read(reinterpret_cast<char*>(&analytics.sharpe_ratio), sizeof(analytics.sharpe_ratio));
        
        // Read trade size analytics
        file.read(reinterpret_cast<char*>(&analytics.avg_trade_size), sizeof(analytics.avg_trade_size));
        file.read(reinterpret_cast<char*>(&analytics.large_trade_count), sizeof(analytics.large_trade_count));
        
        // Read orderbook analytics
        file.read(reinterpret_cast<char*>(&analytics.current_spread), sizeof(analytics.current_spread));
        file.read(reinterpret_cast<char*>(&analytics.current_spread_percent), sizeof(analytics.current_spread_percent));
        file.read(reinterpret_cast<char*>(&analytics.avg_spread), sizeof(analytics.avg_spread));
        file.read(reinterpret_cast<char*>(&analytics.avg_spread_percent), sizeof(analytics.avg_spread_percent));
        file.read(reinterpret_cast<char*>(&analytics.current_imbalance), sizeof(analytics.current_imbalance));
        file.read(reinterpret_cast<char*>(&analytics.avg_imbalance), sizeof(analytics.avg_imbalance));
        file.read(reinterpret_cast<char*>(&analytics.market_depth), sizeof(analytics.market_depth));
        
        // Read OHLCV candles for each timeframe
        for (int tf = 0; tf < static_cast<int>(RenderEngine::TimeFrame::TF_1WEEK) + 1; ++tf) {
            RenderEngine::TimeFrame timeframe = static_cast<RenderEngine::TimeFrame>(tf);
            
            size_t candle_count;
            file.read(reinterpret_cast<char*>(&candle_count), sizeof(candle_count));
            
            // Read each candle
            std::vector<RenderEngine::OHLCVCandle> candles;
            for (size_t j = 0; j < candle_count; ++j) {
                RenderEngine::OHLCVCandle candle;
                file.read(reinterpret_cast<char*>(&candle.timestamp), sizeof(candle.timestamp));
                file.read(reinterpret_cast<char*>(&candle.open), sizeof(candle.open));
                file.read(reinterpret_cast<char*>(&candle.high), sizeof(candle.high));
                file.read(reinterpret_cast<char*>(&candle.low), sizeof(candle.low));
                file.read(reinterpret_cast<char*>(&candle.close), sizeof(candle.close));
                file.read(reinterpret_cast<char*>(&candle.volume), sizeof(candle.volume));
                file.read(reinterpret_cast<char*>(&candle.trade_count), sizeof(candle.trade_count));
                
                candles.push_back(candle);
            }
            
            if (!candles.empty()) {
                analytics.candles[timeframe] = std::move(candles);
            }
            
            // Read current candle if it exists
            bool has_current_candle;
            file.read(reinterpret_cast<char*>(&has_current_candle), sizeof(has_current_candle));
            
            if (has_current_candle) {
                RenderEngine::OHLCVCandle current_candle;
                file.read(reinterpret_cast<char*>(&current_candle.timestamp), sizeof(current_candle.timestamp));
                file.read(reinterpret_cast<char*>(&current_candle.open), sizeof(current_candle.open));
                file.read(reinterpret_cast<char*>(&current_candle.high), sizeof(current_candle.high));
                file.read(reinterpret_cast<char*>(&current_candle.low), sizeof(current_candle.low));
                file.read(reinterpret_cast<char*>(&current_candle.close), sizeof(current_candle.close));
                file.read(reinterpret_cast<char*>(&current_candle.volume), sizeof(current_candle.volume));
                file.read(reinterpret_cast<char*>(&current_candle.trade_count), sizeof(current_candle.trade_count));
                
                analytics.current_candles[timeframe] = current_candle;
            }
        }
        
        // Read volume profile data
        size_t vp_count;
        file.read(reinterpret_cast<char*>(&vp_count), sizeof(vp_count));
        
        for (size_t j = 0; j < vp_count; ++j) {
            VolumeProfileLevel level;
            file.read(reinterpret_cast<char*>(&level.price), sizeof(level.price));
            file.read(reinterpret_cast<char*>(&level.total_volume), sizeof(level.total_volume));
            file.read(reinterpret_cast<char*>(&level.buy_volume), sizeof(level.buy_volume));
            file.read(reinterpret_cast<char*>(&level.sell_volume), sizeof(level.sell_volume));
            
            analytics.session_volume_profile[level.price] = level;
        }
        
        // In a real implementation, we would need to update the processor with the loaded analytics
        // Since the processor stores data internally, we would need a method to inject this data
        // For now, we'll just note that the data was loaded
        std::cout << "[DataPersistence] Loaded analytics for symbol ID: " << symbol_id << std::endl;
    }

    file.close();

    if (file.good()) {
        std::cout << "[DataPersistence] Successfully loaded market analytics from: " << filepath 
                  << " for " << symbol_count << " symbols" << std::endl;
        return true;
    } else {
        std::cerr << "[DataPersistence] Error: Failed to read from file: " << filepath << std::endl;
        return false;
    }
}

void DataPersistence::startPeriodicSave(int interval_seconds) {
    if (periodic_save_running_.load()) {
        std::cout << "[DataPersistence] Periodic save is already running" << std::endl;
        return;
    }

    periodic_save_running_ = true;
    periodic_save_thread_ = std::thread(&DataPersistence::periodicSaveLoop, this, interval_seconds);
    
    std::cout << "[DataPersistence] Started periodic save with interval: " << interval_seconds << " seconds" << std::endl;
}

void DataPersistence::stopPeriodicSave() {
    if (!periodic_save_running_.load()) {
        return;
    }

    periodic_save_running_ = false;
    
    if (periodic_save_thread_.joinable()) {
        periodic_save_thread_.join();
    }
    
    std::cout << "[DataPersistence] Stopped periodic save" << std::endl;
}

std::vector<std::string> DataPersistence::getSavedFiles() const {
    std::vector<std::string> files;
    
    try {
        for (const auto& entry : fs::directory_iterator(data_directory_)) {
            if (entry.is_regular_file() && entry.path().extension() == ".dat") {
                files.push_back(entry.path().filename().string());
            }
        }
        
        // Sort files by modification time (newest first)
        std::sort(files.begin(), files.end(), [&](const std::string& a, const std::string& b) {
            return fs::last_write_time(data_directory_ + "/" + a) > 
                   fs::last_write_time(data_directory_ + "/" + b);
        });
    } catch (const fs::filesystem_error& ex) {
        std::cerr << "[DataPersistence] Error reading directory: " << ex.what() << std::endl;
    }
    
    return files;
}

void DataPersistence::setDataDirectory(const std::string& directory) {
    data_directory_ = directory;
    
    // Create directory if it doesn't exist
    if (!fs::exists(data_directory_)) {
        fs::create_directories(data_directory_);
    }
}

std::string DataPersistence::getDataDirectory() const {
    return data_directory_;
}

std::string DataPersistence::generateFilename(const std::string& prefix) const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << prefix << "_" << std::put_time(std::gmtime(&time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << ms.count() << ".dat";

    return ss.str();
}

std::string DataPersistence::findLatestFile(const std::string& pattern) const {
    std::vector<std::string> files = getSavedFiles();
    
    for (const auto& filename : files) {
        if (filename.find(pattern) != std::string::npos) {
            return filename;
        }
    }
    
    return "";
}

void DataPersistence::setCacheManager(const std::shared_ptr<RenderEngine::CacheManager>& cache_manager) {
    cache_manager_ref_ = cache_manager;
}

void DataPersistence::setMarketDataProcessor(const std::shared_ptr<RenderEngine::MarketDataProcessor>& processor) {
    processor_ref_ = processor;
}

void DataPersistence::periodicSaveLoop(int interval_seconds) {
    while (periodic_save_running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(interval_seconds));

        if (!periodic_save_running_.load()) {
            break;
        }

        // Attempt to lock weak pointers and save data
        if (auto cache_manager = cache_manager_ref_.lock()) {
            std::string filename = generateFilename("periodic_aggregated_data");
            if (saveAggregatedData(cache_manager, filename)) {
                std::cout << "[DataPersistence] Periodic save of aggregated data completed: " << filename << std::endl;
            } else {
                std::cerr << "[DataPersistence] Periodic save of aggregated data failed" << std::endl;
            }
        }

        if (auto processor = processor_ref_.lock()) {
            std::string filename = generateFilename("periodic_market_analytics");
            if (saveMarketAnalytics(processor, filename)) {
                std::cout << "[DataPersistence] Periodic save of market analytics completed: " << filename << std::endl;
            } else {
                std::cerr << "[DataPersistence] Periodic save of market analytics failed" << std::endl;
            }
        }
    }
}

} // namespace Data
} // namespace BTQuant