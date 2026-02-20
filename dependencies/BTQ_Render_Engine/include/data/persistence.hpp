#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../market_data_processor.hpp"
#include "../cache_manager.hpp"

namespace BTQuant {
namespace Data {

/**
 * DataPersistence - Handles saving and loading of aggregated market data to/from disk
 *
 * This class provides functionality to:
 * - Periodically save aggregated data to disk
 * - Load previously saved data on startup
 * - Support multiple data sources
 * - Handle various data types (profiles, indicators, aggregations)
 */
class DataPersistence {
public:
    explicit DataPersistence(const std::string& data_directory = "./data");
    ~DataPersistence();

    // Non-copyable, non-movable
    DataPersistence(const DataPersistence&) = delete;
    DataPersistence& operator=(const DataPersistence&) = delete;
    DataPersistence(DataPersistence&&) = delete;
    DataPersistence& operator=(DataPersistence&&) = delete;

    /**
     * Save aggregated data to disk
     * @param cache_manager Cache manager containing data to save
     * @param filename Optional filename to save to (default: auto-generated)
     * @return True if successful, false otherwise
     */
    bool saveAggregatedData(const std::shared_ptr<RenderEngine::CacheManager>& cache_manager,
                           const std::string& filename = "");

    /**
     * Load aggregated data from disk
     * @param cache_manager Cache manager to load data into
     * @param filename Filename to load from (default: latest saved file)
     * @return True if successful, false otherwise
     */
    bool loadAggregatedData(const std::shared_ptr<RenderEngine::CacheManager>& cache_manager,
                           const std::string& filename = "");

    /**
     * Save market analytics data to disk
     * @param processor Market data processor containing analytics to save
     * @param filename Optional filename to save to (default: auto-generated)
     * @return True if successful, false otherwise
     */
    bool saveMarketAnalytics(const std::shared_ptr<RenderEngine::MarketDataProcessor>& processor,
                            const std::string& filename = "");

    /**
     * Load market analytics data from disk
     * @param processor Market data processor to load data into
     * @param filename Filename to load from (default: latest saved file)
     * @return True if successful, false otherwise
     */
    bool loadMarketAnalytics(const std::shared_ptr<RenderEngine::MarketDataProcessor>& processor,
                            const std::string& filename = "");

    /**
     * Start periodic saving of data
     * @param interval_seconds Interval in seconds between saves
     */
    void startPeriodicSave(int interval_seconds = 300); // Default: 5 minutes

    /**
     * Stop periodic saving of data
     */
    void stopPeriodicSave();

    /**
     * Get list of available saved data files
     * @return Vector of filenames
     */
    std::vector<std::string> getSavedFiles() const;

    /**
     * Set data directory for saving/loading files
     * @param directory Path to data directory
     */
    void setDataDirectory(const std::string& directory);

    /**
     * Get current data directory
     * @return Current data directory path
     */
    std::string getDataDirectory() const;

private:
    std::string data_directory_;
    std::string last_saved_filename_;
    std::thread periodic_save_thread_;
    std::atomic<bool> periodic_save_running_{false};
    std::mutex save_mutex_;

    // References to data sources
    std::weak_ptr<RenderEngine::CacheManager> cache_manager_ref_;
    std::weak_ptr<RenderEngine::MarketDataProcessor> processor_ref_;

    // Helper methods
    std::string generateFilename(const std::string& prefix = "btq_data") const;
    std::string findLatestFile(const std::string& pattern) const;
    void periodicSaveLoop(int interval_seconds);

    // Serialization helpers
    bool serializeCacheEntry(std::ofstream& file, const RenderEngine::CacheKey& key,
                            const RenderEngine::CacheEntry& entry);
    bool deserializeCacheEntry(std::ifstream& file, RenderEngine::CacheKey& key,
                              RenderEngine::CacheEntry& entry);

public:
    // Methods to set data source references for periodic saves
    void setCacheManager(const std::shared_ptr<RenderEngine::CacheManager>& cache_manager);
    void setMarketDataProcessor(const std::shared_ptr<RenderEngine::MarketDataProcessor>& processor);
};

} // namespace Data
} // namespace BTQuant