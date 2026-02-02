#pragma once

#include <atomic>
#include <chrono>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "data/data_types.hpp"

namespace BTQuant {
namespace RenderEngine {

// Enum for different types of cached data
enum class CacheType {
    PROFILE,      // Volume profile data
    INDICATOR,    // Technical indicator values
    AGGREGATION   // Aggregated statistical data
};

// Key structure for cache entries
struct CacheKey {
    CacheType type;
    uint32_t symbol_id;
    TimeFrame timeframe;
    uint64_t bar_timestamp;
    std::string extra_param;  // For indicator names or aggregation types
    
    bool operator==(const CacheKey& other) const {
        return type == other.type &&
               symbol_id == other.symbol_id &&
               timeframe == other.timeframe &&
               bar_timestamp == other.bar_timestamp &&
               extra_param == other.extra_param;
    }
};

// Hash function for CacheKey
struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const {
        std::size_t h1 = std::hash<int>{}(static_cast<int>(k.type));
        std::size_t h2 = std::hash<uint32_t>{}(k.symbol_id);
        std::size_t h3 = std::hash<int>{}(static_cast<int>(k.timeframe));
        std::size_t h4 = std::hash<uint64_t>{}(k.bar_timestamp);
        std::size_t h5 = std::hash<std::string>{}(k.extra_param);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
};

// Cache entry structure
struct CacheEntry {
    std::shared_ptr<void> data;  // Generic pointer to cached data
    size_t size_bytes;
    std::chrono::high_resolution_clock::time_point creation_time;
    std::chrono::high_resolution_clock::time_point last_access_time;
};

// Statistics structure
struct CacheStats {
    size_t entry_count = 0;
    size_t current_size_bytes = 0;
    size_t max_size_bytes = 0;
    uint64_t hit_count = 0;
    uint64_t miss_count = 0;
    
    double getHitRate() const {
        uint64_t total = hit_count + miss_count;
        return total > 0 ? static_cast<double>(hit_count) / total : 0.0;
    }
};

/**
 * CacheManager - Centralized caching system for calculated profiles, indicators, and aggregations
 *
 * This class provides a unified caching solution for:
 * - Volume profile calculations
 * - Technical indicator values
 * - Statistical aggregations
 *
 * Features:
 * - Per-bar caching with timestamp precision
 * - Automatic invalidation on new data arrival
 * - LRU eviction policy when size limits are reached
 * - Thread-safe operations with shared/exclusive locking
 * - Cache statistics tracking
 */
class CacheManager {
public:
    explicit CacheManager(size_t max_cache_size_bytes = 100 * 1024 * 1024); // Default 100MB
    ~CacheManager();

    // Non-copyable, non-movable
    CacheManager(const CacheManager&) = delete;
    CacheManager& operator=(const CacheManager&) = delete;
    CacheManager(CacheManager&&) = delete;
    CacheManager& operator=(CacheManager&&) = delete;

    /**
     * Cache calculated profile data
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar being cached
     * @param profile Volume profile data to cache
     */
    void cacheProfileData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                         const std::vector<VolumeProfileLevel>& profile);

    /**
     * Cache calculated indicator data
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar being cached
     * @param indicator_name Name of the indicator
     * @param indicator_values Calculated indicator values
     */
    void cacheIndicatorData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                           const std::string& indicator_name,
                           const std::vector<double>& indicator_values);

    /**
     * Cache aggregated data
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar being cached
     * @param aggregation_type Type of aggregation
     * @param aggregation_values Aggregated values
     */
    void cacheAggregatedData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                            const std::string& aggregation_type,
                            const std::vector<double>& aggregation_values);

    /**
     * Retrieve cached profile data
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar to retrieve
     * @return Cached profile data if found, std::nullopt otherwise
     */
    std::optional<std::vector<VolumeProfileLevel>> getCachedProfileData(
        uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp);

    /**
     * Retrieve cached indicator data
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar to retrieve
     * @param indicator_name Name of the indicator
     * @return Cached indicator data if found, std::nullopt otherwise
     */
    std::optional<std::vector<double>> getCachedIndicatorData(
        uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
        const std::string& indicator_name);

    /**
     * Retrieve cached aggregation data
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar to retrieve
     * @param aggregation_type Type of aggregation
     * @return Cached aggregation data if found, std::nullopt otherwise
     */
    std::optional<std::vector<double>> getCachedAggregatedData(
        uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
        const std::string& aggregation_type);

    /**
     * Invalidate cache for a specific symbol and timeframe when new data arrives
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame to invalidate
     * @param new_data_timestamp Timestamp of new data (invalidate all entries >= this time)
     */
    void invalidateCache(uint32_t symbol_id, TimeFrame timeframe, uint64_t new_data_timestamp);

    /**
     * Invalidate cache for a specific symbol across all timeframes
     * @param symbol_id Symbol identifier
     */
    void invalidateCacheForSymbol(uint32_t symbol_id);

    /**
     * Invalidate cache for a specific bar (timestamp)
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame
     * @param bar_timestamp Timestamp of the bar to invalidate
     */
    void invalidateCacheForBar(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp);

    /**
     * Clear all caches
     */
    void clearAllCaches();

    /**
     * Check if profile data is cached
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar to check
     * @return True if cached, false otherwise
     */
    bool hasCachedProfileData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp);

    /**
     * Check if indicator data is cached
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar to check
     * @param indicator_name Name of the indicator
     * @return True if cached, false otherwise
     */
    bool hasCachedIndicatorData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                               const std::string& indicator_name);

    /**
     * Check if aggregation data is cached
     * @param symbol_id Symbol identifier
     * @param timeframe Time frame for the data
     * @param bar_timestamp Timestamp of the bar to check
     * @param aggregation_type Type of aggregation
     * @return True if cached, false otherwise
     */
    bool hasCachedAggregatedData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                                const std::string& aggregation_type);

    /**
     * Get cache statistics
     * @return Current cache statistics
     */
    CacheStats getCacheStats() const;

    /**
     * Clean up stale entries based on age
     * @param max_age Maximum age of entries to keep
     */
    void cleanupStaleEntries(std::chrono::seconds max_age);

    /**
     * Serialize cache data to a stream
     * @param out Output stream to write to
     * @return True if successful, false otherwise
     */
    bool serializeToStream(std::ostream& out) const;

    /**
     * Deserialize cache data from a stream
     * @param in Input stream to read from
     * @return True if successful, false otherwise
     */
    bool deserializeFromStream(std::istream& in);

private:
    // Cache storage
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> cache_;
    
    // LRU tracking structures
    std::list<CacheKey> lru_list_;
    std::unordered_map<CacheKey, decltype(lru_list_)::iterator> lru_map_;
    
    // Configuration
    size_t max_cache_size_bytes_;
    std::atomic<size_t> current_cache_size_bytes_;
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    
    // Statistics
    mutable std::atomic<uint64_t> hit_count_{0};
    mutable std::atomic<uint64_t> miss_count_{0};
    
    // Private helper methods
    void evictIfNeeded(size_t new_entry_size);
    void updateAccessTime(const CacheKey& key);
    void invalidateSpecificKey(const CacheKey& key);
    void recordHit();
    void recordMiss();
};

} // namespace RenderEngine
} // namespace BTQuant