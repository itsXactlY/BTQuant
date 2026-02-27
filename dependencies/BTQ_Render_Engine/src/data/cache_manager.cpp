#include "cache_manager.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace BTQuant {
namespace RenderEngine {

// Constructor
CacheManager::CacheManager(size_t max_cache_size_bytes)
    : max_cache_size_bytes_(max_cache_size_bytes), current_cache_size_bytes_(0) {
    std::cout << "[CacheManager] Initialized with max size: " << max_cache_size_bytes << " bytes" << std::endl;
}

// Destructor
CacheManager::~CacheManager() {
    clearAllCaches();
}

// Cache calculated profile data
void CacheManager::cacheProfileData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                                   const std::vector<VolumeProfileLevel>& profile) {
    std::unique_lock lock(mutex_);
    
    CacheKey key{CacheType::PROFILE, symbol_id, timeframe, bar_timestamp};
    
    // Calculate approximate size of the profile data
    size_t profile_size = sizeof(VolumeProfileLevel) * profile.size();
    
    // Check if we need to evict old entries due to size constraints
    evictIfNeeded(profile_size);
    
    // Store the profile data
    auto& entry = cache_[key];
    entry.data = std::make_shared<std::vector<VolumeProfileLevel>>(profile);
    entry.size_bytes = profile_size;
    entry.last_access_time = std::chrono::high_resolution_clock::now();
    entry.creation_time = entry.last_access_time;
    
    current_cache_size_bytes_ += profile_size;
    
    // Update LRU tracking
    lru_list_.push_front(key);
    lru_map_[key] = lru_list_.begin();
}

// Cache calculated indicator data
void CacheManager::cacheIndicatorData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                                     const std::string& indicator_name,
                                     const std::vector<double>& indicator_values) {
    std::unique_lock lock(mutex_);
    
    CacheKey key{CacheType::INDICATOR, symbol_id, timeframe, bar_timestamp, indicator_name};
    
    // Calculate approximate size of the indicator data
    size_t indicator_size = sizeof(double) * indicator_values.size() + indicator_name.size();
    
    // Check if we need to evict old entries due to size constraints
    evictIfNeeded(indicator_size);
    
    // Store the indicator data
    auto& entry = cache_[key];
    entry.data = std::make_shared<std::vector<double>>(indicator_values);
    entry.size_bytes = indicator_size;
    entry.last_access_time = std::chrono::high_resolution_clock::now();
    entry.creation_time = entry.last_access_time;
    
    current_cache_size_bytes_ += indicator_size;
    
    // Update LRU tracking
    lru_list_.push_front(key);
    lru_map_[key] = lru_list_.begin();
}

// Cache aggregated data
void CacheManager::cacheAggregatedData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                                       const std::string& aggregation_type,
                                       const std::vector<double>& aggregation_values) {
    std::unique_lock lock(mutex_);
    
    CacheKey key{CacheType::AGGREGATION, symbol_id, timeframe, bar_timestamp, aggregation_type};
    
    // Calculate approximate size of the aggregation data
    size_t aggregation_size = sizeof(double) * aggregation_values.size() + aggregation_type.size();
    
    // Check if we need to evict old entries due to size constraints
    evictIfNeeded(aggregation_size);
    
    // Store the aggregation data
    auto& entry = cache_[key];
    entry.data = std::make_shared<std::vector<double>>(aggregation_values);
    entry.size_bytes = aggregation_size;
    entry.last_access_time = std::chrono::high_resolution_clock::now();
    entry.creation_time = entry.last_access_time;
    
    current_cache_size_bytes_ += aggregation_size;
    
    // Update LRU tracking
    lru_list_.push_front(key);
    lru_map_[key] = lru_list_.begin();
}

// Retrieve cached profile data
std::optional<std::vector<VolumeProfileLevel>> CacheManager::getCachedProfileData(
    uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp) {
    std::shared_lock lock(mutex_);
    
    CacheKey key{CacheType::PROFILE, symbol_id, timeframe, bar_timestamp};
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Update access time for LRU
        updateAccessTime(key);
        
        auto profile_ptr = std::static_pointer_cast<std::vector<VolumeProfileLevel>>(it->second.data);
        return *profile_ptr;
    }
    
    return std::nullopt;
}

// Retrieve cached indicator data
std::optional<std::vector<double>> CacheManager::getCachedIndicatorData(
    uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
    const std::string& indicator_name) {
    std::shared_lock lock(mutex_);
    
    CacheKey key{CacheType::INDICATOR, symbol_id, timeframe, bar_timestamp, indicator_name};
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Update access time for LRU
        updateAccessTime(key);
        
        auto indicator_ptr = std::static_pointer_cast<std::vector<double>>(it->second.data);
        return *indicator_ptr;
    }
    
    return std::nullopt;
}

// Retrieve cached aggregation data
std::optional<std::vector<double>> CacheManager::getCachedAggregatedData(
    uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
    const std::string& aggregation_type) {
    std::shared_lock lock(mutex_);
    
    CacheKey key{CacheType::AGGREGATION, symbol_id, timeframe, bar_timestamp, aggregation_type};
    
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Update access time for LRU
        updateAccessTime(key);
        
        auto aggregation_ptr = std::static_pointer_cast<std::vector<double>>(it->second.data);
        return *aggregation_ptr;
    }
    
    return std::nullopt;
}

// Invalidate cache for a specific symbol and timeframe when new data arrives
void CacheManager::invalidateCache(uint32_t symbol_id, TimeFrame timeframe, uint64_t new_data_timestamp) {
    std::unique_lock lock(mutex_);
    
    // Find and remove all entries for this symbol/timeframe that are at or after the new data timestamp
    auto it = cache_.begin();
    while (it != cache_.end()) {
        const CacheKey& key = it->first;
        if (key.symbol_id == symbol_id && key.timeframe == timeframe && key.bar_timestamp >= new_data_timestamp) {
            // Remove from LRU tracking
            if (lru_map_.find(key) != lru_map_.end()) {
                lru_list_.erase(lru_map_[key]);
                lru_map_.erase(key);
            }
            
            // Reduce cache size
            current_cache_size_bytes_ -= it->second.size_bytes;
            
            // Remove the entry
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

// Invalidate cache for a specific symbol across all timeframes
void CacheManager::invalidateCacheForSymbol(uint32_t symbol_id) {
    std::unique_lock lock(mutex_);
    
    auto it = cache_.begin();
    while (it != cache_.end()) {
        const CacheKey& key = it->first;
        if (key.symbol_id == symbol_id) {
            // Remove from LRU tracking
            if (lru_map_.find(key) != lru_map_.end()) {
                lru_list_.erase(lru_map_[key]);
                lru_map_.erase(key);
            }
            
            // Reduce cache size
            current_cache_size_bytes_ -= it->second.size_bytes;
            
            // Remove the entry
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

// Invalidate cache for all symbols and timeframes
void CacheManager::clearAllCaches() {
    std::unique_lock lock(mutex_);
    
    cache_.clear();
    lru_list_.clear();
    lru_map_.clear();
    current_cache_size_bytes_ = 0;
}

// Invalidate cache for a specific bar (timestamp)
void CacheManager::invalidateCacheForBar(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp) {
    std::unique_lock lock(mutex_);
    
    CacheKey key{CacheType::PROFILE, symbol_id, timeframe, bar_timestamp};
    invalidateSpecificKey(key);
    
    // Also invalidate any indicator data for this bar
    // Since we can't iterate efficiently with the current key structure, we'll need to find all indicators
    // for this symbol/timeframe/bar combination
    auto it = cache_.begin();
    while (it != cache_.end()) {
        const CacheKey& cache_key = it->first;
        if (cache_key.symbol_id == symbol_id && 
            cache_key.timeframe == timeframe && 
            cache_key.bar_timestamp == bar_timestamp &&
            cache_key.type == CacheType::INDICATOR) {
            invalidateSpecificKey(cache_key);
            it = cache_.begin(); // Restart iteration since we invalidated an element
            continue;
        }
        ++it;
    }
    
    // Also invalidate any aggregation data for this bar
    it = cache_.begin();
    while (it != cache_.end()) {
        const CacheKey& cache_key = it->first;
        if (cache_key.symbol_id == symbol_id && 
            cache_key.timeframe == timeframe && 
            cache_key.bar_timestamp == bar_timestamp &&
            cache_key.type == CacheType::AGGREGATION) {
            invalidateSpecificKey(cache_key);
            it = cache_.begin(); // Restart iteration since we invalidated an element
            continue;
        }
        ++it;
    }
}

// Get cache statistics
CacheStats CacheManager::getCacheStats() const {
    std::shared_lock lock(mutex_);
    
    CacheStats stats;
    stats.entry_count = cache_.size();
    stats.current_size_bytes = current_cache_size_bytes_;
    stats.max_size_bytes = max_cache_size_bytes_;
    stats.hit_count = hit_count_.load();
    stats.miss_count = miss_count_.load();
    
    return stats;
}

// Private helper methods

void CacheManager::evictIfNeeded(size_t new_entry_size) {
    // Check if adding the new entry would exceed the cache size limit
    while (current_cache_size_bytes_ + new_entry_size > max_cache_size_bytes_ && !cache_.empty()) {
        // Evict the least recently used entry
        if (!lru_list_.empty()) {
            CacheKey lru_key = lru_list_.back();
            
            auto it = cache_.find(lru_key);
            if (it != cache_.end()) {
                // Remove from LRU tracking
                lru_map_.erase(lru_key);
                lru_list_.pop_back();
                
                // Reduce cache size
                current_cache_size_bytes_ -= it->second.size_bytes;
                
                // Remove the entry
                cache_.erase(it);
            }
        } else {
            // If LRU list is empty but cache isn't, something is wrong - clear everything
            cache_.clear();
            lru_list_.clear();
            lru_map_.clear();
            current_cache_size_bytes_ = 0;
            break;
        }
    }
}

void CacheManager::updateAccessTime(const CacheKey& key) {
    // Move the accessed key to the front of the LRU list
    if (lru_map_.find(key) != lru_map_.end()) {
        lru_list_.erase(lru_map_[key]);
    }
    
    lru_list_.push_front(key);
    lru_map_[key] = lru_list_.begin();
    
    // Update the access time in the cache entry
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second.last_access_time = std::chrono::high_resolution_clock::now();
    }
}

void CacheManager::invalidateSpecificKey(const CacheKey& key) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        // Remove from LRU tracking
        if (lru_map_.find(key) != lru_map_.end()) {
            lru_list_.erase(lru_map_[key]);
            lru_map_.erase(key);
        }
        
        // Reduce cache size
        current_cache_size_bytes_ -= it->second.size_bytes;
        
        // Remove the entry
        cache_.erase(it);
    }
}

// Cache hit/miss tracking
void CacheManager::recordHit() {
    hit_count_.fetch_add(1);
}

void CacheManager::recordMiss() {
    miss_count_.fetch_add(1);
}

// Method to check if a specific cache entry exists
bool CacheManager::hasCachedProfileData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp) {
    std::shared_lock lock(mutex_);
    
    CacheKey key{CacheType::PROFILE, symbol_id, timeframe, bar_timestamp};
    bool exists = cache_.find(key) != cache_.end();
    
    if (exists) {
        recordHit();
    } else {
        recordMiss();
    }
    
    return exists;
}

bool CacheManager::hasCachedIndicatorData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                                         const std::string& indicator_name) {
    std::shared_lock lock(mutex_);
    
    CacheKey key{CacheType::INDICATOR, symbol_id, timeframe, bar_timestamp, indicator_name};
    bool exists = cache_.find(key) != cache_.end();
    
    if (exists) {
        recordHit();
    } else {
        recordMiss();
    }
    
    return exists;
}

bool CacheManager::hasCachedAggregatedData(uint32_t symbol_id, TimeFrame timeframe, uint64_t bar_timestamp,
                                          const std::string& aggregation_type) {
    std::shared_lock lock(mutex_);
    
    CacheKey key{CacheType::AGGREGATION, symbol_id, timeframe, bar_timestamp, aggregation_type};
    bool exists = cache_.find(key) != cache_.end();
    
    if (exists) {
        recordHit();
    } else {
        recordMiss();
    }
    
    return exists;
}

// Method to clean up stale entries based on age
void CacheManager::cleanupStaleEntries(std::chrono::seconds max_age) {
    std::unique_lock lock(mutex_);
    
    auto now = std::chrono::high_resolution_clock::now();
    auto it = cache_.begin();
    
    while (it != cache_.end()) {
        if (now - it->second.creation_time > max_age) {
            // Entry is too old, remove it
            const CacheKey& key = it->first;
            
            // Remove from LRU tracking
            if (lru_map_.find(key) != lru_map_.end()) {
                lru_list_.erase(lru_map_[key]);
                lru_map_.erase(key);
            }
            
            // Reduce cache size
            current_cache_size_bytes_ -= it->second.size_bytes;
            
            // Remove the entry
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

// Serialization methods
bool CacheManager::serializeToStream(std::ostream& out) const {
    std::shared_lock lock(mutex_);

    // Write number of entries
    size_t entry_count = cache_.size();
    out.write(reinterpret_cast<const char*>(&entry_count), sizeof(entry_count));

    // Write each entry
    for (const auto& [key, entry] : cache_) {
        // Write cache key
        int type_int = static_cast<int>(key.type);
        out.write(reinterpret_cast<const char*>(&type_int), sizeof(type_int));
        out.write(reinterpret_cast<const char*>(&key.symbol_id), sizeof(key.symbol_id));
        int timeframe_int = static_cast<int>(key.timeframe);
        out.write(reinterpret_cast<const char*>(&timeframe_int), sizeof(timeframe_int));
        out.write(reinterpret_cast<const char*>(&key.bar_timestamp), sizeof(key.bar_timestamp));

        // Write extra_param string length and content
        size_t param_len = key.extra_param.length();
        out.write(reinterpret_cast<const char*>(&param_len), sizeof(param_len));
        if (param_len > 0) {
            out.write(key.extra_param.c_str(), param_len);
        }

        // Write entry data
        out.write(reinterpret_cast<const char*>(&entry.size_bytes), sizeof(entry.size_bytes));

        // Write creation and access times
        auto creation_time = entry.creation_time.time_since_epoch().count();
        auto last_access_time = entry.last_access_time.time_since_epoch().count();
        out.write(reinterpret_cast<const char*>(&creation_time), sizeof(creation_time));
        out.write(reinterpret_cast<const char*>(&last_access_time), sizeof(last_access_time));

        // Write the actual cached data based on type
        if (entry.data) {
            // For simplicity, we'll serialize based on cache type
            // In a real implementation, we'd need to properly serialize the actual data
            switch (key.type) {
                case CacheType::PROFILE: {
                    // Cast to vector of VolumeProfileLevel
                    auto profile_ptr = std::static_pointer_cast<std::vector<VolumeProfileLevel>>(entry.data);
                    size_t profile_size = profile_ptr->size();
                    out.write(reinterpret_cast<const char*>(&profile_size), sizeof(profile_size));

                    for (const auto& level : *profile_ptr) {
                        out.write(reinterpret_cast<const char*>(&level.price), sizeof(level.price));
                        out.write(reinterpret_cast<const char*>(&level.total_volume), sizeof(level.total_volume));
                        out.write(reinterpret_cast<const char*>(&level.buy_volume), sizeof(level.buy_volume));
                        out.write(reinterpret_cast<const char*>(&level.sell_volume), sizeof(level.sell_volume));
                    }
                    break;
                }
                case CacheType::INDICATOR: {
                    // Cast to vector of doubles
                    auto indicator_ptr = std::static_pointer_cast<std::vector<double>>(entry.data);
                    size_t indicator_size = indicator_ptr->size();
                    out.write(reinterpret_cast<const char*>(&indicator_size), sizeof(indicator_size));

                    for (double val : *indicator_ptr) {
                        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
                    }
                    break;
                }
                case CacheType::AGGREGATION: {
                    // Cast to vector of doubles
                    auto aggregation_ptr = std::static_pointer_cast<std::vector<double>>(entry.data);
                    size_t aggregation_size = aggregation_ptr->size();
                    out.write(reinterpret_cast<const char*>(&aggregation_size), sizeof(aggregation_size));

                    for (double val : *aggregation_ptr) {
                        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
                    }
                    break;
                }
            }
        }
    }

    return out.good();
}

bool CacheManager::deserializeFromStream(std::istream& in) {
    std::unique_lock lock(mutex_);

    // Clear existing cache
    cache_.clear();
    lru_list_.clear();
    lru_map_.clear();
    current_cache_size_bytes_ = 0;

    // Read number of entries
    size_t entry_count;
    in.read(reinterpret_cast<char*>(&entry_count), sizeof(entry_count));

    // Read each entry
    for (size_t i = 0; i < entry_count; ++i) {
        // Read cache key
        int type_int;
        uint32_t symbol_id;
        int timeframe_int;
        uint64_t bar_timestamp;
        size_t param_len;

        in.read(reinterpret_cast<char*>(&type_int), sizeof(type_int));
        in.read(reinterpret_cast<char*>(&symbol_id), sizeof(symbol_id));
        in.read(reinterpret_cast<char*>(&timeframe_int), sizeof(timeframe_int));
        in.read(reinterpret_cast<char*>(&bar_timestamp), sizeof(bar_timestamp));
        in.read(reinterpret_cast<char*>(&param_len), sizeof(param_len));

        std::string extra_param;
        if (param_len > 0) {
            std::vector<char> param_buf(param_len);
            in.read(param_buf.data(), param_len);
            extra_param = std::string(param_buf.data(), param_len);
        }

        CacheKey key{
            static_cast<CacheType>(type_int),
            symbol_id,
            static_cast<TimeFrame>(timeframe_int),
            bar_timestamp,
            extra_param
        };

        // Read entry data
        size_t size_bytes;
        int64_t creation_time_count, last_access_time_count;

        in.read(reinterpret_cast<char*>(&size_bytes), sizeof(size_bytes));
        in.read(reinterpret_cast<char*>(&creation_time_count), sizeof(creation_time_count));
        in.read(reinterpret_cast<char*>(&last_access_time_count), sizeof(last_access_time_count));

        auto creation_time = std::chrono::high_resolution_clock::time_point(
            std::chrono::nanoseconds(creation_time_count));
        auto last_access_time = std::chrono::high_resolution_clock::time_point(
            std::chrono::nanoseconds(last_access_time_count));

        // Read the actual cached data based on type
        std::shared_ptr<void> data_ptr;

        switch (key.type) {
            case CacheType::PROFILE: {
                size_t profile_size;
                in.read(reinterpret_cast<char*>(&profile_size), sizeof(profile_size));

                auto profile_vec = std::make_shared<std::vector<VolumeProfileLevel>>();
                profile_vec->reserve(profile_size);

                for (size_t j = 0; j < profile_size; ++j) {
                    VolumeProfileLevel level;
                    in.read(reinterpret_cast<char*>(&level.price), sizeof(level.price));
                    in.read(reinterpret_cast<char*>(&level.total_volume), sizeof(level.total_volume));
                    in.read(reinterpret_cast<char*>(&level.buy_volume), sizeof(level.buy_volume));
                    in.read(reinterpret_cast<char*>(&level.sell_volume), sizeof(level.sell_volume));

                    profile_vec->push_back(level);
                }

                data_ptr = std::static_pointer_cast<void>(profile_vec);
                break;
            }
            case CacheType::INDICATOR: {
                size_t indicator_size;
                in.read(reinterpret_cast<char*>(&indicator_size), sizeof(indicator_size));

                auto indicator_vec = std::make_shared<std::vector<double>>();
                indicator_vec->reserve(indicator_size);

                for (size_t j = 0; j < indicator_size; ++j) {
                    double val;
                    in.read(reinterpret_cast<char*>(&val), sizeof(val));
                    indicator_vec->push_back(val);
                }

                data_ptr = std::static_pointer_cast<void>(indicator_vec);
                break;
            }
            case CacheType::AGGREGATION: {
                size_t aggregation_size;
                in.read(reinterpret_cast<char*>(&aggregation_size), sizeof(aggregation_size));

                auto aggregation_vec = std::make_shared<std::vector<double>>();
                aggregation_vec->reserve(aggregation_size);

                for (size_t j = 0; j < aggregation_size; ++j) {
                    double val;
                    in.read(reinterpret_cast<char*>(&val), sizeof(val));
                    aggregation_vec->push_back(val);
                }

                data_ptr = std::static_pointer_cast<void>(aggregation_vec);
                break;
            }
        }

        // Create and store the cache entry
        CacheEntry entry;
        entry.data = data_ptr;
        entry.size_bytes = size_bytes;
        entry.creation_time = creation_time;
        entry.last_access_time = last_access_time;

        cache_[key] = entry;
        current_cache_size_bytes_ += size_bytes;

        // Update LRU tracking
        lru_list_.push_front(key);
        lru_map_[key] = lru_list_.begin();
    }

    return in.good();
}

} // namespace RenderEngine
} // namespace BTQuant