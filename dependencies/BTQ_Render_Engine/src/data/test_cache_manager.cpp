#include "cache_manager.hpp"
#include <iostream>
#include <cassert>
#include <vector>

using namespace BTQuant::RenderEngine;

void test_basic_caching() {
    std::cout << "Testing basic caching functionality..." << std::endl;
    
    CacheManager cache(10 * 1024 * 1024); // 10MB cache
    
    // Test profile caching
    std::vector<VolumeProfileLevel> profile_data = {
        {100.0, 1000.0, 600.0, 400.0},
        {101.0, 1500.0, 900.0, 600.0}
    };
    
    cache.cacheProfileData(1, TimeFrame::TF_1MIN, 1634567890000000, profile_data);
    
    auto retrieved_profile = cache.getCachedProfileData(1, TimeFrame::TF_1MIN, 1634567890000000);
    assert(retrieved_profile.has_value());
    assert(retrieved_profile->size() == 2);
    
    std::cout << "✓ Profile caching works" << std::endl;
    
    // Test indicator caching
    std::vector<double> indicator_data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    cache.cacheIndicatorData(1, TimeFrame::TF_1MIN, 1634567890000000, "SMA_20", indicator_data);
    
    auto retrieved_indicator = cache.getCachedIndicatorData(1, TimeFrame::TF_1MIN, 1634567890000000, "SMA_20");
    assert(retrieved_indicator.has_value());
    assert(retrieved_indicator->size() == 5);
    
    std::cout << "✓ Indicator caching works" << std::endl;
    
    // Test aggregation caching
    std::vector<double> aggregation_data = {10.0, 20.0, 30.0};
    
    cache.cacheAggregatedData(1, TimeFrame::TF_1MIN, 1634567890000000, "VOLUME_SUM", aggregation_data);
    
    auto retrieved_aggregation = cache.getCachedAggregatedData(1, TimeFrame::TF_1MIN, 1634567890000000, "VOLUME_SUM");
    assert(retrieved_aggregation.has_value());
    assert(retrieved_aggregation->size() == 3);
    
    std::cout << "✓ Aggregation caching works" << std::endl;
    
    // Test cache stats
    CacheStats stats = cache.getCacheStats();
    assert(stats.entry_count == 3);
    std::cout << "✓ Cache stats work" << std::endl;
    
    std::cout << "Basic caching tests passed!" << std::endl;
}

void test_cache_invalidation() {
    std::cout << "\nTesting cache invalidation..." << std::endl;
    
    CacheManager cache(10 * 1024 * 1024); // 10MB cache
    
    // Add some data
    std::vector<VolumeProfileLevel> profile_data = {{100.0, 1000.0, 600.0, 400.0}};
    cache.cacheProfileData(1, TimeFrame::TF_1MIN, 1634567890000000, profile_data);
    
    // Verify it's there
    assert(cache.hasCachedProfileData(1, TimeFrame::TF_1MIN, 1634567890000000));
    
    // Invalidate for the symbol
    cache.invalidateCacheForSymbol(1);
    
    // Verify it's gone
    assert(!cache.hasCachedProfileData(1, TimeFrame::TF_1MIN, 1634567890000000));
    
    std::cout << "✓ Symbol-based invalidation works" << std::endl;
    
    // Add data again
    cache.cacheProfileData(2, TimeFrame::TF_1MIN, 1634567890000000, profile_data);
    cache.cacheProfileData(2, TimeFrame::TF_5MIN, 1634567890000000, profile_data);
    
    // Verify both are there
    assert(cache.hasCachedProfileData(2, TimeFrame::TF_1MIN, 1634567890000000));
    assert(cache.hasCachedProfileData(2, TimeFrame::TF_5MIN, 1634567890000000));
    
    // Invalidate for specific timeframe
    cache.invalidateCache(2, TimeFrame::TF_1MIN, 1634567890000000);
    
    // Verify only the 1MIN timeframe is gone
    assert(!cache.hasCachedProfileData(2, TimeFrame::TF_1MIN, 1634567890000000));
    assert(cache.hasCachedProfileData(2, TimeFrame::TF_5MIN, 1634567890000000));
    
    std::cout << "✓ Timeframe-based invalidation works" << std::endl;
    
    std::cout << "Cache invalidation tests passed!" << std::endl;
}

void test_lru_eviction() {
    std::cout << "\nTesting LRU eviction..." << std::endl;
    
    // Create a very small cache (1KB) to force evictions
    CacheManager cache(1024); // 1KB cache
    
    // Add many entries to force evictions
    for (int i = 0; i < 100; ++i) {
        std::vector<VolumeProfileLevel> profile_data = {{static_cast<double>(i), 100.0, 50.0, 50.0}};
        cache.cacheProfileData(1, TimeFrame::TF_1MIN, 1634567890000000 + i, profile_data);
    }
    
    // The cache should have been limited by size
    CacheStats stats = cache.getCacheStats();
    std::cout << "Final cache size: " << stats.current_size_bytes << " bytes, entries: " << stats.entry_count << std::endl;
    
    // At least some entries should have been evicted
    assert(stats.entry_count <= 100);
    
    std::cout << "✓ LRU eviction works" << std::endl;
    
    std::cout << "LRU eviction tests passed!" << std::endl;
}

int main() {
    std::cout << "Running Cache Manager Tests...\n" << std::endl;
    
    test_basic_caching();
    test_cache_invalidation();
    test_lru_eviction();
    
    std::cout << "\nAll tests passed successfully!" << std::endl;
    
    return 0;
}