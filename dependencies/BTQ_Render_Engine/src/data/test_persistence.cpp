#include "persistence.hpp"
#include "cache_manager.hpp"
#include "market_data_processor.hpp"

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

int main() {
    std::cout << "Testing Data Persistence..." << std::endl;

    // Create a cache manager
    auto cache_manager = std::make_shared<BTQuant::RenderEngine::CacheManager>();
    
    // Add some test data to the cache
    std::vector<BTQuant::RenderEngine::VolumeProfileLevel> test_profile = {
        {100.0, 1000.0, 600.0, 400.0},
        {101.0, 1500.0, 900.0, 600.0},
        {102.0, 800.0, 300.0, 500.0}
    };
    
    cache_manager->cacheProfileData(1, BTQuant::RenderEngine::TimeFrame::TF_1MIN, 1234567890, test_profile);
    
    std::vector<double> test_indicator = {1.0, 2.0, 3.0, 4.0, 5.0};
    cache_manager->cacheIndicatorData(1, BTQuant::RenderEngine::TimeFrame::TF_1MIN, 1234567890, "RSI", test_indicator);
    
    std::vector<double> test_aggregation = {10.5, 20.3, 15.7, 18.9};
    cache_manager->cacheAggregatedData(1, BTQuant::RenderEngine::TimeFrame::TF_1MIN, 1234567890, "VOLUME_SUM", test_aggregation);

    // Create a data persistence instance
    BTQuant::Data::DataPersistence persistence("./test_data");
    
    // Set the cache manager reference for periodic saves
    persistence.setCacheManager(cache_manager);
    
    // Save the aggregated data
    std::cout << "Saving aggregated data..." << std::endl;
    bool save_success = persistence.saveAggregatedData(cache_manager, "test_aggregated.dat");
    std::cout << "Save result: " << (save_success ? "SUCCESS" : "FAILED") << std::endl;

    // Create a new cache manager to test loading
    auto new_cache_manager = std::make_shared<BTQuant::RenderEngine::CacheManager>();
    
    // Load the aggregated data
    std::cout << "Loading aggregated data..." << std::endl;
    bool load_success = persistence.loadAggregatedData(new_cache_manager, "test_aggregated.dat");
    std::cout << "Load result: " << (load_success ? "SUCCESS" : "FAILED") << std::endl;

    // Verify that the loaded data matches the original
    auto loaded_profile = new_cache_manager->getCachedProfileData(1, BTQuant::RenderEngine::TimeFrame::TF_1MIN, 1234567890);
    if (loaded_profile.has_value() && loaded_profile->size() == test_profile.size()) {
        std::cout << "Profile data loaded successfully with " << loaded_profile->size() << " levels" << std::endl;
        for (size_t i = 0; i < loaded_profile->size(); ++i) {
            const auto& orig = test_profile[i];
            const auto& loaded = (*loaded_profile)[i];
            if (orig.price == loaded.price && orig.total_volume == loaded.total_volume) {
                std::cout << "  Level " << i << ": OK" << std::endl;
            } else {
                std::cout << "  Level " << i << ": MISMATCH!" << std::endl;
            }
        }
    } else {
        std::cout << "Failed to load profile data or size mismatch" << std::endl;
    }

    // Test market data processor persistence
    auto processor = std::make_shared<BTQuant::RenderEngine::MarketDataProcessor>();
    
    // Save market analytics
    std::cout << "Saving market analytics..." << std::endl;
    bool save_analytics = persistence.saveMarketAnalytics(processor, "test_analytics.dat");
    std::cout << "Market analytics save result: " << (save_analytics ? "SUCCESS" : "FAILED") << std::endl;

    // Load market analytics
    std::cout << "Loading market analytics..." << std::endl;
    bool load_analytics = persistence.loadMarketAnalytics(processor, "test_analytics.dat");
    std::cout << "Market analytics load result: " << (load_analytics ? "SUCCESS" : "FAILED") << std::endl;

    // Test periodic save functionality
    std::cout << "Starting periodic save (5 seconds)..." << std::endl;
    persistence.startPeriodicSave(5);  // Save every 5 seconds
    
    // Wait for 6 seconds to trigger at least one periodic save
    std::this_thread::sleep_for(std::chrono::seconds(6));
    
    // Stop periodic save
    persistence.stopPeriodicSave();
    std::cout << "Stopped periodic save" << std::endl;

    // List saved files
    auto files = persistence.getSavedFiles();
    std::cout << "Saved files: " << files.size() << std::endl;
    for (const auto& file : files) {
        std::cout << "  - " << file << std::endl;
    }

    std::cout << "Data Persistence test completed." << std::endl;
    return 0;
}