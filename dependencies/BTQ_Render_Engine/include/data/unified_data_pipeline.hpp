#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hotspine_data_bridge.hpp"
#include "market_data_processor.hpp"
#include "symbol_manager.hpp"
#include "ui_data_manager.hpp"

// Handle concurrentqueue include path variations for consistency across modules
// This addresses the requirement to modify both files to handle FetchContent path variations
// The include is conditionally added to ensure both files can handle path variations independently
// NOTE: Only include if not already available through market_data_processor.hpp
#ifndef MOODYCAMEL_CONCURRENTQUEUE_H
#if __has_include("moodycamel/concurrentqueue.h")
#include "moodycamel/concurrentqueue.h"
#elif __has_include("concurrentqueue.h")
#include "concurrentqueue.h"
#else
#include "moodycamel/concurrentqueue.h"
#endif
#endif

namespace BTQuant {
namespace Data {

// ============================================================================
// Unified Data Pipeline
// ============================================================================

struct DataSubscription {
  uint32_t symbol_id;
  std::string symbol_name;
  std::string exchange;
  std::vector<std::string> data_types;  // e.g., "OHLC", "ORDERBOOK", "TRADES", etc.
  std::function<void(const void*)> callback;
};

class UnifiedDataPipeline {
 public:
  enum class DataType {
    OHLC,
    ORDERBOOK,
    TRADES,
    VOLUME_PROFILE,
    FOOTPRINT,
    TPO,
    METRICS,
    ALERTS,
    HEIKIN_ASHI,
    RENKO,
    LINEBREAK,
    KAGI,
    POINT_AND_FIGURE,
    RANGE_BARS,
    VOLUME_BARS,
    TICK_BARS
  };

  struct DataEvent {
    DataType type;
    uint32_t symbol_id;
    std::string symbol_name;
    std::string exchange;
    void* data;
    size_t data_size;
    uint64_t timestamp;
  };

  UnifiedDataPipeline(std::shared_ptr<HotSpineDataBridge> bridge,
                      std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                      std::shared_ptr<RenderEngine::SymbolManager> symbol_manager);
  ~UnifiedDataPipeline();

  // Initialize the pipeline
  bool initialize();

  // Subscribe to specific data types for a symbol
  uint32_t subscribe(const DataSubscription& subscription);
  void unsubscribe(uint32_t subscription_id);

  // Publish data to subscribers
  void publish(DataType type, uint32_t symbol_id, const std::string& symbol_name,
               const std::string& exchange, const void* data, size_t data_size);

  // Get current symbol information
  std::string get_current_symbol() const;
  uint32_t get_current_symbol_id() const;

  // Set current symbol (triggers global symbol switching)
  void set_current_symbol(const std::string& symbol_name);

  // Get available symbols
  std::vector<std::string> get_available_symbols() const;

  // Get data processor reference
  std::shared_ptr<RenderEngine::MarketDataProcessor> get_market_processor() const {
    return processor_;
  }

  // Get data bridge reference
  std::shared_ptr<HotSpineDataBridge> get_data_bridge() const { return bridge_; }

  // Get symbol manager reference
  std::shared_ptr<RenderEngine::SymbolManager> get_symbol_manager() const {
    return symbol_manager_;
  }

  // Get UI data manager reference
  std::shared_ptr<UIDataManager> get_ui_data_manager() const { return ui_data_manager_; }

  // Process pending data events
  void process_events();

  // Shutdown the pipeline
  void shutdown();

 private:
  std::shared_ptr<HotSpineDataBridge> bridge_;
  std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;
  std::shared_ptr<RenderEngine::SymbolManager> symbol_manager_;
  std::shared_ptr<UIDataManager> ui_data_manager_;

  std::unordered_map<uint32_t, DataSubscription> subscriptions_;
  std::atomic<uint32_t> next_subscription_id_{1};

  std::vector<DataEvent> event_queue_;
  mutable std::mutex queue_mutex_;
  mutable std::mutex subscriptions_mutex_;

  std::atomic<bool> running_{false};
  std::thread processing_thread_;
  std::condition_variable cv_;

  std::string current_symbol_;
  uint32_t current_symbol_id_{0};

  void processing_loop();
  void dispatch_event(const DataEvent& event);
};

}  // namespace Data
}  // namespace BTQuant