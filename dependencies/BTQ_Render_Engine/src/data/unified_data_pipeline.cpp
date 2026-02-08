#include "../include/data/unified_data_pipeline.hpp"
#include "../include/data/ui_data_manager.hpp"

#include <chrono>
#include <iostream>

namespace BTQuant {
namespace Data {

// ============================================================================
// UnifiedDataPipeline Implementation
// ============================================================================

UnifiedDataPipeline::UnifiedDataPipeline(
    std::shared_ptr<HotSpineDataBridge> bridge,
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<RenderEngine::SymbolManager> symbol_manager)
    : bridge_(bridge), processor_(processor), symbol_manager_(symbol_manager) {
  // Initialize UI data manager
  ui_data_manager_ = std::make_shared<UIDataManager>();

  // Get initial symbol from UI data manager
  current_symbol_ = ui_data_manager_->get_current_symbol();
  if (current_symbol_.empty()) {
    current_symbol_ = "BTCUSDT";  // Default symbol
    ui_data_manager_->set_current_symbol(current_symbol_);
  }

  // Find the symbol ID for the current symbol using symbol manager
  if (symbol_manager_) {
    auto all_symbols = symbol_manager_->getAllSymbols();
    for (const auto& symbol_info : all_symbols) {
      if (symbol_info.symbol == current_symbol_) {
        current_symbol_id_ = symbol_info.id;
        break;
      }
    }
  }
}

UnifiedDataPipeline::~UnifiedDataPipeline() { shutdown(); }

bool UnifiedDataPipeline::initialize() {
  running_ = true;
  processing_thread_ = std::thread(&UnifiedDataPipeline::processing_loop, this);
  return true;
}

uint32_t UnifiedDataPipeline::subscribe(const DataSubscription& subscription) {
  std::lock_guard<std::mutex> lock(subscriptions_mutex_);
  uint32_t id = next_subscription_id_++;
  subscriptions_[id] = subscription;
  return id;
}

void UnifiedDataPipeline::unsubscribe(uint32_t subscription_id) {
  std::lock_guard<std::mutex> lock(subscriptions_mutex_);
  subscriptions_.erase(subscription_id);
}

void UnifiedDataPipeline::publish(DataType type, uint32_t symbol_id, const std::string& symbol_name,
                                  const std::string& exchange, const void* data, size_t data_size) {
  DataEvent event;
  event.type = type;
  event.symbol_id = symbol_id;
  event.symbol_name = symbol_name;
  event.exchange = exchange;
  event.data = const_cast<void*>(data);
  event.data_size = data_size;
  event.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count());

  event_queue_.enqueue(event);

  // Notify processing thread
  cv_.notify_one();
}

std::string UnifiedDataPipeline::get_current_symbol() const { return current_symbol_; }

uint32_t UnifiedDataPipeline::get_current_symbol_id() const { return current_symbol_id_; }

void UnifiedDataPipeline::set_current_symbol(const std::string& symbol_name) {
  current_symbol_ = symbol_name;
  ui_data_manager_->set_current_symbol(symbol_name);

  // Find the symbol ID for the new symbol using symbol manager
  if (symbol_manager_) {
    auto all_symbols = symbol_manager_->getAllSymbols();
    for (const auto& symbol_info : all_symbols) {
      if (symbol_info.symbol == symbol_name) {
        current_symbol_id_ = symbol_info.id;
        break;
      }
    }
  }

  // Notify all interested parties about the symbol change
  DataEvent event;
  event.type = DataType::METRICS;  // Using METRICS as a generic notification type
  event.symbol_id = current_symbol_id_;
  event.symbol_name = symbol_name;
  event.exchange = symbol_manager_ ? symbol_manager_->getSymbolInfo(current_symbol_id_).has_value() ?
                   symbol_manager_->getSymbolInfo(current_symbol_id_)->exchange : "" : "";
  event.data = nullptr;
  event.data_size = 0;
  event.timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                              std::chrono::system_clock::now().time_since_epoch())
                                              .count());

  event_queue_.enqueue(event);

  cv_.notify_one();
}

std::vector<std::string> UnifiedDataPipeline::get_available_symbols() const {
  std::vector<std::string> symbols;

  if (symbol_manager_) {
    auto all_symbols = symbol_manager_->getAllSymbols();
    for (const auto& symbol_info : all_symbols) {
      symbols.push_back(symbol_info.symbol);
    }
  }

  return symbols;
}

void UnifiedDataPipeline::process_events() {
  std::vector<DataEvent> local_queue;
  
  // Dequeue all available events
  DataEvent event;
  while (event_queue_.try_dequeue(event)) {
    local_queue.push_back(std::move(event));
  }

  for (const auto& event : local_queue) {
    dispatch_event(event);
  }
}

void UnifiedDataPipeline::dispatch_event(const DataEvent& event) {
  std::lock_guard<std::mutex> lock(subscriptions_mutex_);

  for (const auto& [id, subscription] : subscriptions_) {
    // Check if subscription matches the event
    if (subscription.symbol_id == 0 || subscription.symbol_id == event.symbol_id) {
      // Check if data type matches - compare with string representations of enum values
      bool type_match = false;
      for (const auto& type_str : subscription.data_types) {
        // Convert enum to string and compare
        std::string event_type_str = std::to_string(static_cast<int>(event.type));
        if (type_str == event_type_str) {
          type_match = true;
          break;
        }
      }

      if (type_match) {
        // Call the subscription callback
        if (subscription.callback) {
          subscription.callback(event.data);
        }
      }
    }
  }
}

void UnifiedDataPipeline::processing_loop() {
  while (running_) {
    // Check if there are events to process
    if (event_queue_.size_approx() > 0) {
      // Process all available events
      std::vector<DataEvent> local_queue;
      
      // Dequeue all available events
      DataEvent event;
      while (event_queue_.try_dequeue(event)) {
        local_queue.push_back(std::move(event));
      }

      for (const auto& event : local_queue) {
        dispatch_event(event);
      }
    } else {
      // No events available, wait for notification
      std::unique_lock<std::mutex> lock(process_mutex_);
      cv_.wait_for(lock, std::chrono::milliseconds(10), [this] { 
        return event_queue_.size_approx() > 0 || !running_; 
      });
    }
  }
}

void UnifiedDataPipeline::shutdown() {
  if (running_) {
    running_ = false;
    cv_.notify_all();

    if (processing_thread_.joinable()) {
      processing_thread_.join();
    }
  }
}

}  // namespace Data
}  // namespace BTQuant