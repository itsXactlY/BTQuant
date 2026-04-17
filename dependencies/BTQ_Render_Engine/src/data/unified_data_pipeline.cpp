#include "data/unified_data_pipeline.hpp"
#include "data/ui_data_manager.hpp"
#include "hotspine_data_bridge.hpp"

#include <iostream>
#include <algorithm>
#include <memory>
#include <chrono>

namespace BTQuant {
namespace Data {

UnifiedDataPipeline::UnifiedDataPipeline(
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
    std::shared_ptr<RenderEngine::SymbolManager> symbol_manager)
    : processor_(std::move(processor)),
      symbol_manager_(std::move(symbol_manager)) {
    ui_data_manager_ = std::make_shared<UIDataManager>();
}

UnifiedDataPipeline::~UnifiedDataPipeline() {
    shutdown();
}

bool UnifiedDataPipeline::initialize() {
    if (running_.load()) return true;

    running_.store(true);
    current_symbol_ = "BTC-USDT";
    current_symbol_id_ = 1;

    std::cout << "[UnifiedDataPipeline] Initializing with symbol: "
              << current_symbol_ << std::endl;

    // Create and start HotSpine data bridge for live market data
    hotspine_bridge_ = std::make_unique<HotSpineDataBridge>("/btquant");
    hotspine_bridge_->setMarketDataProcessor(processor_);

    if (auto result = hotspine_bridge_->start(); !result) {
        std::cerr << "[UnifiedDataPipeline] HotSpine bridge failed: "
                  << result.error() << std::endl;
        std::cerr << "[UnifiedDataPipeline] Running without live data" << std::endl;
    } else {
        std::cout << "[UnifiedDataPipeline] HotSpine bridge connected to shared memory" << std::endl;
    }

    std::cout << "[UnifiedDataPipeline] Initialized with symbol: "
              << current_symbol_ << std::endl;
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

void UnifiedDataPipeline::publish(DataType type, uint32_t symbol_id,
                                   const std::string& symbol_name,
                                   const std::string& exchange,
                                   const void* data, size_t data_size) {
    DataEvent event;
    event.type = type;
    event.symbol_id = symbol_id;
    event.symbol_name = symbol_name;
    event.exchange = exchange;
    event.data = const_cast<void*>(data);
    event.data_size = data_size;
    event.timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());

    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    for (auto& [id, sub] : subscriptions_) {
        bool match = false;
        for (const auto& dt : sub.data_types) {
            if (dt == "ALL" || dt == symbol_name) { match = true; break; }
        }
        if (match && sub.callback) {
            sub.callback(&event);
        }
    }
}

void UnifiedDataPipeline::process_events() {
    // HotSpine bridge runs its own sync thread
}

std::vector<std::string> UnifiedDataPipeline::get_available_symbols() const {
    if (!hotspine_bridge_) return {};
    auto active = hotspine_bridge_->getActiveSymbols();
    std::vector<std::string> result;
    for (auto id : active) {
        result.push_back(hotspine_bridge_->getSymbolName(id));
    }
    return result;
}

void UnifiedDataPipeline::set_current_symbol(const std::string& symbol) {
    current_symbol_ = symbol;
}

std::string UnifiedDataPipeline::get_current_symbol() const {
    return current_symbol_;
}

uint32_t UnifiedDataPipeline::get_current_symbol_id() const {
    return current_symbol_id_;
}

void UnifiedDataPipeline::shutdown() {
    if (!running_.load()) return;
    running_.store(false);

    if (hotspine_bridge_) {
        hotspine_bridge_->stop();
        hotspine_bridge_.reset();
    }

    std::cout << "[UnifiedDataPipeline] Shutdown complete" << std::endl;
}

void UnifiedDataPipeline::processing_loop() {}
void UnifiedDataPipeline::dispatch_event(const DataEvent&) {}

}  // namespace Data
}  // namespace BTQuant
