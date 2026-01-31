#include "detectors/spread_arbitrage_detector.hpp"
#include "detectors/detection_signal.hpp"
#include "detectors/detector_plugin.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>

namespace BTQuant {

// Global HotSpineExtendedReader pointer for factory creation
static HotSpineExtendedReader* g_global_reader = nullptr;

void set_global_reader(HotSpineExtendedReader* reader) {
    g_global_reader = reader;
}

// SpreadSignal implementation
std::string SpreadSignal::to_string() const {
    std::ostringstream oss;
    oss << "SpreadArbitrage(symbol=" << symbol
        << ", buy_exchange=" << buy_exchange
        << ", sell_exchange=" << sell_exchange
        << ", buy_price=" << buy_price
        << ", sell_price=" << sell_price
        << ", spread=" << spread_bps << "bps"
        << ", profit=" << potential_profit_bps << "bps"
        << ", executable=" << (is_executable ? "true" : "false")
        << ")";
    return oss.str();
}

// SpreadArbitrageDetector implementation
SpreadArbitrageDetector::SpreadArbitrageDetector(HotSpineExtendedReader& reader)
    : reader_(&reader) {
    // Constructor implementation
}

std::vector<BTQuant::DetectionSignal> SpreadArbitrageDetector::detect_all(const std::string& symbol) {
    std::vector<BTQuant::DetectionSignal> signals;
    
    if (!reader_) {
        return signals;
    }
    
    // Get orderbook data from all exchanges for this symbol
    auto orderbooks = reader_->get_all_exchange_orderbooks(symbol);
    
    if (orderbooks.size() < 2) {
        // Need at least 2 exchanges for arbitrage
        return signals;
    }
    
    double best_bid = 0.0;
    double best_ask = std::numeric_limits<double>::max();
    std::string best_bid_exchange;
    std::string best_ask_exchange;
    
    // Find best bid and ask across all exchanges
    for (const auto& [exchange, ob] : orderbooks) {
        if (ob.bids.empty() || ob.asks.empty()) {
            continue;
        }
        
        // Best bid is the highest bid price
        if (ob.bids[0].first > best_bid) {
            best_bid = ob.bids[0].first;
            best_bid_exchange = exchange;
        }
        
        // Best ask is the lowest ask price
        if (ob.asks[0].first < best_ask) {
            best_ask = ob.asks[0].first;
            best_ask_exchange = exchange;
        }
    }
    
    // Check if we have valid prices from different exchanges
    if (best_bid > 0 && best_ask < std::numeric_limits<double>::max()
        && best_bid_exchange != best_ask_exchange && best_bid > best_ask) {
        
        // Calculate spread in basis points
        double spread_bps = (best_bid - best_ask) / best_ask * 10000.0;
        
        // Account for fees
        double net_profit_bps = spread_bps - maker_fee_bps_ - taker_fee_bps_;
        
        // Check if profitable
        if (net_profit_bps >= min_profit_bps_) {
            BTQuant::DetectionSignal detection;
            detection.id = "SARB-" + symbol + "-" + std::to_string(detections_);
            detection.type = DetectionType::SPREAD_ARBITRAGE;
            detection.severity = net_profit_bps > 100.0 ? Severity::HIGH : Severity::MEDIUM;
            detection.symbol = symbol;
            detection.exchange = best_ask_exchange + "-" + best_bid_exchange;
            detection.description = "Spread arbitrage opportunity: " +
                                   std::to_string(net_profit_bps) + "bps profit";
            detection.confidence = std::min(1.0, net_profit_bps / 200.0);
            detection.timestamp = reader_->get_current_time_us();
            
            signals.push_back(detection);
            detections_++;
        }
    }
    
    return signals;
}

// IDetector interface implementation
std::string SpreadArbitrageDetector::get_name() const {
    return PLUGIN_NAME;
}

std::string SpreadArbitrageDetector::get_version() const {
    return PLUGIN_VERSION;
}

std::string SpreadArbitrageDetector::get_description() const {
    return PLUGIN_DESCRIPTION;
}

std::optional<BTQuant::DetectionSignal> SpreadArbitrageDetector::detect(const std::string& symbol) {
    auto signals = detect_all(symbol);
    if (!signals.empty()) {
        // The detect_all method already returns DetectionSignal, so we can just return the first one
        return signals[0];
    }
    return std::nullopt;
}

void SpreadArbitrageDetector::update_orderbook(const std::string& exchange, const std::string& symbol) {
    // No special handling needed, the reader already tracks orderbooks
}

void SpreadArbitrageDetector::update_trade(const TradeData& trade) {
    // No special handling needed for trades in this detector
}

void SpreadArbitrageDetector::configure(const std::unordered_map<std::string, BTQuant::Config::ConfigValue>& config) {
    // Simple implementation - just use the values directly
    // In a real implementation, we would parse the ConfigValue variants
    // For now, we'll keep the default values
}

std::unordered_map<std::string, BTQuant::Config::ConfigValue>
SpreadArbitrageDetector::get_configuration() const {
    std::unordered_map<std::string, BTQuant::Config::ConfigValue> config;
    config["min_profit_bps"] = BTQuant::Config::ConfigValue(min_profit_bps_);
    config["maker_fee_bps"] = BTQuant::Config::ConfigValue(maker_fee_bps_);
    config["taker_fee_bps"] = BTQuant::Config::ConfigValue(taker_fee_bps_);
    return config;
}

bool SpreadArbitrageDetector::initialize() {
    // Validate configuration
    if (min_profit_bps_ <= 0) {
        return false;
    }
    if (maker_fee_bps_ < 0 || taker_fee_bps_ < 0) {
        return false;
    }
    return true;
}

void SpreadArbitrageDetector::shutdown() {
    // Cleanup if needed
}

std::vector<std::string> SpreadArbitrageDetector::get_dependencies() const {
    return std::vector<std::string>(PLUGIN_DEPENDENCIES,
                                    PLUGIN_DEPENDENCIES + sizeof(PLUGIN_DEPENDENCIES) / sizeof(PLUGIN_DEPENDENCIES[0]));
}

void SpreadArbitrageDetector::set_dependency(const std::string& name, std::shared_ptr<void> dependency) {
    // Store dependencies if needed
}

} // namespace BTQuant

// Plugin export macro with global reader support
#define BTQUANT_DETECTOR_PLUGIN_WITH_READER(DetectorClass) \
    extern "C" { \
        std::unique_ptr<BTQuant::Detectors::IDetector> create_detector() { \
            auto detector = std::make_unique<DetectorClass>(); \
            detector->set_dependency("HotSpineExtendedReader", BTQuant::Detectors::DetectorRegistry::instance().get_detector("HotSpineExtendedReader")); \
            return detector; \
        } \
        BTQuant::Detectors::PluginInfo get_plugin_info() { \
            return { \
                DetectorClass::PLUGIN_NAME, \
                DetectorClass::PLUGIN_VERSION, \
                DetectorClass::PLUGIN_DESCRIPTION, \
                "", \
                std::vector<std::string>(std::begin(DetectorClass::PLUGIN_DEPENDENCIES), \
                                         std::end(DetectorClass::PLUGIN_DEPENDENCIES)), \
                create_detector \
            }; \
        } \
    }

// Use the standard macro for now - detector will be configured with reader later
BTQUANT_DETECTOR_PLUGIN(BTQuant::SpreadArbitrageDetector)
