#pragma once

#include "hotspine_extended_reader.hpp"
#include "detectors/detector_plugin.hpp"
#include "detectors/detection_signal.hpp"
#include <vector>
#include <optional>
#include <unordered_map>

namespace BTQuant {

struct SpreadSignal {
    std::string symbol;
    std::string buy_exchange;
    std::string sell_exchange;
    double buy_price;
    double sell_price;
    double spread_bps;
    double potential_profit_bps;
    uint64_t timestamp_us;
    bool is_executable;
    
    std::string to_string() const;
};

class SpreadArbitrageDetector : public Detectors::IDetector {
public:
    SpreadArbitrageDetector() = default;
    explicit SpreadArbitrageDetector(HotSpineExtendedReader& reader);
    
    // Plugin metadata
    static constexpr const char PLUGIN_NAME[] = "SpreadArbitrageDetector";
    static constexpr const char PLUGIN_VERSION[] = "1.0.0";
    static constexpr const char PLUGIN_DESCRIPTION[] = "Detects cross-exchange spread arbitrage opportunities";
    static constexpr const char* PLUGIN_DEPENDENCIES[] = {"HotSpineExtendedReader", nullptr};
    
    // Detect arbitrage opportunities for a symbol
    std::vector<BTQuant::DetectionSignal> detect_all(const std::string& symbol) override;
    
    // Original method for getting SpreadSignal results
    std::vector<SpreadSignal> detect_spread_signals(const std::string& symbol);
    
    // Configuration
    void set_min_profit_bps(double profit) { min_profit_bps_ = profit; }
    void set_fees(double maker_fee_bps, double taker_fee_bps) {
        maker_fee_bps_ = maker_fee_bps;
        taker_fee_bps_ = taker_fee_bps;
    }
    
    // Statistics
    uint64_t get_detections() const { return detections_; }
    void reset_statistics() { detections_ = 0; }
    
    // IDetector interface implementation
    std::string get_name() const override;
    std::string get_version() const override;
    std::string get_description() const override;
    std::optional<DetectionSignal> detect(const std::string& symbol) override;
    void update_orderbook(const std::string& exchange, const std::string& symbol) override;
    void update_trade(const TradeData& trade) override;
    void configure(const std::unordered_map<std::string, Config::ConfigValue>& config) override;
    std::unordered_map<std::string, Config::ConfigValue> get_configuration() const override;
    bool initialize() override;
    void shutdown() override;
    std::vector<std::string> get_dependencies() const override;
    void set_dependency(const std::string& name, std::shared_ptr<void> dependency) override;
    
private:
    HotSpineExtendedReader* reader_ = nullptr;
    double maker_fee_bps_ = 10.0;   // 0.1% maker fee
    double taker_fee_bps_ = 20.0;   // 0.2% taker fee
    double min_profit_bps_ = 50.0;  // 0.5% minimum profit
    uint64_t detections_ = 0;
};

} // namespace BTQuant