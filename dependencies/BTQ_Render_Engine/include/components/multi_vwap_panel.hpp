#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>

#include "../market_data_processor.hpp"
#include "panel_base.hpp"
#include "../indicators/anchored_vwap.hpp"

// Forward declaration for TaskScheduler to avoid circular includes
namespace btq {
    class TaskScheduler;
}

namespace BTQuant {

struct VWAPInstance {
    std::string id;
    std::string name;
    ::btq::AnchoredVWAP vwap;
    ImVec4 color;
    bool visible;
    std::string type; // "daily", "weekly", "monthly", "custom"
    uint64_t creation_time;
    
    VWAPInstance(const std::string& _id, const std::string& _name, uint64_t anchor_time, 
                 const ImVec4& _color, const std::string& _type)
        : id(_id), name(_name), vwap(anchor_time), color(_color), visible(true), 
          type(_type), creation_time(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()) {}
};

class MultiVWAPPanel : public PanelBase {
public:
    MultiVWAPPanel(const PanelConfig& config, 
                   std::shared_ptr<RenderEngine::MarketDataProcessor> processor);

    void update(float dt) override;
    void render() override;

private:
    std::shared_ptr<RenderEngine::MarketDataProcessor> processor_;

    // Task scheduler for background calculations
    std::shared_ptr<btq::TaskScheduler> task_scheduler_;

    // Store all VWAP instances
    std::vector<VWAPInstance> vwap_instances_;

    // UI state
    std::string new_vwap_name_ = "Custom VWAP";
    ImVec4 new_vwap_color_ = ImVec4(1.0f, 1.0f, 0.0f, 1.0f); // Yellow
    uint64_t new_vwap_anchor_time_ = 0;

    // Control flags
    bool show_daily_vwap_ = true;
    bool show_weekly_vwap_ = false;
    bool show_monthly_vwap_ = false;

    float update_timer_ = 0.0f;
    const float UPDATE_INTERVAL = 1.0f;  // Update every second
    
    // Private methods
    void update_vwaps();
    void render_vwap_list();
    void render_vwap_controls();
    void render_add_vwap_section();
    
    void add_daily_vwap();
    void add_weekly_vwap();
    void add_monthly_vwap();
    void add_custom_vwap();
    void remove_vwap(const std::string& id);
    
    std::string generate_unique_id();
    uint64_t get_start_of_day(uint64_t timestamp);
    uint64_t get_start_of_week(uint64_t timestamp);
    uint64_t get_start_of_month(uint64_t timestamp);
};

}  // namespace BTQuant