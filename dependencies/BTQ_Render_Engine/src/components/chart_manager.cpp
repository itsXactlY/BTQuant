#include "components/chart_manager.hpp"
namespace BTQuant {
ChartManager::ChartManager(std::shared_ptr<MarketDataProcessor> processor)
    : processor_(processor), next_chart_id_(0) {}
ChartManager::~ChartManager() = default;
uint32_t ChartManager::create_chart(const std::string&, const std::string&, uint32_t,
                                    RenderEngine::TimeFrame) {
  return next_chart_id_++;
}
void ChartManager::update() {}
void ChartManager::populate_chart_data(uint32_t) {}
void ChartManager::update_all_chart_timeframes(RenderEngine::TimeFrame) {}
std::vector<BTQuant::ChartInstance> ChartManager::get_charts_for_symbol(const std::string&) const {
  return {};
}
}  // namespace BTQuant
