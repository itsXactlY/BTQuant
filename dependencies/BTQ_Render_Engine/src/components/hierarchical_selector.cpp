#include "components/hierarchical_selector.hpp"
namespace BTQuant {
const char* HierarchicalSelector::get_timeframe_name(RenderEngine::TimeFrame tf) {
  switch (tf) {
    case RenderEngine::TimeFrame::TF_1MIN:
      return "1m";
    case RenderEngine::TimeFrame::TF_5MIN:
      return "5m";
    case RenderEngine::TimeFrame::TF_15MIN:
      return "15m";
    case RenderEngine::TimeFrame::TF_1HOUR:
      return "1h";
    case RenderEngine::TimeFrame::TF_4HOUR:
      return "4h";
    case RenderEngine::TimeFrame::TF_1DAY:
      return "1D";
    case RenderEngine::TimeFrame::TF_1WEEK:
      return "1W";
    default:
      return "1m";
  }
}
RenderEngine::TimeFrame HierarchicalSelector::get_timeframe_from_index(int index) {
  switch (index) {
    case 0:
      return RenderEngine::TimeFrame::TF_1MIN;
    case 1:
      return RenderEngine::TimeFrame::TF_5MIN;
    case 2:
      return RenderEngine::TimeFrame::TF_15MIN;
    case 3:
      return RenderEngine::TimeFrame::TF_1HOUR;
    case 4:
      return RenderEngine::TimeFrame::TF_4HOUR;
    case 5:
      return RenderEngine::TimeFrame::TF_1DAY;
    case 6:
      return RenderEngine::TimeFrame::TF_1WEEK;
    default:
      return RenderEngine::TimeFrame::TF_1MIN;
  }
}
bool HierarchicalSelector::render(HierarchicalSelectorState&) { return false; }
void HierarchicalSelector::refresh_data(HierarchicalSelectorState&, ChartManager*) {}

}  // namespace BTQuant
