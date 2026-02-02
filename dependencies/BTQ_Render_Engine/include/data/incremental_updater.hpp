#pragma once

#include "market_data_processor.hpp"

namespace BTQuant {
namespace RenderEngine {

/**
 * @brief Process a single trade incrementally
 * Only updates affected analytics without recalculating everything
 */
void processTradeIncrementally(SymbolAnalytics& symbol_data, const TradeData& trade);

}  // namespace RenderEngine
}  // namespace BTQuant