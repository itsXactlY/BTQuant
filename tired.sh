#!/bin/bash

echo "================================================="
echo "🚀 INITIATING THE ARCHITECT'S SCORCHED EARTH PURGE"
echo "================================================="

# 1. DELETE ROGUE HEADERS
echo "[1/4] Amputating rogue headers..."
rm -f include/components/tpoprofilepanel.h
rm -f include/components/domsurfacepanel.h
rm -f include/components/optionanalyticspanel.hpp
rm -f include/components/pricestatisticpanel.h
rm -f include/components/moc_*
rm -f include/components/realtime_dashboard_component.hpp
rm -f include/components/chart_super_node.hpp
rm -f include/components/MarketMicrostructureRenderer.h
rm -rf include/layout/
rm -f include/data/exchange_aggregator.hpp
rm -f include/data/unified_data_pipeline.hpp
rm -f include/data/orderbookhistory.h

# 2. DELETE ROGUE SOURCE FILES
echo "[2/4] Amputating rogue source files..."
rm -rf src/layout/
rm -f src/components/realtime_dashboard_component.cpp
rm -f src/components/chart_super_node.cpp
rm -f src/components/MarketMicrostructureRenderer.cpp
rm -f src/components/domsurfacepanel.cpp
rm -f src/components/tpoprofilepanel.cpp
rm -f src/components/optionanalyticspanel.cpp
rm -f src/components/pricestatisticpanel.cpp
rm -f src/data/orderbookhistory.cpp
rm -f src/data/exchange_aggregator.*
rm -f src/data/unified_data_pipeline.cpp
rm -f src/optimization/performance_optimizer.cpp

# Find and delete test files polluting the src directory
find src -type f -name "test_*.cpp" -delete
find src -type f -name "main_test.cpp" -delete

# 3. PURGE PHANTOM INCLUDES FROM ALL SURVIVING FILES
echo "[3/4] Cauterizing phantom #includes from surviving code..."
BAD_INCLUDES=(
    "components/realtime_dashboard_component.hpp"
    "components/MarketMicrostructureRenderer.h"
    "components/chart_super_node.hpp"
    "components/tpoprofilepanel.h"
    "components/domsurfacepanel.h"
    "components/optionanalyticspanel.hpp"
    "components/pricestatisticpanel.h"
    "layout/dashboard_layout_manager.hpp"
    "layout/layout_presets.hpp"
    "data/exchange_aggregator.hpp"
    "data/unified_data_pipeline.hpp"
)

for inc in "${BAD_INCLUDES[@]}"; do
    # Search all .cpp and .hpp files and delete the line containing the bad include
    find src include -type f \( -name "*.cpp" -o -name "*.hpp" \) -exec sed -i "/$(echo $inc | sed 's/\//\\\//g')/d" {} +
done

# 4. SANITIZE CMAKE
echo "[4/4] Sanitizing CMakeLists.txt..."
if [ -f "CMakeLists.txt" ]; then
    sed -i '/realtime_dashboard_component/d' CMakeLists.txt
    sed -i '/MarketMicrostructureRenderer/d' CMakeLists.txt
    sed -i '/chart_super_node/d' CMakeLists.txt
    sed -i '/tpoprofilepanel\.cpp/d' CMakeLists.txt
    sed -i '/domsurfacepanel\.cpp/d' CMakeLists.txt
    sed -i '/optionanalyticspanel\.cpp/d' CMakeLists.txt
    sed -i '/pricestatisticpanel\.cpp/d' CMakeLists.txt
    sed -i '/dashboard_layout_manager/d' CMakeLists.txt
    sed -i '/layout_presets\.cpp/d' CMakeLists.txt
    sed -i '/exchange_aggregator/d' CMakeLists.txt
    sed -i '/unified_data_pipeline/d' CMakeLists.txt
    sed -i '/performance_optimizer\.cpp/d' CMakeLists.txt
    sed -i '/orderbookhistory\.cpp/d' CMakeLists.txt
    sed -i '/test_/d' CMakeLists.txt
fi

echo "================================================="
echo "✅ PURGE COMPLETE. HYDRA IS DEAD."
echo "Next Step: rm -rf build && ./build_integration.sh"
echo "================================================="