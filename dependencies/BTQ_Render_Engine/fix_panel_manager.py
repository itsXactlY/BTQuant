#!/usr/bin/env python3
import re

with open('src/components/panel_manager.cpp', 'r') as f:
    content = f.read()

# Remove the include statements
includes_to_remove = [
    '#include "../../include/components/trading_orders_panel.hpp"',
    '#include "../../include/components/trading_positions_panel.hpp"',
    '#include "../../include/components/screener_panel.hpp"',
    '#include "../../include/components/risk_metrics_panel.hpp"',
    '#include "../../include/components/risk_analyzer_panel.hpp"',
    '#include "../../include/components/strategy_builder.hpp"',
    '#include "../../include/components/option_analytics_panel.hpp"',
]

for inc in includes_to_remove:
    content = content.replace(inc + '\n', '')
    content = content.replace(inc, '')

# Remove case blocks - patterns to remove
case_patterns = [
    r'    case PanelType::OPTION_ANALYTICS:\n      panel = std::make_unique<BTQuant::RenderEngine::OptionAnalyticsPanel>\(strategy_builder_\.get\(\)\);\n      break;\n',
    r'    case PanelType::TRADING_ORDERS:\n      panel = std::make_unique<TradingOrdersPanel>\(config, order_manager_, position_manager_\);\n      break;\n',
    r'    case PanelType::TRADING_POSITIONS:\n      panel = std::make_unique<TradingPositionsPanel>\(config, position_manager_, risk_assessment_\);\n      break;\n',
    r'    case PanelType::RISK_METRICS:\n      panel = std::make_unique<RiskMetricsPanel>\(config, risk_assessment_, position_manager_\);\n      break;\n',
    r'    case PanelType::SCREENER:\n      panel = std::make_unique<ScreenerPanel>\(config\);\n      break;\n',
    r'    case PanelType::RISK_ANALYZER:\n      panel = std::make_unique<RiskAnalyzerPanel>\(config, bridge_, processor_\);\n      break;\n',
    r'    case PanelType::STRATEGY_BUILDER:\n      panel = std::make_unique<BTQuant::RenderEngine::StrategyBuilder>\(config\);\n      break;\n',
]

for pattern in case_patterns:
    content = re.sub(pattern, '', content)

# Remove case blocks in get_default_panel_title - patterns to remove
title_patterns = [
    r'    case PanelType::TRADING_ORDERS:\n      return "Orders";\n',
    r'    case PanelType::TRADING_POSITIONS:\n      return "Positions";\n',
    r'    case PanelType::RISK_METRICS:\n      return "Risk";\n',
    r'    case PanelType::SCREENER:\n      return "Screener";\n',
    r'    case PanelType::RISK_ANALYZER:\n      return "Risk Analyzer";\n',
    r'    case PanelType::STRATEGY_BUILDER:\n      return "Strategy Builder";\n',
    r'    case PanelType::OPTION_ANALYTICS:\n      return "Option Analytics";\n',
]

for pattern in title_patterns:
    content = re.sub(pattern, '', content)

# Remove strategy_builder_ initialization - find and remove the line
content = re.sub(r'\n  strategy_builder_ = std::make_unique<RenderEngine::StrategyBuilder>\(PanelConfig\{.*?\}\);', '', content)
content = re.sub(r'\n  strategy_builder_\.reset\(\); // Explicitly reset strategy builder before other members', '', content)

# Remove case blocks in set_active_symbol - OPTION_ANALYTICS and SCREENER
set_symbol_patterns = [
    r'      case PanelType::OPTION_ANALYTICS: \{\n        // OptionAnalyticsPanel doesn\'t typically require symbol-specific data\n        break;\n      \}\n',
    r'      case PanelType::SCREENER: \{\n        // Screener panels don\'t typically require symbol-specific data\n        break;\n      \}\n',
]

for pattern in set_symbol_patterns:
    content = re.sub(pattern, '', content)

# Remove OptionAnalyticsPanel dynamic_cast references in save_layout and load_layout
option_patterns = [
    r'    // Option Analytics Panel specific settings\n    else if \(auto\* option_panel = dynamic_cast<BTQuant::RenderEngine::OptionAnalyticsPanel\*>\(panel\.get\(\)\)\) \{\n        settings_json\["active_tab"\] = option_panel->get_active_tab\(\);\n    \}\n',
    r'                // Option Analytics Panel specific settings\n                else if \(auto\* option_panel = dynamic_cast<BTQuant::RenderEngine::OptionAnalyticsPanel\*>\(panel\)\) \{\n                    if \(settings\.contains\("active_tab"\)\) \{\n                        option_panel->set_active_tab\(settings\["active_tab"\]\.get<int>\(\)\);\n                    \}\n                \}\n',
]

for pattern in option_patterns:
    content = re.sub(pattern, '', content)

# Remove trading panels from layout presets
layout_patterns = [
    r'      add_panel\(PanelType::TRADING_ORDERS, "Orders", 2, 2, 1, 1\);\n',
    r'      add_panel\(PanelType::TRADING_POSITIONS, "Positions", 0, 3, 1, 1\);\n',
    r'      add_panel\(PanelType::RISK_METRICS, "Risk", 1, 3, 1, 1\);\n',
    r'      add_panel\(PanelType::RISK_METRICS, "Risk Metrics", 0, 0, 1, 2\);\n',
    r'      add_panel\(PanelType::TRADING_POSITIONS, "Positions", 1, 0, 1, 2\);\n',
    r'      add_panel\(PanelType::RISK_ANALYZER, "Risk Analyzer", 0, 2, 3, 1\);\n',
]

for pattern in layout_patterns:
    content = re.sub(pattern, '', content)

# Fix constructor - remove OrderManager, PositionManager, RiskAssessment parameters
constructor_old = '''PanelManager::PanelManager(
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor,
                           std::shared_ptr<OrderManager> order_manager,
                           std::shared_ptr<PositionManager> position_manager,
                           std::shared_ptr<RiskAssessment> risk_assessment)
    : bridge_(bridge),
      processor_(processor),
      order_manager_(order_manager),
      position_manager_(position_manager),
      risk_assessment_(risk_assessment) {'''

constructor_new = '''PanelManager::PanelManager(
                           std::shared_ptr<RenderEngine::MarketDataProcessor> processor)
    : bridge_(bridge),
      processor_(processor) {'''

content = content.replace(constructor_old, constructor_new)

# Fix context_menus.cpp - remove trading panel handler registrations
context_menu_patterns = [
    r'  context_menu_handlers_\[PanelType::TRADING_ORDERS\] = \[this\]\(PanelBase\* panel\) \{\n    render_generic_context_menu\(panel, "TradingOrdersContextMenu"\);\n  \}\n\n',
    r'  context_menu_handlers_\[PanelType::TRADING_POSITIONS\] = \[this\]\(PanelBase\* panel\) \{\n    render_generic_context_menu\(panel, "TradingPositionsContextMenu"\);\n  \}\n\n',
    r'  context_menu_handlers_\[PanelType::RISK_METRICS\] = \[this\]\(PanelBase\* panel\) \{\n    render_generic_context_menu\(panel, "RiskMetricsContextMenu"\);\n  \}\n\n',
    r'  context_menu_handlers_\[PanelType::SCREENER\] = \[this\]\(PanelBase\* panel\) \{\n    render_generic_context_menu\(panel, "ScreenerContextMenu"\);\n  \}\n\n',
    r'  context_menu_handlers_\[PanelType::RISK_ANALYZER\] = \[this\]\(PanelBase\* panel\) \{\n    render_generic_context_menu\(panel, "RiskAnalyzerContextMenu"\);\n  \}\n\n',
    r'  context_menu_handlers_\[PanelType::STRATEGY_BUILDER\] = \[this\]\(PanelBase\* panel\) \{\n    render_generic_context_menu\(panel, "StrategyBuilderContextMenu"\);\n  \}\n\n',
]

for pattern in context_menu_patterns:
    content = re.sub(pattern, '', content)

with open('src/components/panel_manager.cpp', 'w') as f:
    f.write(content)

print("Done!")
