#!/usr/bin/env python3
import re

filepath = 'dependencies/BTQ_Render_Engine/src/ui/context_menus.cpp'

with open(filepath, 'r') as f:
    content = f.read()

# 1. Remove trading panel includes
includes_to_remove = [
    '#include "components/risk_metrics_panel.hpp"',
    '#include "components/screener_panel.hpp"',
    '#include "components/trading_orders_panel.hpp"',
    '#include "components/trading_positions_panel.hpp"',
    '#include "components/risk_analyzer_panel.hpp"',
    '#include "components/strategy_builder_panel.hpp"',
]

for inc in includes_to_remove:
    content = content.replace(inc + '\n', '')
    content = content.replace(inc, '')

# 2. Remove trading panel handler registrations in initialize_context_menus()
# Pattern: lines containing the handler registration followed by };
trading_handlers = [
    'PanelType::TRADING_ORDERS',
    'PanelType::TRADING_POSITIONS', 
    'PanelType::RISK_METRICS',
    'PanelType::SCREENER',
    'PanelType::RISK_ANALYZER',
    'PanelType::STRATEGY_BUILDER',
]

# Find and remove handler registrations
for handler in trading_handlers:
    # Pattern to match handler registration (multi-line)
    pattern = r'\n\s*context_menu_handlers_\[' + handler + r'\].*?;\n'
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

# 3. Remove orphaned case blocks in render_generic_context_menu
# These are code blocks between valid case statements that have no case label
orphaned_blocks = [
    r'\n        ImGui::Text\("Trading Orders Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Trading Positions Actions:"\);.*?break;\n', 
    r'\n        ImGui::Text\("Risk Metrics Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Screener Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Strategy Builder Actions:"\);.*?break;\n',
]

for pattern in orphaned_blocks:
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

# 4. Remove duplicate }; that might be left over
# Pattern: multiple }; in sequence
content = re.sub(r'\n(\s*};)\1+', r'\n\1', content)

with open(filepath, 'w') as f:
    f.write(content)

print("Done!")
