#!/usr/bin/env python3
import re

with open('dependencies/BTQ_Render_Engine/src/ui/context_menus.cpp', 'r') as f:
    content = f.read()

# Remove trading panel handler registrations in initialize_context_menus()
patterns_to_remove = [
    r'\n\s*context_menu_handlers_\[PanelType::TRADING_ORDERS\].*?;\n',
    r'\n\s*context_menu_handlers_\[PanelType::TRADING_POSITIONS\].*?;\n',
    r'\n\s*context_menu_handlers_\[PanelType::RISK_METRICS\].*?;\n',
    r'\n\s*context_menu_handlers_\[PanelType::SCREENER\].*?;\n',
    r'\n\s*context_menu_handlers_\[PanelType::RISK_ANALYZER\].*?;\n',
    r'\n\s*context_menu_handlers_\[PanelType::STRATEGY_BUILDER\].*?;\n',
]

for pattern in patterns_to_remove:
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

# Remove orphaned case blocks in render_generic_context_menu function
# These are code blocks that appear between case statements without a case label
orphaned_blocks = [
    r'\n        ImGui::Text\("Trading Orders Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Trading Positions Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Risk Metrics Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Screener Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Strategy Builder Actions:"\);.*?break;\n',
]

for pattern in orphaned_blocks:
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

with open('dependencies/BTQ_Render_Engine/src/ui/context_menus.cpp', 'w') as f:
    f.write(content)

print("Done!")
