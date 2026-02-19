#!/usr/bin/env python3
"""
Comprehensive fix for context_menus.cpp - removes trading panel references
"""
import re

filepath = 'dependencies/BTQ_Render_Engine/src/ui/context_menus.cpp'

with open(filepath, 'r') as f:
    lines = f.readlines()

# Find trading panel includes and remove them
trading_includes = [
    'risk_metrics_panel.hpp',
    'screener_panel.hpp',
    'trading_orders_panel.hpp',
    'trading_positions_panel.hpp',
    'risk_analyzer_panel.hpp',
    'strategy_builder_panel.hpp'
]

new_lines = []
skip_next = False
for i, line in enumerate(lines):
    # Skip trading panel includes
    if any(t in line for t in trading_includes):
        continue
    
    new_lines.append(line)

content = ''.join(new_lines)

# Remove trading handler registrations - match full line with handler
trading_handlers = [
    'TRADING_ORDERS',
    'TRADING_POSITIONS',
    'RISK_METRICS',
    'SCREENER',
    'RISK_ANALYZER',
    'STRATEGY_BUILDER'
]

for handler in trading_handlers:
    # Remove the entire line that contains this handler reference
    pattern = rf'.*\b{handler}\b.*\n'
    content = re.sub(pattern, '', content)

# Remove orphaned case blocks - these are between valid case statements
# Pattern: starts with whitespace, then "ImGui::Text("Something Actions:");" 
# and continues until "break;"
orphaned = [
    r'\n        ImGui::Text\("Trading Orders Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Trading Positions Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Risk Metrics Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Screener Actions:"\);.*?break;\n',
    r'\n        ImGui::Text\("Strategy Builder Actions:"\);.*?break;\n',
]

for pattern in orphaned:
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

with open(filepath, 'w') as f:
    f.write(content)

print("Done!")
