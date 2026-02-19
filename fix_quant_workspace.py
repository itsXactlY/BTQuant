#!/usr/bin/env python3
import re

with open('dependencies/BTQ_Render_Engine/src/components/quant_workspace_component.cpp', 'r') as f:
    content = f.read()

# Pattern to match buttons that create trading-related panels
# We need to remove the entire if block that creates these panels

patterns_to_remove = [
    # Match button for SCREENER
    r'\n      if \(ImGui::Button\("Screener"\)\).*?add_panel\(PanelType::SCREENER\);\n      \}',
    
    # Match button for TRADING_ORDERS  
    r'\n      if \(ImGui::Button\("Trading Orders"\)\).*?add_panel\(PanelType::TRADING_ORDERS\);\n      \}',
    
    # Match button for TRADING_POSITIONS
    r'\n      if \(ImGui::Button\("Positions"\)\).*?add_panel\(PanelType::TRADING_POSITIONS\);\n      \}',
    
    # Match button for STRATEGY_BUILDER
    r'\n      if \(ImGui::Button\("Strategy Builder"\)\).*?add_panel\(PanelType::STRATEGY_BUILDER\);\n      \}',
    
    # Match button for RISK_METRICS
    r'\n      if \(ImGui::Button\("Risk Metrics"\)\).*?add_panel\(PanelType::RISK_METRICS\);\n      \}',
    
    # Match button for RISK_ANALYZER
    r'\n      if \(ImGui::Button\("Risk Analyzer"\)\).*?add_panel\(PanelType::RISK_ANALYZER\);\n      \}',
    
    # Match button for OPTION_ANALYTICS
    r'\n      if \(ImGui::Button\("Option Analytics"\)\).*?add_panel\(PanelType::OPTION_ANALYTICS\);\n      \}',
]

for pattern in patterns_to_remove:
    content = re.sub(pattern, '\n', content, flags=re.DOTALL)

with open('dependencies/BTQ_Render_Engine/src/components/quant_workspace_component.cpp', 'w') as f:
    f.write(content)

print("Done!")
