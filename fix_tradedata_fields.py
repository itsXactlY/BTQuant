#!/usr/bin/env python3
"""
Script to fix TradeData field name migrations across all component files.
"""

import re
import os
import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Fix trade.size -> trade.volume
    content = re.sub(r'(\btrade\.)(size)(\b)', r'\1volume\3', content)
    content = re.sub(r'(\bcurrent_trade\.)(size)(\b)', r'\1volume\3', content)
    content = re.sub(r'(\blatest_trade\.)(size)(\b)', r'\1volume\3', content)
    content = re.sub(r'(\bnext_trade\.)(size)(\b)', r'\1volume\3', content)
    content = re.sub(r'(\bprev_trade\.)(size)(\b)', r'\1volume\3', content)
    
    # Fix a.size, b.size -> a.volume, b.volume (in lambdas)
    content = re.sub(r'(\ba\.)(size)(\b)', r'\1volume\3', content)
    content = re.sub(r'(\bb\.)(size)(\b)', r'\1volume\3', content)
    content = re.sub(r'(\bentry\.)(size)(\b)', r'\1volume\3', content)
    
    # Fix trade.timestamp -> trade.timestamp_us (but not timestamp_us)
    content = re.sub(r'(\btrade\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    content = re.sub(r'(\bcurrent_trade\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    content = re.sub(r'(\blatest_trade\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    content = re.sub(r'(\bnext_trade\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    content = re.sub(r'(\bprev_trade\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    
    # Fix a.timestamp, b.timestamp -> a.timestamp_us, b.timestamp_us (in lambdas)
    content = re.sub(r'(\ba\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    content = re.sub(r'(\bb\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    content = re.sub(r'(\bentry\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    
    # Fix trades.back().timestamp -> trades.back().timestamp_us
    content = re.sub(r'(\.back\(\)\.)(timestamp)(?!_us)(\b)', r'\1timestamp_us\3', content)
    
    # Fix trade.is_buy -> trade.is_buy() (method call)
    # But not when it's already a method call
    content = re.sub(r'(\btrade\.)(is_buy)(?!\s*\()(\b)', r'\1is_buy()\3', content)
    content = re.sub(r'(\bcurrent_trade\.)(is_buy)(?!\s*\()(\b)', r'\1is_buy()\3', content)
    content = re.sub(r'(\blatest_trade\.)(is_buy)(?!\s*\()(\b)', r'\1is_buy()\3', content)
    content = re.sub(r'(\bnext_trade\.)(is_buy)(?!\s*\()(\b)', r'\1is_buy()\3', content)
    content = re.sub(r'(\bprev_trade\.)(is_buy)(?!\s*\()(\b)', r'\1is_buy()\3', content)
    
    # Fix bridge_ references - comment them out or replace
    # Replace bridge_ ? bridge_->getExchangeName(...) : "" with just ""
    content = re.sub(r'bridge_\s*\?\s*bridge_->getExchangeName\([^)]+\)\s*:\s*""', '""', content)
    content = re.sub(r'bridge_\s*\?\s*bridge_->getExchangeName\([^)]+\)\s*:\s*"Unknown"', '"Unknown"', content)
    
    # Fix trade.symbol references - TradeData doesn't have symbol field
    content = re.sub(r'trade\.symbol\.empty\(\)\s*\?\s*symbol_name_\s*:\s*trade\.symbol', 'symbol_name_', content)
    
    # Fix RenderEngine::TradeData -> TradeData (it's in BTQuant namespace)
    content = re.sub(r'RenderEngine::TradeData', 'TradeData', content)
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

def main():
    # Find all cpp files in components directory
    component_files = glob.glob('dependencies/BTQ_Render_Engine/src/components/*.cpp')
    
    fixed_count = 0
    for filepath in component_files:
        if fix_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count} files")

if __name__ == '__main__':
    main()
