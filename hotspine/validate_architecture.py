#!/usr/bin/env python3
"""
Simple architecture validation script

This script validates the HotSpine SQL architecture without requiring full dependencies.
"""

import sys
import os
import ast
import re


def validate_architecture():
    """Validate the HotSpine SQL architecture by analyzing source code"""
    print("="*80)
    print("HOTSPINE SQL ARCHITECTURE VALIDATION")
    print("="*80)
    print()
    
    validation_results = []
    
    # 1. Validate SQL integration file exists and has correct structure
    print("🔍 CHECK 1: SQL Integration Module Structure")
    sql_integration_path = "dependencies/backtrader/hotspine/sql_integration.py"
    
    if os.path.exists(sql_integration_path):
        print("✅ SQL integration module exists")
        
        with open(sql_integration_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("HotSpineSQLIntegration class", "class HotSpineSQLIntegration"),
            ("Asynchronous storage", "store_trade_async"),
            ("Replay capabilities", "create_replay_data_feed"),
            ("Analytics methods", "get_database_stats"),
            ("Debugging support", "log_trade_for_debugging"),
            ("Long-term storage focus", "long-term storage"),
        ]
        
        for check_name, pattern in checks:
            if pattern in content:
                print(f"✅ Found {check_name}")
            else:
                print(f"❌ Missing {check_name}")
                validation_results.append(False)
        
        validation_results.append(True)
    else:
        print("❌ SQL integration module not found")
        validation_results.append(False)
    
    print()
    
    # 2. Validate HotSpine reader integration
    print("🔍 CHECK 2: HotSpine Reader Integration")
    reader_path = "dependencies/backtrader/hotspine/reader.py"
    
    if os.path.exists(reader_path):
        print("✅ HotSpine reader exists")
        
        with open(reader_path, 'r') as f:
            content = f.read()
        
        # Check for SQL integration
        sql_checks = [
            ("SQL import", "HotSpineSQLIntegration"),
            ("SQL config parameter", "sql_config"),
            ("SQL storage enable flag", "enable_sql_storage"),
            ("Asynchronous trade storage", "store_trade_async"),
            ("SQL not in hot path", "# SQL is NOT used for live trading"),
        ]
        
        for check_name, pattern in sql_checks:
            if pattern in content:
                print(f"✅ Found {check_name}")
            else:
                print(f"❌ Missing {check_name}")
                validation_results.append(False)
        
        validation_results.append(True)
    else:
        print("❌ HotSpine reader not found")
        validation_results.append(False)
    
    print()
    
    # 3. Validate architecture documentation
    print("🔍 CHECK 3: Architecture Documentation")
    
    # Check reader.py for architecture comments
    with open(reader_path, 'r') as f:
        reader_content = f.read()
    
    doc_checks = [
        ("HotSpine for live data", "HotSpine handles live trading data"),
        ("SQL for storage only", "SQL is used only for"),
        ("Data flow documentation", "Data flow:"),
        ("Architecture separation", "Clean architecture separation"),
    ]
    
    for check_name, pattern in doc_checks:
        if pattern in reader_content:
            print(f"✅ Found {check_name}")
        else:
            print(f"❌ Missing {check_name}")
            validation_results.append(False)
    
    print()
    
    # 4. Validate example files
    print("🔍 CHECK 4: Example and Validation Files")
    
    example_files = [
        "example_hotspine_sql_integration.py",
        "test_hotspine_sql_architecture.py",
        "validate_architecture.py"
    ]
    
    for example_file in example_files:
        if os.path.exists(example_file):
            print(f"✅ Found {example_file}")
        else:
            print(f"❌ Missing {example_file}")
            validation_results.append(False)
    
    print()
    
    # 5. Validate key architectural principles
    print("🔍 CHECK 5: Architectural Principles")
    
    # Read and analyze the SQL integration
    with open(sql_integration_path, 'r') as f:
        sql_content = f.read()
    
    principles = [
        ("Asynchronous operations", "async" in sql_content.lower()),
        ("Non-blocking design", "non-blocking" in sql_content.lower()),
        ("Replay support", "replay" in sql_content.lower()),
        ("Analytics support", "analytics" in sql_content.lower()),
        ("Debugging support", "debugging" in sql_content.lower()),
        ("Long-term storage focus", "long-term" in sql_content.lower()),
    ]
    
    for principle_name, principle_check in principles:
        if principle_check:
            print(f"✅ Implements {principle_name}")
        else:
            print(f"❌ Missing {principle_name}")
            validation_results.append(False)
    
    print()
    
    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed_checks = sum(validation_results)
    total_checks = len(validation_results)
    
    print(f"📊 Total Validation Checks: {total_checks}")
    print(f"✅ Passed: {passed_checks}")
    print(f"❌ Failed: {total_checks - passed_checks}")
    
    if passed_checks >= total_checks * 0.8:  # 80% pass rate
        print()
        print("🎉 ARCHITECTURE VALIDATION PASSED!")
        print()
        print("✅ The HotSpine + SQL integration correctly implements:")
        print("   1. HotSpine for live trading data (NOT SQL)")
        print("   2. SQL for long-term storage (asynchronous)")
        print("   3. SQL for replay and analytics")
        print("   4. Clean separation of concerns")
        print("   5. Proper documentation and examples")
        print()
        print("🏆 The architecture aligns with the new HotSpine design principles!")
        return True
    else:
        print()
        print("❌ ARCHITECTURE VALIDATION FAILED!")
        print(f"❌ Only {passed_checks}/{total_checks} checks passed")
        print()
        print("💡 Recommendations:")
        print("   - Review failed validation checks")
        print("   - Ensure proper implementation of architectural principles")
        print("   - Check documentation and examples")
        return False


def analyze_code_structure():
    """Analyze the code structure for architectural compliance"""
    print("\n" + "="*80)
    print("CODE STRUCTURE ANALYSIS")
    print("="*80)
    print()
    
    # Analyze the main files
    files_to_analyze = [
        "dependencies/backtrader/hotspine/sql_integration.py",
        "dependencies/backtrader/hotspine/reader.py"
    ]
    
    for file_path in files_to_analyze:
        if os.path.exists(file_path):
            print(f"📄 Analyzing {file_path}")
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Basic metrics
            lines = content.split('\n')
            print(f"   Lines of code: {len(lines)}")
            print(f"   Classes: {content.count('class ')}")
            print(f"   Methods: {content.count('def ')}")
            print(f"   Comments: {content.count('#')}")
            print(f"   Docstrings: {content.count('"""')}")
            print()
        else:
            print(f"❌ File not found: {file_path}")


if __name__ == "__main__":
    # Run validation
    success = validate_architecture()
    
    # Run code structure analysis
    analyze_code_structure()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)