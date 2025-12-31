"""
Comprehensive test for the documentation system
Tests all features and integration points
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def test_all_documentation_features():
    """Comprehensive test of all documentation system features"""
    print("🧪 Running Comprehensive Documentation System Test...")
    
    from documentation.documentation_system import DocumentationSystem
    
    # Initialize documentation system
    doc_system = DocumentationSystem()
    
    # Test data
    test_strategy = {
        'id': 'comprehensive_test_strategy',
        'template': 'multi_strategy',
        'parameters': {
            'aggressiveness': 1.5,
            'risk_tolerance': 0.7,
            'time_horizon': 'medium_term'
        }
    }
    
    test_backtest_results = {
        'strategy_id': test_strategy['id'],
        'performance_metrics': {
            'sharpe_ratio': 1.95,
            'max_drawdown': 0.10,
            'win_rate': 0.68,
            'total_return': 0.28
        },
        'risk_profile': {
            'volatility': 0.07,
            'value_at_risk': 0.04
        }
    }
    
    test_selection_results = {
        'strategy': test_strategy,
        'composite_fitness': 0.89,
        'selection_metadata': {'generation': 2, 'rank': 1}
    }
    
    # Test 1: Comprehensive Report Generation
    print("\n1. Testing Comprehensive Report Generation...")
    report = doc_system.generate_comprehensive_report(
        test_strategy, test_backtest_results, test_selection_results
    )
    assert Path(report).exists()
    print("✓ Comprehensive report generated successfully")
    
    # Test 2: Performance Report Generation
    print("\n2. Testing Performance Report Generation...")
    perf_report = doc_system.generate_performance_report(test_strategy, test_backtest_results)
    assert Path(perf_report).exists()
    print("✓ Performance report generated successfully")
    
    # Test 3: Evolutionary Lineage Tracking
    print("\n3. Testing Evolutionary Lineage Tracking...")
    parents = [{
        'id': 'parent_001',
        'template': 'trend_following',
        'parameters': {'trend_strength': 1.0}
    }]
    children = [test_strategy]
    
    lineage_result = doc_system.track_evolutionary_lineage(
        {'generation_number': 1, 'metrics': {'diversity': 0.85}}, parents, children
    )
    assert lineage_result['child_count'] == 1
    print("✓ Evolutionary lineage tracked successfully")
    
    # Test 4: Evolutionary Report Generation
    print("\n4. Testing Evolutionary Report Generation...")
    evo_report = doc_system.generate_evolutionary_report({
        'generations': 1, 'strategies': 2
    })
    assert Path(evo_report).exists()
    print("✓ Evolutionary report generated successfully")
    
    # Test 5: Strategy Visualization
    print("\n5. Testing Strategy Visualization...")
    viz_types = ['performance', 'risk_return', 'drawdown']
    for viz_type in viz_types:
        viz_file = doc_system.create_strategy_visualization(
            test_strategy, test_backtest_results, viz_type
        )
        assert Path(viz_file).exists()
        print(f"✓ {viz_type} visualization created successfully")
    
    # Test 6: Living Archive System
    print("\n6. Testing Living Archive System...")
    archive_file = doc_system.archive_strategy(
        test_strategy, test_backtest_results, test_selection_results
    )
    assert Path(archive_file).exists()
    print("✓ Strategy archived successfully")
    
    # Test 7: System Integration Report
    print("\n7. Testing System Integration Report...")
    system_state = {
        'components': ['StrategyGenerator', 'BacktestEngine', 'EvolutionarySelector'],
        'strategy_count': 1,
        'generation_count': 1
    }
    integration_report = doc_system.generate_system_integration_report(system_state)
    assert Path(integration_report).exists()
    print("✓ System integration report generated successfully")
    
    # Test 8: Lineage Visualization
    print("\n8. Testing Lineage Visualization...")
    lineage_viz = doc_system.get_lineage_visualization()
    assert Path(lineage_viz).exists()
    print("✓ Lineage visualization created successfully")
    
    # Test 9: Process Logging
    print("\n9. Testing Process Logging...")
    log_file = doc_system.log_process('test_comprehensive', {'status': 'success'})
    assert Path(log_file).exists()
    print("✓ Process logged successfully")
    
    # Test 10: Archive Index Verification
    print("\n10. Testing Archive Index...")
    archive_index_path = 'documentation/archive/archive_index.json'
    with open(archive_index_path, 'r') as f:
        archive_index = json.load(f)
    
    assert len(archive_index['strategies']) > 0
    assert len(archive_index['reports']) > 0
    assert len(archive_index['visualizations']) > 0
    print("✓ Archive index verified successfully")
    
    # Test 11: Lineage Database Verification
    print("\n11. Testing Lineage Database...")
    lineage_db_path = 'documentation/lineage/lineage_database.json'
    with open(lineage_db_path, 'r') as f:
        lineage_db = json.load(f)
    
    assert len(lineage_db['generations']) > 0
    assert len(lineage_db['strategy_genealogy']) > 0
    assert len(lineage_db['evolutionary_history']) > 0
    print("✓ Lineage database verified successfully")
    
    # Test 12: Data Consistency Check
    print("\n12. Testing Data Consistency...")
    
    # Check that all generated files are properly indexed
    total_files = 0
    for item_type in ['strategies', 'reports', 'visualizations']:
        for item in archive_index[item_type]:
            file_path = f"documentation/{item['file_path']}"
            assert Path(file_path).exists(), f"Missing file: {file_path}"
            total_files += 1
    
    print(f"✓ All {total_files} files verified and indexed correctly")
    
    # Test 13: Backward Compatibility
    print("\n13. Testing Backward Compatibility...")
    
    # Test that the old generate_report method still works
    old_style_report = doc_system.generate_report(
        test_strategy, test_backtest_results, test_selection_results
    )
    assert Path(old_style_report).exists()
    print("✓ Backward compatibility maintained")
    
    print("\n🎉 All Comprehensive Tests Passed!")
    
    # Print final statistics
    print(f"\n📊 Final Statistics:")
    print(f"   • Archive Index: {len(archive_index['strategies'])} strategies, "
          f"{len(archive_index['reports'])} reports, {len(archive_index['visualizations'])} visualizations")
    print(f"   • Lineage Database: {len(lineage_db['generations'])} generations, "
          f"{len(lineage_db['strategy_genealogy'])} strategies, {len(lineage_db['evolutionary_history'])} events")
    print(f"   • Total Files Generated: {total_files}")
    
    return True

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n🛡️  Testing Error Handling...")
    
    from documentation.documentation_system import DocumentationSystem
    
    doc_system = DocumentationSystem()
    
    # Test with minimal data
    minimal_strategy = {'id': 'minimal_test', 'template': 'simple'}
    minimal_backtest = {
        'strategy_id': 'minimal_test',
        'performance_metrics': {'sharpe_ratio': 1.0},
        'risk_profile': {'volatility': 0.05}
    }
    
    # These should not crash
    try:
        report = doc_system.generate_performance_report(minimal_strategy, minimal_backtest)
        assert Path(report).exists()
        print("✓ Minimal data handling works")
    except Exception as e:
        print(f"❌ Minimal data test failed: {e}")
        return False
    
    # Test lineage with empty data
    try:
        lineage_result = doc_system.track_evolutionary_lineage(
            {'generation_number': 1}, [], []
        )
        assert lineage_result['child_count'] == 0
        print("✓ Empty lineage handling works")
    except Exception as e:
        print(f"❌ Empty lineage test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests"""
    try:
        # Run comprehensive feature test
        if not test_all_documentation_features():
            print("❌ Comprehensive test failed")
            return False
        
        # Run error handling test
        if not test_error_handling():
            print("❌ Error handling test failed")
            return False
        
        print("\n🏆 All Documentation System Tests Passed Successfully!")
        print("\n🎯 The documentation system is fully functional and ready for production use.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)