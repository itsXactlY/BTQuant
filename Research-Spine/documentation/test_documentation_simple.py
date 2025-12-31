"""
Simple test script for the comprehensive documentation system
Tests the documentation system in isolation without external dependencies
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

def test_documentation_system_isolation():
    """Test the documentation system in isolation"""
    print("Testing Documentation System in Isolation...")
    
    # Import only the documentation system
    from documentation.documentation_system import DocumentationSystem
    
    # Create a temporary directory for test outputs
    test_dir = Path('test_documentation_output')
    test_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize documentation system
        doc_system = DocumentationSystem()
        
        # Create mock test data
        mock_strategy = {
            'id': 'test_strategy_001',
            'template': 'mean_reversion',
            'parameters': {
                'mean_reversion_strength': 1.0,
                'volatility_target': 0.01,
                'lookback_period': 20
            },
            'metadata': {
                'generated_by': 'TestGenerator',
                'version': '1.0'
            }
        }
        
        mock_backtest_results = {
            'strategy_id': 'test_strategy_001',
            'template': 'mean_reversion',
            'performance_metrics': {
                'sharpe_ratio': 1.85,
                'max_drawdown': 0.12,
                'win_rate': 0.65,
                'total_return': 0.25,
                'sortino_ratio': 2.1,
                'volatility': 0.08
            },
            'risk_profile': {
                'volatility': 0.08,
                'value_at_risk': 0.05,
                'expected_shortfall': 0.07
            },
            'returns_data': {
                'returns': [0.01, -0.005, 0.015, 0.02, -0.01],
                'num_periods': 5,
                'return_statistics': {
                    'mean': 0.006,
                    'std': 0.012,
                    'min': -0.01,
                    'max': 0.02
                }
            }
        }
        
        mock_selection_results = {
            'strategy': mock_strategy,
            'performance': mock_backtest_results['performance_metrics'],
            'fitness_scores': {
                'sharpe_ratio': 1.85,
                'drawdown': 0.12,
                'win_rate': 0.65
            },
            'composite_fitness': 0.87,
            'selection_metadata': {
                'selection_method': 'test_selection',
                'generation': 1,
                'rank': 1
            }
        }
        
        print(f"\nTest Strategy: {mock_strategy['template']}")
        print(f"Strategy ID: {mock_strategy['id']}")
        
        # Test 1: Comprehensive Report Generation
        print("\n1. Testing Comprehensive Report Generation...")
        comprehensive_report = doc_system.generate_comprehensive_report(
            mock_strategy, mock_backtest_results, mock_selection_results
        )
        print(f"✓ Comprehensive report generated: {comprehensive_report}")
        
        # Verify the report file exists and contains expected data
        assert Path(comprehensive_report).exists(), "Comprehensive report file not created"
        with open(comprehensive_report, 'r') as f:
            report_data = json.load(f)
        assert report_data['strategy']['id'] == mock_strategy['id'], "Strategy ID mismatch in report"
        
        # Test 2: Performance Report Generation
        print("\n2. Testing Performance Report Generation...")
        performance_report = doc_system.generate_performance_report(mock_strategy, mock_backtest_results)
        print(f"✓ Performance report generated: {performance_report}")
        
        assert Path(performance_report).exists(), "Performance report file not created"
        with open(performance_report, 'r') as f:
            perf_data = json.load(f)
        assert perf_data['strategy_id'] == mock_strategy['id'], "Strategy ID mismatch in performance report"
        
        # Test 3: Evolutionary Lineage Tracking
        print("\n3. Testing Evolutionary Lineage Tracking...")
        
        mock_parent_strategy = {
            'id': 'parent_strategy_001',
            'template': 'mean_reversion',
            'parameters': {'mean_reversion_strength': 0.9}
        }
        
        mock_child_strategy = {
            'id': 'child_strategy_001',
            'template': 'mean_reversion',
            'parameters': {'mean_reversion_strength': 1.1}
        }
        
        generation_data = {
            'generation_number': 1,
            'metrics': {
                'population_size': 2,
                'diversity_score': 0.85,
                'fitness_improvement': 0.15
            }
        }
        
        lineage_result = doc_system.track_evolutionary_lineage(
            generation_data, [mock_parent_strategy], [mock_child_strategy]
        )
        print(f"✓ Evolutionary lineage tracked: {lineage_result['child_count']} child strategies")
        assert lineage_result['child_count'] == 1, "Child count mismatch"
        
        # Test 4: Evolutionary Report Generation
        print("\n4. Testing Evolutionary Report Generation...")
        evolutionary_report = doc_system.generate_evolutionary_report({
            'generations': 1,
            'strategies': 2,
            'metrics': generation_data['metrics']
        })
        print(f"✓ Evolutionary report generated: {evolutionary_report}")
        
        assert Path(evolutionary_report).exists(), "Evolutionary report file not created"
        
        # Test 5: Strategy Visualization (basic test - files should be created)
        print("\n5. Testing Strategy Visualization...")
        
        visualization_types = ['performance', 'risk_return', 'drawdown']
        for viz_type in visualization_types:
            viz_file = doc_system.create_strategy_visualization(mock_strategy, mock_backtest_results, viz_type)
            if viz_file:
                print(f"✓ {viz_type} visualization created: {viz_file}")
                assert Path(viz_file).exists(), f"{viz_type} visualization file not created"
        
        # Test 6: Living Archive System
        print("\n6. Testing Living Archive System...")
        archive_file = doc_system.archive_strategy(mock_strategy, mock_backtest_results, mock_selection_results)
        print(f"✓ Strategy archived: {archive_file}")
        
        assert Path(archive_file).exists(), "Archive file not created"
        with open(archive_file, 'r') as f:
            archive_data = json.load(f)
        assert archive_data['strategy']['id'] == mock_strategy['id'], "Strategy ID mismatch in archive"
        
        # Test 7: System Integration Report
        print("\n7. Testing System Integration Report...")
        system_state = {
            'components': ['StrategyGenerator', 'BacktestEngine', 'EvolutionarySelector', 'DocumentationSystem'],
            'strategy_count': 3,
            'generation_count': 1,
            'timestamp': datetime.now().isoformat()
        }
        integration_report = doc_system.generate_system_integration_report(system_state)
        print(f"✓ System integration report generated: {integration_report}")
        
        assert Path(integration_report).exists(), "Integration report file not created"
        
        # Test 8: Lineage Visualization
        print("\n8. Testing Lineage Visualization...")
        lineage_viz = doc_system.get_lineage_visualization()
        if lineage_viz:
            print(f"✓ Lineage visualization created: {lineage_viz}")
            assert Path(lineage_viz).exists(), "Lineage visualization file not created"
        
        # Test 9: Process Logging
        print("\n9. Testing Process Logging...")
        log_file = doc_system.log_process('test_process', {'test_data': 'success'})
        print(f"✓ Process logged: {log_file}")
        
        assert Path(log_file).exists(), "Process log file not created"
        
        # Test 10: Verify Archive Index
        print("\n10. Testing Archive Index...")
        archive_index_path = 'documentation/archive/archive_index.json'
        assert Path(archive_index_path).exists(), "Archive index not created"
        
        with open(archive_index_path, 'r') as f:
            archive_index = json.load(f)
        
        print(f"   - Strategies archived: {len(archive_index['strategies'])}")
        print(f"   - Reports generated: {len(archive_index['reports'])}")
        print(f"   - Visualizations created: {len(archive_index['visualizations'])}")
        
        # Test 11: Verify Lineage Database
        print("\n11. Testing Lineage Database...")
        lineage_db_path = 'documentation/lineage/lineage_database.json'
        assert Path(lineage_db_path).exists(), "Lineage database not created"
        
        with open(lineage_db_path, 'r') as f:
            lineage_db = json.load(f)
        
        print(f"   - Generations tracked: {len(lineage_db['generations'])}")
        print(f"   - Strategies in genealogy: {len(lineage_db['strategy_genealogy'])}")
        print(f"   - Evolutionary events: {len(lineage_db['evolutionary_history'])}")
        
        print("\n🎉 All Documentation System Tests Completed Successfully!")
        
        # Print summary statistics
        print(f"\n📊 Documentation System Summary:")
        print(f"   - Reports generated: {len(archive_index['reports'])}")
        print(f"   - Visualizations created: {len(archive_index['visualizations'])}")
        print(f"   - Strategies archived: {len(archive_index['strategies'])}")
        print(f"   - Lineage generations tracked: {len(lineage_db['generations'])}")
        print(f"   - Process logs created: 1")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up test directory
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    success = test_documentation_system_isolation()
    sys.exit(0 if success else 1)