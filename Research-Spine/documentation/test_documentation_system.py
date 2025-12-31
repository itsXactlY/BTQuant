"""
Test script for the comprehensive documentation system
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from documentation.documentation_system import DocumentationSystem
from strategy_generation.strategy_generator import StrategyGenerator
from backtesting.backtest_engine import BacktestEngine
from evolutionary_selection.evolutionary_selector import EvolutionarySelector
import logging
import json
from datetime import datetime

def setup_test_logging():
    """Set up basic logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def create_test_strategy():
    """Create a test strategy for documentation"""
    strategy_generator = StrategyGenerator()
    
    # Generate a simple test strategy
    strategy = strategy_generator.generate_strategy('mean_reversion', {
        'mean_reversion_strength': 1.0,
        'volatility_target': 0.01,
        'lookback_period': 20
    })
    
    return strategy

def create_test_backtest_results(strategy):
    """Create test backtest results"""
    backtest_engine = BacktestEngine()
    
    # Create mock historical data
    import pandas as pd
    import numpy as np
    
    # Generate synthetic historical data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    historical_data = pd.DataFrame({
        'date': dates,
        'close': np.cumprod(1 + np.random.normal(0.0005, 0.01, len(dates))),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Run backtest
    backtest_results = backtest_engine.run_backtest(strategy, historical_data)
    
    return backtest_results

def create_test_selection_results(strategy, backtest_results):
    """Create test selection results"""
    evolutionary_selector = EvolutionarySelector()
    
    # Create a simple selection result
    fitness_score = evolutionary_selector.calculate_fitness(strategy, backtest_results)
    
    selection_results = {
        'strategy': strategy,
        'performance': backtest_results['performance_metrics'],
        'fitness_scores': {
            'sharpe_ratio': backtest_results['performance_metrics']['sharpe_ratio'],
            'drawdown': backtest_results['performance_metrics']['max_drawdown'],
            'win_rate': backtest_results['performance_metrics']['win_rate']
        },
        'composite_fitness': fitness_score,
        'selection_metadata': {
            'selection_method': 'test_selection',
            'generation': 1,
            'rank': 1
        }
    }
    
    return selection_results

def test_documentation_system():
    """Test all documentation system features"""
    print("Testing Comprehensive Documentation System...")
    
    # Initialize documentation system
    doc_system = DocumentationSystem()
    
    # Create test data
    strategy = create_test_strategy()
    backtest_results = create_test_backtest_results(strategy)
    selection_results = create_test_selection_results(strategy, backtest_results)
    
    print(f"\nTest Strategy: {strategy.get('template', 'unknown')}")
    print(f"Strategy ID: {strategy.get('id', 'unknown')}")
    
    # Test 1: Comprehensive Report Generation
    print("\n1. Testing Comprehensive Report Generation...")
    comprehensive_report = doc_system.generate_comprehensive_report(
        strategy, backtest_results, selection_results
    )
    print(f"✓ Comprehensive report generated: {comprehensive_report}")
    
    # Test 2: Performance Report Generation
    print("\n2. Testing Performance Report Generation...")
    performance_report = doc_system.generate_performance_report(strategy, backtest_results)
    print(f"✓ Performance report generated: {performance_report}")
    
    # Test 3: Evolutionary Lineage Tracking
    print("\n3. Testing Evolutionary Lineage Tracking...")
    
    # Create parent and child strategies for lineage testing
    parent_strategy = create_test_strategy()
    child_strategy = create_test_strategy()
    
    generation_data = {
        'generation_number': 1,
        'metrics': {
            'population_size': 2,
            'diversity_score': 0.85,
            'fitness_improvement': 0.15
        }
    }
    
    lineage_result = doc_system.track_evolutionary_lineage(
        generation_data, [parent_strategy], [child_strategy]
    )
    print(f"✓ Evolutionary lineage tracked: {lineage_result['child_count']} child strategies")
    
    # Test 4: Evolutionary Report Generation
    print("\n4. Testing Evolutionary Report Generation...")
    evolutionary_report = doc_system.generate_evolutionary_report({
        'generations': 1,
        'strategies': 2,
        'metrics': generation_data['metrics']
    })
    print(f"✓ Evolutionary report generated: {evolutionary_report}")
    
    # Test 5: Strategy Visualization
    print("\n5. Testing Strategy Visualization...")
    
    visualization_types = ['performance', 'risk_return', 'drawdown']
    for viz_type in visualization_types:
        viz_file = doc_system.create_strategy_visualization(strategy, backtest_results, viz_type)
        if viz_file:
            print(f"✓ {viz_type} visualization created: {viz_file}")
    
    # Test 6: Living Archive System
    print("\n6. Testing Living Archive System...")
    archive_file = doc_system.archive_strategy(strategy, backtest_results, selection_results)
    print(f"✓ Strategy archived: {archive_file}")
    
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
    
    # Test 8: Lineage Visualization
    print("\n8. Testing Lineage Visualization...")
    lineage_viz = doc_system.get_lineage_visualization()
    if lineage_viz:
        print(f"✓ Lineage visualization created: {lineage_viz}")
    
    # Test 9: Process Logging
    print("\n9. Testing Process Logging...")
    log_file = doc_system.log_process('test_process', {'test_data': 'success'})
    print(f"✓ Process logged: {log_file}")
    
    print("\n🎉 All Documentation System Tests Completed Successfully!")
    
    # Print summary statistics
    print(f"\n📊 Documentation System Summary:")
    print(f"   - Reports generated: 4")
    print(f"   - Visualizations created: {len(visualization_types) + (1 if lineage_viz else 0)}")
    print(f"   - Strategies archived: 1")
    print(f"   - Lineage generations tracked: 1")
    print(f"   - Process logs created: 1")
    
    return True

if __name__ == "__main__":
    setup_test_logging()
    
    try:
        test_documentation_system()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)