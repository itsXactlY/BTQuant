"""
Test Backtesting Framework Integration

Test script to verify the backtesting framework works correctly with the main system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime

# Import from the main package structure
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics.performance_metrics import PerformanceMetrics
from backtesting.metrics.statistical_significance import StatisticalSignificance
from backtesting.validation.out_of_sample_validation import OutOfSampleValidation
from backtesting.optimization.walk_forward_optimization import WalkForwardOptimization
from backtesting.risk_management.risk_management import RiskManagement

def create_sample_data():
    """Create sample historical data"""
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.cumprod(1 + np.random.normal(0.001, 0.01, 500)),
        'high': np.cumprod(1 + np.random.normal(0.001, 0.01, 500)) * 1.01,
        'low': np.cumprod(1 + np.random.normal(0.001, 0.01, 500)) * 0.99,
        'close': np.cumprod(1 + np.random.normal(0.001, 0.01, 500)),
        'volume': np.random.randint(1000, 10000, 500)
    })
    return data

def test_backtesting_framework():
    """Test the complete backtesting framework"""
    print("🧪 Testing Backtesting Framework Integration")
    print("=" * 50)
    
    # Initialize all components
    print("🔧 Initializing components...")
    backtest_engine = BacktestEngine()
    metrics_calculator = PerformanceMetrics()
    stats_calculator = StatisticalSignificance()
    validator = OutOfSampleValidation()
    optimizer = WalkForwardOptimization()
    risk_manager = RiskManagement()
    print("✓ All components initialized")
    
    # Create test data
    print("📊 Creating test data...")
    data = create_sample_data()
    print(f"✓ Created {len(data)} records of historical data")
    
    # Create test strategy
    strategy = {
        'id': 'test_strategy_001',
        'template': 'mean_reversion',
        'parameters': {
            'mean_reversion_strength': 1.0,
            'volatility_target': 0.01
        }
    }
    print("✓ Created test strategy")
    
    # Test 1: Performance Metrics
    print("\n📈 Testing Performance Metrics...")
    returns = np.random.normal(0.001, 0.01, 252)
    metrics = metrics_calculator.calculate_all_metrics(returns)
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.3f}%")
    print(f"  Total Return: {metrics['total_return']:.3f}%")
    print("✓ Performance Metrics working")
    
    # Test 2: Statistical Significance
    print("\n📊 Testing Statistical Significance...")
    p_value = stats_calculator.calculate_p_value(returns)
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {p_value < 0.05}")
    print("✓ Statistical Significance working")
    
    # Test 3: Out-of-Sample Validation
    print("\n🔍 Testing Out-of-Sample Validation...")
    in_sample, out_of_sample = validator.split_data_in_sample_out_of_sample(data, split_ratio=0.7)
    validation_results = validator.validate_strategy_out_of_sample(strategy, in_sample, out_of_sample)
    print(f"  Validation Passed: {validation_results['validation_passed']}")
    print(f"  Overfitting: {validation_results['overfitting_analysis']['overfitting_detected']}")
    print("✓ Out-of-Sample Validation working")
    
    # Test 4: Walk-Forward Optimization
    print("\n🚀 Testing Walk-Forward Optimization...")
    parameter_grid = {
        'mean_reversion_strength': [0.8, 1.0, 1.2],
        'volatility_target': [0.008, 0.01]
    }
    wfo_results = optimizer.perform_walk_forward_optimization(
        strategy, data, parameter_grid, window_size=200, step_size=100
    )
    print(f"  Windows: {len(wfo_results['optimization_windows'])}")
    print(f"  Avg Sharpe: {wfo_results['average_sharpe_ratio']:.3f}")
    print("✓ Walk-Forward Optimization working")
    
    # Test 5: Risk Management
    print("\n🛡️ Testing Risk Management...")
    position_sizing = risk_manager.calculate_position_size(strategy, account_size=100000)
    print(f"  Position Size: ${position_sizing['position_size']:.2f}")
    risk_profile = risk_manager.calculate_strategy_risk_profile(strategy, returns)
    print(f"  Risk Category: {risk_profile['risk_category']}")
    print("✓ Risk Management working")
    
    # Test 6: Full Backtest Engine
    print("\n🔄 Testing Full Backtest Engine Integration...")
    backtest_results = backtest_engine.run_backtest(strategy, data)
    print(f"  Sharpe Ratio: {backtest_results['performance_metrics']['sharpe_ratio']:.3f}")
    print(f"  Risk Category: {backtest_results['risk_profile']['risk_category']}")
    
    # Test comprehensive validation
    validation_results = backtest_engine.run_comprehensive_validation(strategy, data)
    print(f"  Validation Score: {validation_results['validation_score']:.3f}")
    
    # Test full integration
    integration_results = backtest_engine.validate_strategy_with_integration(strategy, data)
    print(f"  Strategy Score: {integration_results['strategy_score']:.3f}")
    print(f"  Recommendation: {integration_results['recommendation']['recommendation']}")
    print("✓ Full Backtest Engine Integration working")
    
    print("\n" + "=" * 50)
    print("🎉 BACKTESTING FRAMEWORK TEST COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    # Summary of what was implemented
    print("\n📋 IMPLEMENTED COMPONENTS:")
    print("  ✓ Comprehensive Performance Metrics (Sharpe, Sortino, Max Drawdown, etc.)")
    print("  ✓ Statistical Significance Testing (p-values, hypothesis tests)")
    print("  ✓ Out-of-Sample Validation Framework")
    print("  ✓ Walk-Forward Optimization")
    print("  ✓ Risk Management and Position Sizing")
    print("  ✓ Integration with Strategy Generation Engine")
    print("  ✓ Comprehensive Testing Framework")
    
    return True

if __name__ == "__main__":
    try:
        success = test_backtesting_framework()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)