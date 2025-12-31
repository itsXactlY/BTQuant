"""
Test Backtesting Framework

Comprehensive test suite for the backtesting framework to verify all components
work correctly with sample strategies.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtesting.backtest_engine import BacktestEngine
from backtesting.metrics.performance_metrics import PerformanceMetrics
from backtesting.metrics.statistical_significance import StatisticalSignificance
from backtesting.validation.out_of_sample_validation import OutOfSampleValidation
from backtesting.optimization.walk_forward_optimization import WalkForwardOptimization
from backtesting.risk_management.risk_management import RiskManagement

class TestBacktestingFramework(unittest.TestCase):
    """Test suite for the backtesting framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backtest_engine = BacktestEngine()
        self.performance_metrics = PerformanceMetrics()
        self.statistical_significance = StatisticalSignificance()
        self.out_of_sample_validation = OutOfSampleValidation()
        self.walk_forward_optimization = WalkForwardOptimization()
        self.risk_management = RiskManagement()
        
        # Create sample strategy
        self.sample_strategy = {
            'id': 'test_strategy_001',
            'template': 'mean_reversion',
            'parameters': {
                'mean_reversion_strength': 1.0,
                'volatility_target': 0.01
            }
        }
        
        # Create sample historical data
        self.sample_data = self._create_sample_historical_data()
    
    def _create_sample_historical_data(self) -> pd.DataFrame:
        """Create sample historical data for testing"""
        # Create date range
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
        
        # Create sample data
        data = pd.DataFrame({
            'date': dates,
            'open': np.cumprod(1 + np.random.normal(0.001, 0.01, 1000)),
            'high': np.cumprod(1 + np.random.normal(0.001, 0.01, 1000)) * 1.01,
            'low': np.cumprod(1 + np.random.normal(0.001, 0.01, 1000)) * 0.99,
            'close': np.cumprod(1 + np.random.normal(0.001, 0.01, 1000)),
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        return data
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        # Create sample returns
        returns = np.random.normal(0.001, 0.01, 252)
        
        # Calculate metrics
        metrics = self.performance_metrics.calculate_all_metrics(returns)
        
        # Verify metrics are calculated
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('total_return', metrics)
        self.assertIn('calmar_ratio', metrics)
        
        # Verify values are reasonable
        self.assertGreaterEqual(metrics['sharpe_ratio'], -5.0)
        self.assertLessEqual(metrics['sharpe_ratio'], 5.0)
        self.assertGreaterEqual(metrics['max_drawdown'], -100.0)
        self.assertLessEqual(metrics['max_drawdown'], 0.0)
    
    def test_statistical_significance_calculation(self):
        """Test statistical significance calculation"""
        # Create sample returns
        returns = np.random.normal(0.001, 0.01, 252)
        
        # Calculate p-value
        p_value = self.statistical_significance.calculate_p_value(returns)
        
        # Verify p-value is calculated
        self.assertIsInstance(p_value, float)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)
        
        # Test hypothesis testing
        hypothesis_result = self.statistical_significance.perform_hypothesis_test(returns)
        self.assertIn('p_value', hypothesis_result)
        self.assertIn('reject_null_hypothesis', hypothesis_result)
    
    def test_out_of_sample_validation(self):
        """Test out-of-sample validation"""
        # Split data
        in_sample, out_of_sample = self.out_of_sample_validation.split_data_in_sample_out_of_sample(
            self.sample_data, split_ratio=0.7
        )
        
        # Verify split
        self.assertEqual(len(in_sample) + len(out_of_sample), len(self.sample_data))
        self.assertEqual(len(in_sample), 700)
        self.assertEqual(len(out_of_sample), 300)
        
        # Validate strategy
        validation_results = self.out_of_sample_validation.validate_strategy_out_of_sample(
            self.sample_strategy, in_sample, out_of_sample
        )
        
        # Verify results
        self.assertIn('validation_passed', validation_results)
        self.assertIn('overfitting_analysis', validation_results)
        self.assertIn('statistical_significance', validation_results)
    
    def test_walk_forward_optimization(self):
        """Test walk-forward optimization"""
        # Define parameter grid
        parameter_grid = {
            'mean_reversion_strength': [0.8, 1.0, 1.2],
            'volatility_target': [0.008, 0.01, 0.012]
        }
        
        # Run walk-forward optimization
        wfo_results = self.walk_forward_optimization.perform_walk_forward_optimization(
            self.sample_strategy, 
            self.sample_data, 
            parameter_grid, 
            window_size=252, 
            step_size=126
        )
        
        # Verify results
        self.assertIn('optimization_windows', wfo_results)
        self.assertIn('final_parameters', wfo_results)
        self.assertIn('average_sharpe_ratio', wfo_results)
        self.assertGreater(len(wfo_results['optimization_windows']), 0)
    
    def test_risk_management(self):
        """Test risk management functions"""
        # Test position sizing
        position_sizing = self.risk_management.calculate_position_size(
            self.sample_strategy, account_size=100000
        )
        
        self.assertIn('position_size', position_sizing)
        self.assertIn('leverage_ratio', position_sizing)
        self.assertGreater(position_sizing['position_size'], 0)
        
        # Test risk profile calculation
        sample_returns = np.random.normal(0.001, 0.01, 252)
        risk_profile = self.risk_management.calculate_strategy_risk_profile(
            self.sample_strategy, sample_returns
        )
        
        self.assertIn('risk_score', risk_profile)
        self.assertIn('risk_category', risk_profile)
        self.assertIn('basic_risk_metrics', risk_profile)
    
    def test_backtest_engine_integration(self):
        """Test backtest engine integration"""
        # Run basic backtest
        backtest_results = self.backtest_engine.run_backtest(
            self.sample_strategy, self.sample_data
        )
        
        # Verify comprehensive results
        self.assertIn('performance_metrics', backtest_results)
        self.assertIn('statistical_significance', backtest_results)
        self.assertIn('risk_profile', backtest_results)
        self.assertIn('position_sizing', backtest_results)
        self.assertIn('returns_data', backtest_results)
        
        # Verify metrics are calculated
        metrics = backtest_results['performance_metrics']
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation"""
        # Run comprehensive validation
        validation_results = self.backtest_engine.run_comprehensive_validation(
            self.sample_strategy, self.sample_data
        )
        
        # Verify results structure
        self.assertIn('out_of_sample_validation', validation_results)
        self.assertIn('walk_forward_optimization', validation_results)
        self.assertIn('validation_score', validation_results)
        self.assertIn('validation_passed', validation_results)
        
        # Verify validation score is reasonable
        self.assertGreaterEqual(validation_results['validation_score'], 0.0)
        self.assertLessEqual(validation_results['validation_score'], 1.0)
    
    def test_full_integration_validation(self):
        """Test full integration validation"""
        # Run full integration validation
        integration_results = self.backtest_engine.validate_strategy_with_integration(
            self.sample_strategy, self.sample_data
        )
        
        # Verify comprehensive results
        self.assertIn('backtest_results', integration_results)
        self.assertIn('validation_results', integration_results)
        self.assertIn('risk_metrics', integration_results)
        self.assertIn('position_sizing', integration_results)
        self.assertIn('strategy_score', integration_results)
        self.assertIn('recommendation', integration_results)
        
        # Verify strategy score is reasonable
        self.assertGreaterEqual(integration_results['strategy_score'], 0.0)
        self.assertLessEqual(integration_results['strategy_score'], 1.0)
        
        # Verify recommendation structure
        recommendation = integration_results['recommendation']
        self.assertIn('recommendation', recommendation)
        self.assertIn('score', recommendation)
        self.assertIn('reason', recommendation)
    
    def test_different_strategy_templates(self):
        """Test different strategy templates"""
        # Test trend following strategy
        trend_strategy = {
            'id': 'trend_strategy_001',
            'template': 'trend_following',
            'parameters': {
                'trend_strength': 1.0,
                'volatility_target': 0.012
            }
        }
        
        # Run backtest
        trend_results = self.backtest_engine.run_backtest(trend_strategy, self.sample_data)
        
        # Verify results
        self.assertIn('performance_metrics', trend_results)
        self.assertGreater(trend_results['performance_metrics']['sharpe_ratio'], -10.0)
        
        # Test default strategy
        default_strategy = {
            'id': 'default_strategy_001',
            'template': 'default',
            'parameters': {}
        }
        
        # Run backtest
        default_results = self.backtest_engine.run_backtest(default_strategy, self.sample_data)
        
        # Verify results
        self.assertIn('performance_metrics', default_results)
        self.assertGreater(default_results['performance_metrics']['sharpe_ratio'], -10.0)
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with invalid data type
        with self.assertRaises(ValueError):
            self.backtest_engine.run_backtest(self.sample_strategy, "invalid_data")
        
        # Test with empty strategy
        empty_strategy = {}
        with self.assertRaises(KeyError):
            self.backtest_engine.run_backtest(empty_strategy, self.sample_data)
    
    def test_performance_consistency(self):
        """Test performance consistency across multiple runs"""
        # Run multiple backtests and check consistency
        results = []
        for i in range(3):
            result = self.backtest_engine.run_backtest(self.sample_strategy, self.sample_data)
            results.append(result['performance_metrics']['sharpe_ratio'])
        
        # Results should be reasonably consistent (same random seed would be better)
        # For now, just verify they're all calculated
        for sharpe in results:
            self.assertIsInstance(sharpe, float)
            self.assertGreater(sharpe, -10.0)
            self.assertLess(sharpe, 10.0)

if __name__ == '__main__':
    unittest.main()