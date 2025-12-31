#!/usr/bin/env python3
"""
Test script for Financial Model Integration

This script demonstrates the integration of Xiaomi MiMo-V2-Flash financial model
with the existing system components.
"""

import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import system components
from financial_models.model_integration import FinancialModelIntegration
from financial_models.data_connectors import DataSourceManager
from financial_models.mimo_v2_flash import MiMoV2FlashModel


def test_financial_model_integration():
    """Test the complete financial model integration"""
    
    logger = logging.getLogger("FinancialModelTest")
    logger.info("Starting Financial Model Integration Test")
    
    try:
        # 1. Test Data Source Manager
        logger.info("\n=== Testing Data Source Manager ===")
        data_manager = DataSourceManager()
        
        # Get sample market data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Test with multiple symbols
        symbols = ['AAPL', 'BTC/USD', 'SPY']
        
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            market_data = data_manager.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe='1D'
            )
            
            logger.info(f"Retrieved {len(market_data)} data points for {symbol}")
            logger.info(f"Data range: {market_data.index[0]} to {market_data.index[-1]}")
            logger.info(f"Price range: ${market_data['close'].min():.2f} to ${market_data['close'].max():.2f}")
            
            # Display sample data
            logger.info(f"Sample data for {symbol}:")
            logger.info(market_data.head(3).to_string())
            
    except Exception as e:
        logger.error(f"Data source test failed: {e}")
        return False
    
    try:
        # 2. Test MiMo-V2-Flash Model
        logger.info("\n=== Testing Xiaomi MiMo-V2-Flash Model ===")
        
        # Initialize the model
        mimo_model = MiMoV2FlashModel()
        
        # Get market data for the model
        market_data = data_manager.get_historical_data(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date,
            timeframe='1D'
        )
        
        # Load data into model
        mimo_model.load_market_data(market_data)
        logger.info("Market data loaded into MiMo-V2-Flash model")
        
        # Train the model
        training_results = mimo_model.train_model(market_data)
        logger.info(f"Model training completed. Validation accuracy: {training_results['validation_accuracy']:.3f}")
        
        # Make predictions
        predictions = mimo_model.predict(market_data.iloc[-30:])  # Predict last 30 days
        logger.info(f"Generated predictions for {len(predictions['price_prediction'])} data points")
        
        # Get model insights
        insights = mimo_model.get_model_insights()
        logger.info(f"Market outlook: {insights['recommendations']['market_outlook']}")
        logger.info(f"Risk assessment: {insights['recommendations']['risk_assessment']}")
        
        # Display strategy recommendations
        for rec in insights['recommendations']['strategy_recommendations']:
            logger.info(f"Strategy recommendation: {rec['type']} ({rec['direction']}) - Confidence: {rec['confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"MiMo-V2-Flash model test failed: {e}")
        return False
    
    try:
        # 3. Test Financial Model Integration
        logger.info("\n=== Testing Financial Model Integration ===")
        
        # Initialize the integration system
        model_integration = FinancialModelIntegration()
        
        # Test strategy generation integration
        integration_results = model_integration.integrate_with_strategy_generation(market_data)
        
        logger.info("Strategy generation integration completed")
        logger.info(f"Market outlook: {integration_results['model_insights']['recommendations']['market_outlook']}")
        
        # Display strategy enhancements
        enhancements = integration_results['strategy_enhancements']
        logger.info("Strategy enhancement recommendations:")
        
        for param, adjustment in enhancements['parameter_adjustments'].items():
            logger.info(f"  {param}: {adjustment}")
        
        for component in enhancements['new_strategy_components']:
            logger.info(f"  New component: {component['component_type']} (weight: {component['weight']:.1f})")
        
        # Test backtesting integration
        sample_backtest_results = {
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.12,
            'total_return': 0.25,
            'win_rate': 0.65
        }
        
        backtest_integration = model_integration.integrate_with_backtesting(
            sample_backtest_results, 
            market_data.iloc[-90:]  # Last 90 days for backtesting
        )
        
        logger.info("Backtesting integration completed")
        logger.info(f"Strategy fit score: {backtest_integration['enhanced_analysis']['strategy_fit']['fit_score']:.3f}")
        logger.info(f"Model contribution: {backtest_integration['enhanced_analysis']['performance_attribution']['model_contribution']:.2f}")
        
        # Test evolutionary selection integration
        sample_population = [
            {'strategy_type': 'momentum', 'fitness': 0.75, 'risk_level': 'medium'},
            {'strategy_type': 'mean_reversion', 'fitness': 0.68, 'risk_level': 'low'},
            {'strategy_type': 'breakout', 'fitness': 0.82, 'risk_level': 'high'}
        ]
        
        enhanced_population = model_integration.integrate_with_evolutionary_selection(sample_population)
        
        logger.info("Evolutionary selection integration completed")
        for i, strategy in enumerate(enhanced_population):
            model_fitness = strategy['fitness']['model_adjusted']
            original_fitness = strategy['fitness']['original']
            logger.info(f"Strategy {i+1}: Original fitness {original_fitness:.3f} -> Model-adjusted {model_fitness:.3f}")
        
    except Exception as e:
        logger.error(f"Financial model integration test failed: {e}")
        return False
    
    try:
        # 4. Test Complete System Integration
        logger.info("\n=== Testing Complete System Integration ===")
        
        # Get integration status
        integration_status = model_integration.get_integration_status()
        
        logger.info("Financial Model Integration Status:")
        logger.info(f"  Models available: {', '.join(integration_status['models_available'])}")
        logger.info(f"  Last integration: {integration_status['last_integration_time']}")
        logger.info(f"  System health: {integration_status['system_health']['status']}")
        
        # Test model persistence
        mimo_model.save_model()
        logger.info("Model saved successfully")
        
        # Load model to verify persistence
        new_model = MiMoV2FlashModel()
        new_model.load_model()
        logger.info("Model loaded successfully")
        
        # Test data preprocessing
        preprocessed_data = data_manager.get_data_for_model(
            symbol='AAPL',
            start_date=start_date,
            end_date=end_date,
            timeframe='1D'
        )
        
        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        logger.info(f"Preprocessed data columns: {list(preprocessed_data.columns)}")
        
    except Exception as e:
        logger.error(f"Complete system integration test failed: {e}")
        return False
    
    logger.info("\n=== Financial Model Integration Test Completed Successfully ===")
    logger.info("✅ Xiaomi MiMo-V2-Flash model integrated")
    logger.info("✅ Data source connectors operational")
    logger.info("✅ Model inference and predictions working")
    logger.info("✅ Strategy enhancement capabilities enabled")
    logger.info("✅ System integration complete")
    
    return True


if __name__ == "__main__":
    success = test_financial_model_integration()
    
    if success:
        print("\n🎉 Financial Model Integration Test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Financial Model Integration Test FAILED")
        sys.exit(1)