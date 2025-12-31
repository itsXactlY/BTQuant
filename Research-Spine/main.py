#!/usr/bin/env python3
"""
Main entry point for the Autonomous Quantitative Research Agency
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from config.config_loader import ConfigLoader
from utils.logging_setup import LoggingSetup
from utils.error_handling import ErrorHandler
from strategy_generation.strategy_generator import StrategyGenerator
from backtesting.backtest_engine import BacktestEngine
from evolutionary_selection.evolutionary_selector import EvolutionarySelector
from documentation.documentation_system import DocumentationSystem
from deployment.deployment_manager import DeploymentManager
from financial_models.model_integration import FinancialModelIntegration
from financial_models.data_connectors import DataSourceManager

def main():
    """Main execution function"""
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # Validate configuration
        if not config_loader.validate_config():
            raise Exception("Invalid system configuration")
        
        # Set up logging
        logging_setup = LoggingSetup(config)
        logging_setup.setup_logging()
        
        # Get logger
        logger = logging_setup.get_logger('main')
        logger.info("Starting Autonomous Quantitative Research Agency")
        
        # Initialize error handler
        error_handler = ErrorHandler()
        
        # Initialize system components
        strategy_generator = StrategyGenerator()
        backtest_engine = BacktestEngine()
        evolutionary_selector = EvolutionarySelector()
        documentation_system = DocumentationSystem()
        deployment_manager = DeploymentManager()
        
        # Initialize financial model integration
        financial_model_integration = FinancialModelIntegration()
        data_source_manager = DataSourceManager()

        logger.info("System components initialized successfully")
        logger.info("Basic scaffolding is functional")
        logger.info("💰 Financial Model Integration: Xiaomi MiMo-V2-Flash model integrated")
        logger.info("   • Advanced market prediction capabilities")
        logger.info("   • Strategy enhancement using model insights")
        logger.info("   • Real-time data integration ready")
        
        # Demonstrate documentation system integration
        logger.info("📚 Documentation System: Ready for comprehensive strategy documentation")
        logger.info("   • Automated strategy documentation generation")
        logger.info("   • Evolutionary lineage tracking")
        logger.info("   • Performance report generation")
        logger.info("   • Strategy visualization tools")
        logger.info("   • Living archive system")
        logger.info("   • Full integration with all components")
        
        return 0
        
    except Exception as e:
        # Basic error handling before logging is set up
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())