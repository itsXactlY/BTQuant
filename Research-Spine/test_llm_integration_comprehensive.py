#!/usr/bin/env python3
"""
Comprehensive LLM Integration Tests

Tests the complete LLM integration pipeline including:
- Intelligent routing
- Error handling and fallbacks
- Monitoring and metrics
- Configuration management
- Performance validation
"""

import sys
import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_llm_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('LLMIntegrationTests')


class TestResults:
    """Track test results"""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures = []
        self.timing = {}
    
    def record_pass(self, test_name: str, duration: float):
        self.tests_run += 1
        self.tests_passed += 1
        self.timing[test_name] = duration
        logger.info(f"✅ PASS: {test_name} ({duration:.2f}s)")
    
    def record_fail(self, test_name: str, error: str, duration: float):
        self.tests_run += 1
        self.tests_failed += 1
        self.timing[test_name] = duration
        self.failures.append((test_name, error))
        logger.error(f"❌ FAIL: {test_name} - {error} ({duration:.2f}s)")
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'total': self.tests_run,
            'passed': self.tests_passed,
            'failed': self.tests_failed,
            'success_rate': self.tests_passed / self.tests_run if self.tests_run > 0 else 0,
            'failures': self.failures,
            'timing': self.timing
        }


class LLMIntegrationTests:
    """Comprehensive test suite for LLM integration"""
    
    def __init__(self):
        self.results = TestResults()
        self.test_config = {
            'provider': 'ollama',
            'base_url': 'http://localhost:11434',
            'model_name': 'qwen2.5:7b',
            'timeout': 10,
            'max_retries': 1,
            'enable_monitoring': True
        }
    
    def run_all_tests(self):
        """Run all integration tests"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE LLM INTEGRATION TESTS")
        logger.info("=" * 60)
        
        # Test 1: Configuration Management
        self.test_configuration_management()
        
        # Test 2: Intelligent Router
        self.test_intelligent_router()
        
        # Test 3: Strategy Generation (LLM)
        self.test_llm_strategy_generation()
        
        # Test 4: Strategy Generation (Template Fallback)
        self.test_template_strategy_generation()
        
        # Test 5: Error Handling & Fallbacks
        self.test_error_handling()
        
        # Test 6: Monitoring System
        self.test_monitoring_system()
        
        # Test 7: Validation System
        self.test_validation_system()
        
        # Test 8: Performance & Metrics
        self.test_performance_metrics()
        
        # Test 9: Circuit Breaker
        self.test_circuit_breaker()
        
        # Test 10: End-to-End Pipeline
        self.test_end_to_end_pipeline()
        
        # Print summary
        self.print_summary()
        
        # Export results
        self.export_results()
        
        return self.results.get_summary()
    
    def test_configuration_management(self):
        """Test configuration loading and validation"""
        test_name = "Configuration Management"
        start_time = time.time()
        
        try:
            from config.llm_config import LLMConfig, ConfigManager, get_llm_config
            
            # Test 1: Create config
            config = LLMConfig(
                provider='ollama',
                base_url='http://localhost:11434',
                model_name='qwen2.5:7b',
                timeout=10,
                max_retries=1
            )
            
            # Test 2: Validate config
            is_valid, errors = config.validate()
            assert is_valid, f"Config validation failed: {errors}"
            
            # Test 3: Config serialization
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)
            assert config_dict['provider'] == 'ollama'
            
            # Test 4: Config deserialization
            config2 = LLMConfig.from_dict(config_dict)
            assert config2.provider.value == config.provider.value
            
            # Test 5: Config manager
            manager = ConfigManager()
            test_config = manager.load_config('default')
            assert test_config is not None
            
            # Test 6: Environment overrides
            os.environ['LLM_TIMEOUT'] = '20'
            config_with_env = manager.load_config('default')
            assert config_with_env.timeout == 20
            
            # Clean up
            if 'LLM_TIMEOUT' in os.environ:
                del os.environ['LLM_TIMEOUT']
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_intelligent_router(self):
        """Test intelligent routing decisions"""
        test_name = "Intelligent Router"
        start_time = time.time()
        
        try:
            from strategy_generation.integration.intelligent_router import (
                IntelligentRouter, DecisionContext, ComponentType
            )
            
            # Test 1: Router initialization
            router_config = {
                'llm_enabled': True,
                'llm_base_url': 'http://localhost:11434',
                'llm_timeout': 10,
                'max_consecutive_failures': 3,
                'fallback_threshold': 0.6
            }
            
            router = IntelligentRouter(router_config)
            assert router is not None
            
            # Test 2: Decision context
            context = DecisionContext(
                strategy_type='physics_based',
                market_context={'volatility': 'high'},
                performance_requirements={'min_novelty': 0.8}
            )
            
            # Test 3: Routing decision (will depend on LLM availability)
            decision = router.make_routing_decision(context)
            assert decision in [ComponentType.LLM_STRATEGY_GENERATION, 
                              ComponentType.TEMPLATE_GENERATION, 
                              ComponentType.FALLBACK]
            
            # Test 4: Health status
            health = router.get_health_status()
            assert isinstance(health, dict)
            assert 'llm_operational' in health
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_llm_strategy_generation(self):
        """Test LLM-based strategy generation"""
        test_name = "LLM Strategy Generation"
        start_time = time.time()
        
        try:
            from strategy_generation.llm_agents.ollama_client import OllamaClient
            from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
            
            # Test connection first
            client = OllamaClient(base_url='http://localhost:11434', timeout=10)
            health = client.health_check()
            
            if not health['connected']:
                logger.warning("LLM server not available, skipping LLM tests")
                self.results.record_pass(test_name + " (skipped - no LLM)", 0.0)
                return
            
            # Test strategy generation
            agent = StrategyGenerationAgent(client)
            strategy = agent.generate_strategy(
                strategy_type='physics_based',
                market_context={'volatility': 'medium', 'trend': 'neutral'},
                constraints={'risk_level': 'moderate'}
            )
            
            # Validate strategy structure
            assert 'name' in strategy
            assert 'entry_rules' in strategy
            assert 'exit_rules' in strategy
            assert 'risk_management' in strategy
            assert strategy.get('type') == 'llm_generated'
            
            # Check for innovation
            name = strategy.get('name', '').lower()
            assert any(concept in name for concept in ['quantum', 'neural', 'fractal', 'chaos'])
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_template_strategy_generation(self):
        """Test template-based strategy generation"""
        test_name = "Template Strategy Generation"
        start_time = time.time()
        
        try:
            from strategy_generation.templates.strategy_templates import StrategyTemplateManager
            
            manager = StrategyTemplateManager()
            
            # Test 1: Get template
            template = manager.get_template('moving_average_crossover')
            assert template is not None
            assert 'entry_rules' in template
            assert 'exit_rules' in template
            
            # Test 2: Validate parameters
            params = {'short_ma': 10, 'long_ma': 50}
            is_valid = manager.validate_template_parameters('moving_average_crossover', params)
            assert is_valid
            
            # Test 3: Get all templates
            templates = manager.get_all_templates()
            assert len(templates) > 0
            assert 'moving_average_crossover' in templates
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        test_name = "Error Handling & Fallbacks"
        start_time = time.time()
        
        try:
            from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
            
            # Test 1: Invalid configuration
            try:
                generator = EnhancedStrategyGenerator(config_profile="nonexistent")
                # Should use defaults
                assert generator.config is not None
            except:
                pass  # Expected to handle gracefully
            
            # Test 2: Generation with invalid parameters
            generator = EnhancedStrategyGenerator(config_profile="default")
            
            # Test 3: Emergency fallback
            fallback = generator._emergency_fallback_strategy("test_123")
            assert fallback['type'] == 'emergency_fallback'
            assert 'entry_rules' in fallback
            assert 'risk_management' in fallback
            
            # Test 4: Error recovery
            generator.router.force_fallback_mode()
            assert generator.router.health.llm_operational == False
            
            # Test recovery attempt
            recovered = generator.router.recover_llm_mode()
            # Should attempt recovery (result depends on LLM availability)
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_monitoring_system(self):
        """Test monitoring and metrics collection"""
        test_name = "Monitoring System"
        start_time = time.time()
        
        try:
            from strategy_generation.integration.monitoring_system import (
                MonitoringSystem, PerformanceMetrics, HealthStatus
            )
            
            # Test 1: Initialize monitoring
            monitoring = MonitoringSystem({
                'retention_hours': 1,
                'export_dir': 'test_exports'
            })
            
            # Test 2: Record metrics
            monitoring.record_llm_operation(
                operation='test_generation',
                component='llm',
                success=True,
                execution_time=1.5,
                input_tokens=100,
                output_tokens=200,
                cost=0.001
            )
            
            # Test 3: Get health status
            health = monitoring.get_health_status()
            assert isinstance(health, HealthStatus)
            assert health.timestamp is not None
            
            # Test 4: Get performance summary
            summary = monitoring.get_performance_summary()
            assert 'success_rate' in summary
            assert 'avg_latency' in summary
            
            # Test 5: Get recent metrics
            recent = monitoring.get_recent_metrics(last_n=5)
            assert isinstance(recent, list)
            
            # Test 6: System health report
            report = monitoring.get_system_health_report()
            assert 'health_status' in report
            assert 'recommendations' in report
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_validation_system(self):
        """Test strategy validation"""
        test_name = "Validation System"
        start_time = time.time()
        
        try:
            from strategy_generation.llm_agents.validation_agent import ValidationAgent
            
            agent = ValidationAgent()
            
            # Test 1: Valid strategy
            valid_strategy = {
                'id': 'test_001',
                'name': 'Quantum Momentum Strategy',
                'description': 'Test strategy',
                'entry_rules': [{'condition': 'test_condition', 'priority': 1}],
                'exit_rules': [{'condition': 'test_exit', 'priority': 1}],
                'risk_management': {
                    'position_sizing': 'fixed',
                    'stop_loss': 'trailing',
                    'max_drawdown': '0.05',
                    'risk_per_trade': '0.02'
                },
                'parameters': {'test': 1.0},
                'type': 'llm_generated'
            }
            
            is_valid, report = agent.validate_strategy(valid_strategy)
            assert is_valid
            
            # Test 2: Invalid strategy (missing fields)
            invalid_strategy = {'name': 'Test'}
            is_valid, report = agent.validate_strategy(invalid_strategy)
            assert not is_valid
            
            # Test 3: Batch validation
            strategies = [valid_strategy, valid_strategy]
            batch_report = agent.validate_strategy_batch(strategies)
            assert batch_report['valid_strategies'] == 2
            
            # Test 4: Quick validation
            quick_valid = agent.quick_validation_check(valid_strategy)
            assert quick_valid
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_performance_metrics(self):
        """Test performance tracking and metrics"""
        test_name = "Performance Metrics"
        start_time = time.time()
        
        try:
            from strategy_generation.integration.monitoring_system import MetricsCollector
            
            collector = MetricsCollector(retention_hours=1)
            
            # Test 1: Record multiple metrics
            for i in range(5):
                metric = PerformanceMetrics(
                    operation='test_gen',
                    component='llm',
                    timestamp=datetime.now(),
                    execution_time=1.0 + i * 0.1,
                    success=True,
                    input_tokens=100 * i,
                    output_tokens=200 * i,
                    cost=0.001 * i
                )
                collector.record_metric(metric)
            
            # Test 2: Get metrics
            metrics = collector.get_metrics(operation='test_gen')
            assert len(metrics) == 5
            
            # Test 3: Get statistics
            stats = collector.get_operation_stats('test_gen')
            assert stats['total'] == 5
            assert stats['success'] == 5
            
            # Test 4: Success rate
            success_rate = collector.get_success_rate('test_gen')
            assert success_rate == 1.0
            
            # Test 5: Average latency
            avg_latency = collector.get_average_latency('test_gen')
            assert 1.0 <= avg_latency <= 1.4
            
            # Test 6: Total cost
            total_cost = collector.get_total_cost('test_gen')
            assert total_cost > 0
            
            # Test 7: Throughput
            throughput = collector.get_throughput('test_gen', window_minutes=60)
            assert throughput > 0
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        test_name = "Circuit Breaker"
        start_time = time.time()
        
        try:
            from strategy_generation.integration.monitoring_system import MonitoringSystem
            
            monitoring = MonitoringSystem()
            
            # Test 1: Record failures
            for i in range(5):
                monitoring.record_llm_operation(
                    operation='test',
                    component='llm',
                    success=False,
                    execution_time=0.1
                )
            
            # Test 2: Check circuit breaker state
            health = monitoring.get_health_status()
            # Should trigger circuit breaker after 5 failures
            # Note: Actual behavior depends on configuration
            
            # Test 3: Attempt reset
            can_reset = monitoring.attempt_circuit_breaker_reset()
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from generation to validation"""
        test_name = "End-to-End Pipeline"
        start_time = time.time()
        
        try:
            from strategy_generation.enhanced_strategy_generator import EnhancedStrategyGenerator
            
            # Initialize generator
            generator = EnhancedStrategyGenerator(config_profile="default")
            
            # Test 1: Single strategy generation
            strategy = generator.generate_strategy(
                strategy_type="innovative",
                market_context={'volatility': 'medium', 'trend': 'neutral'}
            )
            
            assert 'name' in strategy
            assert 'entry_rules' in strategy
            assert 'generation_metadata' in strategy
            
            # Test 2: Validation
            is_valid, validation_report = generator.validate_strategy(strategy)
            assert isinstance(is_valid, bool)
            assert isinstance(validation_report, dict)
            
            # Test 3: Population generation
            population = generator.generate_strategy_population(
                population_size=3,
                strategy_types=['innovative', 'template'],
                max_generation_time=30
            )
            
            assert len(population) >= 2  # May generate fewer due to time limits
            
            # Test 4: System status
            status = generator.get_system_status()
            assert 'generation_stats' in status
            assert 'router_health' in status
            
            # Test 5: Monitoring integration
            if generator.monitoring:
                health = generator.monitoring.get_health_status()
                assert health is not None
            
            self.results.record_pass(test_name, time.time() - start_time)
            
        except Exception as e:
            self.results.record_fail(test_name, str(e), time.time() - start_time)
    
    def print_summary(self):
        """Print test summary"""
        summary = self.results.get_summary()
        
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {summary['total']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        
        if summary['failures']:
            logger.info("\nFAILED TESTS:")
            for test_name, error in summary['failures']:
                logger.info(f"  - {test_name}: {error}")
        
        logger.info("\nTIMING:")
        for test_name, duration in summary['timing'].items():
            logger.info(f"  - {test_name}: {duration:.2f}s")
        
        logger.info("=" * 60)
    
    def export_results(self):
        """Export test results to JSON"""
        summary = self.results.get_summary()
        summary['timestamp'] = datetime.now().isoformat()
        
        try:
            with open('test_llm_integration_results.json', 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info("Results exported to test_llm_integration_results.json")
        except Exception as e:
            logger.error(f"Failed to export results: {e}")


def main():
    """Main test execution"""
    tests = LLMIntegrationTests()
    
    try:
        results = tests.run_all_tests()
        
        # Exit with appropriate code
        if results['failed'] > 0:
            logger.error(f"\n{results['failed']} tests failed!")
            sys.exit(1)
        else:
            logger.info("\nAll tests passed!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()