"""
Enhanced Strategy Generator with LLM Integration

Production-ready strategy generator that seamlessly integrates LLM agents
with legacy components using intelligent routing, comprehensive monitoring,
and robust error handling.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import random

# Import integration components
from strategy_generation.integration.intelligent_router import (
    IntelligentRouter, DecisionContext, ComponentType
)
from strategy_generation.integration.monitoring_system import MonitoringSystem

# Import LLM agents
from strategy_generation.llm_agents.ollama_client import OllamaClient
from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
from strategy_generation.llm_agents.feedback_refinement_agent import FeedbackRefinementAgent
from strategy_generation.llm_agents.validation_agent import ValidationAgent

# Import legacy components
from strategy_generation.templates.strategy_templates import StrategyTemplateManager
from strategy_generation.generators.backtrader_integration import BacktraderStrategyFactory
from strategy_generation.generators.genetic_operators import GeneticOperators
from strategy_generation.generators.novelty_detection import NoveltyDetector

# Import configuration
from config.llm_config import get_llm_config, LLMConfig, GenerationMode


class EnhancedStrategyGenerator:
    """
    Enhanced strategy generator with full LLM integration
    
    Features:
    - Intelligent routing between LLM and legacy components
    - Comprehensive monitoring and metrics
    - Robust error handling and fallback mechanisms
    - Configuration management
    - Circuit breaker pattern
    - Performance optimization
    """
    
    def __init__(self, config_profile: str = "default", enable_monitoring: bool = True):
        """
        Initialize enhanced strategy generator
        
        Args:
            config_profile: Configuration profile name
            enable_monitoring: Whether to enable monitoring system
        """
        self.logger = logging.getLogger('EnhancedStrategyGenerator')
        
        # Load configuration
        self.config: LLMConfig = get_llm_config(config_profile)
        self.config_profile = config_profile
        
        # Initialize monitoring
        self.monitoring: Optional[MonitoringSystem] = None
        if enable_monitoring and self.config.enable_monitoring:
            self.monitoring = MonitoringSystem({
                'retention_hours': self.config.metrics_retention_days * 24,
                'export_dir': 'monitoring_exports',
                'export_interval': 3600  # 1 hour
            })
            self.logger.info("✅ Monitoring system enabled")
        
        # Initialize intelligent router
        router_config = {
            'llm_enabled': self.config.provider.name != 'NONE',
            'llm_base_url': self.config.base_url,
            'llm_timeout': self.config.timeout,
            'llm_max_retries': self.config.max_retries,
            'fallback_threshold': self.config.fallback_threshold,
            'max_consecutive_failures': self.config.max_consecutive_failures,
            'latency_threshold': self.config.latency_threshold,
            'min_novelty_score': self.config.min_novelty_score
        }
        
        self.router = IntelligentRouter(router_config)
        
        # Initialize legacy components
        self.template_manager = StrategyTemplateManager()
        self.backtrader_factory = BacktraderStrategyFactory()
        self.genetic_operators = GeneticOperators()
        self.novelty_detector = NoveltyDetector()
        
        # Initialize LLM components (if enabled)
        self.llm_components = {}
        self._initialize_llm_components()
        
        # Performance tracking
        self.generation_stats = {
            'total_generated': 0,
            'llm_generated': 0,
            'template_generated': 0,
            'fallback_generated': 0,
            'average_generation_time': 0.0,
            'total_generation_time': 0.0
        }
        
        self.logger.info(f"EnhancedStrategyGenerator initialized with profile: {config_profile}")
        self.logger.info(f"Generation mode: {self.config.generation_mode.value}")
        
        # Register monitoring alerts if available
        if self.monitoring:
            self.monitoring.register_alert_handler(self._handle_alert)
    
    def _initialize_llm_components(self) -> None:
        """Initialize LLM components if provider is enabled"""
        if self.config.provider.name == 'NONE':
            self.logger.info("LLM provider disabled, using template-only mode")
            return
        
        try:
            # Initialize Ollama client
            self.llm_components['ollama_client'] = OllamaClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )
            
            # Test connection
            health_check = self.llm_components['ollama_client'].health_check()
            if not health_check['connected']:
                self.logger.warning("LLM server not available, falling back to templates")
                return
            
            # Initialize agents
            self.llm_components['strategy_agent'] = StrategyGenerationAgent(
                self.llm_components['ollama_client']
            )
            self.llm_components['feedback_agent'] = FeedbackRefinementAgent(
                self.llm_components['ollama_client']
            )
            self.llm_components['validation_agent'] = ValidationAgent()
            
            self.logger.info("✅ LLM components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ LLM initialization failed: {str(e)}")
            self.llm_components.clear()
    
    def generate_strategy(self, 
                         strategy_type: str = "innovative",
                         template_name: Optional[str] = None,
                         parameters: Optional[Dict[str, Any]] = None,
                         market_context: Optional[Dict[str, Any]] = None,
                         performance_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a single strategy with intelligent routing
        
        Args:
            strategy_type: Type of strategy to generate
            template_name: Optional template name for template-based generation
            parameters: Optional parameters for template-based generation
            market_context: Market context for LLM generation
            performance_requirements: Performance requirements for routing
            
        Returns:
            Generated strategy dictionary
        """
        start_time = time.time()
        operation_id = f"gen_{int(start_time)}_{random.randint(1000, 9999)}"
        
        self.logger.info(f"🚀 Starting strategy generation: {operation_id}")
        self.logger.info(f"   Strategy type: {strategy_type}")
        self.logger.info(f"   Template: {template_name or 'auto'}")
        
        try:
            # Create decision context
            context = DecisionContext(
                strategy_type=strategy_type,
                market_context=market_context,
                performance_requirements=performance_requirements
            )
            
            # Execute with intelligent routing
            result, component_used, metadata = self.router.execute_with_routing(
                operation='generate_strategy',
                context=context,
                strategy_type=strategy_type,
                template_name=template_name,
                parameters=parameters,
                market_context=market_context
            )
            
            # Check if result is an exception
            if isinstance(result, Exception):
                raise result
            
            # Record monitoring metrics
            execution_time = time.time() - start_time
            if self.monitoring:
                self.monitoring.record_llm_operation(
                    operation='generate_strategy',
                    component=component_used,
                    success=metadata.get('success', True),
                    execution_time=execution_time,
                    error_message=str(result) if not metadata.get('success', True) else None
                )
            
            # Update statistics
            self._update_generation_stats(component_used, execution_time)
            
            # Add metadata
            result['generation_metadata'] = {
                'operation_id': operation_id,
                'component_used': component_used,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat(),
                'config_profile': self.config_profile,
                'generation_mode': self.config.generation_mode.value,
                **metadata
            }
            
            self.logger.info(f"✅ Strategy generation completed: {operation_id}")
            self.logger.info(f"   Component: {component_used}")
            self.logger.info(f"   Time: {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Strategy generation failed: {str(e)}")
            
            # Record failure in monitoring
            if self.monitoring:
                self.monitoring.record_llm_operation(
                    operation='generate_strategy',
                    component='error_handler',
                    success=False,
                    execution_time=execution_time,
                    error_message=str(e)
                )
            
            # Return emergency fallback
            return self._emergency_fallback_strategy(operation_id)
    
    def generate_strategy_population(self,
                                   population_size: int = 10,
                                   strategy_types: Optional[List[str]] = None,
                                   diversity_requirements: Optional[Dict[str, Any]] = None,
                                   max_generation_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Generate a diverse population of strategies
        
        Args:
            population_size: Number of strategies to generate
            strategy_types: List of strategy types to include
            diversity_requirements: Requirements for diversity
            max_generation_time: Maximum time for generation
            
        Returns:
            List of generated strategies
        """
        start_time = time.time()
        self.logger.info(f"🌱 Generating strategy population of size {population_size}")
        
        if strategy_types is None:
            strategy_types = ['innovative', 'physics_based', 'biology_based', 
                            'game_theory', 'complexity_science', 'template']
        
        strategies = []
        generation_time_limit = max_generation_time or self.config.max_generation_time * population_size
        
        for i in range(population_size):
            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > generation_time_limit:
                self.logger.warning(f"Time limit reached after {len(strategies)} strategies")
                break
            
            # Select strategy type
            strategy_type = strategy_types[i % len(strategy_types)]
            if i > len(strategy_types):
                strategy_type = random.choice(strategy_types)
            
            try:
                # Generate individual strategy
                strategy = self.generate_strategy(
                    strategy_type=strategy_type,
                    market_context={'volatility': 'medium', 'trend': 'neutral'},
                    performance_requirements={'min_novelty': self.config.min_novelty_score}
                )
                
                strategies.append(strategy)
                
            except Exception as e:
                self.logger.error(f"Failed to generate strategy {i+1}: {str(e)}")
                # Add fallback strategy
                fallback = self._emergency_fallback_strategy(f"pop_{i:04d}")
                strategies.append(fallback)
        
        # Ensure diversity
        diverse_strategies = self._ensure_diversity(strategies, population_size)
        
        # Calculate population statistics
        population_stats = self._analyze_population(diverse_strategies)
        
        self.logger.info(f"✅ Generated population with {len(diverse_strategies)} strategies")
        self.logger.info(f"   Average novelty: {population_stats['avg_novelty']:.2f}")
        self.logger.info(f"   Average validation: {population_stats['avg_validation']:.2f}")
        self.logger.info(f"   Diversity score: {population_stats['diversity_score']:.2f}")
        
        return diverse_strategies
    
    def refine_strategy(self,
                       strategy: Dict[str, Any],
                       performance_metrics: Dict[str, Any],
                       refinement_iterations: int = 1) -> Dict[str, Any]:
        """
        Refine a strategy based on performance feedback
        
        Args:
            strategy: Strategy to refine
            performance_metrics: Performance metrics from backtesting
            refinement_iterations: Number of refinement iterations
            
        Returns:
            Refined strategy
        """
        start_time = time.time()
        self.logger.info(f"🔬 Refining strategy: {strategy.get('id', 'unknown')}")
        
        current_strategy = strategy.copy()
        
        for iteration in range(refinement_iterations):
            self.logger.info(f"   Refinement iteration {iteration + 1}/{refinement_iterations}")
            
            try:
                # Create decision context
                context = DecisionContext(
                    strategy_type=current_strategy.get('type', 'unknown'),
                    performance_requirements={'target_improvement': 0.2}
                )
                
                # Execute refinement with routing
                result, component_used, metadata = self.router.execute_with_routing(
                    operation='refine_strategy',
                    context=context,
                    strategy=current_strategy,
                    performance_metrics=performance_metrics
                )
                
                if isinstance(result, Exception):
                    self.logger.warning(f"Refinement failed, keeping current version: {str(result)}")
                    break
                
                current_strategy = result
                
                # Record monitoring
                if self.monitoring:
                    self.monitoring.record_llm_operation(
                        operation='refine_strategy',
                        component=component_used,
                        success=metadata.get('success', True),
                        execution_time=time.time() - start_time
                    )
                
            except Exception as e:
                self.logger.error(f"Refinement iteration failed: {str(e)}")
                break
        
        # Add refinement metadata
        current_strategy['refinement_metadata'] = {
            'original_id': strategy.get('id'),
            'iterations': refinement_iterations,
            'total_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Refinement completed in {time.time() - start_time:.2f}s")
        return current_strategy
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a strategy with comprehensive checks
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        start_time = time.time()
        
        try:
            # Create decision context
            context = DecisionContext(
                strategy_type=strategy.get('type', 'unknown')
            )
            
            # Execute validation with routing
            result, component_used, metadata = self.router.execute_with_routing(
                operation='validate_strategy',
                context=context,
                strategy=strategy
            )
            
            if isinstance(result, Exception):
                raise result
            
            # Extract validation results
            if isinstance(result, dict):
                is_valid = result.get('is_valid', False)
                validation_report = result.get('validation_report', {})
            else:
                is_valid = bool(result)
                validation_report = {}
            
            # Record monitoring
            if self.monitoring:
                self.monitoring.record_llm_operation(
                    operation='validate_strategy',
                    component=component_used,
                    success=is_valid,
                    execution_time=time.time() - start_time
                )
            
            self.logger.info(f"✅ Validation completed: {'PASSED' if is_valid else 'FAILED'}")
            return is_valid, validation_report
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {str(e)}")
            
            # Basic validation as fallback
            required_fields = ['name', 'entry_rules', 'exit_rules', 'risk_management']
            is_valid = all(field in strategy for field in required_fields)
            
            validation_report = {
                'overall_score': 0.5 if is_valid else 0.0,
                'failed_checks': [] if is_valid else ['Missing required fields'],
                'error': str(e)
            }
            
            return is_valid, validation_report
    
    def create_backtrader_strategy(self, strategy: Dict[str, Any]) -> type:
        """
        Create a Backtrader strategy class from generated strategy
        
        Args:
            strategy: Generated strategy dictionary
            
        Returns:
            Backtrader strategy class
        """
        self.logger.info(f"Creating Backtrader strategy for: {strategy.get('name', 'unknown')}")
        
        try:
            strategy_class = self.backtrader_factory.create_strategy_class(strategy)
            self.logger.info("✅ Backtrader strategy created successfully")
            return strategy_class
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create Backtrader strategy: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'config_profile': self.config_profile,
            'generation_mode': self.config.generation_mode.value,
            'llm_enabled': self.config.provider.name != 'NONE',
            'monitoring_enabled': self.monitoring is not None,
            'generation_stats': self.generation_stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add router health
        router_health = self.router.get_health_status()
        status['router_health'] = router_health
        
        # Add monitoring status
        if self.monitoring:
            status['monitoring_status'] = {
                'health_status': self.monitoring.get_health_status().to_dict(),
                'performance_summary': self.monitoring.get_performance_summary(),
                'circuit_breaker': self.monitoring.get_circuit_breaker_status()
            }
        
        return status
    
    def _update_generation_stats(self, component: str, execution_time: float) -> None:
        """Update generation statistics"""
        self.generation_stats['total_generated'] += 1
        self.generation_stats['total_generation_time'] += execution_time
        
        if component == 'llm':
            self.generation_stats['llm_generated'] += 1
        elif component == 'template':
            self.generation_stats['template_generated'] += 1
        elif component == 'fallback':
            self.generation_stats['fallback_generated'] += 1
        
        # Calculate average
        if self.generation_stats['total_generated'] > 0:
            self.generation_stats['average_generation_time'] = (
                self.generation_stats['total_generation_time'] / 
                self.generation_stats['total_generated']
            )
    
    def _ensure_diversity(self, strategies: List[Dict[str, Any]], 
                         target_size: int) -> List[Dict[str, Any]]:
        """Ensure population diversity"""
        if len(strategies) <= target_size:
            return strategies
        
        # Use novelty detection to select diverse strategies
        diverse_strategies = []
        remaining = strategies.copy()
        
        while len(diverse_strategies) < target_size and remaining:
            # Select most novel strategy
            most_novel = None
            highest_novelty = -1
            
            for strategy in remaining:
                novelty = self._calculate_novelty(strategy, diverse_strategies)
                if novelty > highest_novelty:
                    highest_novelty = novelty
                    most_novel = strategy
            
            if most_novel:
                diverse_strategies.append(most_novel)
                remaining.remove(most_novel)
        
        return diverse_strategies
    
    def _calculate_novelty(self, strategy: Dict[str, Any], 
                          population: List[Dict[str, Any]]) -> float:
        """Calculate novelty score compared to population"""
        if not population:
            return 1.0
        
        # Base novelty from metadata
        base_novelty = 0.5
        
        # Check strategy type
        strategy_type = strategy.get('type', '')
        if strategy_type == 'llm_generated':
            base_novelty += 0.3
        elif strategy_type == 'template_based':
            base_novelty += 0.1
        
        # Check for innovative concepts
        name = strategy.get('name', '').lower()
        if any(concept in name for concept in ['quantum', 'neural', 'fractal', 'chaos', 'swarm']):
            base_novelty += 0.2
        
        # Compare with existing population
        similarity_scores = []
        for existing in population:
            similarity = self._calculate_similarity(strategy, existing)
            similarity_scores.append(similarity)
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            base_novelty += (1.0 - avg_similarity) * 0.3
        
        return min(1.0, max(0.1, base_novelty))
    
    def _calculate_similarity(self, strategy1: Dict[str, Any], 
                             strategy2: Dict[str, Any]) -> float:
        """Calculate similarity between two strategies"""
        similarity = 0.0
        
        # Compare types
        if strategy1.get('type') == strategy2.get('type'):
            similarity += 0.3
        
        # Compare names (simple string similarity)
        name1 = strategy1.get('name', '').lower()
        name2 = strategy2.get('name', '').lower()
        if name1 == name2:
            similarity += 0.4
        elif any(word in name2 for word in name1.split()):
            similarity += 0.2
        
        # Compare rule count
        rules1 = len(strategy1.get('entry_rules', [])) + len(strategy1.get('exit_rules', []))
        rules2 = len(strategy2.get('entry_rules', [])) + len(strategy2.get('exit_rules', []))
        if rules1 == rules2:
            similarity += 0.3
        
        return min(1.0, similarity)
    
    def _analyze_population(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze population statistics"""
        if not strategies:
            return {'avg_novelty': 0.0, 'avg_validation': 0.0, 'diversity_score': 0.0}
        
        novelty_scores = []
        validation_scores = []
        
        for strategy in strategies:
            # Get novelty from metadata
            metadata = strategy.get('metadata', {})
            novelty = metadata.get('novelty_score', 0.5)
            novelty_scores.append(novelty)
            
            # Get validation score
            validation_report = strategy.get('validation_report', {})
            validation = validation_report.get('overall_score', 0.5)
            validation_scores.append(validation)
        
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        avg_validation = sum(validation_scores) / len(validation_scores) if validation_scores else 0.0
        
        # Calculate diversity score
        diversity_score = self._calculate_population_diversity(strategies)
        
        return {
            'avg_novelty': avg_novelty,
            'avg_validation': avg_validation,
            'diversity_score': diversity_score,
            'population_size': len(strategies)
        }
    
    def _calculate_population_diversity(self, strategies: List[Dict[str, Any]]) -> float:
        """Calculate population diversity score"""
        if len(strategies) < 2:
            return 0.0
        
        # Count unique types
        types = set(s.get('type', 'unknown') for s in strategies)
        type_diversity = len(types) / len(strategies)
        
        # Count unique names (rough measure)
        names = set(s.get('name', '') for s in strategies)
        name_diversity = len(names) / len(strategies)
        
        # Average pairwise novelty
        total_novelty = 0
        comparisons = 0
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i+1:]:
                total_novelty += self._calculate_similarity(s1, s2)
                comparisons += 1
        
        avg_similarity = total_novelty / comparisons if comparisons > 0 else 0.0
        novelty_diversity = 1.0 - avg_similarity
        
        # Combine diversity measures
        diversity_score = (type_diversity + name_diversity + novelty_diversity) / 3
        
        return diversity_score
    
    def _emergency_fallback_strategy(self, operation_id: str) -> Dict[str, Any]:
        """Generate emergency fallback strategy"""
        self.logger.warning("Generating emergency fallback strategy")
        
        return {
            'id': f"emergency_{operation_id}",
            'name': "Emergency Conservative Strategy",
            'type': 'emergency_fallback',
            'description': 'Ultra-conservative emergency strategy',
            'entry_rules': [{
                'condition': 'price_above_sma_200 AND volume > average_volume',
                'priority': 1,
                'weight': 1.0
            }],
            'exit_rules': [{
                'condition': 'price_below_sma_50 OR profit > 5_percent',
                'priority': 1,
                'weight': 1.0
            }],
            'risk_management': {
                'position_sizing': 'conservative',
                'stop_loss': 'tight_2_percent',
                'take_profit': 'moderate',
                'max_drawdown': '0.02',
                'risk_per_trade': '0.01'
            },
            'parameters': {
                'emergency_mode': True,
                'conservatism': 0.99,
                'fallback_reason': 'All generation methods failed'
            },
            'metadata': {
                'generated_by': 'EnhancedStrategyGenerator',
                'generation_method': 'emergency_fallback',
                'timestamp': datetime.now().isoformat(),
                'novelty_score': 0.1,
                'validation_status': 'emergency'
            },
            'validation_report': {
                'overall_score': 0.3,
                'is_valid': True,
                'failed_checks': ['Emergency fallback'],
                'warnings': ['Generated as emergency fallback']
            }
        }
    
    def _handle_alert(self, alert_data: Dict[str, Any]) -> None:
        """Handle monitoring alerts"""
        alert_type = alert_data.get('alert_type', 'unknown')
        data = alert_data.get('data', {})
        
        self.logger.warning(f"🚨 ALERT: {alert_type} - {data}")
        
        # Handle specific alert types
        if alert_type == 'circuit_breaker_opened':
            self.logger.error("Circuit breaker opened - LLM operations suspended")
        
        elif alert_type == 'circuit_breaker_closed':
            self.logger.info("Circuit breaker closed - LLM operations resumed")
        
        elif alert_type == 'low_health_score':
            health_score = data.get('health_score', 0)
            self.logger.warning(f"Low LLM health score: {health_score:.2f}")
        
        elif alert_type == 'high_latency':
            latency = data.get('avg_latency', 0)
            self.logger.warning(f"High LLM latency: {latency:.2f}s")
        
        elif alert_type == 'low_success_rate':
            success_rate = data.get('success_rate', 0)
            self.logger.warning(f"Low LLM success rate: {success_rate:.2%}")
    
    def export_system_report(self, filepath: Optional[str] = None) -> bool:
        """Export comprehensive system report"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"system_report_{timestamp}.json"
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'config_profile': self.config_profile,
                'system_status': self.get_system_status(),
                'generation_stats': self.generation_stats,
                'config': self.config.to_dict()
            }
            
            # Add monitoring data if available
            if self.monitoring:
                report['monitoring'] = {
                    'health_status': self.monitoring.get_health_status().to_dict(),
                    'performance_summary': self.monitoring.get_performance_summary(),
                    'recent_metrics': self.monitoring.get_recent_metrics(last_n=10),
                    'system_health_report': self.monitoring.get_system_health_report()
                }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"✅ System report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export system report: {str(e)}")
            return False