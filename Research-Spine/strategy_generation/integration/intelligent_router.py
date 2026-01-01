"""
Intelligent Router for LLM and Legacy Component Integration

Provides dynamic, context-aware selection between LLM agents and legacy components
with comprehensive error handling, monitoring, and fallback mechanisms.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import random

from strategy_generation.llm_agents.ollama_client import OllamaClient
from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
from strategy_generation.llm_agents.feedback_refinement_agent import FeedbackRefinementAgent
from strategy_generation.llm_agents.validation_agent import ValidationAgent
from strategy_generation.templates.strategy_templates import StrategyTemplateManager


class ComponentType(Enum):
    """Component types for routing decisions"""
    LLM_STRATEGY_GENERATION = "llm_strategy_generation"
    LLM_FEEDBACK_REFINEMENT = "llm_feedback_refinement"
    LLM_VALIDATION = "llm_validation"
    TEMPLATE_GENERATION = "template_generation"
    FALLBACK = "fallback"


class DecisionContext:
    """Context for routing decisions"""
    
    def __init__(self, strategy_type: str = None, market_context: Dict[str, Any] = None, 
                 performance_requirements: Dict[str, Any] = None):
        self.strategy_type = strategy_type
        self.market_context = market_context or {}
        self.performance_requirements = performance_requirements or {}
        self.timestamp = datetime.now()
        self.llm_available = False
        self.llm_health_score = 0.0
        self.system_load = 0.0
        self.previous_failures = 0


class ComponentHealth:
    """Track health and performance of components"""
    
    def __init__(self):
        self.llm_operational = False
        self.llm_latency = 0.0
        self.llm_success_rate = 1.0
        self.llm_failure_count = 0
        self.llm_total_requests = 0
        self.template_success_rate = 1.0
        self.last_health_check = None
        self.consecutive_failures = 0
        
    def update_llm_metrics(self, success: bool, latency: float):
        """Update LLM performance metrics"""
        self.llm_total_requests += 1
        self.llm_latency = latency
        
        if success:
            self.llm_success_rate = (self.llm_success_rate * (self.llm_total_requests - 1) + 1) / self.llm_total_requests
            self.consecutive_failures = 0
        else:
            self.llm_failure_count += 1
            self.llm_success_rate = (self.llm_success_rate * (self.llm_total_requests - 1)) / self.llm_total_requests
            self.consecutive_failures += 1
            
        self.last_health_check = datetime.now()
        
    def get_health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)"""
        if self.llm_total_requests == 0:
            return 0.5  # Neutral score when no data
            
        # Base score on success rate
        base_score = self.llm_success_rate
        
        # Penalize for high latency
        latency_penalty = min(self.llm_latency / 5.0, 0.3)  # Normalize to 5 seconds max
        
        # Penalize for consecutive failures
        failure_penalty = min(self.consecutive_failures * 0.1, 0.4)
        
        health_score = base_score - latency_penalty - failure_penalty
        return max(0.0, min(1.0, health_score))


class IntelligentRouter:
    """
    Intelligent router that dynamically selects between LLM and legacy components
    based on context, health, and performance requirements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the intelligent router
        
        Args:
            config: Configuration dictionary with LLM and routing settings
        """
        self.logger = logging.getLogger('IntelligentRouter')
        self.config = config
        
        # Component health tracking
        self.health = ComponentHealth()
        
        # LLM configuration
        self.llm_enabled = config.get('llm_enabled', True)
        self.llm_base_url = config.get('llm_base_url', 'http://localhost:11434')
        self.llm_timeout = config.get('llm_timeout', 30)
        self.llm_max_retries = config.get('llm_max_retries', 3)
        
        # Routing configuration
        self.fallback_threshold = config.get('fallback_threshold', 0.6)
        self.max_consecutive_failures = config.get('max_consecutive_failures', 3)
        self.latency_threshold = config.get('latency_threshold', 5.0)
        
        # Performance requirements
        self.min_novelty_score = config.get('min_novelty_score', 0.7)
        self.max_generation_time = config.get('max_generation_time', 10.0)
        
        # Initialize components
        self.llm_components = {}
        self.legacy_components = {}
        self._initialize_components()
        
        self.logger.info(f"IntelligentRouter initialized with LLM enabled: {self.llm_enabled}")
        
    def _initialize_components(self) -> None:
        """Initialize LLM and legacy components"""
        # Initialize legacy components
        self.legacy_components['template_manager'] = StrategyTemplateManager()
        
        # Initialize LLM components if enabled
        if self.llm_enabled:
            try:
                self.llm_components['ollama_client'] = OllamaClient(
                    base_url=self.llm_base_url,
                    timeout=self.llm_timeout
                )
                
                # Test connection
                health_check = self.llm_components['ollama_client'].health_check()
                if health_check['connected']:
                    self.health.llm_operational = True
                    self.health.consecutive_failures = 0
                    
                    # Initialize agents
                    self.llm_components['strategy_agent'] = StrategyGenerationAgent(
                        self.llm_components['ollama_client']
                    )
                    self.llm_components['feedback_agent'] = FeedbackRefinementAgent(
                        self.llm_components['ollama_client']
                    )
                    self.llm_components['validation_agent'] = ValidationAgent()
                    
                    self.logger.info("✅ LLM components initialized successfully")
                else:
                    self.logger.warning("⚠️ LLM connection failed, will use fallback")
                    self.health.llm_operational = False
                    
            except Exception as e:
                self.logger.error(f"❌ LLM initialization failed: {str(e)}")
                self.health.llm_operational = False
        else:
            self.logger.info("LLM disabled, using template-based approach")
            self.health.llm_operational = False
    
    def make_routing_decision(self, context: DecisionContext) -> ComponentType:
        """
        Make intelligent routing decision based on context and health
        
        Args:
            context: Decision context with strategy requirements
            
        Returns:
            Component type to use for the request
        """
        # Update context with current health
        context.llm_available = self.health.llm_operational
        context.llm_health_score = self.health.get_health_score()
        
        # Check for critical failures
        if self.health.consecutive_failures >= self.max_consecutive_failures:
            self.logger.warning(f"Too many consecutive failures ({self.health.consecutive_failures}), forcing fallback")
            return ComponentType.FALLBACK
        
        # Check LLM availability and health
        if not self.llm_enabled or not self.health.llm_operational:
            return self._fallback_decision(context)
        
        health_score = self.health.get_health_score()
        
        # Check if health is below threshold
        if health_score < self.fallback_threshold:
            self.logger.warning(f"LLM health low ({health_score:.2f}), considering fallback")
            return self._fallback_decision(context)
        
        # Check latency
        if self.health.llm_latency > self.latency_threshold:
            self.logger.warning(f"LLM latency high ({self.health.llm_latency:.2f}s), considering fallback")
            return self._fallback_decision(context)
        
        # Make decision based on strategy type and requirements
        return self._context_aware_decision(context, health_score)
    
    def _context_aware_decision(self, context: DecisionContext, health_score: float) -> ComponentType:
        """Make context-aware routing decision"""
        
        # High-priority innovative strategies should use LLM if healthy
        if context.strategy_type and any(keyword in context.strategy_type.lower() 
                                        for keyword in ['innovative', 'quantum', 'neural', 'physics', 'biology']):
            if health_score > 0.7:
                self.logger.info(f"Using LLM for innovative strategy (health: {health_score:.2f})")
                return ComponentType.LLM_STRATEGY_GENERATION
            else:
                self.logger.warning(f"LLM health insufficient for innovative strategy, using templates")
                return ComponentType.TEMPLATE_GENERATION
        
        # Performance-critical requirements
        if context.performance_requirements.get('min_novelty', 0) > 0.8:
            if health_score > 0.8:
                self.logger.info(f"High novelty requirement, using LLM (health: {health_score:.2f})")
                return ComponentType.LLM_STRATEGY_GENERATION
            else:
                self.logger.warning(f"Cannot meet high novelty requirement with current LLM health")
                return ComponentType.FALLBACK
        
        # Mixed approach for balanced requirements
        if health_score > 0.75:
            # 80% chance of LLM for good health
            if random.random() > 0.2:
                return ComponentType.LLM_STRATEGY_GENERATION
            else:
                return ComponentType.TEMPLATE_GENERATION
        elif health_score > 0.6:
            # 50% chance of LLM for moderate health
            if random.random() > 0.5:
                return ComponentType.LLM_STRATEGY_GENERATION
            else:
                return ComponentType.TEMPLATE_GENERATION
        else:
            # Use templates for poor health
            return ComponentType.TEMPLATE_GENERATION
    
    def _fallback_decision(self, context: DecisionContext) -> ComponentType:
        """Determine appropriate fallback component"""
        # If we have strategy type, try template generation
        if context.strategy_type:
            return ComponentType.TEMPLATE_GENERATION
        else:
            return ComponentType.FALLBACK
    
    def execute_with_routing(self, operation: str, context: DecisionContext, 
                           *args, **kwargs) -> Tuple[Any, ComponentType, Dict[str, Any]]:
        """
        Execute an operation using intelligent routing
        
        Args:
            operation: Operation name ('generate_strategy', 'refine_strategy', 'validate_strategy')
            context: Decision context
            *args, **kwargs: Operation arguments
            
        Returns:
            Tuple of (result, component_used, metadata)
        """
        start_time = time.time()
        component_used = None
        metadata = {}
        
        try:
            # Make routing decision
            component_type = self.make_routing_decision(context)
            
            # Execute based on component type
            if component_type == ComponentType.LLM_STRATEGY_GENERATION:
                result = self._execute_llm_strategy_generation(*args, **kwargs)
                component_used = 'llm'
                
            elif component_type == ComponentType.LLM_FEEDBACK_REFINEMENT:
                result = self._execute_llm_feedback_refinement(*args, **kwargs)
                component_used = 'llm'
                
            elif component_type == ComponentType.LLM_VALIDATION:
                result = self._execute_llm_validation(*args, **kwargs)
                component_used = 'llm'
                
            elif component_type == ComponentType.TEMPLATE_GENERATION:
                result = self._execute_template_generation(*args, **kwargs)
                component_used = 'template'
                
            else:  # FALLBACK
                result = self._execute_fallback(*args, **kwargs)
                component_used = 'fallback'
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update health metrics
            success = not isinstance(result, Exception)
            if component_used == 'llm':
                self.health.update_llm_metrics(success, execution_time)
            
            # Build metadata
            metadata = {
                'component_used': component_used,
                'execution_time': execution_time,
                'health_score': self.health.get_health_score(),
                'timestamp': datetime.now().isoformat(),
                'success': success
            }
            
            if success:
                self.logger.info(f"✅ Operation '{operation}' completed using {component_used} in {execution_time:.2f}s")
            else:
                self.logger.error(f"❌ Operation '{operation}' failed using {component_used}")
            
            return result, component_used, metadata
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Critical failure in execute_with_routing: {str(e)}")
            
            # Try emergency fallback
            try:
                self.logger.info("🔄 Attempting emergency fallback...")
                result = self._execute_fallback(*args, **kwargs)
                metadata = {
                    'component_used': 'emergency_fallback',
                    'execution_time': execution_time,
                    'health_score': self.health.get_health_score(),
                    'timestamp': datetime.now().isoformat(),
                    'success': True,
                    'error': str(e)
                }
                return result, 'emergency_fallback', metadata
            except Exception as fallback_error:
                metadata = {
                    'component_used': 'none',
                    'execution_time': execution_time,
                    'health_score': self.health.get_health_score(),
                    'timestamp': datetime.now().isoformat(),
                    'success': False,
                    'error': str(e),
                    'fallback_error': str(fallback_error)
                }
                return Exception(f"All execution methods failed: {str(e)} | {str(fallback_error)}"), 'none', metadata
    
    def _execute_llm_strategy_generation(self, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """Execute LLM strategy generation"""
        if 'strategy_agent' not in self.llm_components:
            raise RuntimeError("LLM strategy agent not available")
        
        market_context = kwargs.get('market_context', {'volatility': 'medium', 'trend': 'neutral'})
        constraints = kwargs.get('constraints', {'risk_level': 'moderate'})
        
        strategy = self.llm_components['strategy_agent'].generate_strategy(
            strategy_type=strategy_type,
            market_context=market_context,
            constraints=constraints
        )
        
        # Validate the strategy
        if 'validation_agent' in self.llm_components:
            is_valid, validation_report = self.llm_components['validation_agent'].validate_strategy(strategy)
            strategy['validation_report'] = validation_report
            strategy['validation_status'] = 'passed' if is_valid else 'failed'
        
        return strategy
    
    def _execute_llm_feedback_refinement(self, strategy: Dict[str, Any], 
                                       performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM feedback refinement"""
        if 'feedback_agent' not in self.llm_components:
            raise RuntimeError("LLM feedback agent not available")
        
        refined_strategy = self.llm_components['feedback_agent'].refine_strategy(
            strategy=strategy,
            performance_metrics=performance_metrics
        )
        
        # Validate refined strategy
        if 'validation_agent' in self.llm_components:
            is_valid, validation_report = self.llm_components['validation_agent'].validate_strategy(refined_strategy)
            refined_strategy['validation_report'] = validation_report
        
        return refined_strategy
    
    def _execute_llm_validation(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM validation"""
        if 'validation_agent' not in self.llm_components:
            raise RuntimeError("LLM validation agent not available")
        
        is_valid, validation_report = self.llm_components['validation_agent'].validate_strategy(strategy)
        
        return {
            'is_valid': is_valid,
            'validation_report': validation_report,
            'strategy': strategy
        }
    
    def _execute_template_generation(self, strategy_type: str, **kwargs) -> Dict[str, Any]:
        """Execute template-based generation"""
        template_manager = self.legacy_components['template_manager']
        
        # Select appropriate template based on strategy type
        if 'moving_average' in strategy_type.lower():
            template_name = 'moving_average_crossover'
        elif 'rsi' in strategy_type.lower():
            template_name = 'rsi_mean_reversion'
        elif 'bollinger' in strategy_type.lower():
            template_name = 'bollinger_bands'
        else:
            template_name = 'moving_average_crossover'  # Default
        
        template = template_manager.get_template(template_name)
        parameters = kwargs.get('parameters', template.get('default_parameters', {}))
        
        # Generate strategy structure
        strategy = {
            'id': f"template_{int(time.time())}_{random.randint(1000, 9999)}",
            'name': f"{template_name.replace('_', ' ').title()} Strategy",
            'type': 'template_based',
            'template': template_name,
            'parameters': parameters,
            'entry_rules': self._generate_template_rules(template.get('entry_rules', []), parameters),
            'exit_rules': self._generate_template_rules(template.get('exit_rules', []), parameters),
            'risk_management': template.get('risk_management', {
                'position_sizing': 'fixed_percentage',
                'stop_loss': 'trailing_5_percent',
                'take_profit': 'risk_reward_2_to_1',
                'max_drawdown': '0.05',
                'risk_per_trade': '0.02'
            }),
            'metadata': {
                'generated_by': 'IntelligentRouter',
                'generation_method': 'template_fallback',
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return strategy
    
    def _execute_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute ultimate fallback strategy"""
        self.logger.warning("Using ultimate fallback strategy generation")
        
        return {
            'id': f"fallback_{int(time.time())}_{random.randint(1000, 9999)}",
            'name': "Conservative Fallback Strategy",
            'type': 'fallback',
            'description': 'Ultra-conservative fallback trading strategy',
            'entry_rules': [{
                'condition': 'price_above_sma_200 AND volume_above_average',
                'priority': 1,
                'weight': 1.0
            }],
            'exit_rules': [{
                'condition': 'price_below_sma_50 OR profit_target_reached',
                'priority': 1,
                'weight': 1.0
            }],
            'risk_management': {
                'position_sizing': 'conservative_fixed',
                'stop_loss': 'tight_2_percent',
                'take_profit': 'moderate_3_to_1',
                'max_drawdown': '0.02',
                'risk_per_trade': '0.01'
            },
            'parameters': {
                'conservatism': 0.95,
                'fallback_mode': True,
                'ultra_safe': True
            },
            'metadata': {
                'generated_by': 'IntelligentRouter',
                'generation_method': 'ultimate_fallback',
                'timestamp': datetime.now().isoformat(),
                'reason': 'All other methods failed or unavailable'
            }
        }
    
    def _generate_template_rules(self, rule_configs: List[Dict], parameters: Dict) -> List[Dict[str, Any]]:
        """Generate rules from template configuration"""
        rules = []
        for i, rule_config in enumerate(rule_configs):
            condition = rule_config.get('condition', 'default_condition')
            
            # Replace parameter placeholders
            for param_name, param_value in parameters.items():
                condition = condition.replace(f'{{{param_name}}}', str(param_value))
            
            rules.append({
                'condition': condition,
                'priority': i + 1,
                'weight': rule_config.get('weight', 1.0)
            })
        
        return rules if rules else [{
            'condition': 'default_entry_condition',
            'priority': 1,
            'weight': 1.0
        }]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'llm_operational': self.health.llm_operational,
            'llm_health_score': self.health.get_health_score(),
            'llm_success_rate': self.health.llm_success_rate,
            'llm_latency': self.health.llm_latency,
            'llm_failure_count': self.health.llm_failure_count,
            'consecutive_failures': self.health.consecutive_failures,
            'last_health_check': self.health.last_health_check.isoformat() if self.health.last_health_check else None,
            'router_config': {
                'fallback_threshold': self.fallback_threshold,
                'max_consecutive_failures': self.max_consecutive_failures,
                'latency_threshold': self.latency_threshold,
                'llm_enabled': self.llm_enabled
            }
        }
    
    def force_fallback_mode(self) -> None:
        """Force system into fallback mode"""
        self.logger.warning("Forcing fallback mode")
        self.health.llm_operational = False
        self.health.consecutive_failures = self.max_consecutive_failures
    
    def recover_llm_mode(self) -> bool:
        """Attempt to recover LLM mode"""
        self.logger.info("Attempting LLM recovery...")
        
        try:
            if 'ollama_client' in self.llm_components:
                health_check = self.llm_components['ollama_client'].health_check()
                
                if health_check['connected']:
                    self.health.llm_operational = True
                    self.health.consecutive_failures = 0
                    self.health.llm_failure_count = 0
                    self.logger.info("✅ LLM recovery successful")
                    return True
                else:
                    self.logger.warning("❌ LLM still not connected")
                    return False
            else:
                # Try to reinitialize
                self._initialize_components()
                return self.health.llm_operational
                
        except Exception as e:
            self.logger.error(f"❌ LLM recovery failed: {str(e)}")
            return False