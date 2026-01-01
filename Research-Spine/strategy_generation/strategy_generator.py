"""
Strategy Generator Module

Handles the generation of trading strategies using both LLM agents and traditional templates.
Provides dynamic selection between innovative LLM-generated strategies and legacy template-based approaches.
"""

import logging
from typing import Dict, Any, List, Optional
import random

# Import the new components
from strategy_generation.generators.genetic_operators import GeneticOperators
from strategy_generation.generators.novelty_detection import NoveltyDetector
from strategy_generation.templates.strategy_templates import StrategyTemplateManager
from strategy_generation.generators.backtrader_integration import BacktraderStrategyFactory

# Import LLM agents
from strategy_generation.llm_agents.ollama_client import OllamaClient
from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
from strategy_generation.llm_agents.feedback_refinement_agent import FeedbackRefinementAgent
from strategy_generation.llm_agents.validation_agent import ValidationAgent

class StrategyGenerator:
    """Main class for generating trading strategies with LLM integration"""
    
    def __init__(self, use_llm: bool = True, ollama_base_url: str = "http://localhost:11434"):
        self.logger = logging.getLogger('StrategyGenerator')
        self.logger.info("StrategyGenerator initialized with LLM integration")
        
        # LLM integration configuration
        self.use_llm = use_llm
        self.llm_operational = False
        self.llm_fallback_mode = False
        
        # Initialize legacy components
        self.genetic_operators = GeneticOperators()
        self.novelty_detector = NoveltyDetector()
        self.template_manager = StrategyTemplateManager()
        self.backtrader_factory = BacktraderStrategyFactory()
        
        # Initialize LLM components if enabled
        if self.use_llm:
            self._initialize_llm_components(ollama_base_url)
        else:
            self.logger.info("LLM integration disabled - using template-based approach")
            self.llm_fallback_mode = True
    
    def _initialize_llm_components(self, ollama_base_url: str) -> None:
        """Initialize LLM components and establish connection"""
        try:
            self.logger.info("Initializing LLM components...")
            
            # Initialize Ollama client
            self.ollama_client = OllamaClient(base_url=ollama_base_url)
            
            # Check connection
            health = self.ollama_client.health_check()
            if health['connected']:
                self.logger.info("✅ Ollama client connected successfully")
                
                # Initialize LLM agents
                self.strategy_agent = StrategyGenerationAgent(self.ollama_client)
                self.feedback_agent = FeedbackRefinementAgent(self.ollama_client)
                self.validation_agent = ValidationAgent()
                
                self.llm_operational = True
                self.llm_fallback_mode = False
                
                self.logger.info("✅ LLM agents initialized and operational")
                self.logger.info("🚀 Strategy generation system ready for autonomous innovation")
                
            else:
                self.logger.warning("⚠️  Ollama server not connected - falling back to template-based approach")
                self.llm_operational = False
                self.llm_fallback_mode = True
                
        except Exception as e:
            self.logger.error(f"❌ LLM initialization failed: {str(e)}")
            self.logger.warning("⚠️  Falling back to template-based strategy generation")
            self.llm_operational = False
            self.llm_fallback_mode = True
    
    def _select_generation_method(self, strategy_type: Optional[str] = None) -> str:
        """
        Dynamically select between LLM and legacy generation methods
        
        Args:
            strategy_type: Optional strategy type hint
            
        Returns:
            Generation method to use ('llm', 'template', or 'hybrid')
        """
        # If LLM is operational and we want innovative strategies, use LLM
        if self.llm_operational and self.use_llm:
            if strategy_type and strategy_type.startswith('llm_'):
                return 'llm'
            elif strategy_type and any(innovative in strategy_type for innovative in ['quantum', 'neural', 'evolutionary', 'fractal']):
                return 'llm'
            else:
                # Use LLM for most cases, but allow some template-based strategies for diversity
                return 'llm' if random.random() > 0.2 else 'template'
        else:
            return 'template'
    
    def generate_strategy(self, 
                         strategy_type: str = "innovative",
                         template_name: Optional[str] = None,
                         parameters: Optional[Dict[str, Any]] = None,
                         market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a trading strategy using dynamic selection between LLM and template methods
        
        Args:
            strategy_type: Type of strategy to generate
            template_name: Optional template name for template-based generation
            parameters: Optional parameters for template-based generation
            market_context: Optional market context for LLM generation
            
        Returns:
            Dictionary containing the generated strategy
        """
        generation_method = self._select_generation_method(strategy_type)
        
        self.logger.info(f"Generating strategy using {generation_method} method")
        self.logger.info(f"Strategy type: {strategy_type}")
        
        try:
            if generation_method == 'llm':
                return self._generate_llm_strategy(strategy_type, market_context)
            else:
                return self._generate_template_strategy(strategy_type, template_name, parameters)
                
        except Exception as e:
            self.logger.error(f"❌ Primary generation method failed: {str(e)}")
            self.logger.info("🔄 Attempting fallback generation method...")
            
            # Try fallback method
            fallback_method = 'template' if generation_method == 'llm' else 'llm'
            
            try:
                if fallback_method == 'llm' and self.llm_operational:
                    return self._generate_llm_strategy(strategy_type, market_context)
                else:
                    return self._generate_template_strategy(strategy_type, template_name, parameters)
                    
            except Exception as fallback_error:
                self.logger.error(f"❌ Fallback generation also failed: {str(fallback_error)}")
                raise RuntimeError(f"Both primary and fallback strategy generation failed: {str(e)} | {str(fallback_error)}")
    
    def _generate_llm_strategy(self, 
                              strategy_type: str = "innovative",
                              market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate strategy using LLM agents
        
        Args:
            strategy_type: Type of innovative strategy to generate
            market_context: Current market conditions
            
        Returns:
            LLM-generated strategy dictionary
        """
        if not self.llm_operational or not hasattr(self, 'strategy_agent'):
            raise RuntimeError("LLM components not available for strategy generation")
        
        self.logger.info(f"🧠 Generating LLM-powered strategy: {strategy_type}")
        
        # Generate strategy using LLM agent
        strategy = self.strategy_agent.generate_strategy(
            strategy_type=strategy_type,
            market_context=market_context or {'volatility': 'medium', 'trend': 'neutral'},
            constraints={'risk_level': 'moderate', 'innovation_requirement': 'high'}
        )
        
        # Validate the generated strategy
        is_valid, validation_report = self.validation_agent.validate_strategy(strategy)
        
        if is_valid:
            self.logger.info(f"✅ LLM strategy validation passed (score: {validation_report['overall_score']:.2f})")
            strategy['validation_report'] = validation_report
            strategy['generation_method'] = 'llm_autonomous'
        else:
            self.logger.warning(f"⚠️  LLM strategy validation failed (score: {validation_report['overall_score']:.2f})")
            strategy['validation_report'] = validation_report
            strategy['generation_method'] = 'llm_with_warnings'
        
        return strategy
    
    def _generate_template_strategy(self, 
                                   strategy_type: str = "template",
                                   template_name: Optional[str] = None,
                                   parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate strategy using traditional templates
        
        Args:
            strategy_type: Type of strategy
            template_name: Name of template to use
            parameters: Parameters for the template
            
        Returns:
            Template-generated strategy dictionary
        """
        self.logger.info(f"📝 Generating template-based strategy")
        
        # Use default template if not specified
        if template_name is None:
            template_name = "moving_average_crossover"
        
        # Use default parameters if not specified
        if parameters is None:
            parameters = self._get_default_template_parameters(template_name)
        
        # Validate parameters against template
        if not self.template_manager.validate_template_parameters(template_name, parameters):
            self.logger.error(f"Invalid parameters for template: {template_name}")
            raise ValueError(f"Invalid parameters for template: {template_name}")
        
        # Generate template-based strategy
        strategy = {
            'id': f"template_{random.randint(1000, 9999)}",
            'name': f"{template_name.replace('_', ' ').title()} Strategy",
            'type': 'template_based',
            'template': template_name,
            'parameters': parameters,
            'entry_rules': self._generate_template_entry_rules(template_name, parameters),
            'exit_rules': self._generate_template_exit_rules(template_name, parameters),
            'risk_management': self._generate_template_risk_management(template_name),
            'metadata': {
                'generated_by': 'StrategyGenerator',
                'version': '2.0',
                'generation_method': 'template_based',
                'template_version': self.template_manager.get_template(template_name).get('version', '1.0')
            }
        }
        
        self.logger.info(f"✅ Template strategy generated: {strategy['name']}")
        return strategy
    
    def _get_default_template_parameters(self, template_name: str) -> Dict[str, Any]:
        """Get default parameters for a template"""
        template = self.template_manager.get_template(template_name)
        return template.get('default_parameters', {})
    
    def _generate_template_entry_rules(self, template_name: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate entry rules for template-based strategy"""
        template = self.template_manager.get_template(template_name)
        entry_rules_config = template.get('entry_rules', [])
        
        rules = []
        for i, rule_config in enumerate(entry_rules_config):
            condition = rule_config['condition']
            
            # Replace parameter placeholders
            for param_name, param_value in parameters.items():
                condition = condition.replace(f'{{{param_name}}}', str(param_value))
            
            rules.append({
                'condition': condition,
                'priority': i + 1,
                'weight': rule_config.get('weight', 1.0)
            })
        
        return rules if rules else [{
            'condition': f'{template_name}_entry_condition',
            'priority': 1,
            'weight': 1.0
        }]
    
    def _generate_template_exit_rules(self, template_name: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate exit rules for template-based strategy"""
        template = self.template_manager.get_template(template_name)
        exit_rules_config = template.get('exit_rules', [])
        
        rules = []
        for i, rule_config in enumerate(exit_rules_config):
            condition = rule_config['condition']
            
            # Replace parameter placeholders
            for param_name, param_value in parameters.items():
                condition = condition.replace(f'{{{param_name}}}', str(param_value))
            
            rules.append({
                'condition': condition,
                'priority': i + 1,
                'weight': rule_config.get('weight', 1.0)
            })
        
        return rules if rules else [{
            'condition': f'{template_name}_exit_condition',
            'priority': 1,
            'weight': 1.0
        }]
    
    def _generate_template_risk_management(self, template_name: str) -> Dict[str, Any]:
        """Generate risk management for template-based strategy"""
        template = self.template_manager.get_template(template_name)
        risk_mgmt = template.get('risk_management', {})
        
        # Ensure default risk management values
        default_risk_mgmt = {
            'position_sizing': risk_mgmt.get('position_sizing', 'fixed_percentage'),
            'stop_loss': risk_mgmt.get('stop_loss', 'trailing_5_percent'),
            'take_profit': risk_mgmt.get('take_profit', 'risk_reward_2_to_1'),
            'max_drawdown': risk_mgmt.get('max_drawdown', '0.05'),
            'risk_per_trade': risk_mgmt.get('risk_per_trade', '0.02')
        }
        
        return default_risk_mgmt
    
    def generate_strategy_population(self, 
                                    population_size: int = 10,
                                    strategy_types: Optional[List[str]] = None,
                                    diversity_requirements: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate a diverse population of strategies using both LLM and template methods
        
        Args:
            population_size: Number of strategies to generate
            strategy_types: Optional list of strategy types to include
            diversity_requirements: Requirements for strategy diversity
            
        Returns:
            List of generated strategies
        """
        self.logger.info(f"🌱 Generating diverse strategy population of size: {population_size}")
        
        if strategy_types is None:
            strategy_types = ['innovative', 'physics_based', 'biology_based', 'game_theory', 'complexity_science', 'template']
        
        strategies = []
        
        # Generate mixed population with both LLM and template strategies
        for i in range(population_size):
            # Rotate through strategy types for diversity
            strategy_type = strategy_types[i % len(strategy_types)]
            
            try:
                if strategy_type == 'template':
                    # Generate template-based strategy
                    strategy = self.generate_strategy(
                        strategy_type='template',
                        template_name=random.choice(['moving_average_crossover', 'rsi_mean_reversion', 'bollinger_bands'])
                    )
                else:
                    # Generate LLM-based strategy
                    strategy = self.generate_strategy(strategy_type=strategy_type)
                
                strategies.append(strategy)
                
            except Exception as e:
                self.logger.error(f"Failed to generate strategy {i+1}: {str(e)}")
                # Add fallback strategy
                fallback_strategy = self._generate_fallback_strategy(i)
                strategies.append(fallback_strategy)
        
        # Ensure diversity in the population
        diverse_strategies = self._ensure_population_diversity(strategies, population_size)
        
        self.logger.info(f"✅ Generated diverse population with {len(diverse_strategies)} strategies")
        self.logger.info(f"   LLM strategies: {sum(1 for s in diverse_strategies if s.get('type') == 'llm_generated')}")
        self.logger.info(f"   Template strategies: {sum(1 for s in diverse_strategies if s.get('type') == 'template_based')}")
        
        return diverse_strategies
    
    def _generate_fallback_strategy(self, index: int) -> Dict[str, Any]:
        """Generate a fallback strategy when primary methods fail"""
        fallback_id = f"fallback_{index:04d}"
        
        return {
            'id': fallback_id,
            'name': f"Fallback Strategy {fallback_id}",
            'type': 'fallback',
            'description': 'Automatically generated fallback trading strategy',
            'entry_rules': [{
                'condition': 'price_above_sma_50',
                'priority': 1,
                'weight': 1.0
            }],
            'exit_rules': [{
                'condition': 'price_below_sma_20 OR profit_target_reached',
                'priority': 1,
                'weight': 1.0
            }],
            'risk_management': {
                'position_sizing': 'fixed_percentage',
                'stop_loss': 'trailing_5_percent',
                'take_profit': 'risk_reward_2_to_1',
                'max_drawdown': '0.05',
                'risk_per_trade': '0.02'
            },
            'parameters': {
                'fallback_mode': True,
                'conservatism': 0.8
            },
            'metadata': {
                'generated_by': 'StrategyGenerator',
                'generation_method': 'fallback',
                'fallback_reason': 'Primary generation methods failed'
            }
        }
    
    def _ensure_population_diversity(self, strategies: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
        """Ensure diversity in the strategy population"""
        if len(strategies) <= target_size:
            return strategies
        
        # Use novelty detector to select diverse strategies
        diverse_strategies = []
        remaining_strategies = strategies.copy()
        
        while len(diverse_strategies) < target_size and remaining_strategies:
            # Select strategy that is most novel compared to current diverse set
            most_novel_strategy = None
            highest_novelty_score = -1
            
            for strategy in remaining_strategies:
                novelty_score = self._calculate_strategy_novelty(strategy, diverse_strategies)
                
                if novelty_score > highest_novelty_score:
                    highest_novelty_score = novelty_score
                    most_novel_strategy = strategy
            
            if most_novel_strategy:
                diverse_strategies.append(most_novel_strategy)
                remaining_strategies.remove(most_novel_strategy)
        
        return diverse_strategies
    
    def _calculate_strategy_novelty(self, strategy: Dict[str, Any], population: List[Dict[str, Any]]) -> float:
        """Calculate novelty score for a strategy compared to a population"""
        if not population:
            return 1.0  # High novelty if no comparison
        
        # Simple novelty calculation based on strategy characteristics
        novelty_score = 0.5  # Base score
        
        # Check strategy type
        strategy_type = strategy.get('type', '')
        if strategy_type == 'llm_generated':
            novelty_score += 0.3
        elif strategy_type == 'template_based':
            novelty_score += 0.1
        
        # Check for innovative naming
        name = strategy.get('name', '').lower()
        if any(concept in name for concept in ['quantum', 'neural', 'fractal', 'chaos', 'swarm']):
            novelty_score += 0.2
        
        # Compare with existing strategies
        similarity_scores = []
        for existing_strategy in population:
            similarity = self.calculate_strategy_similarity(strategy, existing_strategy)
            similarity_scores.append(similarity)
        
        if similarity_scores:
            avg_similarity = sum(similarity_scores) / len(similarity_scores)
            novelty_score += (1.0 - avg_similarity) * 0.3  # Higher novelty for less similar strategies
        
        return min(1.0, max(0.1, novelty_score))
    
    def refine_strategy(self, 
                       strategy: Dict[str, Any],
                       performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine a strategy based on performance feedback using LLM agents
        
        Args:
            strategy: Original strategy to refine
            performance_metrics: Backtest performance metrics
            
        Returns:
            Refined strategy dictionary
        """
        if not self.llm_operational or not hasattr(self, 'feedback_agent'):
            self.logger.warning("LLM refinement not available - returning original strategy")
            return strategy
        
        self.logger.info(f"🔬 Refining strategy: {strategy.get('id', 'unknown')}")
        
        try:
            # Use feedback agent to refine the strategy
            refined_strategy = self.feedback_agent.refine_strategy(
                strategy=strategy,
                performance_metrics=performance_metrics
            )
            
            # Validate the refined strategy
            is_valid, validation_report = self.validation_agent.validate_strategy(refined_strategy)
            
            if is_valid:
                self.logger.info(f"✅ Strategy refinement successful (validation score: {validation_report['overall_score']:.2f})")
            else:
                self.logger.warning(f"⚠️  Refined strategy validation issues (score: {validation_report['overall_score']:.2f})")
            
            refined_strategy['validation_report'] = validation_report
            return refined_strategy
            
        except Exception as e:
            self.logger.error(f"❌ Strategy refinement failed: {str(e)}")
            self.logger.info("🔄 Returning original strategy")
            return strategy
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and capabilities"""
        return {
            'llm_enabled': self.use_llm,
            'llm_operational': self.llm_operational,
            'llm_fallback_mode': self.llm_fallback_mode,
            'generation_method': 'hybrid' if self.llm_operational else 'template_only',
            'available_strategy_types': ['innovative', 'physics_based', 'biology_based', 'game_theory', 'complexity_science', 'template'],
            'system_health': 'optimal' if self.llm_operational else 'degraded'
        }
    
    # Legacy methods for backward compatibility
    def generate_strategy_population_legacy(self, template_name: str, population_size: int = 10) -> List[Dict[str, Any]]:
        """Legacy method for template-based population generation"""
        self.logger.info(f"Generating legacy strategy population: {population_size} strategies")
        
        # Get template
        template = self.template_manager.get_template(template_name)
        
        # Generate initial population
        population = self.genetic_operators.create_initial_population(template, population_size)
        
        # Ensure diversity
        diverse_population = self.novelty_detector.ensure_diversity(population, population_size)
        
        self.logger.info(f"Generated diverse population with {len(diverse_population)} strategies")
        return diverse_population
    
    def evolve_strategies(self, population: List[Dict[str, Any]], fitness_scores: List[float],
                          num_generations: int = 5, population_size: int = 10) -> List[Dict[str, Any]]:
        """Evolve a population of strategies using genetic algorithms"""
        self.logger.info(f"Evolving strategies for {num_generations} generations")
        
        current_population = population.copy()
        
        for generation in range(num_generations):
            self.logger.debug(f"Generation {generation + 1}/{num_generations}")
            
            # Select parents
            parents = self.genetic_operators.select_parents(current_population, fitness_scores)
            
            # Create next generation
            next_generation = []
            
            # Keep top performers (elitism)
            elite_size = max(2, int(population_size * 0.1))
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
            for idx in elite_indices:
                next_generation.append(current_population[idx])
            
            # Generate offspring through crossover and mutation
            while len(next_generation) < population_size:
                # Select two parents
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                child1, child2 = self.genetic_operators.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.genetic_operators.mutate(child1)
                child2 = self.genetic_operators.mutate(child2)
                
                # Add to next generation if novel
                if self.novelty_detector.is_novel(child1, next_generation):
                    next_generation.append(child1)
                
                if len(next_generation) < population_size and self.novelty_detector.is_novel(child2, next_generation):
                    next_generation.append(child2)
            
            # Ensure we have exactly population_size strategies
            next_generation = next_generation[:population_size]
            
            # Update population and fitness scores
            current_population = next_generation
            
            # Calculate diversity
            diversity_score = self.novelty_detector.calculate_population_diversity(current_population)
            self.logger.info(f"Generation {generation + 1} completed. Diversity: {diversity_score:.3f}")
        
        return current_population
    
    def create_backtrader_strategy(self, strategy: Dict[str, Any]) -> type:
        """Create a backtrader strategy class from a generated strategy"""
        self.logger.info(f"Creating backtrader strategy for: {strategy.get('name', strategy.get('template', 'unknown'))}")
        return self.backtrader_factory.create_strategy_class(strategy)
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Validate a generated strategy"""
        self.logger.info(f"Validating strategy: {strategy.get('name', strategy.get('template', 'unknown'))}")
        
        # Use appropriate validation method based on strategy type
        if strategy.get('type') == 'llm_generated' and hasattr(self, 'validation_agent'):
            is_valid, validation_report = self.validation_agent.validate_strategy(strategy)
            strategy['validation_report'] = validation_report
            return is_valid
        else:
            # Legacy validation for template-based strategies
            required_fields = ['name', 'entry_rules', 'exit_rules', 'risk_management']
            for field in required_fields:
                if field not in strategy:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            return True
    
    def get_available_templates(self) -> List[str]:
        """Get list of available strategy templates"""
        templates = self.template_manager.get_all_templates()
        return list(templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get detailed information about a strategy template"""
        return self.template_manager.get_template(template_name)
    
    def calculate_strategy_similarity(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> float:
        """Calculate similarity between two strategies"""
        return self.novelty_detector.calculate_strategy_similarity(strategy1, strategy2)
    
    def is_strategy_novel(self, strategy: Dict[str, Any], population: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Check if a strategy is novel compared to a population"""
        return self.novelty_detector.is_novel(strategy, population)