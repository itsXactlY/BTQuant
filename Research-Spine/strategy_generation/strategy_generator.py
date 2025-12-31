"""
Strategy Generator Module

Handles the generation of trading strategies based on templates and parameters.
"""

import logging
from typing import Dict, Any, List, Optional
import random

# Import the new components
from strategy_generation.generators.genetic_operators import GeneticOperators
from strategy_generation.generators.novelty_detection import NoveltyDetector
from strategy_generation.templates.strategy_templates import StrategyTemplateManager
from strategy_generation.generators.backtrader_integration import BacktraderStrategyFactory

class StrategyGenerator:
    """Main class for generating trading strategies"""
     
    def __init__(self):
        self.logger = logging.getLogger('StrategyGenerator')
        self.logger.info("StrategyGenerator initialized")
        
        # Initialize components
        self.genetic_operators = GeneticOperators()
        self.novelty_detector = NoveltyDetector()
        self.template_manager = StrategyTemplateManager()
        self.backtrader_factory = BacktraderStrategyFactory()
         
    def generate_strategy(self, template_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a trading strategy based on a template and parameters
        
        Args:
            template_name: Name of the strategy template to use
            parameters: Dictionary of parameters for the strategy
             
        Returns:
            Dictionary containing the generated strategy
        """
        self.logger.info(f"Generating strategy using template: {template_name}")
        
        # Validate parameters against template
        if not self.template_manager.validate_template_parameters(template_name, parameters):
            self.logger.error(f"Invalid parameters for template: {template_name}")
            raise ValueError(f"Invalid parameters for template: {template_name}")
         
        # Enhanced strategy structure
        strategy = {
            'id': f"strategy_{random.randint(1000, 9999)}",
            'template': template_name,
            'parameters': parameters,
            'metadata': {
                'generated_by': 'StrategyGenerator',
                'version': '2.0',
                'template_version': self.template_manager.get_template(template_name).get('version', '1.0')
            }
        }
         
        self.logger.debug(f"Generated strategy: {strategy}")
        return strategy
    
    def generate_strategy_population(self, template_name: str, population_size: int = 10) -> List[Dict[str, Any]]:
        """
        Generate a population of strategies using genetic algorithms
        
        Args:
            template_name: Name of the strategy template to use
            population_size: Number of strategies to generate
            
        Returns:
            List of generated strategies
        """
        self.logger.info(f"Generating strategy population: {population_size} strategies")
        
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
        """
        Evolve a population of strategies using genetic algorithms
        
        Args:
            population: Initial population of strategies
            fitness_scores: Fitness scores for each strategy
            num_generations: Number of generations to evolve
            population_size: Target population size
            
        Returns:
            Evolved population of strategies
        """
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
        """
        Create a backtrader strategy class from a generated strategy
        
        Args:
            strategy: Generated strategy dictionary
            
        Returns:
            Backtrader Strategy class
        """
        self.logger.info(f"Creating backtrader strategy for: {strategy['template']}")
        return self.backtrader_factory.create_strategy_class(strategy)
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """
        Validate a generated strategy
        
        Args:
            strategy: Strategy dictionary to validate
             
        Returns:
            True if strategy is valid, False otherwise
        """
        self.logger.info(f"Validating strategy: {strategy.get('template', 'unknown')}")
        
        # Check required fields
        required_fields = ['template', 'parameters']
        for field in required_fields:
            if field not in strategy:
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate against template constraints
        try:
            return self.template_manager.validate_template_parameters(strategy['template'], strategy['parameters'])
        except Exception as e:
            self.logger.error(f"Template validation failed: {str(e)}")
            return False
    
    def get_available_templates(self) -> List[str]:
        """
        Get list of available strategy templates
        
        Returns:
            List of template names
        """
        templates = self.template_manager.get_all_templates()
        return list(templates.keys())
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a strategy template
        
        Args:
            template_name: Name of the template
            
        Returns:
            Template information dictionary
        """
        return self.template_manager.get_template(template_name)
    
    def calculate_strategy_similarity(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two strategies
        
        Args:
            strategy1: First strategy
            strategy2: Second strategy
            
        Returns:
            Similarity score between 0 and 1
        """
        return self.novelty_detector.calculate_strategy_similarity(strategy1, strategy2)
    
    def is_strategy_novel(self, strategy: Dict[str, Any], population: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Check if a strategy is novel compared to a population
        
        Args:
            strategy: Strategy to check
            population: Optional population to compare against
            
        Returns:
            True if strategy is novel, False otherwise
        """
        return self.novelty_detector.is_novel(strategy, population)