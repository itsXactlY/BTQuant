"""
Genetic Algorithm Operators for Strategy Generation

Implements crossover, mutation, and selection operators for evolutionary strategy generation.
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple
import logging

class GeneticOperators:
    """Class containing genetic algorithm operators for strategy generation"""
    
    def __init__(self):
        self.logger = logging.getLogger('GeneticOperators')
        self.logger.info("GeneticOperators initialized")
        
    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parent strategies to create two child strategies
        
        Args:
            parent1: First parent strategy
            parent2: Second parent strategy
            
        Returns:
            Tuple containing two child strategies
        """
        self.logger.debug("Performing crossover between two strategies")
        
        # Get parameters from both parents
        params1 = parent1['parameters']
        params2 = parent2['parameters']
        
        # Create child parameters by blending parent parameters
        child1_params = {}
        child2_params = {}
        
        for key in params1.keys():
            if key in params2:
                # For numeric parameters, use weighted average
                if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    # Random weight between parents
                    weight = random.uniform(0.3, 0.7)
                    child1_params[key] = weight * params1[key] + (1 - weight) * params2[key]
                    child2_params[key] = (1 - weight) * params1[key] + weight * params2[key]
                # For other types, randomly choose from parents
                else:
                    child1_params[key] = random.choice([params1[key], params2[key]])
                    child2_params[key] = random.choice([params1[key], params2[key]])
            else:
                # If parameter only exists in one parent, pass it to one child
                child1_params[key] = params1[key]
                child2_params[key] = params1[key]
        
        # Create child strategies
        child1 = {
            'template': parent1['template'],
            'parameters': child1_params,
            'metadata': {
                'generated_by': 'GeneticOperators.crossover',
                'parents': [parent1.get('id', 'unknown'), parent2.get('id', 'unknown')],
                'version': '1.0'
            }
        }
        
        child2 = {
            'template': parent2['template'],
            'parameters': child2_params,
            'metadata': {
                'generated_by': 'GeneticOperators.crossover',
                'parents': [parent1.get('id', 'unknown'), parent2.get('id', 'unknown')],
                'version': '1.0'
            }
        }
        
        self.logger.debug(f"Generated children: {len(child1_params)} params each")
        return child1, child2
    
    def mutate(self, strategy: Dict[str, Any], mutation_rate: float = 0.1, mutation_strength: float = 0.2) -> Dict[str, Any]:
        """
        Mutate a strategy by randomly modifying its parameters
        
        Args:
            strategy: Strategy to mutate
            mutation_rate: Probability of each parameter being mutated (0-1)
            mutation_strength: Strength of mutation for numeric parameters
            
        Returns:
            Mutated strategy
        """
        self.logger.debug(f"Mutating strategy with rate {mutation_rate} and strength {mutation_strength}")
        
        mutated_strategy = strategy.copy()
        mutated_params = strategy['parameters'].copy()
        
        for key, value in strategy['parameters'].items():
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    # For numeric values, add random perturbation
                    perturbation = random.uniform(-mutation_strength, mutation_strength) * value
                    mutated_value = value + perturbation
                    
                    # Ensure we don't get negative values for positive parameters
                    if value > 0 and mutated_value <= 0:
                        mutated_value = value * 0.1  # Reduce to 10% instead of going negative
                    
                    mutated_params[key] = mutated_value
                elif isinstance(value, bool):
                    # Flip boolean values
                    mutated_params[key] = not value
                elif isinstance(value, (list, tuple)) and len(value) > 0:
                    # For lists/tuples, randomly select a different item
                    if len(value) > 1:
                        mutated_params[key] = random.choice(value)
                # For other types, leave unchanged
        
        mutated_strategy['parameters'] = mutated_params
        
        # Update metadata
        if 'metadata' not in mutated_strategy:
            mutated_strategy['metadata'] = {}
        mutated_strategy['metadata']['generated_by'] = 'GeneticOperators.mutate'
        mutated_strategy['metadata']['mutation_info'] = {
            'rate': mutation_rate,
            'strength': mutation_strength
        }
        
        self.logger.debug(f"Mutated {sum(1 for k,v in strategy['parameters'].items() if v != mutated_params.get(k))} parameters")
        return mutated_strategy
    
    def select_parents(self, population: List[Dict[str, Any]], fitness_scores: List[float], num_parents: int = 2) -> List[Dict[str, Any]]:
        """
        Select parent strategies using tournament selection
        
        Args:
            population: List of strategy dictionaries
            fitness_scores: List of corresponding fitness scores
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent strategies
        """
        self.logger.debug(f"Selecting {num_parents} parents from population of {len(population)}")
        
        selected_parents = []
        
        for _ in range(num_parents):
            # Tournament selection
            tournament_size = min(5, len(population))  # Small tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            # Find best in tournament
            best_index = max(tournament_indices, key=lambda i: fitness_scores[i])
            selected_parents.append(population[best_index])
        
        self.logger.debug(f"Selected parents with fitness scores: {[fitness_scores[population.index(p)] for p in selected_parents]}")
        return selected_parents
    
    def create_initial_population(self, template: Dict[str, Any], population_size: int = 10) -> List[Dict[str, Any]]:
        """
        Create an initial population of strategies from a template
        
        Args:
            template: Strategy template with parameter ranges
            population_size: Number of strategies to generate
            
        Returns:
            List of generated strategies
        """
        self.logger.info(f"Creating initial population of {population_size} strategies")
        
        population = []
        
        for i in range(population_size):
            strategy = {
                'template': template['name'],
                'parameters': {},
                'metadata': {
                    'generated_by': 'GeneticOperators.create_initial_population',
                    'population_id': i,
                    'version': '1.0'
                }
            }
            
            # Generate random parameters within template ranges
            for param_name, param_config in template['parameters'].items():
                if 'range' in param_config:
                    min_val, max_val = param_config['range']
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        strategy['parameters'][param_name] = random.randint(min_val, max_val)
                    else:
                        strategy['parameters'][param_name] = random.uniform(min_val, max_val)
                elif 'options' in param_config:
                    strategy['parameters'][param_name] = random.choice(param_config['options'])
                elif 'default' in param_config:
                    strategy['parameters'][param_name] = param_config['default']
            
            population.append(strategy)
        
        self.logger.info(f"Generated initial population with {len(population)} strategies")
        return population