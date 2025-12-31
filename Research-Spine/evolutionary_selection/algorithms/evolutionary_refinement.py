"""
Evolutionary Refinement Algorithms Module

Implements advanced evolutionary algorithms for refining trading strategies.
Includes genetic algorithms, differential evolution, and adaptive parameter tuning.
"""

import numpy as np
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EvolutionaryRefinementResult:
    """Data class for evolutionary refinement results"""
    refined_strategies: List[Dict[str, Any]]
    refinement_history: List[Dict[str, Any]]
    convergence_metrics: Dict[str, Any]

class EvolutionaryRefinement:
    """Class for evolutionary refinement of trading strategies"""
    
    def __init__(self, mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1, population_size: int = 20):
        self.logger = logging.getLogger('EvolutionaryRefinement')
        self.logger.info("EvolutionaryRefinement initialized")
        
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.population_size = population_size
    
    def refine_strategies(self, initial_strategies: List[Dict[str, Any]],
                         backtest_results: List[Dict[str, Any]],
                         fitness_calculator: Any,
                         num_generations: int = 10,
                         refinement_goals: Optional[Dict[str, float]] = None) -> EvolutionaryRefinementResult:
        """
        Refine strategies using evolutionary algorithms
        
        Args:
            initial_strategies: List of initial strategy dictionaries
            backtest_results: List of corresponding backtest results
            fitness_calculator: MultiObjectiveFitness instance
            num_generations: Number of generations to evolve
            refinement_goals: Optional dictionary of target fitness values
            
        Returns:
            EvolutionaryRefinementResult containing refined strategies and metrics
        """
        self.logger.info(f"Starting evolutionary refinement for {num_generations} generations")
        
        # Initialize population
        population = initial_strategies.copy()
        
        # Calculate initial fitness
        fitness_scores = []
        for i, strategy in enumerate(population):
            scores = fitness_calculator.calculate_fitness_scores(strategy, backtest_results[i])
            fitness_scores.append(scores)
        
        # Track refinement history
        refinement_history = []
        
        # Evolutionary loop
        for generation in range(num_generations):
            self.logger.debug(f"Generation {generation + 1}/{num_generations}")
            
            # Calculate generation metrics
            generation_metrics = self._calculate_generation_metrics(population, fitness_scores)
            refinement_history.append({
                'generation': generation + 1,
                'metrics': generation_metrics,
                'population_size': len(population)
            })
            
            # Check for convergence
            if generation > 0 and self._check_convergence(refinement_history):
                self.logger.info(f"Convergence detected at generation {generation + 1}")
                break
            
            # Create next generation
            population, fitness_scores = self._create_next_generation(
                population, fitness_scores, fitness_calculator, backtest_results
            )
        
        # Calculate final convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(refinement_history)
        
        result = EvolutionaryRefinementResult(
            refined_strategies=population,
            refinement_history=refinement_history,
            convergence_metrics=convergence_metrics
        )
        
        self.logger.info(f"Evolutionary refinement completed. Final population: {len(population)}")
        return result
    
    def _create_next_generation(self, population: List[Dict[str, Any]],
                               fitness_scores: List[Dict[str, float]],
                               fitness_calculator: Any,
                               backtest_results: List[Dict[str, Any]]) -> Tuple[List, List]:
        """
        Create next generation using genetic operators
        """
        new_population = []
        new_fitness_scores = []
        
        # Apply elitism - keep top performers
        elite_size = max(2, int(len(population) * self.elitism_rate))
        elite_indices = self._select_elite(population, fitness_scores, elite_size)
        
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            new_fitness_scores.append(fitness_scores[idx].copy())
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            # Select parents
            parent1_idx, parent2_idx = self._select_parents(population, fitness_scores)
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover_strategies(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate_strategy(child1)
            child2 = self._mutate_strategy(child2)
            
            # Add children to new population
            if len(new_population) < self.population_size:
                new_population.append(child1)
                # For now, use parent's fitness (will be recalculated)
                new_fitness_scores.append(fitness_scores[parent1_idx].copy())
            
            if len(new_population) < self.population_size:
                new_population.append(child2)
                new_fitness_scores.append(fitness_scores[parent2_idx].copy())
        
        # Ensure population size is maintained
        new_population = new_population[:self.population_size]
        new_fitness_scores = new_fitness_scores[:self.population_size]
        
        return new_population, new_fitness_scores
    
    def _select_elite(self, population: List[Dict[str, Any]], 
                     fitness_scores: List[Dict[str, float]],
                     elite_size: int) -> List[int]:
        """
        Select elite strategies based on composite fitness
        """
        # Calculate composite fitness scores
        composite_scores = []
        for scores in fitness_scores:
            composite = np.mean(list(scores.values()))
            composite_scores.append(composite)
        
        # Sort by composite score (descending) and select top elite_size
        sorted_indices = np.argsort(composite_scores)[::-1]
        elite_indices = sorted_indices[:elite_size].tolist()
        
        return elite_indices
    
    def _select_parents(self, population: List[Dict[str, Any]], 
                       fitness_scores: List[Dict[str, float]]) -> Tuple[int, int]:
        """
        Select parent strategies using tournament selection
        """
        tournament_size = min(5, len(population))
        
        # Select first parent
        tournament_indices = random.sample(range(len(population)), tournament_size)
        parent1_idx = self._select_tournament_winner(tournament_indices, fitness_scores)
        
        # Select second parent (different from first)
        remaining_indices = [i for i in range(len(population)) if i != parent1_idx]
        tournament_indices = random.sample(remaining_indices, min(tournament_size, len(remaining_indices)))
        parent2_idx = self._select_tournament_winner(tournament_indices, fitness_scores)
        
        return parent1_idx, parent2_idx
    
    def _select_tournament_winner(self, indices: List[int], 
                                  fitness_scores: List[Dict[str, float]]) -> int:
        """
        Select winner from tournament based on fitness
        """
        # Calculate composite fitness for tournament participants
        composite_scores = []
        for idx in indices:
            scores = fitness_scores[idx]
            composite = np.mean(list(scores.values()))
            composite_scores.append(composite)
        
        # Return index of winner (highest composite score)
        winner_idx_in_tournament = np.argmax(composite_scores)
        winner_idx = indices[winner_idx_in_tournament]
        
        return winner_idx
    
    def _crossover_strategies(self, parent1: Dict[str, Any], 
                             parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parent strategies
        """
        # Create copies to avoid modifying originals
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Deep copy parameters
        child1['parameters'] = parent1['parameters'].copy()
        child2['parameters'] = parent2['parameters'].copy()
        
        # Get parameter names
        param_names = list(parent1['parameters'].keys())
        
        # Perform uniform crossover on parameters
        for param in param_names:
            if random.random() < 0.5:
                # Swap parameter values
                child1['parameters'][param] = parent2['parameters'][param]
                child2['parameters'][param] = parent1['parameters'][param]
        
        return child1, child2
    
    def _mutate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate a strategy by modifying its parameters
        """
        mutated_strategy = strategy.copy()
        mutated_strategy['parameters'] = strategy['parameters'].copy()
        
        # Mutate each parameter with mutation_rate probability
        for param_name, param_value in strategy['parameters'].items():
            if random.random() < self.mutation_rate:
                mutated_strategy['parameters'][param_name] = self._mutate_parameter(param_value)
        
        return mutated_strategy
    
    def _mutate_parameter(self, parameter_value: Any) -> Any:
        """
        Mutate a parameter value based on its type
        """
        if isinstance(parameter_value, (int, float)):
            # Numeric parameter - apply Gaussian mutation
            if isinstance(parameter_value, int):
                mutation = max(1, int(abs(parameter_value) * 0.1))
                return parameter_value + random.randint(-mutation, mutation)
            else:
                mutation = abs(parameter_value) * 0.1
                return parameter_value + random.gauss(0, mutation)
        elif isinstance(parameter_value, bool):
            # Boolean parameter - flip
            return not parameter_value
        elif isinstance(parameter_value, str):
            # String parameter - no mutation (for now)
            return parameter_value
        else:
            # Unknown type - return unchanged
            return parameter_value
    
    def _calculate_generation_metrics(self, population: List[Dict[str, Any]],
                                     fitness_scores: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate metrics for the current generation
        """
        if not population:
            return {}
        
        # Calculate composite fitness scores
        composite_scores = [np.mean(list(scores.values())) for scores in fitness_scores]
        
        metrics = {
            'avg_composite_fitness': np.mean(composite_scores),
            'max_composite_fitness': np.max(composite_scores),
            'min_composite_fitness': np.min(composite_scores),
            'fitness_std': np.std(composite_scores),
            'objective_averages': {}
        }
        
        # Calculate averages for each objective
        if fitness_scores:
            objectives = fitness_scores[0].keys()
            for objective in objectives:
                values = [scores[objective] for scores in fitness_scores]
                metrics['objective_averages'][objective] = np.mean(values)
        
        return metrics
    
    def _check_convergence(self, refinement_history: List[Dict[str, Any]]) -> bool:
        """
        Check if the evolutionary process has converged
        """
        if len(refinement_history) < 3:
            return False
        
        # Check if fitness improvement has stalled
        last_3_avg_fitness = [
            history['metrics']['avg_composite_fitness'] 
            for history in refinement_history[-3:]
        ]
        
        # Calculate improvement rate
        improvement = last_3_avg_fitness[-1] - last_3_avg_fitness[0]
        
        # If improvement is less than 1% of the current fitness, consider converged
        current_fitness = last_3_avg_fitness[-1]
        if current_fitness > 0 and improvement < current_fitness * 0.01:
            return True
        
        return False
    
    def _calculate_convergence_metrics(self, refinement_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate convergence metrics from refinement history
        """
        if not refinement_history:
            return {}
        
        metrics = {
            'final_generation': refinement_history[-1]['generation'],
            'initial_avg_fitness': refinement_history[0]['metrics']['avg_composite_fitness'],
            'final_avg_fitness': refinement_history[-1]['metrics']['avg_composite_fitness'],
            'fitness_improvement': 0.0,
            'convergence_rate': 0.0
        }
        
        # Calculate fitness improvement
        if len(refinement_history) > 1:
            metrics['fitness_improvement'] = (
                metrics['final_avg_fitness'] - metrics['initial_avg_fitness']
            )
            
            # Calculate convergence rate (fitness improvement per generation)
            total_generations = len(refinement_history)
            metrics['convergence_rate'] = metrics['fitness_improvement'] / total_generations
        
        return metrics
    
    def adaptive_refinement(self, strategies: List[Dict[str, Any]],
                          backtest_results: List[Dict[str, Any]],
                          fitness_calculator: Any,
                          max_generations: int = 20) -> EvolutionaryRefinementResult:
        """
        Adaptive refinement that adjusts parameters based on population diversity
        """
        self.logger.info("Starting adaptive evolutionary refinement")
        
        # Start with initial parameters
        current_population = strategies.copy()
        
        # Calculate initial fitness
        fitness_scores = []
        for i, strategy in enumerate(current_population):
            scores = fitness_calculator.calculate_fitness_scores(strategy, backtest_results[i])
            fitness_scores.append(scores)
        
        refinement_history = []
        
        for generation in range(max_generations):
            self.logger.debug(f"Adaptive generation {generation + 1}/{max_generations}")
            
            # Calculate diversity metrics
            diversity = self._calculate_population_diversity(current_population, fitness_scores)
            
            # Adjust evolutionary parameters based on diversity
            self._adjust_parameters_based_on_diversity(diversity)
            
            # Calculate generation metrics
            generation_metrics = self._calculate_generation_metrics(current_population, fitness_scores)
            generation_metrics['diversity'] = diversity
            
            refinement_history.append({
                'generation': generation + 1,
                'metrics': generation_metrics,
                'population_size': len(current_population),
                'parameters': {
                    'mutation_rate': self.mutation_rate,
                    'crossover_rate': self.crossover_rate,
                    'elitism_rate': self.elitism_rate
                }
            })
            
            # Check for convergence
            if generation > 0 and self._check_convergence(refinement_history):
                self.logger.info(f"Adaptive convergence detected at generation {generation + 1}")
                break
            
            # Create next generation
            current_population, fitness_scores = self._create_next_generation(
                current_population, fitness_scores, fitness_calculator, backtest_results
            )
        
        # Calculate final convergence metrics
        convergence_metrics = self._calculate_convergence_metrics(refinement_history)
        
        result = EvolutionaryRefinementResult(
            refined_strategies=current_population,
            refinement_history=refinement_history,
            convergence_metrics=convergence_metrics
        )
        
        self.logger.info("Adaptive evolutionary refinement completed")
        return result
    
    def _calculate_population_diversity(self, population: List[Dict[str, Any]],
                                       fitness_scores: List[Dict[str, float]]) -> float:
        """
        Calculate diversity score of the population
        """
        if len(population) <= 1:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        n = len(population)
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self._calculate_strategy_distance(
                    population[i], population[j], 
                    fitness_scores[i], fitness_scores[j]
                )
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        # Diversity is the average distance
        diversity = np.mean(distances)
        
        return diversity
    
    def _calculate_strategy_distance(self, strategy_a: Dict[str, Any], 
                                    strategy_b: Dict[str, Any],
                                    fitness_a: Dict[str, float],
                                    fitness_b: Dict[str, float]) -> float:
        """
        Calculate distance between two strategies
        """
        # Parameter distance
        param_distance = self._calculate_parameter_distance(strategy_a, strategy_b)
        
        # Fitness distance
        fitness_distance = self._calculate_fitness_distance(fitness_a, fitness_b)
        
        # Combined distance
        distance = 0.6 * param_distance + 0.4 * fitness_distance
        
        return distance
    
    def _calculate_parameter_distance(self, strategy_a: Dict[str, Any], 
                                    strategy_b: Dict[str, Any]) -> float:
        """
        Calculate distance based on strategy parameters
        """
        params_a = strategy_a.get('parameters', {})
        params_b = strategy_b.get('parameters', {})
        
        if not params_a or not params_b:
            return 1.0
        
        # Compare common parameters
        common_params = set(params_a.keys()) & set(params_b.keys())
        
        if not common_params:
            return 1.0
        
        # Calculate parameter differences
        total_diff = 0.0
        for param in common_params:
            val_a = params_a[param]
            val_b = params_b[param]
            
            # Handle different parameter types
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # Numeric parameters
                diff = abs(val_a - val_b)
                # Normalize by range
                normalized_diff = diff / (1 + diff)
                total_diff += normalized_diff
            else:
                # Non-numeric parameters
                if val_a == val_b:
                    total_diff += 0.0
                else:
                    total_diff += 1.0
        
        # Average difference
        avg_diff = total_diff / len(common_params)
        
        return avg_diff
    
    def _calculate_fitness_distance(self, fitness_a: Dict[str, float],
                                   fitness_b: Dict[str, float]) -> float:
        """
        Calculate distance based on fitness scores
        """
        # Convert fitness scores to vectors
        objectives = set(fitness_a.keys()) & set(fitness_b.keys())
        
        if not objectives:
            return 1.0
        
        # Create fitness vectors
        fitness_vec_a = np.array([fitness_a[obj] for obj in objectives])
        fitness_vec_b = np.array([fitness_b[obj] for obj in objectives])
        
        # Calculate Euclidean distance
        distance = np.linalg.norm(fitness_vec_a - fitness_vec_b)
        
        # Normalize by number of objectives
        normalized_distance = distance / len(objectives)
        
        return normalized_distance
    
    def _adjust_parameters_based_on_diversity(self, diversity: float):
        """
        Adjust evolutionary parameters based on population diversity
        """
        # If diversity is low, increase mutation rate and decrease elitism
        if diversity < 0.3:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
            self.elitism_rate = max(0.05, self.elitism_rate * 0.9)
        
        # If diversity is high, decrease mutation rate and increase elitism
        elif diversity > 0.7:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.8)
            self.elitism_rate = min(0.2, self.elitism_rate * 1.1)