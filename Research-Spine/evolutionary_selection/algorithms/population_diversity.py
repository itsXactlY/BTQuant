"""
Population Diversity Management Module

Implements algorithms for managing and maintaining diversity in the strategy population.
Includes diversity metrics, niche formation, and adaptive diversity control.
"""

import numpy as np
import logging
import random
from typing import Dict, Any, List, Tuple
from collections import defaultdict

class PopulationDiversityManager:
    """Class for managing population diversity in evolutionary strategy selection"""
    
    def __init__(self, diversity_threshold: float = 0.5, niche_radius: float = 0.3):
        self.logger = logging.getLogger('PopulationDiversityManager')
        self.logger.info("PopulationDiversityManager initialized")
        self.diversity_threshold = diversity_threshold
        self.niche_radius = niche_radius
    
    def manage_diversity(self, population: List[Dict[str, Any]], 
                        fitness_scores: List[Dict[str, float]],
                        target_size: int) -> List[Dict[str, Any]]:
        """
        Manage population diversity to maintain a diverse set of strategies
        
        Args:
            population: List of strategy dictionaries
            fitness_scores: List of corresponding fitness scores
            target_size: Target population size
            
        Returns:
            List of strategies with managed diversity
        """
        self.logger.info(f"Managing diversity for population of {len(population)} strategies")
        
        if len(population) <= target_size:
            return population.copy()
        
        # Calculate current diversity
        current_diversity = self.calculate_population_diversity(population, fitness_scores)
        
        # Apply diversity management based on current diversity level
        if current_diversity < self.diversity_threshold:
            # Low diversity - apply niche formation
            self.logger.debug("Low diversity detected, applying niche formation")
            managed_population = self._apply_niche_formation(population, fitness_scores, target_size)
        else:
            # Sufficient diversity - apply diversity preservation
            self.logger.debug("Sufficient diversity, applying diversity preservation")
            managed_population = self._apply_diversity_preservation(population, fitness_scores, target_size)
        
        self.logger.info(f"Diversity management completed. Final population: {len(managed_population)}")
        return managed_population
    
    def calculate_population_diversity(self, population: List[Dict[str, Any]],
                                      fitness_scores: List[Dict[str, float]]) -> float:
        """
        Calculate overall diversity score of the population
        
        Args:
            population: List of strategy dictionaries
            fitness_scores: List of corresponding fitness scores
            
        Returns:
            Diversity score between 0 and 1
        """
        if len(population) <= 1:
            return 0.0
        
        # Calculate multiple diversity metrics
        parameter_diversity = self._calculate_parameter_diversity(population)
        fitness_diversity = self._calculate_fitness_diversity(fitness_scores)
        
        # Combined diversity score
        diversity_score = 0.6 * parameter_diversity + 0.4 * fitness_diversity
        
        return diversity_score
    
    def _calculate_parameter_diversity(self, population: List[Dict[str, Any]]) -> float:
        """
        Calculate diversity based on strategy parameters
        """
        if len(population) <= 1:
            return 0.0
        
        # Collect all parameter values for each parameter
        param_values = defaultdict(list)
        
        for strategy in population:
            for param_name, param_value in strategy.get('parameters', {}).items():
                param_values[param_name].append(param_value)
        
        # Calculate diversity for each parameter
        param_diversities = []
        
        for param_name, values in param_values.items():
            if len(set(values)) <= 1:
                # All values are the same
                param_diversities.append(0.0)
            else:
                # Calculate coefficient of variation for numeric parameters
                if all(isinstance(v, (int, float)) for v in values):
                    mean = np.mean(values)
                    std = np.std(values)
                    if mean == 0:
                        cv = 0.0
                    else:
                        cv = std / abs(mean)
                    param_diversities.append(min(1.0, cv))
                else:
                    # For non-numeric parameters, use ratio of unique values
                    unique_ratio = len(set(values)) / len(values)
                    param_diversities.append(unique_ratio)
        
        if not param_diversities:
            return 0.0
        
        # Average parameter diversity
        avg_param_diversity = np.mean(param_diversities)
        
        return avg_param_diversity
    
    def _calculate_fitness_diversity(self, fitness_scores: List[Dict[str, float]]) -> float:
        """
        Calculate diversity based on fitness scores
        """
        if len(fitness_scores) <= 1:
            return 0.0
        
        # Collect fitness values for each objective
        objective_values = defaultdict(list)
        
        for scores in fitness_scores:
            for objective, value in scores.items():
                objective_values[objective].append(value)
        
        # Calculate diversity for each objective
        objective_diversities = []
        
        for objective, values in objective_values.items():
            if len(set(values)) <= 1:
                # All values are the same
                objective_diversities.append(0.0)
            else:
                # Calculate coefficient of variation
                mean = np.mean(values)
                std = np.std(values)
                if mean == 0:
                    cv = 0.0
                else:
                    cv = std / abs(mean)
                objective_diversities.append(min(1.0, cv))
        
        if not objective_diversities:
            return 0.0
        
        # Average objective diversity
        avg_objective_diversity = np.mean(objective_diversities)
        
        return avg_objective_diversity
    
    def _apply_niche_formation(self, population: List[Dict[str, Any]],
                              fitness_scores: List[Dict[str, float]],
                              target_size: int) -> List[Dict[str, Any]]:
        """
        Apply niche formation to maintain diversity in low-diversity populations
        """
        # Calculate pairwise distances
        distance_matrix = self._calculate_distance_matrix(population, fitness_scores)
        
        # Find niches (clusters)
        niches = self._find_niches(distance_matrix)
        
        # Select representatives from each niche
        selected_strategies = []
        
        for niche in niches:
            if len(niche) == 1:
                # Single strategy in niche, keep it
                selected_strategies.append(population[niche[0]])
            else:
                # Multiple strategies in niche, select the most fit
                best_idx = self._select_best_in_niche(niche, fitness_scores)
                selected_strategies.append(population[best_idx])
        
        # If we have more strategies than target, apply additional filtering
        if len(selected_strategies) > target_size:
            selected_strategies = self._select_top_strategies(selected_strategies, fitness_scores, target_size)
        
        return selected_strategies
    
    def _apply_diversity_preservation(self, population: List[Dict[str, Any]],
                                     fitness_scores: List[Dict[str, float]],
                                     target_size: int) -> List[Dict[str, Any]]:
        """
        Apply diversity preservation to maintain diversity in diverse populations
        """
        # Calculate pairwise distances
        distance_matrix = self._calculate_distance_matrix(population, fitness_scores)
        
        # Select diverse strategies using maximum distance approach
        selected_indices = self._select_diverse_strategies(distance_matrix, target_size)
        
        # Return selected strategies
        selected_strategies = [population[i] for i in selected_indices]
        
        return selected_strategies
    
    def _calculate_distance_matrix(self, population: List[Dict[str, Any]],
                                  fitness_scores: List[Dict[str, float]]) -> np.ndarray:
        """
        Calculate distance matrix between all strategies
        """
        n = len(population)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    distance = self._calculate_strategy_distance(
                        population[i], population[j],
                        fitness_scores[i], fitness_scores[j]
                    )
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
        
        return distance_matrix
    
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
    
    def _find_niches(self, distance_matrix: np.ndarray) -> List[List[int]]:
        """
        Find niches (clusters) in the population based on distance
        """
        n = distance_matrix.shape[0]
        niches = []
        visited = set()
        
        for i in range(n):
            if i not in visited:
                niche = [i]
                visited.add(i)
                
                # Find all strategies within niche radius
                for j in range(i + 1, n):
                    if distance_matrix[i, j] <= self.niche_radius:
                        niche.append(j)
                        visited.add(j)
                
                niches.append(niche)
        
        return niches
    
    def _select_best_in_niche(self, niche: List[int],
                             fitness_scores: List[Dict[str, float]]) -> int:
        """
        Select the best strategy from a niche
        """
        # Calculate composite fitness for each strategy in niche
        composite_scores = []
        
        for idx in niche:
            scores = fitness_scores[idx]
            composite = np.mean(list(scores.values()))
            composite_scores.append(composite)
        
        # Return index of strategy with highest composite score
        best_idx_in_niche = niche[np.argmax(composite_scores)]
        
        return best_idx_in_niche
    
    def _select_diverse_strategies(self, distance_matrix: np.ndarray,
                                   target_size: int) -> List[int]:
        """
        Select diverse strategies using maximum distance approach
        """
        n = distance_matrix.shape[0]
        selected_indices = []
        
        # Start with the strategy that has the highest average distance to others
        avg_distances = np.mean(distance_matrix, axis=1)
        best_index = np.argmax(avg_distances)
        selected_indices.append(best_index)
        
        # Select remaining strategies based on maximum distance from already selected
        for _ in range(1, target_size):
            max_min_distance = -1
            best_candidate = -1
            
            for i in range(n):
                if i not in selected_indices:
                    # Calculate minimum distance to any selected strategy
                    min_distance = min(distance_matrix[i, j] for j in selected_indices)
                    
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = i
            
            if best_candidate >= 0:
                selected_indices.append(best_candidate)
        
        return selected_indices
    
    def _select_top_strategies(self, strategies: List[Dict[str, Any]],
                              fitness_scores: List[Dict[str, float]],
                              target_size: int) -> List[Dict[str, Any]]:
        """
        Select top strategies based on composite fitness
        """
        # Calculate composite fitness scores
        composite_scores = [np.mean(list(scores.values())) for scores in fitness_scores]
        
        # Sort by composite score (descending)
        sorted_indices = np.argsort(composite_scores)[::-1]
        
        # Select top strategies
        selected_indices = sorted_indices[:target_size]
        
        selected_strategies = [strategies[i] for i in selected_indices]
        
        return selected_strategies
    
    def adaptive_diversity_control(self, population: List[Dict[str, Any]],
                                  fitness_scores: List[Dict[str, float]],
                                  target_size: int,
                                  current_generation: int) -> List[Dict[str, Any]]:
        """
        Adaptive diversity control that adjusts based on generation progress
        """
        self.logger.info(f"Applying adaptive diversity control at generation {current_generation}")
        
        # Calculate current diversity
        current_diversity = self.calculate_population_diversity(population, fitness_scores)
        
        # Adjust diversity threshold based on generation progress
        if current_generation < 5:
            # Early generations - prioritize exploration
            adjusted_threshold = max(0.3, self.diversity_threshold * 0.8)
        elif current_generation < 15:
            # Middle generations - balanced approach
            adjusted_threshold = self.diversity_threshold
        else:
            # Late generations - prioritize exploitation
            adjusted_threshold = min(0.7, self.diversity_threshold * 1.2)
        
        # Apply diversity management with adjusted threshold
        original_threshold = self.diversity_threshold
        self.diversity_threshold = adjusted_threshold
        
        try:
            managed_population = self.manage_diversity(population, fitness_scores, target_size)
        finally:
            # Restore original threshold
            self.diversity_threshold = original_threshold
        
        return managed_population
    
    def calculate_diversity_metrics(self, population: List[Dict[str, Any]],
                                   fitness_scores: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculate comprehensive diversity metrics
        """
        metrics = {
            'overall_diversity': self.calculate_population_diversity(population, fitness_scores),
            'parameter_diversity': self._calculate_parameter_diversity(population),
            'fitness_diversity': self._calculate_fitness_diversity(fitness_scores),
            'niche_count': 0,
            'niche_sizes': []
        }
        
        # Calculate niche metrics
        if len(population) > 1:
            distance_matrix = self._calculate_distance_matrix(population, fitness_scores)
            niches = self._find_niches(distance_matrix)
            
            metrics['niche_count'] = len(niches)
            metrics['niche_sizes'] = [len(niche) for niche in niches]
        
        return metrics