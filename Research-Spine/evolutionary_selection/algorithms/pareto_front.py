"""
Pareto Front Optimization Module

Implements Pareto front optimization for multi-objective strategy selection.
Identifies non-dominated solutions that represent optimal trade-offs between conflicting objectives.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class StrategyParetoPoint:
    """Data class representing a strategy on the Pareto front"""
    strategy: Dict[str, Any]
    fitness_scores: Dict[str, float]
    backtest_results: Dict[str, Any]
    dominance_count: int = 0
    dominated_solutions: List[int] = None
    
    def __post_init__(self):
        if self.dominated_solutions is None:
            self.dominated_solutions = []

class ParetoFrontOptimizer:
    """Class for Pareto front optimization of trading strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('ParetoFrontOptimizer')
        self.logger.info("ParetoFrontOptimizer initialized")
    
    def find_pareto_front(self, strategies: List[Dict[str, Any]], 
                         backtest_results: List[Dict[str, Any]],
                         fitness_calculator: Any) -> List[StrategyParetoPoint]:
        """
        Find the Pareto front from a population of strategies
        
        Args:
            strategies: List of strategy dictionaries
            backtest_results: List of corresponding backtest results
            fitness_calculator: MultiObjectiveFitness instance
            
        Returns:
            List of StrategyParetoPoint objects on the Pareto front
        """
        self.logger.info(f"Finding Pareto front from {len(strategies)} strategies")
        
        # Calculate fitness scores for all strategies
        pareto_points = []
        for i, (strategy, results) in enumerate(zip(strategies, backtest_results)):
            fitness_scores = fitness_calculator.calculate_fitness_scores(strategy, results)
            pareto_points.append(StrategyParetoPoint(
                strategy=strategy,
                fitness_scores=fitness_scores,
                backtest_results=results
            ))
        
        # Find non-dominated solutions
        pareto_front = self._find_non_dominated_solutions(pareto_points)
        
        self.logger.info(f"Found {len(pareto_front)} strategies on Pareto front")
        return pareto_front
    
    def _find_non_dominated_solutions(self, pareto_points: List[StrategyParetoPoint]) -> List[StrategyParetoPoint]:
        """
        Find non-dominated solutions using Pareto dominance
        """
        # Initialize dominance information
        for i, point in enumerate(pareto_points):
            for j, other_point in enumerate(pareto_points):
                if i != j:
                    if self._dominates(point.fitness_scores, other_point.fitness_scores):
                        point.dominated_solutions.append(j)
                    elif self._dominates(other_point.fitness_scores, point.fitness_scores):
                        point.dominance_count += 1
        
        # Find non-dominated solutions (dominance_count == 0)
        pareto_front = [point for point in pareto_points if point.dominance_count == 0]
        
        return pareto_front
    
    def _dominates(self, fitness_a: Dict[str, float], fitness_b: Dict[str, float]) -> bool:
        """
        Check if fitness_a dominates fitness_b
        fitness_a dominates fitness_b if it's better in at least one objective and not worse in any
        """
        dominates = False
        
        for objective in fitness_a.keys():
            if fitness_a[objective] > fitness_b[objective]:
                dominates = True
            elif fitness_a[objective] < fitness_b[objective]:
                return False  # fitness_a is worse in this objective
        
        return dominates
    
    def calculate_pareto_front_metrics(self, pareto_front: List[StrategyParetoPoint]) -> Dict[str, Any]:
        """
        Calculate metrics about the Pareto front
        
        Args:
            pareto_front: List of StrategyParetoPoint objects
            
        Returns:
            Dictionary of Pareto front metrics
        """
        if not pareto_front:
            return {}
        
        metrics = {
            'front_size': len(pareto_front),
            'objective_ranges': {},
            'average_fitness': {},
            'diversity_score': self._calculate_front_diversity(pareto_front)
        }
        
        # Calculate ranges and averages for each objective
        objectives = pareto_front[0].fitness_scores.keys()
        
        for objective in objectives:
            values = [point.fitness_scores[objective] for point in pareto_front]
            metrics['objective_ranges'][objective] = {
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values)
            }
            metrics['average_fitness'][objective] = np.mean(values)
        
        return metrics
    
    def _calculate_front_diversity(self, pareto_front: List[StrategyParetoPoint]) -> float:
        """
        Calculate diversity score of the Pareto front
        Higher score indicates more diverse solutions
        """
        if len(pareto_front) <= 1:
            return 0.0
        
        # Calculate pairwise distances between strategies
        distances = []
        n = len(pareto_front)
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self._calculate_strategy_distance(pareto_front[i], pareto_front[j])
                distances.append(distance)
        
        if not distances:
            return 0.0
        
        # Diversity is the average distance
        diversity = np.mean(distances)
        
        # Normalize by number of objectives
        num_objectives = len(pareto_front[0].fitness_scores)
        normalized_diversity = diversity / num_objectives
        
        return normalized_diversity
    
    def _calculate_strategy_distance(self, point_a: StrategyParetoPoint, point_b: StrategyParetoPoint) -> float:
        """
        Calculate Euclidean distance between two strategies based on fitness scores
        """
        fitness_a = np.array(list(point_a.fitness_scores.values()))
        fitness_b = np.array(list(point_b.fitness_scores.values()))
        
        distance = np.linalg.norm(fitness_a - fitness_b)
        
        return distance
    
    def select_diverse_pareto_solutions(self, pareto_front: List[StrategyParetoPoint], 
                                       num_solutions: int = 5) -> List[StrategyParetoPoint]:
        """
        Select diverse solutions from the Pareto front
        
        Args:
            pareto_front: List of StrategyParetoPoint objects
            num_solutions: Number of diverse solutions to select
            
        Returns:
            List of selected diverse StrategyParetoPoint objects
        """
        if len(pareto_front) <= num_solutions:
            return pareto_front.copy()
        
        # Use clustering to select diverse solutions
        selected_solutions = self._cluster_based_selection(pareto_front, num_solutions)
        
        return selected_solutions
    
    def _cluster_based_selection(self, pareto_front: List[StrategyParetoPoint], 
                                num_solutions: int) -> List[StrategyParetoPoint]:
        """
        Select diverse solutions using simple clustering approach
        """
        # Convert fitness scores to numpy array for clustering
        fitness_matrix = np.array([list(point.fitness_scores.values()) for point in pareto_front])
        
        # Simple approach: select solutions that are farthest apart
        selected_indices = []
        
        # Start with the solution that has the highest composite fitness
        composite_scores = [np.mean(list(point.fitness_scores.values())) for point in pareto_front]
        best_index = np.argmax(composite_scores)
        selected_indices.append(best_index)
        
        # Select remaining solutions based on maximum distance from already selected
        for _ in range(1, num_solutions):
            max_distance = -1
            best_candidate = -1
            
            for i in range(len(pareto_front)):
                if i not in selected_indices:
                    # Calculate minimum distance to any selected solution
                    min_distance = min(
                        self._calculate_strategy_distance(pareto_front[i], pareto_front[j])
                        for j in selected_indices
                    )
                    
                    if min_distance > max_distance:
                        max_distance = min_distance
                        best_candidate = i
            
            if best_candidate >= 0:
                selected_indices.append(best_candidate)
        
        # Return selected solutions
        return [pareto_front[i] for i in selected_indices]