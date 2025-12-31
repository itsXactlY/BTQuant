"""
Strategy Pruning and Redundancy Elimination Module

Implements algorithms for pruning redundant strategies and eliminating 
low-quality or highly similar strategies from the population.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict

class StrategyPruner:
    """Class for pruning and eliminating redundant strategies"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.logger = logging.getLogger('StrategyPruner')
        self.logger.info("StrategyPruner initialized")
        self.similarity_threshold = similarity_threshold
    
    def prune_population(self, strategies: List[Dict[str, Any]], 
                        backtest_results: List[Dict[str, Any]],
                        fitness_scores: List[Dict[str, float]],
                        target_size: int = None) -> List[Dict[str, Any]]:
        """
        Prune population by removing redundant and low-quality strategies
        
        Args:
            strategies: List of strategy dictionaries
            backtest_results: List of corresponding backtest results
            fitness_scores: List of corresponding fitness scores
            target_size: Optional target population size
            
        Returns:
            List of pruned strategies
        """
        self.logger.info(f"Pruning population from {len(strategies)} strategies")
        
        if len(strategies) == 0:
            return []
        
        # Step 1: Remove low-quality strategies
        filtered_strategies, filtered_results, filtered_scores = self._filter_low_quality(
            strategies, backtest_results, fitness_scores
        )
        
        # Step 2: Remove redundant strategies
        pruned_strategies, pruned_results, pruned_scores = self._remove_redundant(
            filtered_strategies, filtered_results, filtered_scores
        )
        
        # Step 3: Apply target size if specified
        if target_size is not None and len(pruned_strategies) > target_size:
            final_strategies, final_results, final_scores = self._select_top_strategies(
                pruned_strategies, pruned_results, pruned_scores, target_size
            )
        else:
            final_strategies, final_results, final_scores = pruned_strategies, pruned_results, pruned_scores
        
        self.logger.info(f"Pruned population to {len(final_strategies)} strategies")
        return final_strategies
    
    def _filter_low_quality(self, strategies: List[Dict[str, Any]], 
                           backtest_results: List[Dict[str, Any]],
                           fitness_scores: List[Dict[str, float]]) -> Tuple[List, List, List]:
        """
        Filter out low-quality strategies based on performance thresholds
        """
        filtered_strategies = []
        filtered_results = []
        filtered_scores = []
        
        for i, (strategy, results, scores) in enumerate(zip(strategies, backtest_results, fitness_scores)):
            # Check if strategy meets minimum quality criteria
            if self._meets_quality_criteria(results, scores):
                filtered_strategies.append(strategy)
                filtered_results.append(results)
                filtered_scores.append(scores)
        
        self.logger.debug(f"Filtered {len(filtered_strategies)} high-quality strategies")
        return filtered_strategies, filtered_results, filtered_scores
    
    def _meets_quality_criteria(self, backtest_results: Dict[str, Any], 
                               fitness_scores: Dict[str, float]) -> bool:
        """
        Check if a strategy meets minimum quality criteria
        """
        metrics = backtest_results['performance_metrics']
        
        # Minimum criteria
        min_sharpe = 0.3
        max_drawdown = 50.0  # percentage
        min_win_rate = 0.4
        
        # Check basic performance criteria
        if (metrics['sharpe_ratio'] < min_sharpe or
            abs(metrics['max_drawdown']) > max_drawdown or
            metrics['win_rate'] < min_win_rate):
            return False
        
        # Check fitness scores
        if fitness_scores['risk_adjusted_fitness'] < 0.3:
            return False
        
        return True
    
    def _remove_redundant(self, strategies: List[Dict[str, Any]], 
                         backtest_results: List[Dict[str, Any]],
                         fitness_scores: List[Dict[str, float]]) -> Tuple[List, List, List]:
        """
        Remove redundant strategies using clustering and similarity analysis
        """
        if len(strategies) <= 1:
            return strategies, backtest_results, fitness_scores
        
        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(strategies, fitness_scores)
        
        # Cluster similar strategies
        clusters = self._find_similarity_clusters(similarity_matrix)
        
        # Select best strategy from each cluster
        pruned_strategies = []
        pruned_results = []
        pruned_scores = []
        
        for cluster in clusters:
            if len(cluster) == 1:
                # Single strategy, keep it
                idx = cluster[0]
                pruned_strategies.append(strategies[idx])
                pruned_results.append(backtest_results[idx])
                pruned_scores.append(fitness_scores[idx])
            else:
                # Multiple similar strategies, keep the best one
                best_idx = self._select_best_from_cluster(cluster, fitness_scores)
                pruned_strategies.append(strategies[best_idx])
                pruned_results.append(backtest_results[best_idx])
                pruned_scores.append(fitness_scores[best_idx])
        
        self.logger.debug(f"Removed redundant strategies, kept {len(pruned_strategies)}")
        return pruned_strategies, pruned_results, pruned_scores
    
    def _calculate_similarity_matrix(self, strategies: List[Dict[str, Any]], 
                                    fitness_scores: List[Dict[str, float]]) -> np.ndarray:
        """
        Calculate similarity matrix between strategies
        """
        n = len(strategies)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity = self._calculate_strategy_similarity(
                        strategies[i], strategies[j], 
                        fitness_scores[i], fitness_scores[j]
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _calculate_strategy_similarity(self, strategy_a: Dict[str, Any], 
                                     strategy_b: Dict[str, Any],
                                     fitness_a: Dict[str, float],
                                     fitness_b: Dict[str, float]) -> float:
        """
        Calculate similarity between two strategies
        """
        # Parameter similarity
        param_similarity = self._calculate_parameter_similarity(strategy_a, strategy_b)
        
        # Fitness similarity
        fitness_similarity = self._calculate_fitness_similarity(fitness_a, fitness_b)
        
        # Combined similarity (weighted average)
        similarity = 0.6 * param_similarity + 0.4 * fitness_similarity
        
        return similarity
    
    def _calculate_parameter_similarity(self, strategy_a: Dict[str, Any], 
                                      strategy_b: Dict[str, Any]) -> float:
        """
        Calculate similarity based on strategy parameters
        """
        params_a = strategy_a.get('parameters', {})
        params_b = strategy_b.get('parameters', {})
        
        if not params_a or not params_b:
            return 0.0
        
        # Compare common parameters
        common_params = set(params_a.keys()) & set(params_b.keys())
        
        if not common_params:
            return 0.0
        
        # Calculate parameter differences
        total_diff = 0.0
        for param in common_params:
            val_a = params_a[param]
            val_b = params_b[param]
            
            # Handle different parameter types
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # Numeric parameters
                diff = abs(val_a - val_b)
                # Normalize by range (simple approach)
                normalized_diff = diff / (1 + diff)
                total_diff += normalized_diff
            else:
                # Non-numeric parameters (e.g., strings, booleans)
                if val_a == val_b:
                    total_diff += 0.0
                else:
                    total_diff += 1.0
        
        # Average difference
        avg_diff = total_diff / len(common_params)
        
        # Similarity is 1 - average difference
        similarity = 1 - avg_diff
        
        return similarity
    
    def _calculate_fitness_similarity(self, fitness_a: Dict[str, float], 
                                    fitness_b: Dict[str, float]) -> float:
        """
        Calculate similarity based on fitness scores
        """
        # Convert fitness scores to vectors
        objectives = set(fitness_a.keys()) & set(fitness_b.keys())
        
        if not objectives:
            return 0.0
        
        # Create fitness vectors
        fitness_vec_a = np.array([fitness_a[obj] for obj in objectives])
        fitness_vec_b = np.array([fitness_b[obj] for obj in objectives])
        
        # Calculate cosine similarity
        dot_product = np.dot(fitness_vec_a, fitness_vec_b)
        norm_a = np.linalg.norm(fitness_vec_a)
        norm_b = np.linalg.norm(fitness_vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        return similarity
    
    def _find_similarity_clusters(self, similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        Find clusters of similar strategies
        """
        n = similarity_matrix.shape[0]
        visited = set()
        clusters = []
        
        for i in range(n):
            if i not in visited:
                cluster = [i]
                visited.add(i)
                
                # Find all strategies similar to this one
                for j in range(i + 1, n):
                    if similarity_matrix[i, j] >= self.similarity_threshold:
                        cluster.append(j)
                        visited.add(j)
                
                clusters.append(cluster)
        
        return clusters
    
    def _select_best_from_cluster(self, cluster: List[int], 
                                 fitness_scores: List[Dict[str, float]]) -> int:
        """
        Select the best strategy from a cluster of similar strategies
        """
        # Calculate composite fitness for each strategy in cluster
        composite_scores = []
        
        for idx in cluster:
            scores = fitness_scores[idx]
            # Simple composite score
            composite = np.mean(list(scores.values()))
            composite_scores.append(composite)
        
        # Return index of strategy with highest composite score
        best_idx_in_cluster = cluster[np.argmax(composite_scores)]
        
        return best_idx_in_cluster
    
    def _select_top_strategies(self, strategies: List[Dict[str, Any]], 
                              backtest_results: List[Dict[str, Any]],
                              fitness_scores: List[Dict[str, float]],
                              target_size: int) -> Tuple[List, List, List]:
        """
        Select top strategies based on composite fitness
        """
        # Calculate composite fitness scores
        composite_scores = []
        for scores in fitness_scores:
            composite = np.mean(list(scores.values()))
            composite_scores.append(composite)
        
        # Sort by composite score (descending)
        sorted_indices = np.argsort(composite_scores)[::-1]
        
        # Select top strategies
        selected_indices = sorted_indices[:target_size]
        
        selected_strategies = [strategies[i] for i in selected_indices]
        selected_results = [backtest_results[i] for i in selected_indices]
        selected_scores = [fitness_scores[i] for i in selected_indices]
        
        return selected_strategies, selected_results, selected_scores