"""
Novelty Detection System

Implements algorithms to ensure diverse strategy generation by detecting and avoiding
similar strategies in the population.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy.spatial import distance
import random

class NoveltyDetector:
    """Detects novelty in generated strategies to ensure diversity"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.logger = logging.getLogger('NoveltyDetector')
        self.logger.info("NoveltyDetector initialized")
        self.similarity_threshold = similarity_threshold
        self.strategy_archive = []  # Archive of seen strategies
        
    def calculate_strategy_similarity(self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]) -> float:
        """
        Calculate similarity between two strategies based on their parameters
        
        Args:
            strategy1: First strategy
            strategy2: Second strategy
            
        Returns:
            Similarity score between 0 (completely different) and 1 (identical)
        """
        # Get parameters from both strategies
        params1 = strategy1['parameters']
        params2 = strategy2['parameters']
        
        # Get common parameter keys
        common_keys = set(params1.keys()) & set(params2.keys())
        
        if not common_keys:
            return 0.0  # No common parameters = completely different
        
        # Normalize parameter values for comparison
        normalized_params = []
        
        for key in common_keys:
            val1, val2 = params1[key], params2[key]
            
            # Handle different parameter types
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize numeric values to 0-1 range based on typical ranges
                if key in ['fast_period', 'slow_period', 'rsi_period', 'bb_period', 'atr_period']:
                    # Time periods typically 1-200
                    norm_val1 = val1 / 200.0
                    norm_val2 = val2 / 200.0
                elif key in ['stop_loss_pct', 'take_profit_pct', 'position_size_pct']:
                    # Percentages 0-1
                    norm_val1 = val1
                    norm_val2 = val2
                elif key in ['bb_std_dev', 'atr_multiplier']:
                    # Multipliers typically 0.5-3.0
                    norm_val1 = (val1 - 0.5) / 2.5
                    norm_val2 = (val2 - 0.5) / 2.5
                else:
                    # Default normalization
                    norm_val1 = val1 / (val1 + val2 + 1e-6)
                    norm_val2 = val2 / (val1 + val2 + 1e-6)
                
                normalized_params.append((norm_val1, norm_val2))
            elif isinstance(val1, bool) and isinstance(val2, bool):
                # Boolean values: 0 for False, 1 for True
                normalized_params.append((1.0 if val1 else 0.0, 1.0 if val2 else 0.0))
            elif isinstance(val1, str) and isinstance(val2, str):
                # String values: 1 if same, 0 if different
                similarity = 1.0 if val1 == val2 else 0.0
                normalized_params.append((similarity, similarity))
        
        if not normalized_params:
            return 0.0
        
        # Convert to arrays for distance calculation
        array1 = np.array([x[0] for x in normalized_params])
        array2 = np.array([x[1] for x in normalized_params])
        
        # Calculate cosine similarity (1 - cosine distance)
        cosine_sim = 1 - distance.cosine(array1, array2)
        
        # Ensure similarity is in valid range
        similarity = max(0.0, min(1.0, cosine_sim))
        
        self.logger.debug(f"Strategy similarity: {similarity:.3f}")
        return similarity
    
    def is_novel(self, strategy: Dict[str, Any], population: List[Dict[str, Any]] = None) -> bool:
        """
        Check if a strategy is novel compared to existing strategies
        
        Args:
            strategy: Strategy to check for novelty
            population: Optional population to compare against (uses archive if None)
            
        Returns:
            True if strategy is novel, False if it's too similar to existing strategies
        """
        comparison_pool = population if population is not None else self.strategy_archive
        
        if not comparison_pool:
            self.logger.debug("No comparison pool available, strategy considered novel")
            return True
        
        # Calculate similarity to all strategies in comparison pool
        similarities = []
        
        for existing_strategy in comparison_pool:
            similarity = self.calculate_strategy_similarity(strategy, existing_strategy)
            similarities.append(similarity)
            
            # Early exit if we find a very similar strategy
            if similarity > self.similarity_threshold:
                self.logger.debug(f"Found similar strategy with similarity {similarity:.3f} > threshold {self.similarity_threshold}")
                return False
        
        # Check if maximum similarity is below threshold
        max_similarity = max(similarities) if similarities else 0.0
        is_novel = max_similarity <= self.similarity_threshold
        
        self.logger.debug(f"Strategy novelty check: max_similarity={max_similarity:.3f}, threshold={self.similarity_threshold}, novel={is_novel}")
        return is_novel
    
    def ensure_diversity(self, population: List[Dict[str, Any]], target_size: int) -> List[Dict[str, Any]]:
        """
        Ensure diversity in a population by removing similar strategies
        
        Args:
            population: Current population of strategies
            target_size: Target population size
            
        Returns:
            Diverse subset of the population
        """
        if len(population) <= target_size:
            self.logger.debug("Population already at or below target size")
            return population
        
        self.logger.info(f"Ensuring diversity in population: {len(population)} -> {target_size}")
        
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(population), len(population)))
        
        for i in range(len(population)):
            for j in range(i, len(population)):
                if i == j:
                    similarity_matrix[i][j] = 1.0  # Strategy is identical to itself
                else:
                    similarity = self.calculate_strategy_similarity(population[i], population[j])
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity
        
        # Select diverse strategies using greedy algorithm
        selected_indices = []
        remaining_indices = list(range(len(population)))
        
        # Start with the most "central" strategy (highest average similarity)
        avg_similarities = similarity_matrix.mean(axis=1)
        start_index = np.argmax(avg_similarities)
        selected_indices.append(start_index)
        remaining_indices.remove(start_index)
        
        # Greedily add strategies that are most different from already selected ones
        while len(selected_indices) < target_size and remaining_indices:
            # Calculate minimum similarity to already selected strategies
            min_similarities = []
            
            for idx in remaining_indices:
                similarities_to_selected = [similarity_matrix[idx][selected] for selected in selected_indices]
                min_similarity = min(similarities_to_selected) if similarities_to_selected else 0.0
                min_similarities.append((idx, min_similarity))
            
            # Select strategy with lowest minimum similarity (most different)
            if min_similarities:
                next_index = min(min_similarities, key=lambda x: x[1])[0]
                selected_indices.append(next_index)
                remaining_indices.remove(next_index)
            else:
                break
        
        # Create diverse population
        diverse_population = [population[i] for i in selected_indices]
        
        self.logger.info(f"Diversity ensured: {len(diverse_population)} strategies selected")
        return diverse_population
    
    def add_to_archive(self, strategy: Dict[str, Any]) -> None:
        """
        Add a strategy to the novelty archive
        
        Args:
            strategy: Strategy to add to archive
        """
        self.strategy_archive.append(strategy)
        self.logger.debug(f"Added strategy to archive. Archive size: {len(self.strategy_archive)}")
    
    def clear_archive(self) -> None:
        """Clear the strategy archive"""
        self.strategy_archive = []
        self.logger.info("Strategy archive cleared")
    
    def get_archive_size(self) -> int:
        """Get the current size of the strategy archive"""
        return len(self.strategy_archive)
    
    def calculate_population_diversity(self, population: List[Dict[str, Any]]) -> float:
        """
        Calculate overall diversity score for a population
        
        Args:
            population: Population to analyze
            
        Returns:
            Diversity score between 0 (all identical) and 1 (all completely different)
        """
        if len(population) <= 1:
            return 1.0  # Single strategy is maximally diverse
        
        # Calculate all pairwise similarities
        similarities = []
        n = len(population)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = self.calculate_strategy_similarity(population[i], population[j])
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Diversity is 1 - average similarity
        avg_similarity = sum(similarities) / len(similarities)
        diversity_score = 1.0 - avg_similarity
        
        self.logger.debug(f"Population diversity score: {diversity_score:.3f}")
        return diversity_score