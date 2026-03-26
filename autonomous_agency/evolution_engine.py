"""
Evolution Engine for the Autonomous Quantitative Research Agency

This module implements genetic algorithm-inspired evolution mechanisms for
strategy refinement, mutation, and pruning. It evolves strategies through
iterative improvement while maintaining mathematical elegance and robustness.
"""

import os
import logging
import random
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import hashlib

from .config import config
from .evaluator import ValidationResult, EvaluationMetrics
from .hypothesis_generator import Hypothesis


@dataclass
class StrategyGenome:
    """Genetic representation of a trading strategy"""
    strategy_name: str
    hypothesis_id: str
    generation: int
    fitness_score: float

    # Core strategy parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Strategy DNA (serialized hypothesis and implementation details)
    dna: Dict[str, Any] = field(default_factory=dict)

    # Evolution metadata
    parent_strategies: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    crossover_history: List[Dict[str, Any]] = field(default_factory=list)

    # Performance lineage
    performance_history: List[Dict[str, Any]] = field(default_factory=list)

    # Elegance metrics
    complexity_score: float = 0.0
    mathematical_beauty: float = 0.0
    computational_efficiency: float = 0.0


@dataclass
class EvolutionResult:
    """Result of an evolution cycle"""
    generation: int
    population_size: int
    elite_strategies: List[StrategyGenome]
    evolved_strategies: List[StrategyGenome]
    pruned_strategies: List[str]
    new_hypotheses: List[Hypothesis]
    evolution_metrics: Dict[str, float]


class EvolutionOperator(ABC):
    """Abstract base class for evolution operators"""

    @abstractmethod
    def apply(self, genome: StrategyGenome, **kwargs) -> StrategyGenome:
        """Apply the evolution operator to a genome"""
        pass

    @abstractmethod
    def get_probability(self) -> float:
        """Get the probability of applying this operator"""
        pass


class ParameterMutation(EvolutionOperator):
    """Parameter mutation operator"""

    def __init__(self, mutation_rate: float = 0.1, mutation_strength: float = 0.2):
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

    def apply(self, genome: StrategyGenome, **kwargs) -> StrategyGenome:
        """Apply parameter mutation"""
        mutated_genome = StrategyGenome(
            strategy_name=f"{genome.strategy_name}_mut_{random.randint(1000, 9999)}",
            hypothesis_id=genome.hypothesis_id,
            generation=genome.generation + 1,
            fitness_score=0.0,  # Will be recalculated
            parameters=genome.parameters.copy(),
            dna=genome.dna.copy(),
            parent_strategies=[genome.strategy_name],
            mutation_history=genome.mutation_history.copy(),
            crossover_history=genome.crossover_history.copy(),
            performance_history=genome.performance_history.copy(),
            complexity_score=genome.complexity_score,
            mathematical_beauty=genome.mathematical_beauty,
            computational_efficiency=genome.computational_efficiency
        )

        # Mutate parameters
        for param_name, param_value in mutated_genome.parameters.items():
            if random.random() < self.mutation_rate:
                mutated_value = self._mutate_parameter(param_value)
                mutated_genome.parameters[param_name] = mutated_value

                # Record mutation
                mutated_genome.mutation_history.append({
                    'generation': genome.generation + 1,
                    'parameter': param_name,
                    'old_value': param_value,
                    'new_value': mutated_value,
                    'operator': 'parameter_mutation'
                })

        return mutated_genome

    def _mutate_parameter(self, value: Any) -> Any:
        """Mutate a parameter value"""
        if isinstance(value, (int, float)):
            # Gaussian mutation
            if isinstance(value, int):
                mutation = np.random.normal(0, abs(value) * self.mutation_strength)
                return int(value + mutation)
            else:
                mutation = np.random.normal(0, abs(value) * self.mutation_strength)
                return value + mutation
        elif isinstance(value, bool):
            # Flip with probability
            return not value if random.random() < 0.5 else value
        elif isinstance(value, list):
            # Mutate list elements
            mutated_list = value.copy()
            if mutated_list and random.random() < 0.5:
                idx = random.randint(0, len(mutated_list) - 1)
                mutated_list[idx] = self._mutate_parameter(mutated_list[idx])
            return mutated_list
        else:
            # No mutation for other types
            return value

    def get_probability(self) -> float:
        return 0.3  # 30% chance of parameter mutation


class StructureMutation(EvolutionOperator):
    """Strategy structure mutation operator"""

    def __init__(self, mutation_rate: float = 0.05):
        self.mutation_rate = mutation_rate

    def apply(self, genome: StrategyGenome, **kwargs) -> StrategyGenome:
        """Apply structure mutation"""
        mutated_genome = StrategyGenome(
            strategy_name=f"{genome.strategy_name}_struct_mut_{random.randint(1000, 9999)}",
            hypothesis_id=genome.hypothesis_id,
            generation=genome.generation + 1,
            fitness_score=0.0,
            parameters=genome.parameters.copy(),
            dna=genome.dna.copy(),
            parent_strategies=[genome.strategy_name],
            mutation_history=genome.mutation_history.copy(),
            crossover_history=genome.crossover_history.copy(),
            performance_history=genome.performance_history.copy(),
            complexity_score=genome.complexity_score,
            mathematical_beauty=genome.mathematical_beauty,
            computational_efficiency=genome.computational_efficiency
        )

        # Apply structural changes to DNA
        if random.random() < self.mutation_rate:
            # Add new indicator
            if 'indicators' in mutated_genome.dna:
                new_indicator = self._generate_random_indicator()
                mutated_genome.dna['indicators'].append(new_indicator)

                mutated_genome.mutation_history.append({
                    'generation': genome.generation + 1,
                    'type': 'structure_mutation',
                    'operation': 'add_indicator',
                    'details': new_indicator
                })

        if random.random() < self.mutation_rate:
            # Modify signal logic
            if 'signal_logic' in mutated_genome.dna:
                modified_logic = self._modify_signal_logic(mutated_genome.dna['signal_logic'])
                mutated_genome.dna['signal_logic'] = modified_logic

                mutated_genome.mutation_history.append({
                    'generation': genome.generation + 1,
                    'type': 'structure_mutation',
                    'operation': 'modify_signal_logic',
                    'details': 'signal_logic_modified'
                })

        return mutated_genome

    def _generate_random_indicator(self) -> Dict[str, Any]:
        """Generate a random indicator configuration"""
        indicators = ['SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'STOCH', 'CCI', 'MFI']
        indicator = random.choice(indicators)

        if indicator == 'SMA':
            return {'type': 'SMA', 'period': random.randint(5, 50)}
        elif indicator == 'EMA':
            return {'type': 'EMA', 'period': random.randint(5, 50)}
        elif indicator == 'RSI':
            return {'type': 'RSI', 'period': random.randint(7, 21)}
        elif indicator == 'MACD':
            return {
                'type': 'MACD',
                'fastperiod': random.randint(8, 20),
                'slowperiod': random.randint(21, 40),
                'signalperiod': random.randint(5, 15)
            }
        else:
            return {'type': indicator, 'period': random.randint(10, 30)}

    def _modify_signal_logic(self, logic: Dict[str, Any]) -> Dict[str, Any]:
        """Modify signal generation logic"""
        # Simple modifications for now
        modified_logic = logic.copy()

        if 'conditions' in modified_logic:
            if random.random() < 0.5 and len(modified_logic['conditions']) > 1:
                # Remove a condition
                idx = random.randint(0, len(modified_logic['conditions']) - 1)
                removed = modified_logic['conditions'].pop(idx)
            elif random.random() < 0.3:
                # Add a simple condition
                new_condition = {
                    'indicator': 'SMA',
                    'comparison': 'crosses_above',
                    'value': random.randint(10, 50)
                }
                modified_logic['conditions'].append(new_condition)

        return modified_logic

    def get_probability(self) -> float:
        return 0.1  # 10% chance of structure mutation


class CrossoverOperator(EvolutionOperator):
    """Crossover operator for combining strategy traits"""

    def __init__(self, crossover_rate: float = 0.2):
        self.crossover_rate = crossover_rate

    def apply(self, genome1: StrategyGenome, genome2: StrategyGenome, **kwargs) -> StrategyGenome:
        """Apply crossover between two genomes"""
        child_genome = StrategyGenome(
            strategy_name=f"crossover_{random.randint(1000, 9999)}",
            hypothesis_id=f"{genome1.hypothesis_id}_{genome2.hypothesis_id}",
            generation=max(genome1.generation, genome2.generation) + 1,
            fitness_score=0.0,
            parameters={},
            dna={},
            parent_strategies=[genome1.strategy_name, genome2.strategy_name],
            mutation_history=[],
            crossover_history=[{
                'generation': max(genome1.generation, genome2.generation) + 1,
                'parents': [genome1.strategy_name, genome2.strategy_name],
                'crossover_type': 'parameter_crossover'
            }],
            performance_history=[],
            complexity_score=(genome1.complexity_score + genome2.complexity_score) / 2,
            mathematical_beauty=(genome1.mathematical_beauty + genome2.mathematical_beauty) / 2,
            computational_efficiency=(genome1.computational_efficiency + genome2.computational_efficiency) / 2
        )

        # Parameter crossover
        all_params = set(genome1.parameters.keys()) | set(genome2.parameters.keys())
        for param in all_params:
            if param in genome1.parameters and param in genome2.parameters:
                # Average numeric parameters
                if isinstance(genome1.parameters[param], (int, float)) and isinstance(genome2.parameters[param], (int, float)):
                    child_genome.parameters[param] = (genome1.parameters[param] + genome2.parameters[param]) / 2
                else:
                    # Random choice for other types
                    child_genome.parameters[param] = random.choice([genome1.parameters[param], genome2.parameters[param]])
            elif param in genome1.parameters:
                child_genome.parameters[param] = genome1.parameters[param]
            else:
                child_genome.parameters[param] = genome2.parameters[param]

        # DNA crossover
        for key in set(genome1.dna.keys()) | set(genome2.dna.keys()):
            if key in genome1.dna and key in genome2.dna:
                # Merge DNA elements
                if isinstance(genome1.dna[key], list) and isinstance(genome2.dna[key], list):
                    # Combine lists
                    combined = genome1.dna[key] + genome2.dna[key]
                    # Remove duplicates if they exist
                    if combined and isinstance(combined[0], dict):
                        seen = set()
                        unique = []
                        for item in combined:
                            item_hash = hashlib.md5(json.dumps(item, sort_keys=True).encode()).hexdigest()
                            if item_hash not in seen:
                                seen.add(item_hash)
                                unique.append(item)
                        child_genome.dna[key] = unique[:len(unique)//2 + 1]  # Take roughly half
                    else:
                        child_genome.dna[key] = combined[:len(combined)//2 + 1]
                else:
                    # Random choice
                    child_genome.dna[key] = random.choice([genome1.dna[key], genome2.dna[key]])
            elif key in genome1.dna:
                child_genome.dna[key] = genome1.dna[key]
            else:
                child_genome.dna[key] = genome2.dna[key]

        return child_genome

    def get_probability(self) -> float:
        return 0.2  # 20% chance of crossover


class StrategyEvolutionEngine:
    """Main evolution engine for strategy optimization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.evolution_dir = Path(config.evolution_results_dir)
        self.evolution_dir.mkdir(parents=True, exist_ok=True)

        # Evolution operators
        self.operators = [
            ParameterMutation(),
            StructureMutation(),
            CrossoverOperator()
        ]

        # Evolution parameters
        self.population_size = config.evolution_population_size
        self.elitism_rate = config.evolution_elitism_rate
        self.mutation_rate = config.evolution_mutation_rate
        self.crossover_rate = config.evolution_crossover_rate

        # Strategy population
        self.population: List[StrategyGenome] = []
        self.generation = 0

        # Evolution history
        self.evolution_history: List[EvolutionResult] = []

    def initialize_population(self, validation_results: List[ValidationResult]) -> List[StrategyGenome]:
        """
        Initialize population from validated strategies

        Args:
            validation_results: Results from strategy evaluation

        Returns:
            Initial population of strategy genomes
        """
        population = []

        # Sort by overall score and take top performers
        sorted_results = sorted(validation_results,
                              key=lambda x: x.evaluation_metrics.overall_score,
                              reverse=True)

        for result in sorted_results[:self.population_size]:
            genome = self._create_genome_from_validation(result)
            population.append(genome)

        # Fill remaining slots with random variations if needed
        while len(population) < self.population_size:
            if sorted_results:
                base_result = random.choice(sorted_results)
                genome = self._create_genome_from_validation(base_result)
                # Apply some initial mutation
                mutation_op = ParameterMutation(mutation_rate=0.2)
                genome = mutation_op.apply(genome)
                population.append(genome)

        self.population = population
        self.generation = 0

        self.logger.info(f"Initialized population with {len(population)} strategies")
        return population

    def evolve_population(self, validation_results: List[ValidationResult]) -> EvolutionResult:
        """
        Evolve the population for one generation

        Args:
            validation_results: Latest validation results

        Returns:
            Evolution result for this generation
        """
        try:
            # Update fitness scores
            self._update_fitness_scores(validation_results)

            # Sort population by fitness
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)

            # Elitism: keep top performers
            elite_count = int(self.population_size * self.elitism_rate)
            elite_strategies = self.population[:elite_count]

            # Generate new strategies through evolution
            evolved_strategies = []
            new_hypotheses = []

            # Apply evolution operators
            while len(evolved_strategies) < (self.population_size - elite_count):
                operation = random.random()

                if operation < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = random.sample(self.population[:self.population_size//2], 2)
                    crossover_op = CrossoverOperator()
                    child = crossover_op.apply(parent1, parent2)
                    evolved_strategies.append(child)

                elif operation < (self.crossover_rate + self.mutation_rate):
                    # Mutation
                    parent = random.choice(self.population[:self.population_size//2])
                    mutation_op = random.choice([ParameterMutation(), StructureMutation()])
                    mutant = mutation_op.apply(parent)
                    evolved_strategies.append(mutant)

                else:
                    # Random new hypothesis (exploration)
                    hypothesis = self._generate_exploratory_hypothesis()
                    if hypothesis:
                        new_hypotheses.append(hypothesis)

            # Prune weak strategies
            pruned_strategies = self._prune_population()

            # Update population
            self.population = elite_strategies + evolved_strategies
            self.generation += 1

            # Calculate evolution metrics
            evolution_metrics = self._calculate_evolution_metrics()

            # Create evolution result
            evolution_result = EvolutionResult(
                generation=self.generation,
                population_size=len(self.population),
                elite_strategies=elite_strategies,
                evolved_strategies=evolved_strategies,
                pruned_strategies=pruned_strategies,
                new_hypotheses=new_hypotheses,
                evolution_metrics=evolution_metrics
            )

            # Save evolution result
            self._save_evolution_result(evolution_result)
            self.evolution_history.append(evolution_result)

            self.logger.info(f"Completed evolution generation {self.generation}")
            return evolution_result

        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
            return None

    def _create_genome_from_validation(self, validation_result: ValidationResult) -> StrategyGenome:
        """Create a strategy genome from validation result"""

        # Extract parameters from backtest result (this would need to be implemented based on strategy structure)
        parameters = getattr(validation_result.backtest_result, 'parameters', {})

        # Create basic DNA structure
        dna = {
            'hypothesis': validation_result.backtest_result.hypothesis_id,
            'indicators': [],  # Would be populated from strategy analysis
            'signal_logic': {},  # Would be populated from strategy analysis
            'risk_management': {}  # Would be populated from strategy analysis
        }

        genome = StrategyGenome(
            strategy_name=validation_result.strategy_name,
            hypothesis_id=validation_result.backtest_result.hypothesis_id,
            generation=0,
            fitness_score=validation_result.evaluation_metrics.overall_score,
            parameters=parameters,
            dna=dna,
            complexity_score=self._calculate_complexity_score(validation_result),
            mathematical_beauty=self._calculate_mathematical_beauty(validation_result),
            computational_efficiency=self._calculate_computational_efficiency(validation_result)
        )

        return genome

    def _update_fitness_scores(self, validation_results: List[ValidationResult]):
        """Update fitness scores for population based on latest validation"""

        result_dict = {result.strategy_name: result for result in validation_results}

        for genome in self.population:
            if genome.strategy_name in result_dict:
                result = result_dict[genome.strategy_name]
                genome.fitness_score = result.evaluation_metrics.overall_score

                # Update performance history
                genome.performance_history.append({
                    'generation': self.generation,
                    'fitness_score': genome.fitness_score,
                    'validation_status': result.validation_status,
                    'rejection_reasons': result.rejection_reasons,
                    'refinement_suggestions': result.refinement_suggestions
                })

    def _prune_population(self) -> List[str]:
        """Prune weak strategies from population"""

        # Identify strategies to prune (bottom performers)
        prune_count = max(1, int(self.population_size * 0.1))  # Prune 10% worst
        pruned_strategies = []

        if len(self.population) > self.population_size:
            # Sort by fitness and remove weakest
            self.population.sort(key=lambda x: x.fitness_score, reverse=True)
            pruned = self.population[self.population_size:]
            pruned_strategies = [genome.strategy_name for genome in pruned]
            self.population = self.population[:self.population_size]

        return pruned_strategies

    def _generate_exploratory_hypothesis(self) -> Optional[Hypothesis]:
        """Generate a new exploratory hypothesis for innovation"""

        # This would integrate with the hypothesis generator
        # For now, return None (would be implemented when hypothesis generator is available)
        return None

    def _calculate_complexity_score(self, validation_result: ValidationResult) -> float:
        """Calculate strategy complexity score"""

        # Simple complexity based on number of parameters and indicators
        # Lower complexity is generally better (Occam's razor)
        param_count = len(getattr(validation_result.backtest_result, 'parameters', {}))
        complexity = min(param_count / 20, 1.0)  # Normalize to 0-1

        return complexity

    def _calculate_mathematical_beauty(self, validation_result: ValidationResult) -> float:
        """Calculate mathematical beauty score"""

        # This is subjective - based on symmetry, elegance of logic, etc.
        # For now, use a combination of consistency and robustness
        metrics = validation_result.evaluation_metrics
        beauty = (metrics.consistency_score + metrics.robustness_score) / 2

        return beauty

    def _calculate_computational_efficiency(self, validation_result: ValidationResult) -> float:
        """Calculate computational efficiency score"""

        # Based on backtest performance metrics (simulated time, etc.)
        # For now, assume reasonable efficiency
        return 0.8

    def _calculate_evolution_metrics(self) -> Dict[str, float]:
        """Calculate metrics for the evolution process"""

        if not self.population:
            return {}

        fitness_scores = [genome.fitness_score for genome in self.population]

        return {
            'mean_fitness': np.mean(fitness_scores),
            'max_fitness': max(fitness_scores),
            'min_fitness': min(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'diversity_score': self._calculate_diversity_score(),
            'convergence_score': self._calculate_convergence_score()
        }

    def _calculate_diversity_score(self) -> float:
        """Calculate population diversity"""

        if len(self.population) < 2:
            return 0.0

        # Simple diversity based on parameter differences
        param_vectors = []
        for genome in self.population:
            param_vector = []
            for param_name in sorted(genome.parameters.keys()):
                param_value = genome.parameters[param_name]
                if isinstance(param_value, (int, float)):
                    param_vector.append(float(param_value))
                elif isinstance(param_value, bool):
                    param_vector.append(1.0 if param_value else 0.0)
                else:
                    # Hash string representations for diversity
                    param_vector.append(hash(str(param_value)) % 1000 / 1000.0)

            if param_vector:
                param_vectors.append(param_vector)

        if not param_vectors:
            return 0.0

        # Calculate average pairwise distance
        distances = []
        for i in range(len(param_vectors)):
            for j in range(i+1, len(param_vectors)):
                if len(param_vectors[i]) == len(param_vectors[j]):
                    dist = np.linalg.norm(np.array(param_vectors[i]) - np.array(param_vectors[j]))
                    distances.append(dist)

        if distances:
            avg_distance = np.mean(distances)
            # Normalize diversity score (higher = more diverse)
            diversity = min(avg_distance / 10, 1.0)
        else:
            diversity = 0.0

        return diversity

    def _calculate_convergence_score(self) -> float:
        """Calculate population convergence"""

        if not self.population:
            return 0.0

        fitness_scores = [genome.fitness_score for genome in self.population]
        fitness_std = np.std(fitness_scores)

        # Lower standard deviation = higher convergence
        convergence = 1.0 - min(fitness_std, 1.0)

        return convergence

    def _save_evolution_result(self, evolution_result: EvolutionResult):
        """Save evolution result to disk"""

        try:
            result_file = self.evolution_dir / f"evolution_gen_{evolution_result.generation}.json"

            result_dict = {
                'generation': evolution_result.generation,
                'population_size': evolution_result.population_size,
                'elite_strategies': [
                    {
                        'name': genome.strategy_name,
                        'fitness': genome.fitness_score,
                        'generation': genome.generation
                    } for genome in evolution_result.elite_strategies
                ],
                'evolved_strategies_count': len(evolution_result.evolved_strategies),
                'pruned_strategies': evolution_result.pruned_strategies,
                'new_hypotheses_count': len(evolution_result.new_hypotheses),
                'evolution_metrics': evolution_result.evolution_metrics
            }

            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save evolution result: {e}")

    def get_best_strategies(self, count: int = 5) -> List[StrategyGenome]:
        """Get the best performing strategies"""

        sorted_population = sorted(self.population,
                                 key=lambda x: x.fitness_score,
                                 reverse=True)
        return sorted_population[:count]

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process"""

        if not self.evolution_history:
            return {}

        latest_result = self.evolution_history[-1]

        return {
            'current_generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max([g.fitness_score for g in self.population]) if self.population else 0,
            'mean_fitness': np.mean([g.fitness_score for g in self.population]) if self.population else 0,
            'evolution_metrics': latest_result.evolution_metrics if latest_result else {},
            'total_evolved_strategies': sum(len(r.evolved_strategies) for r in self.evolution_history),
            'total_pruned_strategies': sum(len(r.pruned_strategies) for r in self.evolution_history)
        }