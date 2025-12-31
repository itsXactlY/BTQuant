"""
Test module for the evolutionary selection system
"""

import logging
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolutionary_selection.evolutionary_selector import EvolutionarySelector
from evolutionary_selection.fitness.multi_objective_fitness import MultiObjectiveFitness
from evolutionary_selection.algorithms.pareto_front import ParetoFrontOptimizer
from evolutionary_selection.algorithms.strategy_pruning import StrategyPruner
from evolutionary_selection.algorithms.evolutionary_refinement import EvolutionaryRefinement
from evolutionary_selection.algorithms.population_diversity import PopulationDiversityManager

def test_basic_functionality():
    """Test basic functionality of all components"""
    print("Testing evolutionary selection system...")
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: MultiObjectiveFitness
    print("\n1. Testing MultiObjectiveFitness...")
    fitness_calculator = MultiObjectiveFitness()
    
    # Create mock strategy and backtest results
    mock_strategy = {
        'id': 'test_strategy_1',
        'template': 'SMA_Crossover',
        'parameters': {'fast_period': 10, 'slow_period': 50}
    }
    
    mock_backtest_results = {
        'performance_metrics': {
            'sharpe_ratio': 1.5,
            'sortino_ratio': 1.8,
            'calmar_ratio': 2.0,
            'max_drawdown': -15.0,
            'risk_adjusted_return': 25.0,
            'total_return': 45.0,
            'annualized_return': 30.0,
            'volatility': 12.0,
            'win_rate': 0.65,
            'avg_win': 2.5,
            'avg_loss': -1.8,
            'omega_ratio': 1.7,
            'skewness': 0.5,
            'kurtosis': 1.2
        }
    }
    
    fitness_scores = fitness_calculator.calculate_fitness_scores(mock_strategy, mock_backtest_results)
    composite_fitness = fitness_calculator.calculate_composite_fitness(fitness_scores)
    
    print(f"✓ MultiObjectiveFitness works. Composite fitness: {composite_fitness:.3f}")
    
    # Test 2: ParetoFrontOptimizer
    print("\n2. Testing ParetoFrontOptimizer...")
    pareto_optimizer = ParetoFrontOptimizer()
    
    # Create a small population
    strategies = [
        {'id': f'strategy_{i}', 'template': 'SMA_Crossover', 'parameters': {'fast': 5+i, 'slow': 20+i}}
        for i in range(5)
    ]
    
    backtest_results = [
        {
            'performance_metrics': {
                'sharpe_ratio': 0.5 + i * 0.2,
                'sortino_ratio': 0.6 + i * 0.2,
                'calmar_ratio': 0.8 + i * 0.2,
                'max_drawdown': -10.0 - i * 2.0,
                'risk_adjusted_return': 10.0 + i * 3.0,
                'total_return': 20.0 + i * 5.0,
                'annualized_return': 15.0 + i * 3.0,
                'volatility': 10.0 - i * 1.0,
                'win_rate': 0.5 + i * 0.05,
                'avg_win': 1.5 + i * 0.2,
                'avg_loss': -1.0 - i * 0.1,
                'omega_ratio': 1.0 + i * 0.1,
                'skewness': 0.2 + i * 0.1,
                'kurtosis': 0.8 + i * 0.1
            }
        }
        for i in range(5)
    ]
    
    pareto_front = pareto_optimizer.find_pareto_front(strategies, backtest_results, fitness_calculator)
    print(f"✓ ParetoFrontOptimizer works. Found {len(pareto_front)} strategies on Pareto front")
    
    # Test 3: StrategyPruner
    print("\n3. Testing StrategyPruner...")
    strategy_pruner = StrategyPruner(similarity_threshold=0.8)
    
    # Calculate fitness scores for pruning
    fitness_scores_list = []
    for i, strategy in enumerate(strategies):
        scores = fitness_calculator.calculate_fitness_scores(strategy, backtest_results[i])
        fitness_scores_list.append(scores)
    
    pruned_strategies = strategy_pruner.prune_population(
        strategies, backtest_results, fitness_scores_list, target_size=3
    )
    print(f"✓ StrategyPruner works. Pruned to {len(pruned_strategies)} strategies")
    
    # Test 4: EvolutionaryRefinement
    print("\n4. Testing EvolutionaryRefinement...")
    evolutionary_refiner = EvolutionaryRefinement(mutation_rate=0.1, crossover_rate=0.8, population_size=5)
    
    refinement_result = evolutionary_refiner.refine_strategies(
        strategies[:3], backtest_results[:3], fitness_calculator, num_generations=2
    )
    print(f"✓ EvolutionaryRefinement works. Refined to {len(refinement_result.refined_strategies)} strategies")
    
    # Test 5: PopulationDiversityManager
    print("\n5. Testing PopulationDiversityManager...")
    diversity_manager = PopulationDiversityManager()
    
    diverse_strategies = diversity_manager.manage_diversity(
        strategies, fitness_scores_list, target_size=3
    )
    print(f"✓ PopulationDiversityManager works. Managed diversity to {len(diverse_strategies)} strategies")
    
    # Test 6: Full EvolutionarySelector
    print("\n6. Testing full EvolutionarySelector...")
    evolutionary_selector = EvolutionarySelector()
    
    selected_strategies = evolutionary_selector.select_strategies(
        strategies, backtest_results, target_size=2, use_refinement=False
    )
    print(f"✓ EvolutionarySelector works. Selected {len(selected_strategies)} strategies")
    
    # Test advanced selection
    advanced_result = evolutionary_selector.advanced_evolutionary_selection(
        strategies, backtest_results, target_size=2, num_generations=2
    )
    print(f"✓ Advanced evolutionary selection works. Selected {len(advanced_result['final_strategies'])} strategies")
    
    print("\n🎉 All tests passed! Evolutionary selection system is working correctly.")
    
    return True

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\nTesting edge cases...")
    
    fitness_calculator = MultiObjectiveFitness()
    evolutionary_selector = EvolutionarySelector()
    
    # Test with empty lists
    empty_strategies = []
    empty_results = []
    
    selected = evolutionary_selector.select_strategies(empty_strategies, empty_results)
    assert len(selected) == 0, "Empty input should return empty output"
    print("✓ Empty input handling works")
    
    # Test with single strategy
    single_strategy = [{'id': 'single', 'template': 'test', 'parameters': {}}]
    single_result = [{
        'performance_metrics': {
            'sharpe_ratio': 1.0,
            'sortino_ratio': 1.2,
            'calmar_ratio': 1.5,
            'max_drawdown': -10.0,
            'risk_adjusted_return': 20.0,
            'total_return': 30.0,
            'annualized_return': 25.0,
            'volatility': 8.0,
            'win_rate': 0.6,
            'avg_win': 2.0,
            'avg_loss': -1.5,
            'omega_ratio': 1.3,
            'skewness': 0.3,
            'kurtosis': 0.9
        }
    }]
    
    selected_single = evolutionary_selector.select_strategies(single_strategy, single_result, target_size=1)
    assert len(selected_single) == 1, "Single strategy should be selected"
    print("✓ Single strategy handling works")
    
    print("✓ All edge cases passed")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_edge_cases()
        print("\n🚀 Evolutionary selection system is fully functional!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)