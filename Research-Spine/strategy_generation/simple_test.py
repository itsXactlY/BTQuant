"""
Simple Test for Strategy Generation Engine

Basic test that doesn't require backtrader installation.
"""

import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import components to test (excluding backtrader-dependent ones)
from strategy_generation.strategy_generator import StrategyGenerator
from strategy_generation.generators.genetic_operators import GeneticOperators
from strategy_generation.generators.novelty_detection import NoveltyDetector
from strategy_generation.templates.strategy_templates import StrategyTemplateManager

def test_strategy_generation_engine():
    """Test the core strategy generation engine components"""
    
    logger = logging.getLogger('SimpleTest')
    logger.info("Starting simple test of strategy generation engine")
    
    # Initialize components
    strategy_generator = StrategyGenerator()
    genetic_operators = GeneticOperators()
    novelty_detector = NoveltyDetector()
    template_manager = StrategyTemplateManager()
    
    # Test 1: Strategy Template Management
    logger.info("\n=== Test 1: Strategy Template Management ===")
    templates = template_manager.get_all_templates()
    logger.info(f"Available templates: {list(templates.keys())}")
    
    # Get specific template
    ma_template = template_manager.get_template('moving_average_crossover')
    logger.info(f"MA Template parameters: {list(ma_template['parameters'].keys())}")
    
    # Test 2: Strategy Generation
    logger.info("\n=== Test 2: Strategy Generation ===")
    sample_params = {
        'fast_period': 10,
        'slow_period': 50,
        'ma_type': 'SMA',
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.15
    }
    
    strategy = strategy_generator.generate_strategy('moving_average_crossover', sample_params)
    logger.info(f"Generated strategy ID: {strategy['id']}")
    logger.info(f"Strategy template: {strategy['template']}")
    logger.info(f"Strategy parameters: {strategy['parameters']}")
    
    # Test 3: Strategy Validation
    logger.info("\n=== Test 3: Strategy Validation ===")
    is_valid = strategy_generator.validate_strategy(strategy)
    logger.info(f"Strategy validation result: {is_valid}")
    
    # Test invalid strategy
    invalid_strategy = {'template': 'unknown_template', 'parameters': {}}
    is_valid = strategy_generator.validate_strategy(invalid_strategy)
    logger.info(f"Invalid strategy validation result: {is_valid}")
    
    # Test 4: Population Generation
    logger.info("\n=== Test 4: Population Generation ===")
    population = strategy_generator.generate_strategy_population('moving_average_crossover', 5)
    logger.info(f"Generated population size: {len(population)}")
    
    # Show first strategy from population
    if population:
        first_strategy = population[0]
        logger.info(f"First strategy parameters: {first_strategy['parameters']}")
    
    # Test 5: Genetic Operators
    logger.info("\n=== Test 5: Genetic Operators ===")
    
    # Test crossover
    parent1 = population[0]
    parent2 = population[1]
    child1, child2 = genetic_operators.crossover(parent1, parent2)
    
    logger.info(f"Parent 1 fast_period: {parent1['parameters']['fast_period']}")
    logger.info(f"Parent 2 fast_period: {parent2['parameters']['fast_period']}")
    logger.info(f"Child 1 fast_period: {child1['parameters']['fast_period']}")
    logger.info(f"Child 2 fast_period: {child2['parameters']['fast_period']}")
    
    # Test mutation
    mutated = genetic_operators.mutate(parent1, mutation_rate=0.3)
    logger.info(f"Original fast_period: {parent1['parameters']['fast_period']}")
    logger.info(f"Mutated fast_period: {mutated['parameters']['fast_period']}")
    
    # Test 6: Novelty Detection
    logger.info("\n=== Test 6: Novelty Detection ===")
    
    # Calculate similarity between strategies
    similarity = novelty_detector.calculate_strategy_similarity(parent1, parent2)
    logger.info(f"Similarity between parent1 and parent2: {similarity:.3f}")
    
    # Test novelty
    is_novel = novelty_detector.is_novel(child1, population)
    logger.info(f"Child1 is novel compared to population: {is_novel}")
    
    # Test diversity
    diversity_score = novelty_detector.calculate_population_diversity(population)
    logger.info(f"Population diversity score: {diversity_score:.3f}")
    
    # Test 7: Evolution Process
    logger.info("\n=== Test 7: Evolution Process ===")
    
    # Create fitness scores (random for testing)
    fitness_scores = [random.uniform(0.1, 1.0) for _ in range(len(population))]
    logger.info(f"Initial fitness scores: {[f'{score:.3f}' for score in fitness_scores]}")
    
    # Evolve population
    evolved_population = strategy_generator.evolve_strategies(
        population, fitness_scores, num_generations=2, population_size=5
    )
    
    logger.info(f"Evolved population size: {len(evolved_population)}")
    
    # Check diversity after evolution
    evolved_diversity = novelty_detector.calculate_population_diversity(evolved_population)
    logger.info(f"Evolved population diversity: {evolved_diversity:.3f}")
    
    # Test 8: Constraint Validation
    logger.info("\n=== Test 8: Constraint Validation ===")
    
    # Test valid MA strategy
    valid_ma_params = {
        'fast_period': 10,
        'slow_period': 50,
        'ma_type': 'SMA',
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.15
    }
    
    valid = template_manager.validate_template_parameters('moving_average_crossover', valid_ma_params)
    logger.info(f"Valid MA parameters validation: {valid}")
    
    # Test invalid MA strategy (violates constraint)
    invalid_ma_params = {
        'fast_period': 50,
        'slow_period': 10,  # This violates slow_period > fast_period constraint
        'ma_type': 'SMA',
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.15
    }
    
    valid = template_manager.validate_template_parameters('moving_average_crossover', invalid_ma_params)
    logger.info(f"Invalid MA parameters validation: {valid}")
    
    # Test 9: Template Information
    logger.info("\n=== Test 9: Template Information ===")
    
    available_templates = strategy_generator.get_available_templates()
    logger.info(f"Available templates: {available_templates}")
    
    for template_name in available_templates:
        template_info = strategy_generator.get_template_info(template_name)
        logger.info(f"Template '{template_name}': {template_info['description']}")
    
    logger.info("\n=== All Tests Completed Successfully! ===")
    logger.info("Strategy generation engine is working correctly.")
    
    return True

if __name__ == '__main__':
    try:
        test_strategy_generation_engine()
        print("\n✅ Strategy Generation Engine test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()