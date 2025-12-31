"""
Integration example showing how the documentation system works with other components
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def demonstrate_integration():
    """Demonstrate how the documentation system integrates with other components"""
    print("🔄 Demonstrating Documentation System Integration...")
    
    from documentation.documentation_system import DocumentationSystem
    
    # Initialize documentation system
    doc_system = DocumentationSystem()
    
    # Simulate data from other components
    print("\n📦 Simulating data flow from other components...")
    
    # 1. Strategy Generation Component
    print("\n1. 🤖 Strategy Generation Component -> Documentation System")
    
    mock_strategy = {
        'id': 'integration_strategy_001',
        'template': 'trend_following',
        'parameters': {
            'trend_strength': 1.2,
            'volatility_target': 0.015,
            'lookback_period': 50,
            'stop_loss': 0.05,
            'take_profit': 0.10
        },
        'metadata': {
            'generated_by': 'StrategyGenerator',
            'version': '2.0',
            'generation_timestamp': datetime.now().isoformat()
        }
    }
    
    print(f"   Strategy generated: {mock_strategy['id']} ({mock_strategy['template']})")
    
    # 2. Backtesting Engine Component
    print("\n2. 📊 Backtesting Engine Component -> Documentation System")
    
    mock_backtest_results = {
        'strategy_id': mock_strategy['id'],
        'template': mock_strategy['template'],
        'performance_metrics': {
            'sharpe_ratio': 2.15,
            'max_drawdown': 0.08,
            'win_rate': 0.72,
            'total_return': 0.32,
            'sortino_ratio': 2.4,
            'volatility': 0.06,
            'calmar_ratio': 4.0,
            'profit_factor': 1.85
        },
        'risk_profile': {
            'volatility': 0.06,
            'value_at_risk': 0.03,
            'expected_shortfall': 0.045,
            'max_drawdown_duration': 14,
            'recovery_factor': 2.5
        },
        'statistical_significance': {
            'p_value': 0.021,
            'significant': True,
            'confidence_interval': [0.28, 0.36]
        },
        'returns_data': {
            'returns': [0.012, 0.008, -0.003, 0.015, 0.021, -0.007, 0.018],
            'num_periods': 7,
            'return_statistics': {
                'mean': 0.0106,
                'std': 0.0102,
                'min': -0.007,
                'max': 0.021
            }
        },
        'metadata': {
            'backtested_by': 'BacktestEngine',
            'version': '2.0',
            'backtest_timestamp': datetime.now().isoformat()
        }
    }
    
    print(f"   Backtest completed: Sharpe {mock_backtest_results['performance_metrics']['sharpe_ratio']:.2f}, "
          f"Max DD {mock_backtest_results['performance_metrics']['max_drawdown']:.2%}")
    
    # 3. Evolutionary Selection Component
    print("\n3. 🧬 Evolutionary Selection Component -> Documentation System")
    
    mock_selection_results = {
        'strategy': mock_strategy,
        'performance': mock_backtest_results['performance_metrics'],
        'fitness_scores': {
            'sharpe_ratio': 2.15,
            'drawdown': 0.08,
            'win_rate': 0.72,
            'risk_adjusted_return': 0.87,
            'consistency': 0.91
        },
        'composite_fitness': 0.93,
        'selection_metadata': {
            'selection_method': 'pareto_front_optimization',
            'generation': 3,
            'rank': 1,
            'population_size': 25,
            'selected_from': 50,
            'selection_timestamp': datetime.now().isoformat()
        }
    }
    
    print(f"   Strategy selected: Rank {mock_selection_results['selection_metadata']['rank']}, "
          f"Fitness {mock_selection_results['composite_fitness']:.2f}")
    
    # 4. Documentation System Integration
    print("\n4. 📚 Documentation System Integration")
    
    # Generate comprehensive documentation
    print("   📝 Generating comprehensive report...")
    comprehensive_report = doc_system.generate_comprehensive_report(
        mock_strategy, mock_backtest_results, mock_selection_results
    )
    print(f"   ✓ Comprehensive report: {Path(comprehensive_report).name}")
    
    print("   📊 Generating performance report...")
    performance_report = doc_system.generate_performance_report(mock_strategy, mock_backtest_results)
    print(f"   ✓ Performance report: {Path(performance_report).name}")
    
    print("   🎨 Creating visualizations...")
    visualizations = []
    for viz_type in ['performance', 'risk_return', 'drawdown']:
        viz_file = doc_system.create_strategy_visualization(mock_strategy, mock_backtest_results, viz_type)
        if viz_file:
            visualizations.append(viz_type)
            print(f"   ✓ {viz_type} visualization: {Path(viz_file).name}")
    
    print("   🗄️ Archiving strategy...")
    archive_file = doc_system.archive_strategy(mock_strategy, mock_backtest_results, mock_selection_results)
    print(f"   ✓ Strategy archive: {Path(archive_file).name}")
    
    # 5. Evolutionary Lineage Tracking
    print("\n5. 🧬 Evolutionary Lineage Tracking")
    
    # Simulate evolutionary process
    parent_strategies = [
        {
            'id': 'parent_strategy_001',
            'template': 'trend_following',
            'parameters': {'trend_strength': 1.0}
        },
        {
            'id': 'parent_strategy_002',
            'template': 'mean_reversion',
            'parameters': {'mean_reversion_strength': 0.9}
        }
    ]
    
    child_strategies = [mock_strategy]
    
    generation_data = {
        'generation_number': 3,
        'metrics': {
            'population_size': 50,
            'diversity_score': 0.88,
            'fitness_improvement': 0.12,
            'convergence_rate': 0.05,
            'novelty_score': 0.75
        }
    }
    
    lineage_result = doc_system.track_evolutionary_lineage(
        generation_data, parent_strategies, child_strategies
    )
    
    print(f"   ✓ Lineage tracked: {lineage_result['generation_record']['generation_number']} generations")
    print(f"   ✓ Parents: {lineage_result['parent_count']}, Children: {lineage_result['child_count']}")
    
    # Generate evolutionary report
    evolutionary_report = doc_system.generate_evolutionary_report({
        'generations': 3,
        'strategies': 150,
        'metrics': generation_data['metrics']
    })
    print(f"   ✓ Evolutionary report: {Path(evolutionary_report).name}")
    
    # 6. System Integration
    print("\n6. 🔗 System Integration Report")
    
    system_state = {
        'components': [
            {'name': 'StrategyGenerator', 'version': '2.0', 'status': 'active'},
            {'name': 'BacktestEngine', 'version': '2.0', 'status': 'active'},
            {'name': 'EvolutionarySelector', 'version': '2.0', 'status': 'active'},
            {'name': 'DocumentationSystem', 'version': '2.0', 'status': 'active'},
            {'name': 'DeploymentManager', 'version': '1.0', 'status': 'standby'}
        ],
        'strategy_count': 150,
        'generation_count': 3,
        'active_strategies': 5,
        'archived_strategies': 145,
        'timestamp': datetime.now().isoformat()
    }
    
    integration_report = doc_system.generate_system_integration_report(system_state)
    print(f"   ✓ Integration report: {Path(integration_report).name}")
    
    # 7. Process Logging
    print("\n7. 📝 Process Logging")
    
    process_log = doc_system.log_process('strategy_evolution_pipeline', {
        'strategy_id': mock_strategy['id'],
        'generation': 3,
        'fitness_score': 0.93,
        'performance_metrics': mock_backtest_results['performance_metrics'],
        'selection_rank': 1,
        'timestamp': datetime.now().isoformat()
    })
    print(f"   ✓ Process log: {Path(process_log).name}")
    
    # 8. Summary
    print("\n📊 Integration Summary:")
    print(f"   • Strategy: {mock_strategy['id']} ({mock_strategy['template']})")
    print(f"   • Performance: Sharpe {mock_backtest_results['performance_metrics']['sharpe_ratio']:.2f}")
    print(f"   • Selection: Rank {mock_selection_results['selection_metadata']['rank']}, Fitness {mock_selection_results['composite_fitness']:.2f}")
    print(f"   • Documentation: {len(visualizations)} visualizations, 2 reports, 1 archive")
    print(f"   • Lineage: {generation_data['generation_number']} generations tracked")
    
    # 9. Verify Archive Index
    print("\n🗄️ Archive Index Status:")
    archive_index_path = 'documentation/archive/archive_index.json'
    with open(archive_index_path, 'r') as f:
        archive_index = json.load(f)
    
    print(f"   • Total strategies archived: {len(archive_index['strategies'])}")
    print(f"   • Total reports generated: {len(archive_index['reports'])}")
    print(f"   • Total visualizations created: {len(archive_index['visualizations'])}")
    
    # 10. Verify Lineage Database
    print("\n🧬 Lineage Database Status:")
    lineage_db_path = 'documentation/lineage/lineage_database.json'
    with open(lineage_db_path, 'r') as f:
        lineage_db = json.load(f)
    
    print(f"   • Generations tracked: {len(lineage_db['generations'])}")
    print(f"   • Strategies in genealogy: {len(lineage_db['strategy_genealogy'])}")
    print(f"   • Evolutionary events recorded: {len(lineage_db['evolutionary_history'])}")
    
    print("\n🎉 Documentation System Integration Demonstration Complete!")
    print("\n📁 Generated Files:")
    print(f"   • Reports: {len(archive_index['reports'])} files")
    print(f"   • Visualizations: {len(archive_index['visualizations'])} files")
    print(f"   • Archives: {len(archive_index['strategies'])} files")
    print(f"   • Logs: 1 file")
    print(f"   • Database files: 2 (lineage_db, archive_index)")
    
    return True

if __name__ == "__main__":
    success = demonstrate_integration()
    sys.exit(0 if success else 1)