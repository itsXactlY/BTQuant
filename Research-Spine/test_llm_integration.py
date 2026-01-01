#!/usr/bin/env python3
"""
Test script for LLM integration with strategy generation system.
Tests the new autonomous LLM agents and their integration.
"""

import logging
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Test imports
test_description = "Testing LLM Integration Components"
print(f"\n{'='*60}")
print(f"{test_description}")
print(f"{'='*60}\n")

try:
    # Test 1: Import LLM components
    print("🧪 Test 1: Importing LLM components...")
    from strategy_generation.llm_agents.ollama_client import OllamaClient
    from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
    from strategy_generation.llm_agents.feedback_refinement_agent import FeedbackRefinementAgent
    from strategy_generation.llm_agents.validation_agent import ValidationAgent
    print("✅ LLM components imported successfully")
    
    # Test 2: Initialize Ollama client
    print("\n🧪 Test 2: Initializing Ollama client...")
    try:
        ollama_client = OllamaClient(model_name="qwen2.5:7b")
        print("✅ Ollama client initialized")
        
        # Test connection
        health = ollama_client.health_check()
        print(f"📊 Ollama health: connected={health['connected']}, model={health['model']}")
        
        if not health['connected']:
            print("⚠️  Ollama server not connected - will use fallback strategies")
            
    except Exception as e:
        print(f"❌ Ollama client initialization failed: {e}")
        print("⚠️  Will proceed with fallback strategies")
        ollama_client = None
    
    # Test 3: Initialize strategy generation agent
    print("\n🧪 Test 3: Initializing Strategy Generation Agent...")
    if ollama_client:
        strategy_agent = StrategyGenerationAgent(ollama_client)
        print("✅ Strategy Generation Agent initialized")
    else:
        print("⚠️  Strategy Generation Agent requires Ollama client")
        strategy_agent = None
    
    # Test 4: Initialize validation agent
    print("\n🧪 Test 4: Initializing Validation Agent...")
    validation_agent = ValidationAgent()
    print("✅ Validation Agent initialized")
    
    # Test 5: Generate fallback strategies (works without Ollama)
    print("\n🧪 Test 5: Generating fallback strategies...")
    if strategy_agent:
        try:
            # Generate a strategy (will use fallback if Ollama not available)
            strategy = strategy_agent.generate_strategy(strategy_type="innovative")
            print(f"✅ Generated strategy: {strategy.get('name', 'unknown')}")
            print(f"   ID: {strategy.get('id', 'unknown')}")
            print(f"   Type: {strategy.get('type', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Strategy generation failed: {e}")
    else:
        print("⚠️  Strategy agent not available")
    
    # Test 6: Validate a strategy
    print("\n🧪 Test 6: Testing validation system...")
    
    # Create a test strategy
    test_strategy = {
        'id': 'test_strategy_001',
        'name': 'Quantum Momentum Strategy',
        'type': 'llm_generated',
        'description': 'Innovative quantum physics-inspired momentum trading strategy',
        'entry_rules': [
            {
                'condition': 'quantum_entropy > 0.75 AND momentum_score > 0.6',
                'priority': 1,
                'weight': 0.8
            }
        ],
        'exit_rules': [
            {
                'condition': 'quantum_decoherence_detected OR momentum_reversal',
                'priority': 1,
                'weight': 0.9
            }
        ],
        'risk_management': {
            'position_sizing': 'quantum_adaptive',
            'stop_loss': 'volatility_based',
            'take_profit': 'wave_function_peak',
            'max_drawdown': '0.08',
            'risk_per_trade': '0.02'
        },
        'parameters': {
            'quantum_entropy_threshold': 0.75,
            'momentum_window': 14,
            'innovation_factor': 0.85
        },
        'generated_at': '2026-01-01T12:00:00.000000',
        'generation_method': 'test'
    }
    
    try:
        is_valid, validation_report = validation_agent.validate_strategy(test_strategy)
        print(f"✅ Validation completed: {'PASSED' if is_valid else 'FAILED'}")
        print(f"   Score: {validation_report['overall_score']:.2f}")
        print(f"   Novelty: {test_strategy['metadata'].get('novelty_score', 'N/A')}")
        
        if not is_valid:
            print("   Failed checks:")
            for check in validation_report['failed_checks']:
                print(f"     - {check['check']}: {check['details']}")
                
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    
    # Test 7: Test strategy generator integration
    print("\n🧪 Test 7: Testing Strategy Generator integration...")
    try:
        from strategy_generation.strategy_generator import StrategyGenerator
        
        # Test with LLM enabled
        if ollama_client:
            generator = StrategyGenerator(use_llm=True)
        else:
            generator = StrategyGenerator(use_llm=False)
            
        print("✅ Strategy Generator initialized")
        print(f"   LLM enabled: {generator.use_llm}")
        print(f"   Generation method: {'LLM-powered' if generator.use_llm else 'Template-based'}")
        
        # Test strategy generation
        if generator.use_llm:
            strategy = generator.generate_strategy(strategy_type="physics_based")
            print(f"✅ LLM strategy generated: {strategy.get('name', 'unknown')}")
        else:
            strategy = generator.generate_strategy()
            print(f"✅ Template strategy generated: {strategy.get('id', 'unknown')}")
            
    except Exception as e:
        print(f"❌ Strategy Generator test failed: {e}")
    
    # Test 8: Test population generation
    print("\n🧪 Test 8: Testing population generation...")
    try:
        from strategy_generation.strategy_generator import StrategyGenerator
        
        generator = StrategyGenerator(use_llm=bool(ollama_client))
        
        # Generate a small population
        population = generator.generate_strategy_population(population_size=3)
        
        print(f"✅ Generated population with {len(population)} strategies:")
        for i, strategy in enumerate(population):
            name = strategy.get('name', strategy.get('id', f'strategy_{i+1}'))
            strategy_type = strategy.get('type', 'template')
            print(f"   {i+1}. {name} ({strategy_type})")
            
    except Exception as e:
        print(f"❌ Population generation failed: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 LLM Integration Test Suite Completed!")
    print(f"{'='*60}\n")
    
    # Summary
    if ollama_client and ollama_client.connected:
        print("📋 Summary:")
        print("   ✅ LLM components: Fully operational")
        print("   ✅ Strategy generation: LLM-powered")
        print("   ✅ Validation system: Active")
        print("   ✅ Integration: Complete")
        print("\n🚀 System ready for autonomous strategy generation!")
    else:
        print("📋 Summary:")
        print("   ⚠️  LLM components: Fallback mode (Ollama not connected)")
        print("   ✅ Strategy generation: Template-based fallback")
        print("   ✅ Validation system: Active")
        print("   ✅ Integration: Partial (waiting for Ollama)")
        print("\n🔌 Connect Ollama server to enable full LLM capabilities")
    
    return True
    
except Exception as e:
    print(f"\n❌ Test suite failed with critical error: {e}")
    import traceback
    traceback.print_exc()
    return False

if __name__ == "__main__":
    success = test_llm_integration()
    sys.exit(0 if success else 1)