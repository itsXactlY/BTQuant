#!/usr/bin/env python3
"""
Simple test for LLM integration components.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🧪 Testing LLM Integration Components")
print("=" * 50)

try:
    # Test imports
    print("\n1. Testing imports...")
    from strategy_generation.llm_agents.ollama_client import OllamaClient
    from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
    from strategy_generation.llm_agents.feedback_refinement_agent import FeedbackRefinementAgent
    from strategy_generation.llm_agents.validation_agent import ValidationAgent
    print("✅ All LLM components imported successfully")
    
    # Test Ollama client
    print("\n2. Testing Ollama client...")
    try:
        client = OllamaClient()
        health = client.health_check()
        print(f"📊 Ollama status: connected={health['connected']}")
        if health['connected']:
            print("✅ Ollama client working")
        else:
            print("⚠️  Ollama not connected (expected if server not running)")
    except Exception as e:
        print(f"❌ Ollama client error: {e}")
    
    # Test validation agent
    print("\n3. Testing validation agent...")
    try:
        validator = ValidationAgent()
        
        # Test strategy
        test_strategy = {
            'id': 'test_001',
            'name': 'Quantum Strategy',
            'type': 'llm_generated',
            'description': 'Test quantum strategy',
            'entry_rules': [{'condition': 'quantum_signal > 0.7', 'priority': 1}],
            'exit_rules': [{'condition': 'signal_reversal', 'priority': 1}],
            'risk_management': {
                'position_sizing': 'adaptive',
                'stop_loss': 'volatility_based',
                'max_drawdown': '0.08',
                'risk_per_trade': '0.02'
            },
            'parameters': {'quantum_threshold': 0.7}
        }
        
        is_valid, report = validator.validate_strategy(test_strategy)
        print(f"✅ Validation: {'PASSED' if is_valid else 'FAILED'} (score: {report['overall_score']:.2f})")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 LLM Integration Test Completed!")
    print("\nNext steps:")
    print("1. Start Ollama server with Mistral-3:8B model")
    print("2. Run full system integration tests")
    print("3. Generate novel strategies using LLM agents")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)