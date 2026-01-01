"""
LLM Agents Package

Autonomous LLM-powered agents for innovative strategy generation and refinement.
Integrates with Ollama and Mistral-3:8B for creative, unconventional trading strategies.
"""

from strategy_generation.llm_agents.ollama_client import OllamaClient
from strategy_generation.llm_agents.strategy_generation_agent import StrategyGenerationAgent
from strategy_generation.llm_agents.feedback_refinement_agent import FeedbackRefinementAgent
from strategy_generation.llm_agents.validation_agent import ValidationAgent

__all__ = [
    'OllamaClient',
    'StrategyGenerationAgent', 
    'FeedbackRefinementAgent',
    'ValidationAgent'
]