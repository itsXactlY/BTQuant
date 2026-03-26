"""
Hypothesis Generator for the Autonomous Quantitative Research Agency

This module generates novel trading strategy hypotheses using AI models,
leveraging market data patterns, mathematical concepts, and evolutionary insights.
"""

import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import aiohttp

from .config import config


@dataclass
class Hypothesis:
    """Represents a generated trading strategy hypothesis"""
    id: str
    title: str
    description: str
    strategy_type: str
    complexity: str
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]
    parameters: Dict[str, Any]
    rationale: str
    expected_performance: Dict[str, float]
    generated_at: datetime
    generation_context: Dict[str, Any]


class HypothesisGenerator:
    """AI-powered hypothesis generation engine"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.generation_history: List[Hypothesis] = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def generate_hypotheses(
        self,
        num_hypotheses: int = config.max_hypotheses_per_cycle,
        strategy_types: Optional[List[str]] = None,
        complexity_levels: Optional[List[str]] = None,
        market_context: Optional[Dict[str, Any]] = None
    ) -> List[Hypothesis]:
        """
        Generate novel trading strategy hypotheses using AI

        Args:
            num_hypotheses: Number of hypotheses to generate
            strategy_types: Types of strategies to focus on
            complexity_levels: Complexity levels to target
            market_context: Current market conditions and data insights

        Returns:
            List of generated hypotheses
        """
        if strategy_types is None:
            strategy_types = config.strategy_types
        if complexity_levels is None:
            complexity_levels = config.hypothesis_complexity_levels

        hypotheses = []

        # Generate hypotheses in parallel
        tasks = []
        for i in range(num_hypotheses):
            task = self._generate_single_hypothesis(
                hypothesis_index=i,
                strategy_types=strategy_types,
                complexity_levels=complexity_levels,
                market_context=market_context
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Hypothesis generation failed: {result}")
                continue
            if result:
                hypotheses.append(result)
                self.generation_history.append(result)

        self.logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses

    async def _generate_single_hypothesis(
        self,
        hypothesis_index: int,
        strategy_types: List[str],
        complexity_levels: List[str],
        market_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Hypothesis]:
        """Generate a single hypothesis using AI"""

        # Prepare AI prompt
        prompt = self._build_generation_prompt(
            strategy_types=strategy_types,
            complexity_levels=complexity_levels,
            market_context=market_context,
            hypothesis_index=hypothesis_index
        )

        try:
            # Call AI model
            ai_response = await self._call_ai_model(prompt)

            # Parse and validate response
            hypothesis_data = self._parse_ai_response(ai_response)

            if hypothesis_data:
                hypothesis = Hypothesis(
                    id=f"hyp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hypothesis_index}",
                    generated_at=datetime.now(),
                    generation_context={
                        "strategy_types": strategy_types,
                        "complexity_levels": complexity_levels,
                        "market_context": market_context,
                        "ai_model": config.ai_model_name,
                        "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt
                    },
                    **hypothesis_data
                )
                return hypothesis

        except Exception as e:
            self.logger.error(f"Failed to generate hypothesis {hypothesis_index}: {e}")
            return None

    def _build_generation_prompt(
        self,
        strategy_types: List[str],
        complexity_levels: List[str],
        market_context: Optional[Dict[str, Any]],
        hypothesis_index: int
    ) -> str:
        """Build the AI prompt for hypothesis generation"""

        base_prompt = f"""
You are an expert quantitative researcher designing novel trading strategies.
Generate a unique, innovative trading strategy hypothesis that could outperform traditional approaches.

Strategy Requirements:
- Strategy Types: {', '.join(strategy_types)}
- Complexity Levels: {', '.join(complexity_levels)}
- Focus on mathematical elegance and market efficiency
- Consider current market regime: {market_context.get('regime', 'unknown') if market_context else 'unknown'}

Previous successful strategies in our lineage:
{self._get_lineage_context()}

Generate a hypothesis in the following JSON format:
{{
    "title": "Descriptive strategy name",
    "description": "Brief explanation of the strategy concept",
    "strategy_type": "One of: {', '.join(strategy_types)}",
    "complexity": "One of: {', '.join(complexity_levels)}",
    "indicators": ["List of technical indicators used"],
    "entry_conditions": ["Specific entry rules"],
    "exit_conditions": ["Specific exit rules"],
    "risk_management": {{
        "stop_loss": "Stop loss mechanism",
        "position_sizing": "Position sizing method",
        "max_drawdown": "Maximum drawdown limit"
    }},
    "parameters": {{
        "Parameter_name": "default_value",
        ...
    }},
    "rationale": "Mathematical and market reasoning behind the strategy",
    "expected_performance": {{
        "sharpe_ratio": 1.5,
        "win_rate": 0.60,
        "max_drawdown": 0.12,
        "profit_factor": 1.8
    }}
}}

Ensure the strategy is:
1. Mathematically sound
2. Implementable in a backtesting framework
3. Different from common strategies
4. Adaptive to market conditions
5. Risk-managed appropriately

Be creative and think outside traditional trading paradigms!
"""

        return base_prompt

    async def _call_ai_model(self, prompt: str) -> str:
        """Call the configured AI model"""

        if not self.session:
            raise RuntimeError("HTTP session not initialized")

        payload = {
            "model": config.ai_model_name,
            "prompt": prompt,
            "temperature": config.ai_temperature,
            "max_tokens": config.ai_max_tokens,
            "stream": False
        }

        try:
            async with self.session.post(
                f"{config.ai_model_endpoint}/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"AI model API error: {response.status}")

                result = await response.json()
                return result.get("choices", [{}])[0].get("text", "")

        except Exception as e:
            self.logger.error(f"AI model call failed: {e}")
            raise

    def _parse_ai_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse AI response into hypothesis data"""

        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON found in AI response")
                return None

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Validate required fields
            required_fields = [
                "title", "description", "strategy_type", "complexity",
                "indicators", "entry_conditions", "exit_conditions",
                "risk_management", "parameters", "rationale", "expected_performance"
            ]

            if not all(field in data for field in required_fields):
                self.logger.warning("Missing required fields in hypothesis")
                return None

            return data

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response as JSON: {e}")
            return None

    def _get_lineage_context(self) -> str:
        """Get context from previous successful strategies"""

        if not self.generation_history:
            return "No previous strategies - starting fresh!"

        # Get top 3 most recent successful hypotheses
        recent_successes = [
            h for h in self.generation_history[-10:]
            if h.expected_performance.get("sharpe_ratio", 0) > 1.0
        ][:3]

        if not recent_successes:
            return "Building on fundamental trading principles..."

        context = "Recent successful strategies:\n"
        for hyp in recent_successes:
            context += f"- {hyp.title}: {hyp.strategy_type} with Sharpe {hyp.expected_performance.get('sharpe_ratio', 'N/A')}\n"

        return context

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about hypothesis generation"""

        if not self.generation_history:
            return {"total_generated": 0}

        strategy_counts = {}
        complexity_counts = {}
        avg_performance = {
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }

        for hyp in self.generation_history:
            strategy_counts[hyp.strategy_type] = strategy_counts.get(hyp.strategy_type, 0) + 1
            complexity_counts[hyp.complexity] = complexity_counts.get(hyp.complexity, 0) + 1

            for key in avg_performance:
                avg_performance[key] += hyp.expected_performance.get(key, 0)

        num_hyp = len(self.generation_history)
        for key in avg_performance:
            avg_performance[key] /= num_hyp

        return {
            "total_generated": num_hyp,
            "strategy_distribution": strategy_counts,
            "complexity_distribution": complexity_counts,
            "average_performance": avg_performance
        }