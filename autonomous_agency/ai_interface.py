"""
AI Interface for hypothesis generation using MiMo-V2-Flash and Kilo Code CLI
"""

import subprocess
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import re

from .config import AIModelConfig


@dataclass
class StrategyHypothesis:
    """Represents a generated trading strategy hypothesis"""
    id: str
    name: str
    description: str
    indicators: List[str]
    entry_conditions: List[str]
    exit_conditions: List[str]
    parameters: Dict[str, Any]
    rationale: str
    mathematical_beauty_score: float
    expected_regime: str
    risk_profile: str


class AIHypothesisGenerator:
    """Generates trading strategy hypotheses using AI models"""

    def __init__(self, config: AIModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def generate_hypotheses(self, context: Dict[str, Any]) -> List[StrategyHypothesis]:
        """
        Generate novel trading strategy hypotheses

        Args:
            context: Dictionary containing market context, previous performance, etc.

        Returns:
            List of strategy hypotheses
        """
        prompt = self._build_hypothesis_prompt(context)

        try:
            # Call MiMo-V2-Flash model
            response = await self._call_ai_model(prompt)

            # Parse and validate hypotheses
            hypotheses = self._parse_hypotheses(response)

            # Filter for mathematical beauty and feasibility
            valid_hypotheses = [
                h for h in hypotheses
                if self._validate_hypothesis(h) and h.mathematical_beauty_score > 0.7
            ]

            self.logger.info(f"Generated {len(valid_hypotheses)} valid hypotheses")
            return valid_hypotheses

        except Exception as e:
            self.logger.error(f"Failed to generate hypotheses: {e}")
            return []

    def _build_hypothesis_prompt(self, context: Dict[str, Any]) -> str:
        """Build a comprehensive prompt for hypothesis generation"""

        market_regime = context.get('market_regime', 'neutral')
        successful_patterns = context.get('successful_patterns', [])
        failed_patterns = context.get('failed_patterns', [])
        available_indicators = context.get('available_indicators', [])

        prompt = f"""
You are an expert quantitative researcher designing novel algorithmic trading strategies.
Your goal is to create strategies with mathematical beauty, robustness, and real-world resilience.

MARKET CONTEXT:
- Current regime: {market_regime}
- Recent successful patterns: {', '.join(successful_patterns[:5])}
- Patterns to avoid: {', '.join(failed_patterns[:5])}
- Available indicators: {', '.join(available_indicators)}

REQUIREMENTS:
1. Mathematical elegance: Use harmonic ratios, fractal relationships, or elegant mathematical formulations
2. Robustness: Strategies should work across different market conditions
3. Innovation: Combine indicators in novel ways, not obvious combinations
4. Risk management: Include proper position sizing and stop losses
5. Interpretability: Clear, logical signal generation

Generate 5 novel trading strategy hypotheses. Each hypothesis should include:

STRATEGY NAME: [Creative, descriptive name]
DESCRIPTION: [Brief explanation of the strategy concept]
INDICATORS: [List of technical indicators used]
ENTRY CONDITIONS: [Specific buy/short conditions]
EXIT CONDITIONS: [Specific sell/cover conditions]
PARAMETERS: [Key parameters with suggested ranges]
RATIONALE: [Why this should work mathematically]
MATHEMATICAL BEAUTY: [Score 0-1, explain why]
EXPECTED REGIME: [Best market conditions]
RISK PROFILE: [Conservative/Moderate/Aggressive]

Focus on strategies that exhibit mathematical beauty through:
- Golden ratio relationships (1.618)
- Fibonacci sequences
- Harmonic oscillators
- Fractal patterns
- Elegant signal combinations

Output in JSON format with key "hypotheses" containing an array of strategy objects.
"""

        return prompt

    async def _call_ai_model(self, prompt: str) -> str:
        """Call the MiMo-V2-Flash model"""
        # This is a placeholder - actual implementation depends on the model API
        # For now, return a mock response

        if self.config.api_endpoint:
            # Real API call would go here
            pass
        else:
            # Mock response for development
            return self._get_mock_response()

    def _get_mock_response(self) -> str:
        """Mock AI response for development"""
        mock_hypotheses = [
            {
                "name": "Golden Ratio Momentum",
                "description": "Momentum strategy using golden ratio relationships between RSI and MACD",
                "indicators": ["RSI", "MACD", "EMA"],
                "entry_conditions": ["RSI crosses above 30", "MACD signal > 0", "Price > EMA(21)"],
                "exit_conditions": ["RSI crosses below 70", "MACD signal crosses below 0"],
                "parameters": {"rsi_period": 14, "macd_fast": 12, "macd_slow": 26, "ema_period": 21},
                "rationale": "Golden ratio (1.618) relationship between MACD periods creates harmonic convergence",
                "mathematical_beauty_score": 0.85,
                "expected_regime": "trending",
                "risk_profile": "moderate"
            },
            {
                "name": "Fractal Williams Oscillator",
                "description": "Williams %R with fractal dimension filtering",
                "indicators": ["WilliamsR", "FractalDimension", "ATR"],
                "entry_conditions": ["WilliamsR < -80", "FractalDimension < 1.5", "ATR > ATR_mean"],
                "exit_conditions": ["WilliamsR > -20", "FractalDimension > 1.8"],
                "parameters": {"williams_period": 14, "fractal_period": 20, "atr_period": 14},
                "rationale": "Fractal dimension measures market efficiency, combining with momentum oscillator",
                "mathematical_beauty_score": 0.82,
                "expected_regime": "volatile",
                "risk_profile": "aggressive"
            }
        ]

        return json.dumps({"hypotheses": mock_hypotheses})

    def _parse_hypotheses(self, response: str) -> List[StrategyHypothesis]:
        """Parse AI response into StrategyHypothesis objects"""
        try:
            data = json.loads(response)
            hypotheses = []

            for i, h_data in enumerate(data.get('hypotheses', [])):
                hypothesis = StrategyHypothesis(
                    id=f"hyp_{i}_{hash(h_data['name']) % 10000}",
                    name=h_data['name'],
                    description=h_data['description'],
                    indicators=h_data['indicators'],
                    entry_conditions=h_data['entry_conditions'],
                    exit_conditions=h_data['exit_conditions'],
                    parameters=h_data['parameters'],
                    rationale=h_data['rationale'],
                    mathematical_beauty_score=h_data['mathematical_beauty_score'],
                    expected_regime=h_data['expected_regime'],
                    risk_profile=h_data['risk_profile']
                )
                hypotheses.append(hypothesis)

            return hypotheses

        except Exception as e:
            self.logger.error(f"Failed to parse hypotheses: {e}")
            return []

    def _validate_hypothesis(self, hypothesis: StrategyHypothesis) -> bool:
        """Validate hypothesis for feasibility and safety"""
        # Check for required fields
        if not all([hypothesis.name, hypothesis.indicators, hypothesis.entry_conditions]):
            return False

        # Check for dangerous patterns
        dangerous_patterns = ['infinite', 'nan', 'null', 'divide by zero']
        text_to_check = ' '.join([
            hypothesis.description,
            hypothesis.rationale,
            str(hypothesis.parameters)
        ]).lower()

        if any(pattern in text_to_check for pattern in dangerous_patterns):
            return False

        # Check parameter ranges
        for param, value in hypothesis.parameters.items():
            if isinstance(value, (int, float)):
                if value <= 0 or value > 1000:  # Reasonable bounds
                    return False

        return True


class KiloCodeStrategyFactory:
    """Uses Kilo Code CLI to generate strategy code from hypotheses"""

    def __init__(self, config: AIModelConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_strategy_code(self, hypothesis: StrategyHypothesis) -> Optional[str]:
        """
        Generate Python strategy code using Kilo Code CLI

        Args:
            hypothesis: Strategy hypothesis to convert to code

        Returns:
            Generated Python code as string, or None if failed
        """

        if not self.config.kilo_code_cli_path:
            self.logger.warning("Kilo Code CLI path not configured, using template generation")
            return self._generate_template_code(hypothesis)

        prompt = self._build_code_generation_prompt(hypothesis)

        try:
            # Call Kilo Code CLI
            result = subprocess.run(
                [self.config.kilo_code_cli_path, 'generate', '--prompt', prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                code = result.stdout.strip()
                if self._validate_generated_code(code):
                    return code
                else:
                    self.logger.error("Generated code failed validation")
                    return None
            else:
                self.logger.error(f"Kilo Code CLI failed: {result.stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to generate code with Kilo Code CLI: {e}")
            return self._generate_template_code(hypothesis)

    def _build_code_generation_prompt(self, hypothesis: StrategyHypothesis) -> str:
        """Build prompt for code generation"""

        prompt = f"""
Generate a Backtrader strategy class based on this hypothesis:

STRATEGY NAME: {hypothesis.name}
DESCRIPTION: {hypothesis.description}

INDICATORS: {', '.join(hypothesis.indicators)}
ENTRY CONDITIONS: {'; '.join(hypothesis.entry_conditions)}
EXIT CONDITIONS: {'; '.join(hypothesis.exit_conditions)}
PARAMETERS: {hypothesis.parameters}

REQUIREMENTS:
1. Inherit from BaseStrategy
2. Use proper indicator initialization
3. Implement buy_or_short_condition() and sell_or_cover_condition() methods
4. Include proper parameter definitions
5. Add comments explaining the mathematical logic
6. Follow BTQuant coding standards

Generate clean, well-documented Python code for a Backtrader strategy.
"""

        return prompt

    def _generate_template_code(self, hypothesis: StrategyHypothesis) -> str:
        """Generate strategy code using templates when CLI is not available"""

        template = f'''
from backtrader.strategies.base import BaseStrategy
from backtrader import indicators as btind
import backtrader as bt

class {self._sanitize_class_name(hypothesis.name)}(BaseStrategy):
    """
    {hypothesis.name}
    {hypothesis.description}

    Generated from hypothesis: {hypothesis.id}
    Mathematical beauty score: {hypothesis.mathematical_beauty_score}
    """

    params = (
        {self._format_parameters(hypothesis.parameters)}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize indicators
        {self._generate_indicator_initialization(hypothesis.indicators)}

    def buy_or_short_condition(self):
        """Entry conditions based on hypothesis"""
        {self._generate_entry_conditions(hypothesis.entry_conditions)}
        return False  # Placeholder

    def sell_or_cover_condition(self):
        """Exit conditions based on hypothesis"""
        {self._generate_exit_conditions(hypothesis.exit_conditions)}
        return False  # Placeholder
'''

        return template

    def _sanitize_class_name(self, name: str) -> str:
        """Sanitize strategy name for use as class name"""
        # Remove special characters and spaces
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '', name.replace(' ', '_'))
        return sanitized if sanitized else "GeneratedStrategy"

    def _format_parameters(self, parameters: Dict[str, Any]) -> str:
        """Format parameters for Backtrader params tuple"""
        param_lines = []
        for key, value in parameters.items():
            if isinstance(value, (int, float)):
                param_lines.append(f'        ("{key}", {value}),')
            else:
                param_lines.append(f'        ("{key}", {repr(value)}),')
        return '\n'.join(param_lines)

    def _generate_indicator_initialization(self, indicators: List[str]) -> str:
        """Generate indicator initialization code"""
        init_lines = []
        for indicator in indicators:
            if indicator.lower() == 'rsi':
                init_lines.append('        self.rsi = btind.RSI(self.data.close, period=14)')
            elif indicator.lower() == 'macd':
                init_lines.append('        self.macd = btind.MACD(self.data.close)')
            elif indicator.lower() == 'ema':
                init_lines.append('        self.ema = btind.EMA(self.data.close, period=21)')
            # Add more indicators as needed
        return '\n'.join(init_lines) if init_lines else '        # No indicators specified'

    def _generate_entry_conditions(self, conditions: List[str]) -> str:
        """Generate entry condition code"""
        code_lines = ['        # Entry conditions:']
        for condition in conditions:
            code_lines.append(f'        # {condition}')
        code_lines.append('        # TODO: Implement actual conditions')
        return '\n'.join(code_lines)

    def _generate_exit_conditions(self, conditions: List[str]) -> str:
        """Generate exit condition code"""
        code_lines = ['        # Exit conditions:']
        for condition in conditions:
            code_lines.append(f'        # {condition}')
        code_lines.append('        # TODO: Implement actual conditions')
        return '\n'.join(code_lines)

    def _validate_generated_code(self, code: str) -> bool:
        """Basic validation of generated code"""
        try:
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False