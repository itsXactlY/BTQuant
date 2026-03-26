"""
Strategy Factory for the Autonomous Quantitative Research Agency

This module converts AI-generated hypotheses into executable BTQuant strategy classes
using the Kilo Code CLI for automated code generation.
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

from .config import config
from .ai_interface import StrategyHypothesis


@dataclass
class GeneratedStrategy:
    """Represents a generated strategy implementation"""
    hypothesis_id: str
    strategy_name: str
    code_path: str
    class_name: str
    parameters: Dict[str, Any]
    indicators: List[str]
    generated_at: str
    validation_status: str = "pending"


class StrategyFactory:
    """Factory for converting hypotheses into executable strategies"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.generated_strategies: List[GeneratedStrategy] = []
        self.output_dir = Path(config.strategy_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_strategy(self, hypothesis: StrategyHypothesis) -> Optional[GeneratedStrategy]:
        """
        Generate an executable strategy from a hypothesis using Kilo Code CLI

        Args:
            hypothesis: The hypothesis to convert into a strategy

        Returns:
            Generated strategy object or None if generation failed
        """
        try:
            # Create strategy specification
            spec = self._create_strategy_spec(hypothesis)

            # Generate code using Kilo Code CLI
            code = self._generate_code_with_kilo(spec)

            if not code:
                self.logger.error(f"Failed to generate code for hypothesis {hypothesis.id}")
                return None

            # Validate and format the generated code
            validated_code = self._validate_and_format_code(code, hypothesis)

            if not validated_code:
                self.logger.error(f"Code validation failed for hypothesis {hypothesis.id}")
                return None

            # Save the strategy
            strategy = self._save_strategy(hypothesis, validated_code)

            if strategy:
                self.generated_strategies.append(strategy)
                self.logger.info(f"Successfully generated strategy: {strategy.strategy_name}")

            return strategy

        except Exception as e:
            self.logger.error(f"Strategy generation failed for {hypothesis.id}: {e}")
            return None

    def _create_strategy_spec(self, hypothesis: StrategyHypothesis) -> Dict[str, Any]:
        """Create a detailed specification for code generation"""

        spec = {
            "hypothesis_id": hypothesis.id,
            "strategy_name": self._generate_strategy_name(hypothesis),
            "description": hypothesis.description,
            "strategy_type": hypothesis.strategy_type,
            "complexity": hypothesis.complexity,
            "indicators": hypothesis.indicators,
            "entry_conditions": hypothesis.entry_conditions,
            "exit_conditions": hypothesis.exit_conditions,
            "risk_management": hypothesis.risk_management,
            "parameters": hypothesis.parameters,
            "rationale": hypothesis.rationale,
            "framework": "backtrader",
            "base_class": "bt.Strategy",
            "imports": [
                "import backtrader as bt",
                "import numpy as np",
                "import pandas as pd",
                "from datetime import datetime, timedelta"
            ],
            "required_methods": [
                "__init__",
                "next",
                "notify_order",
                "notify_trade"
            ],
            "data_requirements": {
                "ohlcv": True,
                "volume": True,
                "additional_data": []
            }
        }

        return spec

    def _generate_strategy_name(self, hypothesis: StrategyHypothesis) -> str:
        """Generate a unique, descriptive strategy name"""

        # Clean and format the title
        name = hypothesis.title.replace(" ", "_").replace("-", "_")
        name = ''.join(c for c in name if c.isalnum() or c == '_')

        # Add timestamp for uniqueness
        timestamp = hypothesis.generated_at.strftime("%Y%m%d_%H%M%S")

        return f"{name}_{timestamp}"

    def _generate_code_with_kilo(self, spec: Dict[str, Any]) -> Optional[str]:
        """Generate strategy code using Kilo Code CLI"""

        # Create a temporary specification file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(spec, f, indent=2)
            spec_file = f.name

        try:
            # Prepare the Kilo Code CLI command
            cmd = [
                "kilo", "generate-strategy",
                "--spec", spec_file,
                "--framework", "backtrader",
                "--output-format", "python",
                "--validate-syntax", "true"
            ]

            # Add any additional CLI options from config
            if hasattr(config, 'kilo_cli_options'):
                cmd.extend(config.kilo_cli_options)

            self.logger.info(f"Running Kilo Code CLI: {' '.join(cmd)}")

            # Execute the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.kilo_timeout_seconds,
                cwd=config.project_root
            )

            if result.returncode != 0:
                self.logger.error(f"Kilo Code CLI failed: {result.stderr}")
                return None

            # Extract the generated code from stdout
            generated_code = result.stdout.strip()

            if not generated_code:
                self.logger.error("Kilo Code CLI produced no output")
                return None

            return generated_code

        except subprocess.TimeoutExpired:
            self.logger.error(f"Kilo Code CLI timed out after {config.kilo_timeout_seconds} seconds")
            return None
        except FileNotFoundError:
            self.logger.error("Kilo Code CLI not found. Please ensure it's installed and in PATH")
            return None
        except Exception as e:
            self.logger.error(f"Error running Kilo Code CLI: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(spec_file)
            except:
                pass

    def _validate_and_format_code(self, code: str, hypothesis: StrategyHypothesis) -> Optional[str]:
        """Validate and format the generated code"""

        try:
            # Basic syntax validation
            compile(code, '<generated>', 'exec')

            # Check for required components
            required_elements = [
                f"class {self._generate_strategy_name(hypothesis)}",
                "def __init__",
                "def next",
                "bt.Strategy"
            ]

            for element in required_elements:
                if element not in code:
                    self.logger.error(f"Generated code missing required element: {element}")
                    return None

            # Format the code using black if available
            try:
                formatted_code = self._format_code_with_black(code)
                return formatted_code
            except:
                # If black fails, return original code
                return code

        except SyntaxError as e:
            self.logger.error(f"Generated code has syntax errors: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Code validation failed: {e}")
            return None

    def _format_code_with_black(self, code: str) -> str:
        """Format code using black formatter"""

        try:
            result = subprocess.run(
                ["black", "--diff", "--quiet", "-"],
                input=code,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                # Code is already formatted
                return code

            # Format the code
            result = subprocess.run(
                ["black", "-"],
                input=code,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return result.stdout
            else:
                # Return original if formatting fails
                return code

        except FileNotFoundError:
            # Black not available, return original
            return code

    def _save_strategy(self, hypothesis: StrategyHypothesis, code: str) -> Optional[GeneratedStrategy]:
        """Save the generated strategy to disk"""

        try:
            strategy_name = self._generate_strategy_name(hypothesis)
            filename = f"{strategy_name}.py"
            filepath = self.output_dir / filename

            # Add header comments
            header = f'''"""
Generated Strategy: {strategy_name}

Auto-generated from hypothesis: {hypothesis.id}
Title: {hypothesis.title}
Description: {hypothesis.description}
Strategy Type: {hypothesis.strategy_type}
Complexity: {hypothesis.complexity}

Generated at: {hypothesis.generated_at.isoformat()}
Rationale: {hypothesis.rationale}
"""

'''

            full_code = header + code

            # Write to file
            with open(filepath, 'w') as f:
                f.write(full_code)

            # Create strategy object
            strategy = GeneratedStrategy(
                hypothesis_id=hypothesis.id,
                strategy_name=strategy_name,
                code_path=str(filepath),
                class_name=strategy_name,
                parameters=hypothesis.parameters,
                indicators=hypothesis.indicators,
                generated_at=hypothesis.generated_at.isoformat(),
                validation_status="generated"
            )

            return strategy

        except Exception as e:
            self.logger.error(f"Failed to save strategy: {e}")
            return None

    def get_strategy_by_hypothesis(self, hypothesis_id: str) -> Optional[GeneratedStrategy]:
        """Get a generated strategy by hypothesis ID"""

        for strategy in self.generated_strategies:
            if strategy.hypothesis_id == hypothesis_id:
                return strategy
        return None

    def list_strategies(self) -> List[GeneratedStrategy]:
        """List all generated strategies"""

        return self.generated_strategies.copy()

    def cleanup_old_strategies(self, keep_recent: int = 100) -> int:
        """Clean up old strategy files, keeping only the most recent ones"""

        if len(self.generated_strategies) <= keep_recent:
            return 0

        # Sort by generation time (newest first)
        sorted_strategies = sorted(
            self.generated_strategies,
            key=lambda s: s.generated_at,
            reverse=True
        )

        # Strategies to remove
        to_remove = sorted_strategies[keep_recent:]

        removed_count = 0
        for strategy in to_remove:
            try:
                os.unlink(strategy.code_path)
                self.generated_strategies.remove(strategy)
                removed_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to remove strategy {strategy.strategy_name}: {e}")

        self.logger.info(f"Cleaned up {removed_count} old strategies")
        return removed_count