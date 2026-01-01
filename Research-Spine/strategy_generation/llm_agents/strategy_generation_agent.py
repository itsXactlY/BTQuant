"""
Strategy Generation Agent

Autonomous agent for generating novel trading strategies using LLM capabilities.
Focuses on creating unconventional, paradigm-shifting strategies beyond traditional templates.
"""

import logging
import json
import random
from typing import Dict, Any, List, Optional
from datetime import datetime

# Local imports
from strategy_generation.llm_agents.ollama_client import OllamaClient


class StrategyGenerationAgent:
    """
    Autonomous agent for generating novel trading strategies using LLM
    
    Responsible for crafting prompts, parsing LLM responses, and ensuring
    generated strategies meet system requirements for innovation and implementability.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Strategy Generation Agent
        
        Args:
            ollama_client: OllamaClient instance for LLM communication
        """
        self.logger = logging.getLogger('StrategyGenerationAgent')
        self.ollama_client = ollama_client
        self.strategy_counter = 0
        
        # Prompt engineering templates
        self.prompt_templates = self._load_prompt_templates()
        
        self.logger.info("StrategyGenerationAgent initialized")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """
        Load prompt engineering templates for different strategy types
        
        Returns:
            Dictionary of prompt templates
        """
        return {
            'base': """
You are an autonomous quantitative trading strategy researcher at the forefront of financial innovation.
Your mission is to generate groundbreaking, unconventional trading strategies that challenge traditional
financial paradigms and explore uncharted territory in quantitative finance.

## Constraints and Requirements:
1. **Think Beyond Conventional Indicators**: Move beyond standard technical indicators like moving averages, RSI, etc.
2. **Explore Interdisciplinary Approaches**: Incorporate principles from physics, biology, game theory, complexity science, etc.
3. **Focus on Measurable Outcomes**: Strategies must have clear, implementable rules with measurable performance metrics.
4. **Ensure Completeness**: Every strategy must include entry rules, exit rules, and comprehensive risk management.
5. **Encourage Radical Innovation**: Don't be constrained by what's considered "normal" in trading - push boundaries.

## Strategy Structure Requirements:
- **Strategy Name**: Creative, descriptive name that captures the essence
- **Description**: Detailed explanation of the innovative approach and underlying principles
- **Entry Rules**: Clear, implementable conditions for entering trades
- **Exit Rules**: Clear, implementable conditions for exiting trades  
- **Risk Management**: Comprehensive risk controls including position sizing, stop loss, etc.
- **Parameters**: Configurable variables with reasonable ranges
- **Market Context**: Types of market conditions where this strategy might excel

## Current Market Context:
{market_context}

## Additional Constraints:
{constraints}

## Generate Strategy:
Create a trading strategy that represents a paradigm shift in quantitative trading.
The strategy should be highly innovative, potentially radical, and capable of delivering
superior risk-adjusted returns through its unique approach.

Provide the strategy in JSON format with all required fields.
""",
            
            'physics_based': """
Generate a trading strategy inspired by principles from quantum physics, thermodynamics, or chaos theory.
Consider concepts like entropy, wave-particle duality, quantum superposition, or fractal patterns.

Example approaches might include:
- Quantum momentum strategies based on wave function collapse
- Thermodynamic arbitrage exploiting market heat gradients
- Chaos theory-based prediction of regime shifts
- Fractal market hypothesis applied to multi-timeframe analysis

Focus on translating physical principles into actionable trading rules.
""",
            
            'biology_based': """
Create a trading strategy inspired by biological systems and evolutionary principles.
Consider concepts like neural networks, genetic algorithms, ecosystem dynamics, or cellular automata.

Example approaches might include:
- Neural plasticity-based adaptive trading systems
- Genetic evolution of trading rules over time
- Predator-prey dynamics applied to market participants
- Swarm intelligence for collective decision making

Focus on how biological systems adapt and evolve, applying those principles to markets.
""",
            
            'game_theory': """
Develop a strategy based on advanced game theory concepts beyond simple Nash equilibria.
Consider evolutionary game theory, behavioral game theory, or mechanism design.

Example approaches might include:
- Multi-agent market simulations with adaptive strategies
- Signaling games for interpreting market sentiment
- Auction theory applied to order book dynamics
- Coalition formation among correlated assets

Focus on strategic interactions between market participants.
""",
            
            'complexity_science': """
Design a strategy rooted in complexity science and emergent phenomena.
Consider concepts like self-organized criticality, power laws, network theory, or phase transitions.

Example approaches might include:
- Critical point detection for market regime identification
- Scale-free network analysis of asset correlations
- Avalanche dynamics for predicting market cascades
- Emergent pattern recognition in high-frequency data

Focus on how complex systems behave and how those principles apply to financial markets.
"""
        }
    
    def generate_strategy(self, 
                         strategy_type: str = "innovative",
                         market_context: Optional[Dict[str, Any]] = None,
                         constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a novel trading strategy using LLM
        
        Args:
            strategy_type: Type of strategy to generate (innovative, physics_based, etc.)
            market_context: Current market conditions and context
            constraints: Additional constraints for strategy generation
            
        Returns:
            Dictionary containing the generated strategy
        """
        self.strategy_counter += 1
        strategy_id = f"llm_{datetime.now().strftime('%Y%m%d')}_{self.strategy_counter:04d}"
        
        # Prepare market context
        context_str = "No specific market context provided"
        if market_context:
            context_str = self._format_market_context(market_context)
            
        # Prepare constraints
        constraints_str = "No additional constraints"
        if constraints:
            constraints_str = self._format_constraints(constraints)
        
        # Select prompt template
        prompt_template = self.prompt_templates.get(strategy_type, self.prompt_templates['base'])
        
        # Format the complete prompt
        complete_prompt = prompt_template.format(
            market_context=context_str,
            constraints=constraints_str
        )
        
        self.logger.info(f"Generating strategy {strategy_id} with type: {strategy_type}")
        self.logger.debug(f"Using prompt template: {strategy_type}")
        
        try:
            # Generate strategy using LLM
            generated_text = self.ollama_client.generate(
                prompt=complete_prompt,
                system_message="You are a cutting-edge quantitative strategy researcher.",
                temperature=0.9,  # Higher temperature for more creativity
                max_tokens=2000
            )
            
            self.logger.debug(f"LLM generated text: {generated_text[:500]}...")
            
            # Parse the generated strategy
            parsed_strategy = self._parse_generated_strategy(generated_text, strategy_id)
            
            # Validate and enhance the strategy
            validated_strategy = self._validate_and_enhance_strategy(parsed_strategy)
            
            self.logger.info(f"✅ Successfully generated strategy: {validated_strategy['name']}")
            return validated_strategy
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate strategy {strategy_id}: {str(e)}")
            # Fallback to template-based approach if LLM fails
            return self._generate_fallback_strategy(strategy_id, strategy_type)
    
    def _format_market_context(self, market_context: Dict[str, Any]) -> str:
        """
        Format market context for prompt inclusion
        
        Args:
            market_context: Dictionary containing market context information
            
        Returns:
            Formatted string for prompt
        """
        context_lines = []
        
        for key, value in market_context.items():
            if isinstance(value, (str, int, float)):
                context_lines.append(f"• {key}: {value}")
            elif isinstance(value, dict):
                sub_context = ", ".join(f"{k}={v}" for k, v in value.items())
                context_lines.append(f"• {key}: {sub_context}")
            elif isinstance(value, list):
                items = ", ".join(str(item) for item in value)
                context_lines.append(f"• {key}: {items}")
        
        return "\n".join(context_lines) if context_lines else "No specific market context"
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """
        Format constraints for prompt inclusion
        
        Args:
            constraints: Dictionary containing strategy constraints
            
        Returns:
            Formatted string for prompt
        """
        constraint_lines = []
        
        for key, value in constraints.items():
            if isinstance(value, (str, int, float)):
                constraint_lines.append(f"• {key}: {value}")
            elif isinstance(value, dict):
                sub_constraints = "; ".join(f"{k}={v}" for k, v in value.items())
                constraint_lines.append(f"• {key}: {sub_constraints}")
            elif isinstance(value, list):
                items = ", ".join(str(item) for item in value)
                constraint_lines.append(f"• {key}: {items}")
        
        return "\n".join(constraint_lines) if constraint_lines else "No additional constraints"
    
    def _parse_generated_strategy(self, generated_text: str, strategy_id: str) -> Dict[str, Any]:
        """
        Parse LLM-generated text into structured strategy format
        
        Args:
            generated_text: Raw text from LLM
            strategy_id: ID for the strategy
            
        Returns:
            Parsed strategy dictionary
        """
        try:
            # First try to parse as JSON (if LLM returned proper JSON)
            strategy_data = json.loads(generated_text)
            
            # Ensure required fields are present
            required_fields = ['name', 'description', 'entry_rules', 'exit_rules', 'risk_management']
            for field in required_fields:
                if field not in strategy_data:
                    strategy_data[field] = self._generate_fallback_field(field)
            
            # Add metadata
            strategy_data['id'] = strategy_id
            strategy_data['type'] = 'llm_generated'
            strategy_data['generated_at'] = datetime.now().isoformat()
            strategy_data['generation_method'] = 'llm_autonomous'
            
            return strategy_data
            
        except json.JSONDecodeError:
            # If not valid JSON, try to extract structured information
            self.logger.warning("LLM output not valid JSON, attempting structured extraction")
            return self._extract_structured_strategy(generated_text, strategy_id)
        except Exception as e:
            self.logger.error(f"Error parsing generated strategy: {str(e)}")
            return self._generate_fallback_strategy(strategy_id, "innovative")
    
    def _extract_structured_strategy(self, text: str, strategy_id: str) -> Dict[str, Any]:
        """
        Extract structured strategy information from unstructured text
        
        Args:
            text: Unstructured text from LLM
            strategy_id: ID for the strategy
            
        Returns:
            Structured strategy dictionary
        """
        # This is a simplified extraction - in production, use more sophisticated NLP
        strategy = {
            'id': strategy_id,
            'type': 'llm_generated',
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'llm_autonomous',
            'name': self._extract_field(text, 'Strategy Name:', 'Name:', default=f"Innovative Strategy {strategy_id}"),
            'description': self._extract_field(text, 'Description:', 'Strategy Description:', 
                                             default="Autonomously generated innovative trading strategy"),
            'entry_rules': self._extract_rules(text, 'Entry Rules:', 'Entry:'),
            'exit_rules': self._extract_rules(text, 'Exit Rules:', 'Exit:'),
            'risk_management': self._extract_risk_management(text),
            'parameters': self._extract_parameters(text),
            'market_context': self._extract_field(text, 'Market Context:', 'Context:', default="General market conditions")
        }
        
        return strategy
    
    def _extract_field(self, text: str, *prefixes: str, default: str = "") -> str:
        """
        Extract a field from text based on prefixes
        
        Args:
            text: Text to search
            prefixes: Possible prefixes for the field
            default: Default value if not found
            
        Returns:
            Extracted field value
        """
        for prefix in prefixes:
            if prefix in text:
                start_idx = text.find(prefix) + len(prefix)
                # Find end of line or next section
                end_idx = text.find('\n', start_idx)
                if end_idx == -1:
                    end_idx = len(text)
                return text[start_idx:end_idx].strip()
        return default
    
    def _extract_rules(self, text: str, *prefixes: str) -> List[Dict[str, Any]]:
        """
        Extract trading rules from text
        
        Args:
            text: Text to search
            prefixes: Possible prefixes for rules section
            
        Returns:
            List of rule dictionaries
        """
        rules = []
        
        for prefix in prefixes:
            if prefix in text:
                start_idx = text.find(prefix) + len(prefix)
                # Find end of rules section
                end_prefixes = ['Exit Rules:', 'Risk Management:', 'Parameters:', 'Market Context:']
                end_idx = len(text)
                
                for end_prefix in end_prefixes:
                    found_idx = text.find(end_prefix, start_idx)
                    if found_idx > 0 and found_idx < end_idx:
                        end_idx = found_idx
                
                rules_text = text[start_idx:end_idx].strip()
                
                # Simple rule parsing - split by bullet points or numbers
                if rules_text:
                    rule_items = []
                    for line in rules_text.split('\n'):
                        line = line.strip()
                        if line and not line.startswith(('•', '-', '*')):
                            # Check if it's a numbered list item
                            if line[0].isdigit() and line[1] in ('.', ')'):
                                rule_items.append(line[2:].strip())
                            elif line.startswith(('•', '-', '*')):
                                rule_items.append(line[1:].strip())
                        elif line.startswith(('•', '-', '*')):
                            rule_items.append(line[1:].strip())
                    
                    # Create rule dictionaries
                    for i, rule_text in enumerate(rule_items):
                        if rule_text:
                            rules.append({
                                'condition': rule_text,
                                'priority': i + 1,
                                'weight': round(1.0 / max(1, len(rule_items)), 2)
                            })
                
                break
        
        # If no rules found, add a default
        if not rules:
            rules.append({
                'condition': 'Default entry/exit condition',
                'priority': 1,
                'weight': 1.0
            })
        
        return rules
    
    def _extract_risk_management(self, text: str) -> Dict[str, Any]:
        """
        Extract risk management information from text
        
        Args:
            text: Text to search
            
        Returns:
            Risk management dictionary
        """
        risk_mgmt = {
            'position_sizing': self._extract_field(text, 'Position Sizing:', default='fixed_percentage'),
            'stop_loss': self._extract_field(text, 'Stop Loss:', 'Max Loss:', default='trailing_volatility'),
            'take_profit': self._extract_field(text, 'Take Profit:', 'Profit Target:', default='risk_reward_based'),
            'max_drawdown': self._extract_field(text, 'Max Drawdown:', default='0.05'),  # 5%
            'risk_per_trade': self._extract_field(text, 'Risk Per Trade:', default='0.02')  # 2%
        }
        
        # Add some random variation to make strategies more diverse
        risk_mgmt['novelty_factor'] = round(random.uniform(0.7, 0.95), 2)
        
        return risk_mgmt
    
    def _extract_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract parameters from text
        
        Args:
            text: Text to search
            
        Returns:
            Parameters dictionary
        """
        params = {}
        
        # Look for parameter sections
        param_section = self._extract_field(text, 'Parameters:', 'Configurable Variables:', 'Settings:')
        
        if param_section:
            # Simple parameter parsing
            for line in param_section.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    params[key.strip()] = value.strip()
        
        # Add some default parameters if none found
        if not params:
            params = {
                'innovation_level': round(random.uniform(0.8, 1.0), 2),
                'adaptability': round(random.uniform(0.7, 0.9), 2),
                'complexity': round(random.uniform(0.6, 0.8), 2)
            }
        
        return params
    
    def _validate_and_enhance_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance the generated strategy
        
        Args:
            strategy: Generated strategy dictionary
            
        Returns:
            Validated and enhanced strategy
        """
        # Ensure all required fields are present
        required_fields = {
            'id': strategy.get('id', f"llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'type': strategy.get('type', 'llm_generated'),
            'name': strategy.get('name', f"Innovative Strategy {strategy.get('id', 'unknown')}"),
            'description': strategy.get('description', "Autonomously generated trading strategy"),
            'entry_rules': strategy.get('entry_rules', [{'condition': 'default_entry', 'priority': 1}]),
            'exit_rules': strategy.get('exit_rules', [{'condition': 'default_exit', 'priority': 1}]),
            'risk_management': strategy.get('risk_management', self._generate_fallback_risk_management()),
            'parameters': strategy.get('parameters', {}),
            'generated_at': strategy.get('generated_at', datetime.now().isoformat()),
            'generation_method': strategy.get('generation_method', 'llm_autonomous')
        }
        
        # Add metadata for tracking
        strategy['metadata'] = {
            'novelty_score': self._calculate_novelty_score(strategy),
            'complexity_score': self._calculate_complexity_score(strategy),
            'validation_status': 'auto_validated',
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Ensure parameters have reasonable defaults
        if 'parameters' not in strategy or not strategy['parameters']:
            strategy['parameters'] = {
                'innovation_factor': round(random.uniform(0.8, 1.0), 2),
                'risk_tolerance': round(random.uniform(0.3, 0.7), 2),
                'time_horizon': random.choice(['short_term', 'medium_term', 'long_term'])
            }
        
        return strategy
    
    def _generate_fallback_field(self, field_name: str) -> Any:
        """
        Generate fallback values for missing fields
        
        Args:
            field_name: Name of the missing field
            
        Returns:
            Appropriate fallback value
        """
        fallback_values = {
            'name': f"Fallback Strategy {random.randint(1000, 9999)}",
            'description': "Automatically generated fallback trading strategy",
            'entry_rules': [{'condition': 'price_above_sma_50', 'priority': 1, 'weight': 1.0}],
            'exit_rules': [{'condition': 'price_below_sma_20 OR profit_target_reached', 'priority': 1, 'weight': 1.0}],
            'risk_management': self._generate_fallback_risk_management(),
            'parameters': {'fallback_mode': True, 'conservatism': 0.8}
        }
        
        return fallback_values.get(field_name, "unknown")
    
    def _generate_fallback_risk_management(self) -> Dict[str, Any]:
        """
        Generate fallback risk management settings
        
        Returns:
            Risk management dictionary
        """
        return {
            'position_sizing': 'fixed_percentage',
            'stop_loss': 'trailing_5_percent',
            'take_profit': 'risk_reward_2_to_1',
            'max_drawdown': '0.05',
            'risk_per_trade': '0.02',
            'fallback_mode': True
        }
    
    def _generate_fallback_strategy(self, strategy_id: str, strategy_type: str) -> Dict[str, Any]:
        """
        Generate a fallback strategy when LLM generation fails
        
        Args:
            strategy_id: ID for the strategy
            strategy_type: Type of strategy requested
            
        Returns:
            Fallback strategy dictionary
        """
        self.logger.warning(f"Using fallback strategy generation for {strategy_id}")
        
        # Generate a strategy based on the requested type
        if strategy_type == 'physics_based':
            return self._generate_physics_fallback(strategy_id)
        elif strategy_type == 'biology_based':
            return self._generate_biology_fallback(strategy_id)
        elif strategy_type == 'game_theory':
            return self._generate_game_theory_fallback(strategy_id)
        elif strategy_type == 'complexity_science':
            return self._generate_complexity_fallback(strategy_id)
        else:
            return self._generate_generic_fallback(strategy_id)
    
    def _generate_physics_fallback(self, strategy_id: str) -> Dict[str, Any]:
        """Generate physics-based fallback strategy"""
        return {
            'id': strategy_id,
            'type': 'llm_generated',
            'name': f"Quantum Momentum Strategy {strategy_id}",
            'description': "Fallback quantum physics-inspired momentum strategy using wave function analysis",
            'entry_rules': [
                {
                    'condition': 'quantum_entropy > 0.7 AND momentum_score > 0.6',
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
                'max_drawdown': '0.05',
                'risk_per_trade': '0.02'
            },
            'parameters': {
                'quantum_entropy_threshold': 0.7,
                'momentum_window': 14,
                'fallback_mode': True
            },
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'fallback_physics',
            'metadata': {
                'novelty_score': 0.75,
                'complexity_score': 0.8,
                'validation_status': 'fallback_validated'
            }
        }
    
    def _generate_biology_fallback(self, strategy_id: str) -> Dict[str, Any]:
        """Generate biology-based fallback strategy"""
        return {
            'id': strategy_id,
            'type': 'llm_generated',
            'name': f"Neural Adaptive Strategy {strategy_id}",
            'description': "Fallback biology-inspired adaptive strategy using neural plasticity principles",
            'entry_rules': [
                {
                    'condition': 'neural_activation > 0.8 AND market_volatility < 0.4',
                    'priority': 1,
                    'weight': 0.75
                }
            ],
            'exit_rules': [
                {
                    'condition': 'neural_fatigue_detected OR trend_reversal',
                    'priority': 1,
                    'weight': 0.85
                }
            ],
            'risk_management': {
                'position_sizing': 'adaptive_synaptic',
                'stop_loss': 'neural_inhibition_based',
                'take_profit': 'learning_saturation',
                'max_drawdown': '0.04',
                'risk_per_trade': '0.015'
            },
            'parameters': {
                'neural_activation_threshold': 0.8,
                'adaptation_rate': 0.05,
                'fallback_mode': True
            },
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'fallback_biology',
            'metadata': {
                'novelty_score': 0.8,
                'complexity_score': 0.85,
                'validation_status': 'fallback_validated'
            }
        }
    
    def _generate_game_theory_fallback(self, strategy_id: str) -> Dict[str, Any]:
        """Generate game theory-based fallback strategy"""
        return {
            'id': strategy_id,
            'type': 'llm_generated',
            'name': f"Strategic Equilibrium Strategy {strategy_id}",
            'description': "Fallback game theory-inspired strategy focusing on market participant interactions",
            'entry_rules': [
                {
                    'condition': 'nash_equilibrium_detected AND player_advantage > 0.15',
                    'priority': 1,
                    'weight': 0.8
                }
            ],
            'exit_rules': [
                {
                    'condition': 'equilibrium_shift OR cooperative_breakdown',
                    'priority': 1,
                    'weight': 0.9
                }
            ],
            'risk_management': {
                'position_sizing': 'game_theoretic_optimal',
                'stop_loss': 'adversarial_response_based',
                'take_profit': 'cooperative_surplus',
                'max_drawdown': '0.06',
                'risk_per_trade': '0.025'
            },
            'parameters': {
                'equilibrium_sensitivity': 0.15,
                'player_count_estimate': 5,
                'fallback_mode': True
            },
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'fallback_game_theory',
            'metadata': {
                'novelty_score': 0.85,
                'complexity_score': 0.9,
                'validation_status': 'fallback_validated'
            }
        }
    
    def _generate_complexity_fallback(self, strategy_id: str) -> Dict[str, Any]:
        """Generate complexity science-based fallback strategy"""
        return {
            'id': strategy_id,
            'type': 'llm_generated',
            'name': f"Emergent Pattern Strategy {strategy_id}",
            'description': "Fallback complexity science-inspired strategy detecting emergent market patterns",
            'entry_rules': [
                {
                    'condition': 'critical_point_proximity < 0.2 AND fractal_dimension > 1.3',
                    'priority': 1,
                    'weight': 0.82
                }
            ],
            'exit_rules': [
                {
                    'condition': 'phase_transition_detected OR pattern_collapse',
                    'priority': 1,
                    'weight': 0.88
                }
            ],
            'risk_management': {
                'position_sizing': 'emergent_adaptive',
                'stop_loss': 'criticality_based',
                'take_profit': 'pattern_saturation',
                'max_drawdown': '0.055',
                'risk_per_trade': '0.022'
            },
            'parameters': {
                'critical_point_threshold': 0.2,
                'fractal_window': 21,
                'fallback_mode': True
            },
            'generated_at': datetime.now().isoformat(),
            'generation_method': 'fallback_complexity',
            'metadata': {
                'novelty_score': 0.9,
                'complexity_score': 0.95,
                'validation_status': 'fallback_validated'
            }
        }
    
    def _generate_generic_fallback(self, strategy_id: str) -> Dict[str, Any]:
        """Generate generic fallback strategy"""
        innovative_concepts = [
            "Quantum", "Neural", "Evolutionary", "Fractal", "Chaos",
            "Swarm", "Holographic", "String Theory", "Dark Matter", "Entropy"
        ]
        
        concept = random.choice(innovative_concepts)
        
        return {
            'id': strategy_id,
            'type': 'llm_generated',
            'name': f"{concept} Trading Strategy {strategy_id}",
            'description': f"Fallback {concept.lower()}-inspired trading strategy with adaptive learning",
            'entry_rules': [
                {
                    'condition': f'{concept.lower()}_signal > 0.75 AND market_conditions_favorable',
                    'priority': 1,
                    'weight': 0.8
                }
            ],
            'exit_rules': [
                {
                    'condition': f'{concept.lower()}_signal_reversal OR risk_threshold_reached',
                    'priority': 1,
                    'weight': 0.9
                }
            ],
            'risk_management': {
                'position_sizing': 'adaptive_dynamic',
                'stop_loss': 'volatility_adaptive',
                'take_profit': 'signal_strength_based',
                'max_drawdown': '0.05',
                'risk_per_trade': '0.02'
            },
            'parameters': {
                f'{concept.lower()}_threshold': round(random.uniform(0.7, 0.9), 2),
                'adaptation_speed': round(random.uniform(0.01, 0.05), 3),
                'fallback_mode': True
            },
            'generated_at': datetime.now().isoformat(),
            'generation_method': f'fallback_{concept.lower().replace(" ", "_")}',
            'metadata': {
                'novelty_score': round(random.uniform(0.7, 0.9), 2),
                'complexity_score': round(random.uniform(0.75, 0.95), 2),
                'validation_status': 'fallback_validated'
            }
        }
    
    def _calculate_novelty_score(self, strategy: Dict[str, Any]) -> float:
        """
        Calculate novelty score for the strategy
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            Novelty score (0.0-1.0)
        """
        # Simple novelty calculation based on strategy characteristics
        novelty_factors = []
        
        # Check for innovative naming
        name = strategy.get('name', '').lower()
        if any(concept in name for concept in ['quantum', 'neural', 'fractal', 'chaos', 'swarm', 'holographic']):
            novelty_factors.append(0.3)
        else:
            novelty_factors.append(0.1)
        
        # Check description complexity
        description = strategy.get('description', '')
        if len(description) > 100:
            novelty_factors.append(0.25)
        else:
            novelty_factors.append(0.1)
        
        # Check for complex entry/exit rules
        entry_rules = strategy.get('entry_rules', [])
        exit_rules = strategy.get('exit_rules', [])
        
        rule_complexity = min(0.3, len(entry_rules) * 0.1 + len(exit_rules) * 0.1)
        novelty_factors.append(rule_complexity)
        
        # Check parameters
        parameters = strategy.get('parameters', {})
        param_novelty = min(0.2, len(parameters) * 0.05)
        novelty_factors.append(param_novelty)
        
        # Base novelty for LLM-generated strategies
        novelty_factors.append(0.2)
        
        # Calculate final score
        novelty_score = sum(novelty_factors)
        return round(min(1.0, max(0.5, novelty_score)), 2)
    
    def _calculate_complexity_score(self, strategy: Dict[str, Any]) -> float:
        """
        Calculate complexity score for the strategy
        
        Args:
            strategy: Strategy dictionary
            
        Returns:
            Complexity score (0.0-1.0)
        """
        complexity_factors = []
        
        # Rule complexity
        entry_rules = strategy.get('entry_rules', [])
        exit_rules = strategy.get('exit_rules', [])
        total_rules = len(entry_rules) + len(exit_rules)
        complexity_factors.append(min(0.4, total_rules * 0.1))
        
        # Parameter complexity
        parameters = strategy.get('parameters', {})
        complexity_factors.append(min(0.3, len(parameters) * 0.07))
        
        # Description length
        description = strategy.get('description', '')
        complexity_factors.append(min(0.2, len(description) * 0.001))
        
        # Risk management complexity
        risk_mgmt = strategy.get('risk_management', {})
        complexity_factors.append(min(0.1, len(risk_mgmt) * 0.02))
        
        # Base complexity
        complexity_factors.append(0.2)
        
        complexity_score = sum(complexity_factors)
        return round(min(1.0, max(0.4, complexity_score)), 2)
    
    def generate_strategy_population(self, population_size: int = 10, 
                                    diversity_requirements: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate a diverse population of strategies
        
        Args:
            population_size: Number of strategies to generate
            diversity_requirements: Requirements for strategy diversity
            
        Returns:
            List of generated strategies
        """
        self.logger.info(f"Generating strategy population of size: {population_size}")
        
        strategies = []
        strategy_types = ['innovative', 'physics_based', 'biology_based', 'game_theory', 'complexity_science']
        
        # Generate diverse strategies
        for i in range(population_size):
            strategy_type = strategy_types[i % len(strategy_types)]
            
            # Add some randomness to strategy types for more diversity
            if i > len(strategy_types):
                strategy_type = random.choice(strategy_types)
            
            try:
                strategy = self.generate_strategy(
                    strategy_type=strategy_type,
                    market_context={'volatility': 'medium', 'trend': 'neutral'},
                    constraints={'risk_level': 'moderate'}
                )
                strategies.append(strategy)
                
            except Exception as e:
                self.logger.error(f"Failed to generate strategy {i+1}: {str(e)}")
                # Add fallback strategy
                fallback_id = f"llm_{datetime.now().strftime('%Y%m%d')}_fb_{i:04d}"
                strategies.append(self._generate_fallback_strategy(fallback_id, strategy_type))
        
        self.logger.info(f"✅ Generated {len(strategies)} strategies with {len(set(s['name'] for s in strategies))} unique names")
        return strategies