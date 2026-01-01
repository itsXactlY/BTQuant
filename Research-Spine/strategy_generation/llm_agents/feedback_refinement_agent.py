"""
Feedback Refinement Agent

Autonomous agent for iterative improvement of trading strategies through performance-based feedback loops.
Analyzes strategy performance and generates constructive feedback to guide LLM in creating better versions.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Local imports
from strategy_generation.llm_agents.ollama_client import OllamaClient


class FeedbackRefinementAgent:
    """
    Autonomous agent for refining strategies through performance feedback
    
    Analyzes backtest results, identifies strengths and weaknesses, and generates
    constructive feedback to guide the LLM in creating improved strategy versions.
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize Feedback Refinement Agent
        
        Args:
            ollama_client: OllamaClient instance for LLM communication
        """
        self.logger = logging.getLogger('FeedbackRefinementAgent')
        self.ollama_client = ollama_client
        
        # Feedback templates
        self.feedback_templates = self._load_feedback_templates()
        
        self.logger.info("FeedbackRefinementAgent initialized")
    
    def _load_feedback_templates(self) -> Dict[str, str]:
        """
        Load feedback templates for different refinement scenarios
        
        Returns:
            Dictionary of feedback templates
        """
        return {
            'performance_analysis': """
## Strategy Performance Analysis

**Backtest Results:**
- Sharpe Ratio: {sharpe_ratio}
- Max Drawdown: {max_drawdown}%
- Win Rate: {win_rate}%
- Profit Factor: {profit_factor}
- Total Return: {total_return}%

**Strengths Identified:**
{strengths}

**Weaknesses Identified:**
{weaknesses}

**Key Observations:**
{observations}

## Refinement Goals

1. **Improve Performance Metrics:**
   - Target Sharpe Ratio: {target_sharpe}
   - Target Max Drawdown: {target_drawdown}%
   - Target Win Rate: {target_win_rate}%

2. **Address Specific Weaknesses:**
   {specific_improvements}

3. **Enhance Robustness:**
   - Improve performance across different market regimes
   - Reduce sensitivity to parameter changes
   - Enhance risk management effectiveness

## Strategy Refinement Instructions

Based on the performance analysis, generate an improved version of this strategy that:

1. **Maintains Successful Elements:**
   - Preserve the core innovative concept: {core_concept}
   - Keep effective components: {effective_components}
   - Maintain successful risk management approaches

2. **Addresses Identified Weaknesses:**
   - {weakness_solutions}

3. **Enhances Overall Performance:**
   - Improve risk-adjusted returns while maintaining novelty
   - Enhance adaptability to changing market conditions
   - Optimize parameter settings for better robustness

4. **Innovative Improvements:**
   - {innovative_improvements}

Generate the refined strategy in JSON format, ensuring it maintains the original innovative approach
while incorporating these performance-based improvements.
""",
            
            'risk_management': """
## Risk Management Refinement

**Current Risk Profile:**
- Max Drawdown: {max_drawdown}%
- Risk Per Trade: {risk_per_trade}%
- Position Sizing: {position_sizing}
- Stop Loss Effectiveness: {stop_loss_effectiveness}

**Risk Analysis:**
{risk_analysis}

**Refinement Focus:**
1. **Drawdown Reduction:** Target {target_drawdown}% maximum drawdown
2. **Risk-Reward Optimization:** Improve {risk_reward_metric}
3. **Position Sizing:** Enhance {position_sizing_aspect}
4. **Stop Loss Strategy:** Refine {stop_loss_aspect}

Generate an improved risk management approach that maintains the strategy's innovative edge
while significantly improving its risk profile and drawdown characteristics.
""",
            
            'entry_exit_optimization': """
## Entry/Exit Rule Optimization

**Current Rule Performance:**
- Entry Rule Effectiveness: {entry_effectiveness}%
- Exit Rule Effectiveness: {exit_effectiveness}%
- Win Rate: {win_rate}%
- Average Trade Duration: {avg_duration}

**Rule Analysis:**
{rule_analysis}

**Optimization Goals:**
1. **Improve Entry Timing:** {entry_improvement_goal}
2. **Enhance Exit Strategy:** {exit_improvement_goal}
3. **Reduce False Signals:** {false_signal_reduction}
4. **Optimize Trade Duration:** {duration_optimization}

Generate refined entry and exit rules that maintain the strategy's core innovative approach
while significantly improving trade timing, win rate, and overall effectiveness.
""",
            
            'parameter_optimization': """
## Parameter Optimization

**Current Parameters:**
{current_parameters}

**Parameter Sensitivity Analysis:**
{sensitivity_analysis}

**Optimization Objectives:**
1. **Reduce Sensitivity:** Make strategy more robust to parameter variations
2. **Improve Stability:** Enhance performance consistency across different market conditions
3. **Optimize Settings:** Find parameter combinations that maximize {optimization_metric}

Generate an optimized parameter set and potentially new adaptive parameter mechanisms
that improve the strategy's robustness and performance consistency.
"""
        }
    
    def refine_strategy(self, 
                       strategy: Dict[str, Any],
                       performance_metrics: Dict[str, Any],
                       market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Refine a strategy based on performance feedback
        
        Args:
            strategy: Original strategy to refine
            performance_metrics: Backtest performance metrics
            market_context: Current market context
            
        Returns:
            Refined strategy dictionary
        """
        strategy_id = strategy.get('id', 'unknown')
        strategy_name = strategy.get('name', 'Unknown Strategy')
        
        self.logger.info(f"Refining strategy {strategy_id}: {strategy_name}")
        
        try:
            # Analyze performance
            analysis = self._analyze_performance(performance_metrics)
            
            # Generate refinement feedback
            refinement_feedback = self._generate_refinement_feedback(strategy, analysis)
            
            # Create refinement prompt
            refinement_prompt = self._create_refinement_prompt(strategy, analysis, refinement_feedback)
            
            # Generate refined strategy using LLM
            refined_text = self.ollama_client.generate(
                prompt=refinement_prompt,
                system_message="You are a strategy optimization expert refining trading approaches based on performance data.",
                temperature=0.6,  # Slightly lower for more focused improvements
                max_tokens=1500
            )
            
            # Parse the refined strategy
            refined_strategy = self._parse_refined_strategy(refined_text, strategy, analysis)
            
            self.logger.info(f"✅ Successfully refined strategy {strategy_id}")
            return refined_strategy
            
        except Exception as e:
            self.logger.error(f"❌ Failed to refine strategy {strategy_id}: {str(e)}")
            # Return original strategy with refinement metadata
            return self._create_fallback_refinement(strategy, analysis)
    
    def _analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze strategy performance metrics
        
        Args:
            metrics: Performance metrics dictionary
            
        Returns:
            Performance analysis dictionary
        """
        analysis = {
            'raw_metrics': metrics,
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'profit_factor': metrics.get('profit_factor', 0.0),
            'total_return': metrics.get('total_return', 0.0),
            'strengths': [],
            'weaknesses': [],
            'observations': []
        }
        
        # Identify strengths
        if analysis['sharpe_ratio'] > 1.5:
            analysis['strengths'].append(f"Excellent risk-adjusted returns (Sharpe: {analysis['sharpe_ratio']:.2f})")
        elif analysis['sharpe_ratio'] > 1.0:
            analysis['strengths'].append(f"Good risk-adjusted returns (Sharpe: {analysis['sharpe_ratio']:.2f})")
        
        if analysis['win_rate'] > 60:
            analysis['strengths'].append(f"High win rate ({analysis['win_rate']:.1f}%)")
        elif analysis['win_rate'] > 50:
            analysis['strengths'].append(f"Positive win rate ({analysis['win_rate']:.1f}%)")
        
        if analysis['profit_factor'] > 2.0:
            analysis['strengths'].append(f"Excellent profit factor ({analysis['profit_factor']:.2f})")
        elif analysis['profit_factor'] > 1.5:
            analysis['strengths'].append(f"Good profit factor ({analysis['profit_factor']:.2f})")
        
        # Identify weaknesses
        if analysis['max_drawdown'] > 20:
            analysis['weaknesses'].append(f"High maximum drawdown ({analysis['max_drawdown']:.1f}%)")
        elif analysis['max_drawdown'] > 10:
            analysis['weaknesses'].append(f"Moderate drawdown ({analysis['max_drawdown']:.1f}%)")
        
        if analysis['sharpe_ratio'] < 0.5:
            analysis['weaknesses'].append(f"Low Sharpe ratio ({analysis['sharpe_ratio']:.2f})")
        elif analysis['sharpe_ratio'] < 1.0:
            analysis['weaknesses'].append(f"Suboptimal Sharpe ratio ({analysis['sharpe_ratio']:.2f})")
        
        if analysis['win_rate'] < 40:
            analysis['weaknesses'].append(f"Low win rate ({analysis['win_rate']:.1f}%)")
        elif analysis['win_rate'] < 50:
            analysis['weaknesses'].append(f"Suboptimal win rate ({analysis['win_rate']:.1f}%)")
        
        if analysis['profit_factor'] < 1.0:
            analysis['weaknesses'].append(f"Negative profit factor ({analysis['profit_factor']:.2f})")
        elif analysis['profit_factor'] < 1.2:
            analysis['weaknesses'].append(f"Low profit factor ({analysis['profit_factor']:.2f})")
        
        # Add observations
        if len(analysis['strengths']) == 0:
            analysis['observations'].append("Strategy shows limited strengths - significant improvement needed")
        
        if len(analysis['weaknesses']) > 3:
            analysis['observations'].append("Multiple significant weaknesses identified")
        
        # Set improvement targets
        analysis['target_sharpe'] = min(2.5, analysis['sharpe_ratio'] * 1.3 + 0.2)
        analysis['target_drawdown'] = max(5.0, analysis['max_drawdown'] * 0.7)
        analysis['target_win_rate'] = min(70.0, analysis['win_rate'] * 1.2 + 5.0)
        
        return analysis
    
    def _generate_refinement_feedback(self, strategy: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate constructive refinement feedback
        
        Args:
            strategy: Original strategy
            analysis: Performance analysis
            
        Returns:
            Refinement feedback dictionary
        """
        feedback = {
            'core_concept': strategy.get('description', 'innovative trading approach'),
            'effective_components': [],
            'specific_improvements': [],
            'weakness_solutions': [],
            'innovative_improvements': []
        }
        
        # Identify effective components based on strategy type
        strategy_type = strategy.get('type', 'llm_generated')
        strategy_name = strategy.get('name', '').lower()
        
        if 'quantum' in strategy_name or strategy_type == 'physics_based':
            feedback['effective_components'].append("quantum physics-inspired approach")
            feedback['innovative_improvements'].append("enhanced quantum entropy calculations")
        elif 'neural' in strategy_name or strategy_type == 'biology_based':
            feedback['effective_components'].append("neural network adaptation principles")
            feedback['innovative_improvements'].append("improved synaptic learning mechanisms")
        elif 'game' in strategy_name or strategy_type == 'game_theory':
            feedback['effective_components'].append("game theoretic equilibrium analysis")
            feedback['innovative_improvements'].append("advanced multi-player market simulations")
        else:
            feedback['effective_components'].append("innovative interdisciplinary approach")
            feedback['innovative_improvements'].append("cross-domain concept integration")
        
        # Generate specific improvements based on weaknesses
        for weakness in analysis['weaknesses']:
            if 'drawdown' in weakness:
                feedback['specific_improvements'].append("implement adaptive position sizing to reduce drawdown")
                feedback['weakness_solutions'].append("add dynamic risk management based on volatility regimes")
            elif 'Sharpe' in weakness:
                feedback['specific_improvements'].append("optimize risk-reward balance to improve Sharpe ratio")
                feedback['weakness_solutions'].append("refine entry/exit timing to enhance risk-adjusted returns")
            elif 'win rate' in weakness:
                feedback['specific_improvements'].append("enhance signal filtering to improve win rate")
                feedback['weakness_solutions'].append("add confirmation indicators to reduce false signals")
            elif 'profit factor' in weakness:
                feedback['specific_improvements'].append("improve profit targeting and loss cutting mechanisms")
                feedback['weakness_solutions'].append("optimize position sizing relative to signal strength")
        
        return feedback
    
    def _create_refinement_prompt(self, strategy: Dict[str, Any], analysis: Dict[str, Any], 
                                  feedback: Dict[str, Any]) -> str:
        """
        Create refinement prompt for LLM
        
        Args:
            strategy: Original strategy
            analysis: Performance analysis
            feedback: Refinement feedback
            
        Returns:
            Formatted refinement prompt
        """
        # Format strengths and weaknesses
        strengths_str = "\n".join([f"  + {strength}" for strength in analysis['strengths']]) or "  (None identified)"
        weaknesses_str = "\n".join([f"  - {weakness}" for weakness in analysis['weaknesses']]) or "  (None identified)"
        observations_str = "\n".join([f"  • {obs}" for obs in analysis['observations']]) or "  (None)"
        
        # Format specific improvements
        specific_improvements = "\n".join([f"  • {imp}" for imp in feedback['specific_improvements']]) or "  (None specified)"
        weakness_solutions = "\n".join([f"  • {sol}" for sol in feedback['weakness_solutions']]) or "  (None specified)"
        innovative_improvements = "\n".join([f"  • {imp}" for imp in feedback['innovative_improvements']]) or "  (None specified)"
        effective_components = ", ".join(feedback['effective_components']) or "core innovative concept"
        
        # Use the performance analysis template
        prompt = self.feedback_templates['performance_analysis'].format(
            sharpe_ratio=f"{analysis['sharpe_ratio']:.2f}",
            max_drawdown=f"{analysis['max_drawdown']:.1f}",
            win_rate=f"{analysis['win_rate']:.1f}",
            profit_factor=f"{analysis['profit_factor']:.2f}",
            total_return=f"{analysis['total_return']:.1f}",
            strengths=strengths_str,
            weaknesses=weaknesses_str,
            observations=observations_str,
            target_sharpe=f"{analysis['target_sharpe']:.2f}",
            target_drawdown=f"{analysis['target_drawdown']:.1f}",
            target_win_rate=f"{analysis['target_win_rate']:.1f}",
            specific_improvements=specific_improvements,
            core_concept=feedback['core_concept'],
            effective_components=effective_components,
            weakness_solutions=weakness_solutions,
            innovative_improvements=innovative_improvements
        )
        
        return prompt
    
    def _parse_refined_strategy(self, refined_text: str, original_strategy: Dict[str, Any], 
                                analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse refined strategy from LLM output
        
        Args:
            refined_text: Raw text from LLM
            original_strategy: Original strategy for reference
            analysis: Performance analysis
            
        Returns:
            Parsed refined strategy
        """
        try:
            # Try to parse as JSON
            refined_strategy = json.loads(refined_text)
            
            # Ensure it maintains core identity but has improvements
            refined_strategy['id'] = original_strategy['id'] + "_v2"
            refined_strategy['original_id'] = original_strategy['id']
            refined_strategy['type'] = original_strategy['type']
            refined_strategy['refinement_iteration'] = original_strategy.get('refinement_iteration', 0) + 1
            refined_strategy['refined_at'] = datetime.now().isoformat()
            
            # Add refinement metadata
            refined_strategy['refinement_metadata'] = {
                'original_performance': analysis['raw_metrics'],
                'improvement_targets': {
                    'sharpe_ratio': analysis['target_sharpe'],
                    'max_drawdown': analysis['target_drawdown'],
                    'win_rate': analysis['target_win_rate']
                },
                'refinement_focus': analysis['weaknesses'],
                'refinement_method': 'llm_feedback_loop'
            }
            
            # Ensure all required fields are present
            required_fields = ['name', 'description', 'entry_rules', 'exit_rules', 'risk_management', 'parameters']
            for field in required_fields:
                if field not in refined_strategy:
                    refined_strategy[field] = original_strategy.get(field, self._get_default_field(field))
            
            return refined_strategy
            
        except json.JSONDecodeError:
            self.logger.warning("Refined strategy not valid JSON, attempting structured extraction")
            return self._extract_refined_strategy_structured(refined_text, original_strategy, analysis)
        except Exception as e:
            self.logger.error(f"Error parsing refined strategy: {str(e)}")
            return self._create_fallback_refinement(original_strategy, analysis)
    
    def _extract_refined_strategy_structured(self, text: str, original_strategy: Dict[str, Any], 
                                             analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract refined strategy from unstructured text
        
        Args:
            text: Unstructured text from LLM
            original_strategy: Original strategy
            analysis: Performance analysis
            
        Returns:
            Structured refined strategy
        """
        # Create a refined version based on the original with improvements
        refined_strategy = original_strategy.copy()
        
        # Add refinement metadata
        refined_strategy['id'] = original_strategy['id'] + "_v2"
        refined_strategy['original_id'] = original_strategy['id']
        refined_strategy['refinement_iteration'] = original_strategy.get('refinement_iteration', 0) + 1
        refined_strategy['refined_at'] = datetime.now().isoformat()
        
        # Enhance the strategy based on analysis
        refined_strategy['name'] = f"Refined {original_strategy['name']}"
        refined_strategy['description'] = f"{original_strategy['description']} - Performance Optimized"
        
        # Add refinement metadata
        refined_strategy['refinement_metadata'] = {
            'original_performance': analysis['raw_metrics'],
            'improvement_targets': {
                'sharpe_ratio': analysis['target_sharpe'],
                'max_drawdown': analysis['target_drawdown'],
                'win_rate': analysis['target_win_rate']
            },
            'refinement_focus': analysis['weaknesses'],
            'refinement_method': 'structured_extraction'
        }
        
        # Apply some automatic improvements based on weaknesses
        if any('drawdown' in weakness for weakness in analysis['weaknesses']):
            if 'risk_management' in refined_strategy:
                refined_strategy['risk_management']['max_drawdown'] = str(min(10.0, analysis['target_drawdown']))
                refined_strategy['risk_management']['position_sizing'] = 'adaptive_conservative'
        
        if any('win rate' in weakness for weakness in analysis['weaknesses']):
            if 'entry_rules' in refined_strategy and refined_strategy['entry_rules']:
                # Add confirmation to entry rules
                refined_strategy['entry_rules'][0]['condition'] += ' AND confirmation_signal'
        
        return refined_strategy
    
    def _create_fallback_refinement(self, original_strategy: Dict[str, Any], 
                                    analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create fallback refinement when LLM refinement fails
        
        Args:
            original_strategy: Original strategy
            analysis: Performance analysis
            
        Returns:
            Fallback refined strategy
        """
        self.logger.warning(f"Using fallback refinement for strategy {original_strategy.get('id', 'unknown')}")
        
        # Create a refined version with basic improvements
        refined_strategy = original_strategy.copy()
        
        # Add refinement metadata
        refined_strategy['id'] = original_strategy['id'] + "_v2_fallback"
        refined_strategy['original_id'] = original_strategy['id']
        refined_strategy['refinement_iteration'] = original_strategy.get('refinement_iteration', 0) + 1
        refined_strategy['refined_at'] = datetime.now().isoformat()
        refined_strategy['name'] = f"[Fallback] Refined {original_strategy['name']}"
        
        # Apply basic improvements based on analysis
        improvements_made = []
        
        # Improve risk management if drawdown is high
        if analysis['max_drawdown'] > 15:
            if 'risk_management' in refined_strategy:
                refined_strategy['risk_management']['max_drawdown'] = str(max(5.0, analysis['max_drawdown'] * 0.6))
                improvements_made.append(f"Reduced max drawdown target to {refined_strategy['risk_management']['max_drawdown']}%")
        
        # Enhance entry rules if win rate is low
        if analysis['win_rate'] < 45:
            if 'entry_rules' in refined_strategy and refined_strategy['entry_rules']:
                original_condition = refined_strategy['entry_rules'][0]['condition']
                refined_strategy['entry_rules'][0]['condition'] = f"{original_condition} AND volume_confirmation"
                improvements_made.append("Added volume confirmation to entry rules")
        
        # Add refinement metadata
        refined_strategy['refinement_metadata'] = {
            'original_performance': analysis['raw_metrics'],
            'improvements_made': improvements_made,
            'fallback_refinement': True,
            'refinement_focus': analysis['weaknesses']
        }
        
        return refined_strategy
    
    def _get_default_field(self, field_name: str) -> Any:
        """
        Get default values for missing fields
        
        Args:
            field_name: Name of the missing field
            
        Returns:
            Appropriate default value
        """
        defaults = {
            'name': 'Refined Trading Strategy',
            'description': 'Performance-optimized trading strategy',
            'entry_rules': [{'condition': 'optimized_entry_signal', 'priority': 1}],
            'exit_rules': [{'condition': 'optimized_exit_signal', 'priority': 1}],
            'risk_management': {
                'position_sizing': 'adaptive',
                'stop_loss': 'dynamic',
                'take_profit': 'risk_based',
                'max_drawdown': '0.08',
                'risk_per_trade': '0.015'
            },
            'parameters': {
                'optimization_level': 'high',
                'adaptability': 0.85
            }
        }
        
        return defaults.get(field_name, "unknown")
    
    def create_refinement_plan(self, strategy: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a detailed refinement plan without generating full refined strategy
        
        Args:
            strategy: Original strategy
            performance_metrics: Performance metrics
            
        Returns:
            Refinement plan dictionary
        """
        analysis = self._analyze_performance(performance_metrics)
        feedback = self._generate_refinement_feedback(strategy, analysis)
        
        return {
            'strategy_id': strategy.get('id', 'unknown'),
            'strategy_name': strategy.get('name', 'Unknown'),
            'current_performance': analysis['raw_metrics'],
            'strengths': analysis['strengths'],
            'weaknesses': analysis['weaknesses'],
            'improvement_targets': {
                'sharpe_ratio': analysis['target_sharpe'],
                'max_drawdown': analysis['target_drawdown'],
                'win_rate': analysis['target_win_rate']
            },
            'refinement_focus': feedback['specific_improvements'],
            'innovation_preservation': feedback['effective_components'],
            'refinement_priority': 'high' if len(analysis['weaknesses']) > 2 else 'medium'
        }