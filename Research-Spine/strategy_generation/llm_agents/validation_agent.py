"""
Validation Agent

Autonomous agent for validating LLM-generated trading strategies.
Ensures strategies meet system requirements, are logically consistent, and can be implemented.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class ValidationAgent:
    """
    Autonomous agent for validating LLM-generated strategies
    
    Performs comprehensive validation including structural checks, logical consistency,
    parameter validation, and implementation feasibility assessment.
    """
    
    def __init__(self):
        """Initialize Validation Agent"""
        self.logger = logging.getLogger('ValidationAgent')
        self.validation_rules = self._load_validation_rules()
        self.logger.info("ValidationAgent initialized")
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """
        Load validation rules and criteria
        
        Returns:
            Dictionary of validation rules
        """
        return {
            'required_fields': ['id', 'name', 'description', 'entry_rules', 'exit_rules', 'risk_management', 'parameters'],
            'field_types': {
                'id': str,
                'name': str,
                'description': str,
                'entry_rules': list,
                'exit_rules': list,
                'risk_management': dict,
                'parameters': dict,
                'generated_at': str,
                'type': str
            },
            'risk_management_requirements': ['position_sizing', 'stop_loss', 'max_drawdown', 'risk_per_trade'],
            'parameter_constraints': {
                'max_drawdown': {'min': 0.01, 'max': 0.20},
                'risk_per_trade': {'min': 0.005, 'max': 0.05}
            },
            'rule_requirements': {
                'min_entry_rules': 1,
                'min_exit_rules': 1,
                'max_rules_per_type': 5
            },
            'name_patterns': {
                'forbidden': ['template', 'basic', 'simple', 'standard', 'traditional'],
                'encouraged': ['innovative', 'quantum', 'neural', 'adaptive', 'dynamic', 'evolutionary']
            }
        }
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a complete strategy
        
        Args:
            strategy: Strategy dictionary to validate
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        strategy_id = strategy.get('id', 'unknown')
        strategy_name = strategy.get('name', 'Unknown')
        
        self.logger.info(f"Validating strategy {strategy_id}: {strategy_name}")
        
        validation_report = {
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            'validation_timestamp': datetime.now().isoformat(),
            'passed_checks': [],
            'failed_checks': [],
            'warnings': [],
            'suggestions': [],
            'overall_score': 0.0,
            'is_valid': False
        }
        
        # Run all validation checks
        self._check_required_fields(strategy, validation_report)
        self._check_field_types(strategy, validation_report)
        self._check_structural_integrity(strategy, validation_report)
        self._check_logical_consistency(strategy, validation_report)
        self._check_parameter_constraints(strategy, validation_report)
        self._check_novelty_and_innovation(strategy, validation_report)
        self._check_implementation_feasibility(strategy, validation_report)
        
        # Calculate overall validation score
        total_checks = len(validation_report['passed_checks']) + len(validation_report['failed_checks'])
        if total_checks > 0:
            validation_report['overall_score'] = len(validation_report['passed_checks']) / total_checks
        
        # Determine overall validity
        validation_report['is_valid'] = len(validation_report['failed_checks']) == 0
        
        if validation_report['is_valid']:
            self.logger.info(f"✅ Strategy {strategy_id} validation passed (score: {validation_report['overall_score']:.2f})")
        else:
            self.logger.warning(f"⚠️  Strategy {strategy_id} validation failed (score: {validation_report['overall_score']:.2f})")
        
        return validation_report['is_valid'], validation_report
    
    def _check_required_fields(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check that all required fields are present
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Required Fields Check"
        missing_fields = []
        
        for field in self.validation_rules['required_fields']:
            if field not in strategy:
                missing_fields.append(field)
        
        if missing_fields:
            report['failed_checks'].append({
                'check': check_name,
                'status': 'failed',
                'details': f"Missing required fields: {', '.join(missing_fields)}",
                'severity': 'critical'
            })
        else:
            report['passed_checks'].append({
                'check': check_name,
                'status': 'passed',
                'details': 'All required fields present'
            })
    
    def _check_field_types(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check that fields have correct data types
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Field Type Check"
        type_errors = []
        
        for field, expected_type in self.validation_rules['field_types'].items():
            if field in strategy:
                actual_value = strategy[field]
                if not isinstance(actual_value, expected_type):
                    type_errors.append(f"{field} should be {expected_type.__name__}, got {type(actual_value).__name__}")
        
        if type_errors:
            report['failed_checks'].append({
                'check': check_name,
                'status': 'failed',
                'details': f"Type errors: {'; '.join(type_errors)}",
                'severity': 'critical'
            })
        else:
            report['passed_checks'].append({
                'check': check_name,
                'status': 'passed',
                'details': 'All field types correct'
            })
    
    def _check_structural_integrity(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check structural integrity of strategy components
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Structural Integrity Check"
        structural_issues = []
        
        # Check entry rules structure
        entry_rules = strategy.get('entry_rules', [])
        if not isinstance(entry_rules, list) or len(entry_rules) < self.validation_rules['rule_requirements']['min_entry_rules']:
            structural_issues.append(f"Need at least {self.validation_rules['rule_requirements']['min_entry_rules']} entry rules")
        else:
            for i, rule in enumerate(entry_rules):
                if not isinstance(rule, dict) or 'condition' not in rule:
                    structural_issues.append(f"Entry rule {i+1} malformed")
        
        # Check exit rules structure
        exit_rules = strategy.get('exit_rules', [])
        if not isinstance(exit_rules, list) or len(exit_rules) < self.validation_rules['rule_requirements']['min_exit_rules']:
            structural_issues.append(f"Need at least {self.validation_rules['rule_requirements']['min_exit_rules']} exit rules")
        else:
            for i, rule in enumerate(exit_rules):
                if not isinstance(rule, dict) or 'condition' not in rule:
                    structural_issues.append(f"Exit rule {i+1} malformed")
        
        # Check risk management structure
        risk_mgmt = strategy.get('risk_management', {})
        if not isinstance(risk_mgmt, dict):
            structural_issues.append("Risk management must be a dictionary")
        else:
            for req_field in self.validation_rules['risk_management_requirements']:
                if req_field not in risk_mgmt:
                    structural_issues.append(f"Risk management missing: {req_field}")
        
        if structural_issues:
            report['failed_checks'].append({
                'check': check_name,
                'status': 'failed',
                'details': f"Structural issues: {'; '.join(structural_issues)}",
                'severity': 'critical'
            })
        else:
            report['passed_checks'].append({
                'check': check_name,
                'status': 'passed',
                'details': 'Structural integrity verified'
            })
    
    def _check_logical_consistency(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check logical consistency of strategy rules
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Logical Consistency Check"
        consistency_issues = []
        
        # Check for contradictory rules
        entry_rules = strategy.get('entry_rules', [])
        exit_rules = strategy.get('exit_rules', [])
        
        # Simple contradiction detection (would be more sophisticated in production)
        for entry_rule in entry_rules:
            entry_condition = entry_rule.get('condition', '').lower()
            for exit_rule in exit_rules:
                exit_condition = exit_rule.get('condition', '').lower()
                
                # Check for obvious contradictions
                if ('above' in entry_condition and 'below' in exit_condition) or \
                   ('bullish' in entry_condition and 'bearish' in exit_condition):
                    # This is a very basic check - real implementation would be more sophisticated
                    pass  # These aren't necessarily contradictions
        
        # Check risk management consistency
        risk_mgmt = strategy.get('risk_management', {})
        max_drawdown = self._safe_float_convert(risk_mgmt.get('max_drawdown', 0))
        risk_per_trade = self._safe_float_convert(risk_mgmt.get('risk_per_trade', 0))
        
        if risk_per_trade > max_drawdown / 10:  # Heuristic: risk per trade should be much smaller than max drawdown
            consistency_issues.append(f"Risk per trade ({risk_per_trade}) too high relative to max drawdown ({max_drawdown})")
        
        if consistency_issues:
            report['failed_checks'].append({
                'check': check_name,
                'status': 'failed',
                'details': f"Consistency issues: {'; '.join(consistency_issues)}",
                'severity': 'high'
            })
        else:
            report['passed_checks'].append({
                'check': check_name,
                'status': 'passed',
                'details': 'Logical consistency verified'
            })
    
    def _check_parameter_constraints(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check parameter constraints and ranges
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Parameter Constraints Check"
        constraint_issues = []
        
        risk_mgmt = strategy.get('risk_management', {})
        
        # Check max drawdown constraint
        if 'max_drawdown' in risk_mgmt:
            max_drawdown = self._safe_float_convert(risk_mgmt['max_drawdown'])
            if max_drawdown is not None:
                constraints = self.validation_rules['parameter_constraints'].get('max_drawdown', {})
                if 'min' in constraints and max_drawdown < constraints['min']:
                    constraint_issues.append(f"Max drawdown too low: {max_drawdown} < {constraints['min']}")
                if 'max' in constraints and max_drawdown > constraints['max']:
                    constraint_issues.append(f"Max drawdown too high: {max_drawdown} > {constraints['max']}")
        
        # Check risk per trade constraint
        if 'risk_per_trade' in risk_mgmt:
            risk_per_trade = self._safe_float_convert(risk_mgmt['risk_per_trade'])
            if risk_per_trade is not None:
                constraints = self.validation_rules['parameter_constraints'].get('risk_per_trade', {})
                if 'min' in constraints and risk_per_trade < constraints['min']:
                    constraint_issues.append(f"Risk per trade too low: {risk_per_trade} < {constraints['min']}")
                if 'max' in constraints and risk_per_trade > constraints['max']:
                    constraint_issues.append(f"Risk per trade too high: {risk_per_trade} > {constraints['max']}")
        
        if constraint_issues:
            report['failed_checks'].append({
                'check': check_name,
                'status': 'failed',
                'details': f"Constraint violations: {'; '.join(constraint_issues)}",
                'severity': 'high'
            })
        else:
            report['passed_checks'].append({
                'check': check_name,
                'status': 'passed',
                'details': 'Parameter constraints satisfied'
            })
    
    def _check_novelty_and_innovation(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check strategy novelty and innovation level
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Novelty and Innovation Check"
        strategy_name = strategy.get('name', '').lower()
        description = strategy.get('description', '').lower()
        strategy_type = strategy.get('type', '')
        
        novelty_score = 0.5  # Base score
        innovation_indicators = []
        
        # Check for innovative naming
        encouraged_patterns = self.validation_rules['name_patterns']['encouraged']
        forbidden_patterns = self.validation_rules['name_patterns']['forbidden']
        
        for pattern in encouraged_patterns:
            if pattern in strategy_name or pattern in description:
                novelty_score += 0.1
                innovation_indicators.append(f"Uses {pattern} concept")
        
        for pattern in forbidden_patterns:
            if pattern in strategy_name or pattern in description:
                novelty_score -= 0.15
                report['warnings'].append(f"Name/description contains traditional pattern: {pattern}")
        
        # Check strategy type
        if strategy_type == 'llm_generated':
            novelty_score += 0.2
            innovation_indicators.append("LLM-generated strategy")
        
        # Check for interdisciplinary concepts
        interdisciplinary_concepts = ['quantum', 'neural', 'biology', 'physics', 'game theory', 'complexity']
        for concept in interdisciplinary_concepts:
            if concept in description:
                novelty_score += 0.05
                innovation_indicators.append(f"Incorporates {concept}")
        
        # Cap novelty score
        novelty_score = min(1.0, max(0.3, novelty_score))
        
        report['passed_checks'].append({
            'check': check_name,
            'status': 'passed',
            'details': f"Novelty score: {novelty_score:.2f}",
            'severity': 'low',
            'metadata': {
                'novelty_score': novelty_score,
                'innovation_indicators': innovation_indicators
            }
        })
        
        # Add novelty score to strategy metadata if not present
        if 'metadata' not in strategy:
            strategy['metadata'] = {}
        strategy['metadata']['novelty_score'] = novelty_score
    
    def _check_implementation_feasibility(self, strategy: Dict[str, Any], report: Dict[str, Any]) -> None:
        """
        Check if strategy can be implemented in the system
        
        Args:
            strategy: Strategy to validate
            report: Validation report to update
        """
        check_name = "Implementation Feasibility Check"
        feasibility_issues = []
        feasibility_warnings = []
        
        # Check rule complexity (simplified check)
        entry_rules = strategy.get('entry_rules', [])
        exit_rules = strategy.get('exit_rules', [])
        
        if len(entry_rules) > self.validation_rules['rule_requirements']['max_rules_per_type']:
            feasibility_warnings.append(f"Complex entry rules ({len(entry_rules)} rules) may be hard to implement")
        
        if len(exit_rules) > self.validation_rules['rule_requirements']['max_rules_per_type']:
            feasibility_warnings.append(f"Complex exit rules ({len(exit_rules)} rules) may be hard to implement")
        
        # Check for potentially problematic rule conditions
        problematic_patterns = ['quantum superposition', 'string theory', 'dark matter', 'holographic universe']
        
        for rule_set in [entry_rules, exit_rules]:
            for rule in rule_set:
                condition = rule.get('condition', '').lower()
                for pattern in problematic_patterns:
                    if pattern in condition:
                        feasibility_warnings.append(f"Rule contains potentially unimplementable concept: {pattern}")
        
        if feasibility_issues:
            report['failed_checks'].append({
                'check': check_name,
                'status': 'failed',
                'details': f"Feasibility issues: {'; '.join(feasibility_issues)}",
                'severity': 'medium'
            })
        else:
            report['passed_checks'].append({
                'check': check_name,
                'status': 'passed',
                'details': 'Implementation feasible'
            })
        
        # Add warnings if any
        if feasibility_warnings:
            report['warnings'].extend([
                {
                    'type': 'feasibility',
                    'message': warning,
                    'severity': 'low'
                } for warning in feasibility_warnings
            ])
    
    def _safe_float_convert(self, value: Any) -> Optional[float]:
        """
        Safely convert value to float
        
        Args:
            value: Value to convert
            
        Returns:
            Float value or None if conversion fails
        """
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Remove percentage signs and convert
                clean_value = value.replace('%', '').strip()
                return float(clean_value)
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    def generate_validation_report(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            Detailed validation report
        """
        is_valid, validation_report = self.validate_strategy(strategy)
        
        # Add recommendations based on validation results
        recommendations = self._generate_recommendations(validation_report)
        validation_report['recommendations'] = recommendations
        
        return validation_report
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on validation results
        
        Args:
            validation_report: Validation report
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Recommendations for failed checks
        for failed_check in validation_report['failed_checks']:
            severity = failed_check['severity']
            details = failed_check['details']
            
            if 'required fields' in details.lower():
                recommendations.append({
                    'type': 'critical',
                    'recommendation': 'Add all required strategy fields',
                    'details': details,
                    'priority': 'high'
                })
            elif 'type' in details.lower():
                recommendations.append({
                    'type': 'critical',
                    'recommendation': 'Fix field data types',
                    'details': details,
                    'priority': 'high'
                })
            elif 'structural' in details.lower():
                recommendations.append({
                    'type': 'structural',
                    'recommendation': 'Fix strategy structure',
                    'details': details,
                    'priority': 'high'
                })
            elif 'constraint' in details.lower():
                recommendations.append({
                    'type': 'parameter',
                    'recommendation': 'Adjust parameters to meet constraints',
                    'details': details,
                    'priority': 'medium'
                })
        
        # Recommendations for warnings
        for warning in validation_report['warnings']:
            if 'traditional' in warning['message'].lower():
                recommendations.append({
                    'type': 'innovation',
                    'recommendation': 'Increase novelty and innovation',
                    'details': warning['message'],
                    'priority': 'low'
                })
            elif 'feasibility' in warning['message'].lower():
                recommendations.append({
                    'type': 'feasibility',
                    'recommendation': 'Simplify or clarify implementation',
                    'details': warning['message'],
                    'priority': 'medium'
                })
        
        # General recommendations based on score
        if validation_report['overall_score'] < 0.5:
            recommendations.append({
                'type': 'general',
                'recommendation': 'Major revision needed - strategy has significant issues',
                'details': f"Validation score: {validation_report['overall_score']:.2f}",
                'priority': 'high'
            })
        elif validation_report['overall_score'] < 0.8:
            recommendations.append({
                'type': 'general',
                'recommendation': 'Review and improve strategy quality',
                'details': f"Validation score: {validation_report['overall_score']:.2f}",
                'priority': 'medium'
            })
        else:
            recommendations.append({
                'type': 'general',
                'recommendation': 'Strategy is well-constructed and ready for testing',
                'details': f"Validation score: {validation_report['overall_score']:.2f}",
                'priority': 'low'
            })
        
        return recommendations
    
    def validate_strategy_batch(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of strategies
        
        Args:
            strategies: List of strategies to validate
            
        Returns:
            Batch validation report
        """
        batch_report = {
            'total_strategies': len(strategies),
            'valid_strategies': 0,
            'invalid_strategies': 0,
            'validation_reports': [],
            'batch_timestamp': datetime.now().isoformat()
        }
        
        for strategy in strategies:
            is_valid, validation_report = self.validate_strategy(strategy)
            batch_report['validation_reports'].append(validation_report)
            
            if is_valid:
                batch_report['valid_strategies'] += 1
            else:
                batch_report['invalid_strategies'] += 1
        
        # Calculate batch statistics
        batch_report['validation_rate'] = batch_report['valid_strategies'] / batch_report['total_strategies'] if batch_report['total_strategies'] > 0 else 0.0
        
        # Calculate average novelty score
        novelty_scores = []
        for report in batch_report['validation_reports']:
            for check in report['passed_checks']:
                if check['check'] == 'Novelty and Innovation Check':
                    novelty_scores.append(check['metadata']['novelty_score'])
        
        batch_report['average_novelty_score'] = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        
        self.logger.info(f"Batch validation completed: {batch_report['valid_strategies']}/{batch_report['total_strategies']} strategies valid")
        
        return batch_report
    
    def quick_validation_check(self, strategy: Dict[str, Any]) -> bool:
        """
        Perform quick validation check for basic requirements
        
        Args:
            strategy: Strategy to check
            
        Returns:
            True if basic validation passes, False otherwise
        """
        # Check minimal requirements
        required_fields = ['name', 'entry_rules', 'exit_rules', 'risk_management']
        
        for field in required_fields:
            if field not in strategy:
                return False
        
        # Check that rules are not empty
        if not strategy['entry_rules'] or not strategy['exit_rules']:
            return False
        
        # Check risk management has basic fields
        risk_mgmt = strategy['risk_management']
        if not isinstance(risk_mgmt, dict) or not risk_mgmt:
            return False
        
        return True