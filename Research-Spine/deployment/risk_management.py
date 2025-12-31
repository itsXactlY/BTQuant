"""
Risk Management Module

Handles risk management, position sizing, and performance tracking for live deployments.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass
class RiskParameters:
    """Data class representing risk management parameters"""
    max_risk_per_trade: float = 0.02  # 2% of account balance
    max_drawdown: float = 0.10  # 10% of account balance
    position_size_method: str = 'fixed_fractional'  # 'fixed', 'fixed_fractional', 'volatility_based'
    leverage_limit: float = 1.0  # 1:1 leverage
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.10  # 10% take profit

@dataclass
class PositionSize:
    """Data class representing calculated position size"""
    symbol: str
    quantity: float
    risk_amount: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float

class RiskManager:
    """Risk management and position sizing calculator"""
    
    def __init__(self, risk_parameters: Optional[RiskParameters] = None):
        self.logger = logging.getLogger('RiskManager')
        self.risk_parameters = risk_parameters or RiskParameters()
        
    def calculate_position_size(self, account_balance: float, 
                              symbol: str, entry_price: float) -> PositionSize:
        """Calculate position size based on risk parameters"""
        
        # Calculate risk amount based on max risk per trade
        risk_amount = account_balance * self.risk_parameters.max_risk_per_trade
        
        # Calculate stop loss price
        if self.risk_parameters.stop_loss_pct > 0:
            stop_loss_price = entry_price * (1 - self.risk_parameters.stop_loss_pct)
        else:
            stop_loss_price = entry_price * 0.95  # Default 5% stop loss
            
        # Calculate take profit price
        if self.risk_parameters.take_profit_pct > 0:
            take_profit_price = entry_price * (1 + self.risk_parameters.take_profit_pct)
        else:
            take_profit_price = entry_price * 1.10  # Default 10% take profit
            
        # Calculate risk-reward ratio
        risk_per_unit = entry_price - stop_loss_price
        reward_per_unit = take_profit_price - entry_price
        risk_reward_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 1.0
        
        # Calculate position size based on risk amount
        if self.risk_parameters.position_size_method == 'fixed_fractional':
            # Fixed fractional position sizing
            quantity = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        elif self.risk_parameters.position_size_method == 'fixed':
            # Fixed position size (e.g., 100 shares)
            quantity = 100  # Default fixed size
        else:  # volatility_based
            # Volatility-based position sizing (simplified)
            quantity = risk_amount / (risk_per_unit * 2)  # More conservative
            
        # Apply leverage limit
        max_position_value = account_balance * self.risk_parameters.leverage_limit
        max_quantity = max_position_value / entry_price
        quantity = min(quantity, max_quantity)
        
        return PositionSize(
            symbol=symbol,
            quantity=quantity,
            risk_amount=risk_amount,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_reward_ratio=risk_reward_ratio
        )
        
    def check_risk_limits(self, account_balance: float, 
                         current_drawdown: float) -> Dict[str, Any]:
        """Check if current risk exposure exceeds limits"""
        
        risk_check = {
            'account_balance': account_balance,
            'current_drawdown': current_drawdown,
            'max_drawdown_limit': self.risk_parameters.max_drawdown,
            'risk_violations': []
        }
        
        # Check drawdown limit
        if current_drawdown > self.risk_parameters.max_drawdown:
            risk_check['risk_violations'].append({
                'type': 'drawdown_limit_exceeded',
                'current': current_drawdown,
                'limit': self.risk_parameters.max_drawdown,
                'severity': 'critical'
            })
            
        return risk_check
        
    def calculate_portfolio_risk(self, positions: List[Dict[str, Any]], 
                                account_balance: float) -> Dict[str, Any]:
        """Calculate overall portfolio risk metrics"""
        
        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        for position in positions:
            position_value = position['quantity'] * position['current_price']
            total_exposure += position_value
            total_unrealized_pnl += position.get('unrealized_pnl', 0)
            
        portfolio_risk = {
            'total_exposure': total_exposure,
            'exposure_percentage': total_exposure / account_balance if account_balance > 0 else 0,
            'unrealized_pnl': total_unrealized_pnl,
            'unrealized_pnl_percentage': total_unrealized_pnl / account_balance if account_balance > 0 else 0,
            'concentration_risk': self._calculate_concentration_risk(positions, total_exposure)
        }
        
        return portfolio_risk
        
    def _calculate_concentration_risk(self, positions: List[Dict[str, Any]], 
                                    total_exposure: float) -> Dict[str, Any]:
        """Calculate concentration risk metrics"""
        
        if not positions or total_exposure <= 0:
            return {
                'max_position_percentage': 0.0,
                'top_3_concentration': 0.0,
                'position_count': 0
            }
            
        position_values = []
        for position in positions:
            position_value = position['quantity'] * position['current_price']
            position_values.append(position_value)
            
        position_values.sort(reverse=True)
        
        max_position_percentage = (position_values[0] / total_exposure) if position_values else 0.0
        top_3_concentration = sum(position_values[:3]) / total_exposure if len(position_values) >= 3 else 0.0
        
        return {
            'max_position_percentage': max_position_percentage,
            'top_3_concentration': top_3_concentration,
            'position_count': len(positions)
        }
        
    def get_risk_parameters(self) -> RiskParameters:
        """Get current risk parameters"""
        return self.risk_parameters
        
    def set_risk_parameters(self, risk_parameters: RiskParameters):
        """Set risk parameters"""
        self.risk_parameters = risk_parameters
        self.logger.info(f"Updated risk parameters: {risk_parameters}")

class PerformanceTracker:
    """Performance tracking and drift detection"""
    
    def __init__(self):
        self.logger = logging.getLogger('PerformanceTracker')
        self.performance_history = []
        self.baseline_metrics = {}
        
    def track_performance(self, deployment_id: str, strategy_id: str, 
                         metrics: Dict[str, Any]):
        """Track performance metrics for a deployment"""
        
        timestamp = datetime.utcnow().isoformat() + 'Z'
        performance_record = {
            'deployment_id': deployment_id,
            'strategy_id': strategy_id,
            'timestamp': timestamp,
            'metrics': metrics
        }
        
        self.performance_history.append(performance_record)
        
        # Check for performance drift
        drift_analysis = self.detect_performance_drift(deployment_id, strategy_id, metrics)
        
        return drift_analysis
        
    def set_baseline_metrics(self, deployment_id: str, strategy_id: str, 
                            baseline_metrics: Dict[str, Any]):
        """Set baseline metrics for drift detection"""
        
        baseline_key = f"{deployment_id}_{strategy_id}"
        self.baseline_metrics[baseline_key] = baseline_metrics
        self.logger.info(f"Set baseline metrics for {deployment_id}: {baseline_metrics}")
        
    def detect_performance_drift(self, deployment_id: str, strategy_id: str, 
                                current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance drift from baseline"""
        
        baseline_key = f"{deployment_id}_{strategy_id}"
        
        if baseline_key not in self.baseline_metrics:
            return {
                'drift_detected': False,
                'message': 'No baseline metrics available for comparison'
            }
            
        baseline_metrics = self.baseline_metrics[baseline_key]
        drift_analysis = {}
        drift_detected = False
        
        # Compare key metrics
        key_metrics = ['sharpe_ratio', 'win_rate', 'max_drawdown', 'profit_factor']
        
        for metric in key_metrics:
            if metric in baseline_metrics and metric in current_metrics:
                baseline_value = baseline_metrics[metric]
                current_value = current_metrics[metric]
                
                # Calculate percentage change
                if baseline_value != 0:
                    change_pct = abs((current_value - baseline_value) / baseline_value) * 100
                else:
                    change_pct = abs(current_value) * 100
                    
                drift_analysis[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'change_pct': change_pct,
                    'drift': change_pct > 20  # 20% threshold for drift
                }
                
                if drift_analysis[metric]['drift']:
                    drift_detected = True
                    
        return {
            'drift_detected': drift_detected,
            'drift_analysis': drift_analysis,
            'message': 'Performance drift detected' if drift_detected else 'Performance within expected range'
        }
        
    def get_performance_history(self, deployment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get performance history for a specific deployment or all deployments"""
        
        if deployment_id:
            return [record for record in self.performance_history 
                   if record['deployment_id'] == deployment_id]
        return self.performance_history
        
    def calculate_rolling_performance(self, deployment_id: str, window_size: int = 10) -> Dict[str, Any]:
        """Calculate rolling performance metrics"""
        
        deployment_history = [record for record in self.performance_history 
                            if record['deployment_id'] == deployment_id]
        
        if len(deployment_history) < window_size:
            return {'error': 'Insufficient data for rolling analysis'}
            
        # Get most recent records
        recent_records = deployment_history[-window_size:]
        
        # Calculate rolling averages
        rolling_metrics = {}
        
        for metric_name in ['pnl', 'drawdown', 'win_rate']:
            values = []
            for record in recent_records:
                if metric_name in record['metrics']:
                    values.append(record['metrics'][metric_name])
                    
            if values:
                rolling_metrics[f'rolling_{metric_name}'] = {
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
                
        return rolling_metrics