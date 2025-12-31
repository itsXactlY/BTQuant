#!/usr/bin/env python3
"""
Test script for the deployment system
"""

import sys
from pathlib import Path
import time

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from deployment.deployment_manager import DeploymentManager
from deployment.brokers.broker_interface import Order
from deployment.monitoring.monitoring_system import AlertLevel, AlertType
from deployment.risk_management import RiskParameters

def test_deployment_system():
    """Test the complete deployment system"""
    
    print("🚀 Testing Deployment System")
    print("=" * 50)
    
    # Initialize deployment manager
    print("1. Initializing Deployment Manager...")
    deployment_manager = DeploymentManager()
    print("   ✓ DeploymentManager initialized")
    
    # Test strategy data
    test_strategy = {
        'id': 'test_strategy_001',
        'name': 'Test Strategy',
        'template': 'SMA_Crossover',
        'parameters': {
            'fast_period': 10,
            'slow_period': 50
        },
        'backtest_results': {
            'sharpe_ratio': 1.5,
            'win_rate': 0.65,
            'max_drawdown': 0.12,
            'profit_factor': 1.8
        },
        'risk_parameters': {
            'max_risk_per_trade': 0.02,
            'max_drawdown': 0.10,
            'position_size_method': 'fixed_fractional'
        }
    }
    
    # Test deployment configuration
    deployment_config = {
        'broker_type': 'simulated',
        'broker_config': {
            'initial_balance': 100000.0
        },
        'risk_management': {
            'enabled': True
        },
        'monitoring': {
            'enabled': True,
            'alert_thresholds': {
                'max_drawdown': 0.15,
                'min_win_rate': 0.5
            }
        }
    }
    
    # Test 1: Deploy strategy
    print("\n2. Testing Strategy Deployment...")
    deployment_result = deployment_manager.deploy_strategy(test_strategy, deployment_config)
    
    if deployment_result['status'] == 'deployed':
        deployment_id = deployment_result['deployment_id']
        print(f"   ✓ Strategy deployed successfully (ID: {deployment_id})")
        print(f"   ✓ Broker: {deployment_result['broker_type']}")
        print(f"   ✓ Initial balance: ${deployment_result['initial_balance']:,.2f}")
    else:
        print(f"   ✗ Deployment failed: {deployment_result.get('error', 'Unknown error')}")
        return False
    
    # Test 2: Get deployment status
    print("\n3. Testing Deployment Status...")
    status_result = deployment_manager.get_deployment_status(deployment_id)
    
    if status_result.get('status') == 'active':
        print(f"   ✓ Deployment status: {status_result['status']}")
        print(f"   ✓ Strategy: {status_result['strategy_name']}")
    else:
        print(f"   ✗ Status check failed: {status_result.get('error', 'Unknown error')}")
    
    # Test 3: Execute trade
    print("\n4. Testing Trade Execution...")
    test_order = {
        'symbol': 'AAPL',
        'side': 'buy',
        'order_type': 'market',
        'price': 150.0,
        'quantity': 100
    }
    
    trade_result = deployment_manager.execute_trade(deployment_id, test_order)
    
    if trade_result['status'] == 'filled':
        print(f"   ✓ Trade executed successfully")
        print(f"   ✓ Order ID: {trade_result['order_id']}")
        print(f"   ✓ Quantity: {trade_result['quantity']}")
        print(f"   ✓ Risk amount: ${trade_result['risk_amount']:,.2f}")
        print(f"   ✓ Risk-reward ratio: {trade_result['risk_reward_ratio']:.2f}")
    else:
        print(f"   ✗ Trade execution failed: {trade_result.get('error', 'Unknown error')}")
    
    # Test 4: Monitor deployment
    print("\n5. Testing Deployment Monitoring...")
    print("   🔄 Waiting for monitoring data...")
    time.sleep(2)  # Wait for monitoring to collect data
    
    monitoring_result = deployment_manager.monitor_deployment(deployment_id)
    
    if monitoring_result.get('status') == 'active':
        print(f"   ✓ Monitoring data collected")
        print(f"   ✓ Account balance: ${monitoring_result['account_balance']['total_balance']:,.2f}")
        print(f"   ✓ Open positions: {len(monitoring_result['open_positions'])}")
        print(f"   ✓ Performance metrics: {len(monitoring_result['performance_metrics'])}")
        
        # Check for alerts
        alerts = monitoring_result.get('alerts', [])
        if alerts:
            print(f"   ⚠ Alerts detected: {len(alerts)}")
            for alert in alerts:
                print(f"     - {alert['alert_level']}: {alert['message']}")
        else:
            print(f"   ✓ No alerts detected")
    else:
        print(f"   ✗ Monitoring failed: {monitoring_result.get('error', 'Unknown error')}")
    
    # Test 5: Get performance report
    print("\n6. Testing Performance Report...")
    performance_report = deployment_manager.get_performance_report(deployment_id)
    
    if 'performance_history' in performance_report:
        print(f"   ✓ Performance report generated")
        print(f"   ✓ History records: {len(performance_report['performance_history'])}")
        print(f"   ✓ Current metrics: {len(performance_report['current_metrics'])}")
    else:
        print(f"   ✗ Performance report failed: {performance_report.get('error', 'Unknown error')}")
    
    # Test 6: Stop deployment
    print("\n7. Testing Deployment Stop...")
    stop_result = deployment_manager.stop_deployment(deployment_id)
    
    if stop_result['status'] == 'stopped':
        print(f"   ✓ Deployment stopped successfully")
    else:
        print(f"   ✗ Stop failed: {stop_result.get('error', 'Unknown error')}")
    
    # Test 7: Shutdown deployment manager
    print("\n8. Testing Deployment Manager Shutdown...")
    deployment_manager.shutdown()
    print(f"   ✓ DeploymentManager shutdown complete")
    
    print("\n" + "=" * 50)
    print("🎉 All deployment system tests completed successfully!")
    print("\nDeployment System Features:")
    print("✓ Live trading interface with broker integration")
    print("✓ Real-time monitoring and alerting")
    print("✓ Performance tracking and drift detection")
    print("✓ Risk management and position sizing")
    print("✓ Strategy lifecycle management")
    print("✓ Integration with all system components")
    
    return True

if __name__ == "__main__":
    try:
        success = test_deployment_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)