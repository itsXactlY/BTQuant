"""
Strategy Lifecycle Manager Module

Manages the complete lifecycle of deployed strategies from deployment to retirement.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

class StrategyStatus(Enum):
    """Strategy lifecycle status"""
    PENDING = "pending"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    RETIRED = "retired"
    FAILED = "failed"

@dataclass
class StrategyDeployment:
    """Data class representing a strategy deployment"""
    deployment_id: str
    strategy_id: str
    strategy_name: str
    status: StrategyStatus
    deployment_timestamp: str
    broker_type: str
    initial_balance: float
    risk_parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class StrategyLifecycleManager:
    """Manages the lifecycle of deployed strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger('StrategyLifecycleManager')
        self.deployments = {}  # deployment_id -> StrategyDeployment
        self.strategy_history = []  # List of all deployment events
        
    def create_deployment(self, strategy: Dict[str, Any], 
                         broker_config: Dict[str, Any]) -> StrategyDeployment:
        """Create a new strategy deployment"""
        
        deployment_id = str(uuid.uuid4())
        deployment_timestamp = datetime.utcnow().isoformat() + 'Z'
        
        # Create deployment record
        deployment = StrategyDeployment(
            deployment_id=deployment_id,
            strategy_id=strategy.get('id', 'unknown'),
            strategy_name=strategy.get('name', strategy.get('template', 'unknown')),
            status=StrategyStatus.PENDING,
            deployment_timestamp=deployment_timestamp,
            broker_type=broker_config.get('broker_type', 'simulated'),
            initial_balance=broker_config.get('initial_balance', 100000.0),
            risk_parameters=strategy.get('risk_parameters', {}),
            metadata={
                'strategy_config': strategy,
                'broker_config': broker_config
            }
        )
        
        # Store deployment
        self.deployments[deployment_id] = deployment
        
        # Record deployment event
        self._record_event(deployment_id, 'deployment_created', 
                          f"Strategy deployment created: {deployment.strategy_name}")
        
        self.logger.info(f"Created deployment {deployment_id} for strategy {deployment.strategy_name}")
        
        return deployment
        
    def deploy_strategy(self, deployment_id: str) -> bool:
        """Deploy a strategy"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        if deployment.status != StrategyStatus.PENDING:
            self.logger.error(f"Cannot deploy strategy in status: {deployment.status}")
            return False
            
        # Update status
        deployment.status = StrategyStatus.DEPLOYED
        
        # Record deployment event
        self._record_event(deployment_id, 'strategy_deployed', 
                          f"Strategy deployed: {deployment.strategy_name}")
        
        self.logger.info(f"Deployed strategy {deployment.strategy_name} (ID: {deployment_id})")
        
        return True
        
    def activate_strategy(self, deployment_id: str) -> bool:
        """Activate a deployed strategy"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        if deployment.status != StrategyStatus.DEPLOYED:
            self.logger.error(f"Cannot activate strategy in status: {deployment.status}")
            return False
            
        # Update status
        deployment.status = StrategyStatus.ACTIVE
        
        # Record activation event
        self._record_event(deployment_id, 'strategy_activated', 
                          f"Strategy activated: {deployment.strategy_name}")
        
        self.logger.info(f"Activated strategy {deployment.strategy_name} (ID: {deployment_id})")
        
        return True
        
    def pause_strategy(self, deployment_id: str) -> bool:
        """Pause an active strategy"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        if deployment.status != StrategyStatus.ACTIVE:
            self.logger.error(f"Cannot pause strategy in status: {deployment.status}")
            return False
            
        # Update status
        deployment.status = StrategyStatus.PAUSED
        
        # Record pause event
        self._record_event(deployment_id, 'strategy_paused', 
                          f"Strategy paused: {deployment.strategy_name}")
        
        self.logger.info(f"Paused strategy {deployment.strategy_name} (ID: {deployment_id})")
        
        return True
        
    def resume_strategy(self, deployment_id: str) -> bool:
        """Resume a paused strategy"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        if deployment.status != StrategyStatus.PAUSED:
            self.logger.error(f"Cannot resume strategy in status: {deployment.status}")
            return False
            
        # Update status
        deployment.status = StrategyStatus.ACTIVE
        
        # Record resume event
        self._record_event(deployment_id, 'strategy_resumed', 
                          f"Strategy resumed: {deployment.strategy_name}")
        
        self.logger.info(f"Resumed strategy {deployment.strategy_name} (ID: {deployment_id})")
        
        return True
        
    def stop_strategy(self, deployment_id: str) -> bool:
        """Stop a strategy"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        if deployment.status not in [StrategyStatus.ACTIVE, StrategyStatus.PAUSED, StrategyStatus.DEPLOYED]:
            self.logger.error(f"Cannot stop strategy in status: {deployment.status}")
            return False
            
        # Update status
        deployment.status = StrategyStatus.STOPPED
        
        # Record stop event
        self._record_event(deployment_id, 'strategy_stopped', 
                          f"Strategy stopped: {deployment.strategy_name}")
        
        self.logger.info(f"Stopped strategy {deployment.strategy_name} (ID: {deployment_id})")
        
        return True
        
    def retire_strategy(self, deployment_id: str) -> bool:
        """Retire a strategy"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        if deployment.status not in [StrategyStatus.STOPPED, StrategyStatus.FAILED]:
            self.logger.error(f"Cannot retire strategy in status: {deployment.status}")
            return False
            
        # Update status
        deployment.status = StrategyStatus.RETIRED
        
        # Record retirement event
        self._record_event(deployment_id, 'strategy_retired', 
                          f"Strategy retired: {deployment.strategy_name}")
        
        self.logger.info(f"Retired strategy {deployment.strategy_name} (ID: {deployment_id})")
        
        return True
        
    def mark_strategy_failed(self, deployment_id: str, reason: str) -> bool:
        """Mark a strategy as failed"""
        
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
            
        deployment = self.deployments[deployment_id]
        
        # Update status
        deployment.status = StrategyStatus.FAILED
        
        # Record failure event
        self._record_event(deployment_id, 'strategy_failed', 
                          f"Strategy failed: {deployment.strategy_name}. Reason: {reason}")
        
        self.logger.error(f"Strategy {deployment.strategy_name} (ID: {deployment_id}) failed: {reason}")
        
        return True
        
    def get_deployment(self, deployment_id: str) -> Optional[StrategyDeployment]:
        """Get deployment information"""
        return self.deployments.get(deployment_id)
        
    def get_all_deployments(self) -> List[StrategyDeployment]:
        """Get all deployments"""
        return list(self.deployments.values())
        
    def get_deployment_history(self, deployment_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get deployment history for a specific deployment or all deployments"""
        
        if deployment_id:
            return [event for event in self.strategy_history 
                   if event['deployment_id'] == deployment_id]
        return self.strategy_history
        
    def get_active_deployments(self) -> List[StrategyDeployment]:
        """Get all active deployments"""
        return [deployment for deployment in self.deployments.values() 
               if deployment.status == StrategyStatus.ACTIVE]
        
    def _record_event(self, deployment_id: str, event_type: str, message: str):
        """Record a lifecycle event"""
        
        event = {
            'event_id': str(uuid.uuid4()),
            'deployment_id': deployment_id,
            'event_type': event_type,
            'message': message,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        self.strategy_history.append(event)
        
    def cleanup_completed_deployments(self):
        """Clean up completed (retired/failed) deployments"""
        
        completed_deployments = [deployment_id for deployment_id, deployment in self.deployments.items() 
                                if deployment.status in [StrategyStatus.RETIRED, StrategyStatus.FAILED]]
        
        for deployment_id in completed_deployments:
            deployment = self.deployments[deployment_id]
            self.logger.info(f"Cleaning up completed deployment: {deployment.strategy_name}")
            del self.deployments[deployment_id]
            
        return len(completed_deployments)