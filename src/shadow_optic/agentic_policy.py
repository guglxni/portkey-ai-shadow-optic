"""
Agentic Policy Framework for Shadow-Optic.

Implements the control plane for autonomous prompt governance:
- AgenticPolicy: Schema for cluster-level policies stored in Qdrant
- ClusterState: State machine for prompt cluster lifecycle
- RoutingRule: Conditional routing logic based on metrics
- EvolutionRecord: Audit trail for policy changes

The policy framework enables closed-loop optimization:
1. Detect - Watchdog identifies drifting/expensive clusters
2. Diagnose - Shadow-Optic evaluates challenger models
3. Optimize - DSPy rewrites prompts for target models
4. Deploy - Updated policy pushed to Portkey
"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ClusterState(str, Enum):
    """State machine states for prompt clusters.
    
    Lifecycle:
        EMBRYONIC -> STABLE -> DRIFTING -> UNSTABLE
                       ^          |
                       +----------+
                    (after optimization)
    """
    EMBRYONIC = "embryonic"   # New cluster, insufficient data
    STABLE = "stable"         # Healthy, within thresholds
    DRIFTING = "drifting"     # Variance detected, monitoring
    UNSTABLE = "unstable"     # Action required (DSPy/fallback)


class TriggerType(str, Enum):
    """Types of triggers that can activate the watchdog."""
    DRIFT = "drift"           # Semantic/statistical drift detected
    COST = "cost"             # Cost exceeds target
    QUALITY = "quality"       # Evaluation score below threshold
    LATENCY = "latency"       # Response time exceeds SLA
    VOLUME = "volume"         # Traffic spike detected


class ActionType(str, Enum):
    """Actions the system can take in response to triggers."""
    NONE = "none"                     # No action required
    FALLBACK = "fallback"             # Switch to fallback model
    SWITCH_PROVIDER = "switch_provider"  # Change provider
    DSPY_OPTIMIZE = "dspy_optimize"   # Trigger DSPy optimization
    ALERT = "alert"                   # Notify human
    THROTTLE = "throttle"             # Rate limit cluster
    AUTO_DEPLOY = "auto_deploy"       # Deploy optimized prompt


class OptimizationMetric(str, Enum):
    """Metrics that can be optimized."""
    FAITHFULNESS = "faithfulness"
    QUALITY = "quality"
    CONCISENESS = "conciseness"
    LATENCY = "latency"
    COST = "cost"
    COMPOSITE = "composite"


# =============================================================================
# Policy Models
# =============================================================================

class OptimizationTarget(BaseModel):
    """Target constraints for prompt optimization."""
    metric: OptimizationMetric = Field(
        default=OptimizationMetric.FAITHFULNESS,
        description="Primary metric to optimize"
    )
    min_score: float = Field(
        default=0.85, 
        ge=0.0, 
        le=1.0,
        description="Minimum acceptable score (0-1)"
    )
    max_cost_per_1k: float = Field(
        default=0.01,
        ge=0.0,
        description="Maximum cost per 1000 tokens in USD"
    )
    max_latency_ms: int = Field(
        default=2000,
        ge=0,
        description="Maximum acceptable latency in milliseconds"
    )
    
    def is_satisfied(
        self,
        score: float,
        cost_per_1k: float,
        latency_ms: int
    ) -> bool:
        """Check if current metrics satisfy the target."""
        return (
            score >= self.min_score and
            cost_per_1k <= self.max_cost_per_1k and
            latency_ms <= self.max_latency_ms
        )


class RoutingRule(BaseModel):
    """Conditional routing rule for the policy engine.
    
    Example:
        {"condition": "faithfulness < 0.9", "action": "fallback_to_gpt4"}
    """
    condition: str = Field(
        ...,
        description="Condition expression (e.g., 'faithfulness < 0.9')"
    )
    action: str = Field(
        ...,
        description="Action to take (e.g., 'fallback_to_gpt4', 'switch_provider')"
    )
    priority: int = Field(
        default=0,
        description="Higher priority rules evaluated first"
    )
    enabled: bool = Field(
        default=True,
        description="Whether this rule is active"
    )
    
    def evaluate(self, metrics: Dict[str, float]) -> bool:
        """Evaluate the condition against provided metrics.
        
        Supports simple comparisons: <, >, <=, >=, ==, !=
        """
        if not self.enabled:
            return False
            
        # Parse condition (simple parser for now)
        import re
        match = re.match(r'(\w+)\s*(<=|>=|<|>|==|!=)\s*([\d.]+)', self.condition)
        if not match:
            logger.warning(f"Could not parse condition: {self.condition}")
            return False
        
        metric_name, operator, threshold = match.groups()
        threshold = float(threshold)
        
        if metric_name not in metrics:
            return False
        
        value = metrics[metric_name]
        
        ops = {
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: abs(a - b) < 0.0001,
            '!=': lambda a, b: abs(a - b) >= 0.0001,
        }
        
        return ops[operator](value, threshold)


class EvolutionRecord(BaseModel):
    """Audit record for policy/template changes."""
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    action: ActionType = Field(..., description="Action taken")
    change_description: str = Field(..., description="What was changed")
    improvement: Optional[str] = Field(
        None, 
        description="Measured improvement (e.g., '+12% accuracy')"
    )
    triggered_by: Optional[TriggerType] = Field(
        None,
        description="What triggered this change"
    )
    model_before: Optional[str] = None
    model_after: Optional[str] = None
    prompt_version_before: Optional[str] = None
    prompt_version_after: Optional[str] = None


class AgenticPolicy(BaseModel):
    """
    The core policy schema for autonomous prompt governance.
    
    Stored as Qdrant payload on each template cluster, enabling:
    - Intent-based routing (finance, code, creative, etc.)
    - Sensitivity classification (PII handling)
    - Dynamic optimization targets
    - Conditional routing rules
    - Evolution audit trail
    
    Example Qdrant payload:
    ```json
    {
        "cluster_id": "finance_earnings_v1",
        "state": "STABLE",
        "agentic_policy": {
            "intent": "financial_analysis",
            "sensitivity": "high",
            "optimization_target": {...},
            "routing_rules": [...]
        }
    }
    ```
    """
    # Identity
    cluster_id: str = Field(..., description="Unique cluster identifier")
    version: str = Field(default="1.0.0", description="Policy version")
    
    # Classification
    intent: str = Field(
        default="general",
        description="Semantic intent (e.g., 'financial_analysis', 'code_generation')"
    )
    sensitivity: str = Field(
        default="internal",
        description="Data sensitivity: 'public', 'internal', 'restricted'"
    )
    
    # State Machine
    state: ClusterState = Field(
        default=ClusterState.EMBRYONIC,
        description="Current cluster lifecycle state"
    )
    
    # Optimization
    optimization_target: OptimizationTarget = Field(
        default_factory=OptimizationTarget,
        description="Target metrics for optimization"
    )
    
    # Routing
    routing_rules: List[RoutingRule] = Field(
        default_factory=list,
        description="Ordered list of routing rules"
    )
    
    # Model Configuration
    primary_model: str = Field(
        default="gpt-5-mini",
        description="Primary model for this cluster"
    )
    fallback_model: Optional[str] = Field(
        default="gpt-4o",
        description="Fallback model if primary fails"
    )
    
    # DSPy Configuration
    dspy_enabled: bool = Field(
        default=False,
        description="Whether DSPy optimization is enabled for this cluster"
    )
    dspy_signature: Optional[str] = Field(
        None,
        description="DSPy signature class name (e.g., 'EarningsSummary')"
    )
    dspy_optimized_prompt: Optional[str] = Field(
        None,
        description="The optimized prompt from MIPROv2"
    )
    
    # Governance
    auto_deploy: bool = Field(
        default=False,
        description="Whether to auto-deploy optimizations without human review"
    )
    owner: Optional[str] = Field(
        None,
        description="Team or individual responsible for this cluster"
    )
    
    # Metrics (updated by Watchdog)
    current_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Current metric values (faithfulness, quality, cost, etc.)"
    )
    variance: float = Field(
        default=0.0,
        ge=0.0,
        description="Statistical variance in cluster (drift indicator)"
    )
    
    # History
    evolution_history: List[EvolutionRecord] = Field(
        default_factory=list,
        description="Audit trail of policy changes"
    )
    
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    
    @field_validator('sensitivity')
    @classmethod
    def validate_sensitivity(cls, v):
        allowed = {'public', 'internal', 'restricted'}
        if v not in allowed:
            raise ValueError(f"sensitivity must be one of {allowed}")
        return v
    
    def should_trigger_optimization(
        self,
        drift_threshold: float = 0.15,
        quality_threshold: float = 0.85
    ) -> Optional[TriggerType]:
        """Check if optimization should be triggered.
        
        Returns the trigger type if action needed, None otherwise.
        """
        # Drift check
        if self.variance > drift_threshold:
            return TriggerType.DRIFT
        
        # Quality check
        quality = self.current_metrics.get('quality', 1.0)
        faithfulness = self.current_metrics.get('faithfulness', 1.0)
        if quality < quality_threshold or faithfulness < quality_threshold:
            return TriggerType.QUALITY
        
        # Cost check
        cost = self.current_metrics.get('cost_per_1k', 0.0)
        if cost > self.optimization_target.max_cost_per_1k:
            return TriggerType.COST
        
        # Latency check
        latency = self.current_metrics.get('latency_ms', 0)
        if latency > self.optimization_target.max_latency_ms:
            return TriggerType.LATENCY
        
        return None
    
    def get_applicable_rule(
        self,
        metrics: Dict[str, float]
    ) -> Optional[RoutingRule]:
        """Get the first matching routing rule for current metrics."""
        sorted_rules = sorted(
            self.routing_rules, 
            key=lambda r: r.priority, 
            reverse=True
        )
        for rule in sorted_rules:
            if rule.evaluate(metrics):
                return rule
        return None
    
    def add_evolution_record(
        self,
        action: ActionType,
        change_description: str,
        improvement: Optional[str] = None,
        triggered_by: Optional[TriggerType] = None
    ) -> None:
        """Add a new evolution record to the history."""
        self.evolution_history.append(EvolutionRecord(
            action=action,
            change_description=change_description,
            improvement=improvement,
            triggered_by=triggered_by
        ))
        self.updated_at = datetime.now(timezone.utc)
    
    def transition_state(self, new_state: ClusterState) -> None:
        """Transition to a new state with validation."""
        valid_transitions = {
            ClusterState.EMBRYONIC: {ClusterState.STABLE, ClusterState.DRIFTING},
            ClusterState.STABLE: {ClusterState.DRIFTING, ClusterState.UNSTABLE},
            ClusterState.DRIFTING: {ClusterState.STABLE, ClusterState.UNSTABLE},
            ClusterState.UNSTABLE: {ClusterState.STABLE, ClusterState.DRIFTING},
        }
        
        if new_state not in valid_transitions.get(self.state, set()):
            logger.warning(
                f"Invalid state transition: {self.state} -> {new_state}"
            )
        
        old_state = self.state
        self.state = new_state
        self.add_evolution_record(
            action=ActionType.ALERT,
            change_description=f"State transition: {old_state} -> {new_state}"
        )
        logger.info(f"Cluster {self.cluster_id} transitioned: {old_state} -> {new_state}")
    
    def should_trigger(self, metrics: Dict[str, float]) -> bool:
        """Check if current metrics should trigger any action.
        
        Args:
            metrics: Current cluster metrics (quality, variance, etc.)
            
        Returns:
            True if any trigger condition is met
        """
        return self.get_trigger_type(metrics) is not None
    
    def get_trigger_type(self, metrics: Dict[str, float]) -> Optional[TriggerType]:
        """Determine which trigger type applies based on metrics.
        
        Priority: QUALITY > DRIFT > LATENCY > COST
        
        Args:
            metrics: Current cluster metrics
            
        Returns:
            The applicable TriggerType, or None if healthy
        """
        quality = metrics.get("quality", 1.0)
        faithfulness = metrics.get("faithfulness", 1.0)
        variance = metrics.get("variance", 0.0)
        latency = metrics.get("latency_p95", metrics.get("latency_ms", 0))
        
        # Quality issues take priority
        if quality < self.optimization_target.min_score or faithfulness < 0.85:
            return TriggerType.QUALITY
        
        # Then drift
        if variance > 0.15:
            return TriggerType.DRIFT
        
        # Then latency
        if latency > self.optimization_target.max_latency_ms:
            return TriggerType.LATENCY
        
        # Then cost
        cost = metrics.get("cost_per_1k", 0.0)
        if cost > self.optimization_target.max_cost_per_1k:
            return TriggerType.COST
        
        return None
    
    def resolve_action(self, trigger: TriggerType) -> ActionType:
        """Resolve what action to take for a given trigger.
        
        Args:
            trigger: The trigger type detected
            
        Returns:
            The action to take
        """
        # Unstable state always triggers fallback
        if self.state == ClusterState.UNSTABLE:
            return ActionType.FALLBACK
        
        # DSPy-enabled clusters can optimize
        if self.dspy_enabled:
            if trigger in (TriggerType.DRIFT, TriggerType.QUALITY):
                return ActionType.DSPY_OPTIMIZE
        
        # Default to alert if DSPy not enabled
        if trigger in (TriggerType.DRIFT, TriggerType.QUALITY):
            return ActionType.ALERT
        
        if trigger == TriggerType.LATENCY:
            return ActionType.FALLBACK
        
        if trigger == TriggerType.COST:
            return ActionType.ALERT
        
        return ActionType.NONE
    
    def record_optimization(
        self,
        action: ActionType,
        description: str,
        metrics_before: Optional[Dict[str, float]] = None,
        metrics_after: Optional[Dict[str, float]] = None
    ) -> None:
        """Record an optimization attempt in evolution history.
        
        Args:
            action: The action taken
            description: What was done
            metrics_before: Metrics before optimization
            metrics_after: Metrics after optimization
        """
        improvement = None
        if metrics_before and metrics_after:
            before_q = metrics_before.get("quality", 0)
            after_q = metrics_after.get("quality", 0)
            if before_q > 0:
                improvement = f"{((after_q - before_q) / before_q) * 100:.1f}% quality improvement"
        
        self.evolution_history.append(EvolutionRecord(
            action=action,
            change_description=description,
            improvement=improvement,
            triggered_by=TriggerType.DRIFT
        ))
        self.updated_at = datetime.now(timezone.utc)
    
    def can_auto_deploy(self) -> bool:
        """Check if this policy allows automatic deployment.
        
        Restricted sensitivity data cannot be auto-deployed.
        """
        if self.sensitivity == "restricted":
            return False
        return self.auto_deploy
    
    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert this policy to Qdrant-compatible payload format."""
        return {
            "cluster_id": self.cluster_id,
            "state": self.state.value,
            "intent": self.intent,
            "sensitivity": self.sensitivity,
            "agentic_policy": self.model_dump(mode='json'),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_qdrant_payload(cls, payload: Dict[str, Any]) -> "AgenticPolicy":
        """Reconstruct AgenticPolicy from Qdrant payload."""
        if "agentic_policy" in payload:
            policy_data = payload["agentic_policy"]
            if isinstance(policy_data, str):
                import json
                policy_data = json.loads(policy_data)
            return cls(**policy_data)
        else:
            # Legacy format - construct from flat payload
            return cls(
                cluster_id=payload.get("cluster_id", "unknown"),
                state=ClusterState(payload.get("state", "embryonic")),
                intent=payload.get("intent", "general"),
                sensitivity=payload.get("sensitivity", "internal"),
            )


# =============================================================================
# Policy Factory
# =============================================================================

def create_default_policy(
    cluster_id: str,
    intent: str = "general",
    sensitivity: str = "internal"
) -> AgenticPolicy:
    """Create a default policy for a new cluster.
    
    Args:
        cluster_id: Unique identifier for the cluster
        intent: Semantic intent classification
        sensitivity: Data sensitivity level
        
    Returns:
        AgenticPolicy with sensible defaults
    """
    # Default routing rules based on intent
    routing_rules = [
        RoutingRule(
            condition="faithfulness < 0.85",
            action="fallback_to_gpt4",
            priority=100
        ),
        RoutingRule(
            condition="latency > 3000",
            action="switch_provider",
            priority=50
        ),
    ]
    
    # Adjust for sensitivity
    if sensitivity == "restricted":
        routing_rules.insert(0, RoutingRule(
            condition="sensitivity == restricted",
            action="use_local_model",
            priority=200,
            enabled=True
        ))
    
    return AgenticPolicy(
        cluster_id=cluster_id,
        intent=intent,
        sensitivity=sensitivity,
        routing_rules=routing_rules,
        state=ClusterState.EMBRYONIC,
    )


def policy_to_qdrant_payload(policy: AgenticPolicy) -> Dict[str, Any]:
    """Convert AgenticPolicy to Qdrant-compatible payload format."""
    return {
        "cluster_id": policy.cluster_id,
        "state": policy.state.value,
        "intent": policy.intent,
        "sensitivity": policy.sensitivity,
        "agentic_policy": policy.model_dump(mode='json'),
        "updated_at": policy.updated_at.isoformat(),
    }


def policy_from_qdrant_payload(payload: Dict[str, Any]) -> AgenticPolicy:
    """Reconstruct AgenticPolicy from Qdrant payload."""
    if "agentic_policy" in payload:
        return AgenticPolicy(**payload["agentic_policy"])
    else:
        # Legacy format - construct from flat payload
        return AgenticPolicy(
            cluster_id=payload.get("cluster_id", "unknown"),
            state=ClusterState(payload.get("state", "embryonic")),
            intent=payload.get("intent", "general"),
            sensitivity=payload.get("sensitivity", "internal"),
        )
