"""
Tests for AgenticPolicy models and state machine.

Tests the complete agentic metadata system including:
- ClusterState enum and transitions
- TriggerType detection
- ActionType resolution
- AgenticPolicy Pydantic model
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from shadow_optic.agentic_policy import (
    ClusterState,
    TriggerType,
    ActionType,
    AgenticPolicy,
    OptimizationTarget,
    EvolutionRecord,
    policy_to_qdrant_payload,
    policy_from_qdrant_payload,
)


# =============================================================================
# ClusterState Tests
# =============================================================================

class TestClusterState:
    """Tests for ClusterState enum."""
    
    def test_all_states_defined(self):
        """Verify all expected states exist."""
        assert ClusterState.EMBRYONIC.value == "embryonic"
        assert ClusterState.STABLE.value == "stable"
        assert ClusterState.DRIFTING.value == "drifting"
        assert ClusterState.UNSTABLE.value == "unstable"
    
    def test_state_from_string(self):
        """Test creating state from string value."""
        assert ClusterState("embryonic") == ClusterState.EMBRYONIC
        assert ClusterState("stable") == ClusterState.STABLE
    
    def test_invalid_state_raises(self):
        """Test that invalid state raises ValueError."""
        with pytest.raises(ValueError):
            ClusterState("invalid_state")


# =============================================================================
# TriggerType Tests
# =============================================================================

class TestTriggerType:
    """Tests for TriggerType enum."""
    
    def test_trigger_types_exist(self):
        """Verify trigger types exist."""
        assert TriggerType.DRIFT.value == "drift"
        assert TriggerType.QUALITY.value == "quality"
        assert TriggerType.COST.value == "cost"
        assert TriggerType.LATENCY.value == "latency"


# =============================================================================
# ActionType Tests
# =============================================================================

class TestActionType:
    """Tests for ActionType enum."""
    
    def test_action_types_exist(self):
        """Verify action types exist."""
        assert ActionType.FALLBACK.value == "fallback"
        assert ActionType.DSPY_OPTIMIZE.value == "dspy_optimize"
        assert ActionType.ALERT.value == "alert"


# =============================================================================
# OptimizationTarget Tests
# =============================================================================

class TestOptimizationTarget:
    """Tests for OptimizationTarget model."""
    
    def test_default_values(self):
        """Test default target values."""
        target = OptimizationTarget()
        
        assert target.min_score == 0.85
        assert target.max_cost_per_1k == 0.01
        assert target.max_latency_ms == 2000
    
    def test_custom_values(self):
        """Test custom target values."""
        target = OptimizationTarget(
            min_score=0.95,
            max_cost_per_1k=0.005,
            max_latency_ms=1000
        )
        
        assert target.min_score == 0.95
        assert target.max_cost_per_1k == 0.005
        assert target.max_latency_ms == 1000
    
    def test_is_satisfied(self):
        """Test is_satisfied method."""
        target = OptimizationTarget(
            min_score=0.85,
            max_cost_per_1k=0.01,
            max_latency_ms=2000
        )
        
        # Should be satisfied
        assert target.is_satisfied(0.90, 0.008, 1500) is True
        
        # Score too low
        assert target.is_satisfied(0.70, 0.008, 1500) is False
        
        # Cost too high
        assert target.is_satisfied(0.90, 0.02, 1500) is False
        
        # Latency too high
        assert target.is_satisfied(0.90, 0.008, 3000) is False


# =============================================================================
# EvolutionRecord Tests
# =============================================================================

class TestEvolutionRecord:
    """Tests for EvolutionRecord model."""
    
    def test_create_record(self):
        """Test creating an evolution record."""
        record = EvolutionRecord(
            action=ActionType.DSPY_OPTIMIZE,
            change_description="Optimization triggered by drift"
        )
        
        assert record.action == ActionType.DSPY_OPTIMIZE
        assert "drift" in record.change_description.lower()
    
    def test_record_with_improvement(self):
        """Test record with improvement tracking."""
        record = EvolutionRecord(
            action=ActionType.DSPY_OPTIMIZE,
            change_description="Quality improved",
            improvement="+15% quality"
        )
        
        assert record.improvement == "+15% quality"


# =============================================================================
# AgenticPolicy Tests
# =============================================================================

class TestAgenticPolicy:
    """Comprehensive tests for AgenticPolicy model."""
    
    def test_default_state_is_embryonic(self):
        """New policies should start in embryonic state."""
        policy = AgenticPolicy(
            cluster_id="test-cluster",
            intent="customer_support"
        )
        assert policy.state == ClusterState.EMBRYONIC
    
    def test_policy_serialization(self):
        """Test policy can be serialized to dict."""
        policy = AgenticPolicy(
            cluster_id="test-cluster",
            intent="customer_support"
        )
        data = policy.model_dump()
        
        assert data["cluster_id"] == "test-cluster"
        assert data["intent"] == "customer_support"
    
    def test_policy_json_serialization(self):
        """Test policy can be serialized to JSON."""
        policy = AgenticPolicy(
            cluster_id="test-cluster",
            intent="customer_support"
        )
        json_str = policy.model_dump_json()
        
        assert "test-cluster" in json_str
    
    def test_should_trigger_optimization_drift(self):
        """Test drift detection logic."""
        policy = AgenticPolicy(
            cluster_id="drift-test",
            intent="test",
            variance=0.25  # High variance
        )
        
        trigger = policy.should_trigger_optimization()
        
        assert trigger == TriggerType.DRIFT
    
    def test_should_trigger_optimization_quality(self):
        """Test quality degradation detection."""
        policy = AgenticPolicy(
            cluster_id="quality-test",
            intent="test",
            current_metrics={"quality": 0.70, "faithfulness": 0.75}
        )
        
        trigger = policy.should_trigger_optimization()
        
        assert trigger == TriggerType.QUALITY
    
    def test_no_trigger_when_healthy(self):
        """Test no trigger when metrics are good."""
        policy = AgenticPolicy(
            cluster_id="healthy-test",
            intent="test",
            variance=0.05,
            current_metrics={"quality": 0.95, "faithfulness": 0.95}
        )
        
        trigger = policy.should_trigger_optimization()
        
        assert trigger is None
    
    def test_state_transition(self):
        """Test state transitions."""
        policy = AgenticPolicy(
            cluster_id="transition-test",
            intent="test"
        )
        
        # Start embryonic
        assert policy.state == ClusterState.EMBRYONIC
        
        # Transition to drifting
        policy.transition_state(ClusterState.DRIFTING)
        assert policy.state == ClusterState.DRIFTING
        
        # Transition to stable
        policy.transition_state(ClusterState.STABLE)
        assert policy.state == ClusterState.STABLE
    
    def test_add_evolution_record(self):
        """Test adding evolution records."""
        policy = AgenticPolicy(
            cluster_id="record-test",
            intent="test"
        )
        
        policy.add_evolution_record(
            action=ActionType.DSPY_OPTIMIZE,
            change_description="Test optimization"
        )
        
        assert len(policy.evolution_history) == 1
        assert policy.evolution_history[0].action == ActionType.DSPY_OPTIMIZE
    
    def test_resolve_action(self):
        """Test action resolution."""
        policy = AgenticPolicy(
            cluster_id="action-test",
            intent="test",
            dspy_enabled=True
        )
        
        action = policy.resolve_action(TriggerType.QUALITY)
        assert action == ActionType.DSPY_OPTIMIZE
    
    def test_resolve_action_no_dspy(self):
        """Test action resolution without DSPy."""
        policy = AgenticPolicy(
            cluster_id="action-test",
            intent="test",
            dspy_enabled=False
        )
        
        action = policy.resolve_action(TriggerType.QUALITY)
        assert action == ActionType.ALERT
    
    def test_can_auto_deploy_restricted(self):
        """Test that restricted data can't auto-deploy."""
        policy = AgenticPolicy(
            cluster_id="restricted-test",
            intent="test",
            sensitivity="restricted",
            auto_deploy=True
        )
        
        assert policy.can_auto_deploy() is False
    
    def test_can_auto_deploy_internal(self):
        """Test that internal data can auto-deploy."""
        policy = AgenticPolicy(
            cluster_id="internal-test",
            intent="test",
            sensitivity="internal",
            auto_deploy=True
        )
        
        assert policy.can_auto_deploy() is True


# =============================================================================
# Qdrant Payload Tests
# =============================================================================

class TestQdrantPayload:
    """Tests for Qdrant payload conversion."""
    
    def test_policy_to_qdrant_payload(self):
        """Test converting policy to Qdrant payload."""
        policy = AgenticPolicy(
            cluster_id="qdrant-test",
            intent="customer_support",
            sensitivity="internal"
        )
        
        payload = policy_to_qdrant_payload(policy)
        
        assert payload["cluster_id"] == "qdrant-test"
        assert payload["intent"] == "customer_support"
        assert "agentic_policy" in payload
    
    def test_policy_from_qdrant_payload(self):
        """Test reconstructing policy from Qdrant payload."""
        original = AgenticPolicy(
            cluster_id="restore-test",
            intent="billing",
            sensitivity="internal",
            state=ClusterState.STABLE
        )
        
        payload = policy_to_qdrant_payload(original)
        restored = policy_from_qdrant_payload(payload)
        
        assert restored.cluster_id == original.cluster_id
        assert restored.intent == original.intent
        assert restored.state == original.state
