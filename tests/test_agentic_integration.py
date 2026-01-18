"""
Integration tests for the complete agentic flow.

Tests end-to-end integration between:
- AgenticPolicy
- AgenticWatchdog  
- DSPyOptimizer

Verifies the complete autonomous prompt governance loop.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

from shadow_optic.agentic_policy import (
    AgenticPolicy,
    ClusterState,
    TriggerType,
    ActionType,
    OptimizationTarget,
    EvolutionRecord,
    policy_to_qdrant_payload,
    policy_from_qdrant_payload,
    create_default_policy,
)
from shadow_optic.watchdog import (
    WatchdogConfig,
    AgenticWatchdog,
    WatchdogEvent,
    WatchdogEventHandler,
)
from shadow_optic.dspy_optimizer import (
    DSPyOptimizerConfig,
    OptimizationStrategy,
    SignatureBuilder,
    DSPySignature,
)


# =============================================================================
# Policy Lifecycle Integration Tests
# =============================================================================

class TestPolicyLifecycle:
    """Tests for complete policy lifecycle."""
    
    def test_create_policy_and_serialize(self):
        """Test creating a policy and round-tripping through Qdrant format."""
        # Create policy
        policy = AgenticPolicy(
            cluster_id="lifecycle-test",
            intent="customer_support",
            sensitivity="internal",
            dspy_enabled=True,
            optimization_target=OptimizationTarget(
                min_score=0.90,
                max_cost_per_1k=0.008,
                max_latency_ms=1500
            )
        )
        
        # Serialize to Qdrant
        payload = policy_to_qdrant_payload(policy)
        
        # Deserialize back
        restored = policy_from_qdrant_payload(payload)
        
        # Verify
        assert restored.cluster_id == policy.cluster_id
        assert restored.intent == policy.intent
        assert restored.dspy_enabled == policy.dspy_enabled
        assert restored.optimization_target.min_score == 0.90
    
    def test_policy_state_transitions(self):
        """Test complete state machine transitions."""
        policy = AgenticPolicy(
            cluster_id="state-test",
            intent="test"
        )
        
        # Start EMBRYONIC
        assert policy.state == ClusterState.EMBRYONIC
        
        # Mature to STABLE
        policy.transition_state(ClusterState.STABLE)
        assert policy.state == ClusterState.STABLE
        assert len(policy.evolution_history) == 1
        
        # Detect drift
        policy.transition_state(ClusterState.DRIFTING)
        assert policy.state == ClusterState.DRIFTING
        assert len(policy.evolution_history) == 2
        
        # Recover to STABLE
        policy.transition_state(ClusterState.STABLE)
        assert policy.state == ClusterState.STABLE
        assert len(policy.evolution_history) == 3
    
    def test_policy_evolution_tracking(self):
        """Test that all actions are properly recorded."""
        policy = AgenticPolicy(
            cluster_id="evolution-test",
            intent="test"
        )
        
        # Simulate optimization
        policy.record_optimization(
            action=ActionType.DSPY_OPTIMIZE,
            description="MIPROv2 optimization triggered",
            metrics_before={"quality": 0.75},
            metrics_after={"quality": 0.92}
        )
        
        # Simulate fallback
        policy.add_evolution_record(
            action=ActionType.FALLBACK,
            change_description="Fallback to gpt-4o",
            triggered_by=TriggerType.QUALITY
        )
        
        assert len(policy.evolution_history) == 2
        assert policy.evolution_history[0].action == ActionType.DSPY_OPTIMIZE
        assert "improvement" in str(policy.evolution_history[0].improvement).lower() or policy.evolution_history[0].improvement is not None
        assert policy.evolution_history[1].action == ActionType.FALLBACK


# =============================================================================
# Watchdog + Policy Integration Tests
# =============================================================================

class TestWatchdogPolicyIntegration:
    """Tests for watchdog and policy integration."""
    
    @pytest.fixture
    def watchdog_with_policy(self):
        """Create watchdog with a cached policy."""
        mock_qdrant = MagicMock()
        mock_qdrant.scroll = AsyncMock(return_value=([], None))
        mock_qdrant.set_payload = AsyncMock()
        
        config = WatchdogConfig(
            scan_interval_seconds=10,
            drift_variance_threshold=0.15,
            quality_min_score=0.85,
            auto_optimize=True
        )
        
        watchdog = AgenticWatchdog(
            qdrant_client=mock_qdrant,
            config=config
        )
        
        # Pre-cache a policy
        policy = AgenticPolicy(
            cluster_id="integration-test",
            intent="customer_support",
            dspy_enabled=True,
            state=ClusterState.STABLE,
            variance=0.05
        )
        watchdog._policy_cache["integration-test"] = policy
        
        return watchdog, policy
    
    def test_trigger_detection_updates_policy(self, watchdog_with_policy):
        """Test that trigger detection properly identifies issues."""
        watchdog, policy = watchdog_with_policy
        
        # Simulate degraded metrics
        metrics = {
            "quality": 0.70,  # Below threshold
            "faithfulness": 0.75,
            "variance": 0.08,
            "latency_ms": 500,
            "cost_per_1k": 0.005
        }
        
        trigger = watchdog._check_triggers(policy, metrics)
        
        assert trigger == TriggerType.QUALITY
    
    def test_action_determination_with_dspy(self, watchdog_with_policy):
        """Test that DSPy-enabled policies get DSPy optimization action."""
        watchdog, policy = watchdog_with_policy
        
        action, new_state = watchdog._determine_action(
            policy,
            TriggerType.QUALITY,
            {"quality": 0.70}
        )
        
        # With auto_optimize=True and dspy_enabled=True
        assert action == ActionType.DSPY_OPTIMIZE
        assert new_state == ClusterState.UNSTABLE


# =============================================================================
# DSPy + Policy Integration Tests
# =============================================================================

class TestDSPyPolicyIntegration:
    """Tests for DSPy optimizer and policy integration."""
    
    def test_signature_from_genie_template(self):
        """Test building DSPy signature from PromptGenie result."""
        genie_result = {
            "template": "Given the context: {context}\n\nAnswer this question: {question}",
            "metadata": {
                "intent_tags": ["question_answering"],
                "task_type": "generation"
            },
            "variables": {
                "context": {"description": "Background information", "type": "text"},
                "question": {"description": "The user's question", "type": "text"}
            }
        }
        
        sig = SignatureBuilder.from_genie_template(genie_result)
        
        assert len(sig.input_fields) == 2
        assert sig.output_fields[0].name == "generated_text"
    
    def test_config_matches_policy_requirements(self):
        """Test that DSPy config satisfies policy targets."""
        # Create a policy with specific targets
        policy = AgenticPolicy(
            cluster_id="dspy-integration",
            intent="code_generation",
            optimization_target=OptimizationTarget(
                min_score=0.90,
                max_cost_per_1k=0.005,
                max_latency_ms=1000
            )
        )
        
        # Create matching DSPy config
        config = DSPyOptimizerConfig(
            teacher_model="gpt-5",
            student_model="gpt-5-mini",
            target_quality_score=policy.optimization_target.min_score,
            strategy=OptimizationStrategy.MIPRO_V2
        )
        
        assert config.target_quality_score >= policy.optimization_target.min_score


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_default_policy(self):
        """Test default policy creation."""
        policy = create_default_policy(
            cluster_id="factory-test",
            intent="billing",
            sensitivity="restricted"
        )
        
        assert policy.cluster_id == "factory-test"
        assert policy.intent == "billing"
        assert policy.sensitivity == "restricted"
        assert policy.state == ClusterState.EMBRYONIC
        assert len(policy.routing_rules) >= 2  # Has default rules
    
    def test_restricted_policy_has_extra_rules(self):
        """Test that restricted sensitivity adds extra rules."""
        public_policy = create_default_policy(
            cluster_id="public-test",
            sensitivity="public"
        )
        
        restricted_policy = create_default_policy(
            cluster_id="restricted-test",
            sensitivity="restricted"
        )
        
        # Restricted should have additional security rule
        assert len(restricted_policy.routing_rules) > len(public_policy.routing_rules)


# =============================================================================
# Complete Flow Tests
# =============================================================================

class TestCompleteAgenticFlow:
    """Tests for the complete agentic flow."""
    
    def test_detect_optimize_deploy_flow(self):
        """Test the complete detect -> optimize -> deploy flow."""
        # 1. Create a stable cluster
        policy = create_default_policy(
            cluster_id="full-flow-test",
            intent="customer_support"
        )
        policy.state = ClusterState.STABLE
        policy.dspy_enabled = True
        
        # 2. Simulate quality degradation
        policy.current_metrics = {
            "quality": 0.72,
            "faithfulness": 0.75
        }
        
        # 3. Check for trigger
        trigger = policy.should_trigger_optimization()
        assert trigger == TriggerType.QUALITY
        
        # 4. Resolve action
        action = policy.resolve_action(trigger)
        assert action == ActionType.DSPY_OPTIMIZE
        
        # 5. Record optimization
        policy.record_optimization(
            action=action,
            description="Optimized with MIPROv2",
            metrics_before={"quality": 0.72},
            metrics_after={"quality": 0.93}
        )
        
        # 6. Check auto-deploy eligibility
        policy.auto_deploy = True
        assert policy.can_auto_deploy() is True
        
        # 7. Transition back to stable
        policy.transition_state(ClusterState.STABLE)
        
        # Verify audit trail
        assert len(policy.evolution_history) >= 2
        assert policy.state == ClusterState.STABLE
    
    def test_restricted_data_blocks_auto_deploy(self):
        """Test that restricted data cannot auto-deploy."""
        policy = AgenticPolicy(
            cluster_id="restricted-flow",
            intent="financial_analysis",
            sensitivity="restricted",
            auto_deploy=True  # Requested but should be blocked
        )
        
        # Even with auto_deploy=True, restricted data can't auto-deploy
        assert policy.can_auto_deploy() is False
