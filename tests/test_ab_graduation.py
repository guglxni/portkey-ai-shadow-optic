"""
Tests for P2 Enhancement: A/B Graduation Controller.

Tests cover:
- Experiment stage transitions
- Quality gate checks
- Promotion/rollback decisions
- Portkey config generation
- Statistical significance testing
- Experiment management
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from shadow_optic.p2_enhancements.ab_graduation import (
    AutomatedRolloutController,
    PortkeyConfigManager,
    ExperimentManager,
    ExperimentStage,
    ExperimentMetrics,
    StageConfig,
    PromotionDecision,
    PromotionCheckResult,
    ExperimentHistory,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def controller():
    """Create an AutomatedRolloutController instance."""
    return AutomatedRolloutController(
        experiment_id="test-exp-001",
        challenger_model="deepseek-v3",
        production_model="gpt-4o-mini"
    )


@pytest.fixture
def config_manager():
    """Create a PortkeyConfigManager instance."""
    return PortkeyConfigManager()


@pytest.fixture
def experiment_manager():
    """Create an ExperimentManager instance."""
    return ExperimentManager()


@pytest.fixture
def passing_challenger_metrics():
    """Metrics that pass all quality gates."""
    return ExperimentMetrics(
        total_requests=10000,
        successful_responses=9900,
        refusals=50,
        errors=50,
        avg_latency_ms=150,
        p50_latency_ms=120,
        p95_latency_ms=280,
        p99_latency_ms=450,
        avg_quality_score=0.88,
        avg_faithfulness=0.92,
        cost_per_request=0.0002,
        total_cost=2.0
    )


@pytest.fixture
def failing_challenger_metrics():
    """Metrics that fail quality gates."""
    return ExperimentMetrics(
        total_requests=1000,
        successful_responses=800,
        refusals=100,
        errors=100,
        avg_latency_ms=500,
        p50_latency_ms=400,
        p95_latency_ms=800,
        p99_latency_ms=1200,
        avg_quality_score=0.60,
        avg_faithfulness=0.70,
        cost_per_request=0.0005,
        total_cost=0.5
    )


@pytest.fixture
def production_metrics():
    """Baseline production metrics."""
    return ExperimentMetrics(
        total_requests=100000,
        successful_responses=99500,
        refusals=300,
        errors=200,
        avg_latency_ms=180,
        p50_latency_ms=150,
        p95_latency_ms=320,
        p99_latency_ms=500,
        avg_quality_score=0.90,
        avg_faithfulness=0.95,
        cost_per_request=0.002,
        total_cost=200.0
    )


# =============================================================================
# ExperimentMetrics Tests
# =============================================================================

class TestExperimentMetrics:
    """Tests for ExperimentMetrics dataclass."""
    
    def test_success_rate(self, passing_challenger_metrics):
        """Test success rate calculation."""
        assert passing_challenger_metrics.success_rate == 0.99
    
    def test_error_rate(self, passing_challenger_metrics):
        """Test error rate calculation."""
        assert passing_challenger_metrics.error_rate == 0.005
    
    def test_refusal_rate(self, passing_challenger_metrics):
        """Test refusal rate calculation."""
        assert passing_challenger_metrics.refusal_rate == 0.005
    
    def test_zero_requests(self):
        """Test rates with zero requests."""
        metrics = ExperimentMetrics()
        
        assert metrics.success_rate == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.refusal_rate == 0.0


# =============================================================================
# StageConfig Tests
# =============================================================================

class TestStageConfig:
    """Tests for StageConfig dataclass."""
    
    def test_weight_calculation(self):
        """Test production/challenger weight calculation."""
        config = StageConfig(
            traffic_percentage=0.10,
            min_samples=1000,
            min_duration_hours=12
        )
        
        assert config.production_weight == 90
        assert config.challenger_weight == 10
    
    def test_weight_full_traffic(self):
        """Test weights at 100% traffic."""
        config = StageConfig(
            traffic_percentage=1.0,
            min_samples=0,
            min_duration_hours=0
        )
        
        assert config.production_weight == 0
        assert config.challenger_weight == 100
    
    def test_weight_shadow(self):
        """Test weights at 0% traffic (shadow)."""
        config = StageConfig(
            traffic_percentage=0.0,
            min_samples=100,
            min_duration_hours=24
        )
        
        assert config.production_weight == 100
        assert config.challenger_weight == 0


# =============================================================================
# PortkeyConfigManager Tests
# =============================================================================

class TestPortkeyConfigManager:
    """Tests for PortkeyConfigManager."""
    
    def test_get_shadow_config(self, config_manager):
        """Test shadow stage config."""
        config = config_manager.get_stage_config(
            ExperimentStage.SHADOW,
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["strategy"]["mode"] == "fallback"
        assert config["targets"][0]["weight"] == 100
    
    def test_get_canary_config(self, config_manager):
        """Test canary stage config."""
        config = config_manager.get_stage_config(
            ExperimentStage.CANARY,
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["strategy"]["mode"] == "loadbalance"
        assert config["targets"][0]["weight"] == 99  # Production
        assert config["targets"][1]["weight"] == 1   # Challenger
    
    def test_get_graduated_config(self, config_manager):
        """Test graduated stage config."""
        config = config_manager.get_stage_config(
            ExperimentStage.GRADUATED,
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["targets"][0]["weight"] == 90
        assert config["targets"][1]["weight"] == 10
    
    def test_get_ramping_config(self, config_manager):
        """Test ramping stage config."""
        config = config_manager.get_stage_config(
            ExperimentStage.RAMPING,
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["targets"][0]["weight"] == 50
        assert config["targets"][1]["weight"] == 50
    
    def test_get_majority_config(self, config_manager):
        """Test majority stage config."""
        config = config_manager.get_stage_config(
            ExperimentStage.MAJORITY,
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["targets"][0]["weight"] == 10  # Production
        assert config["targets"][1]["weight"] == 90  # Challenger
    
    def test_get_full_config(self, config_manager):
        """Test full rollout config."""
        config = config_manager.get_stage_config(
            ExperimentStage.FULL,
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["strategy"]["mode"] == "fallback"
        assert config["targets"][0]["weight"] == 100  # Challenger is now primary
    
    def test_sticky_session_config(self, config_manager):
        """Test sticky session configuration."""
        config = config_manager.get_sticky_session_config(
            ExperimentStage.GRADUATED,
            "gpt-4o-mini",
            "deepseek-v3",
            session_ttl=7200
        )
        
        assert "sticky_session" in config["strategy"]
        assert config["strategy"]["sticky_session"]["ttl"] == 7200
        assert "hash_fields" in config["strategy"]["sticky_session"]
    
    def test_rollback_config(self, config_manager):
        """Test rollback config generation."""
        config = config_manager.create_rollback_config("gpt-4o-mini")
        
        assert config["strategy"]["mode"] == "fallback"
        assert config["targets"][0]["weight"] == 100
        assert config["targets"][0]["override_params"]["model"] == "gpt-4o-mini"
    
    def test_kill_switch_config(self, config_manager):
        """Test kill switch config generation."""
        config = config_manager.create_kill_switch_config(
            "gpt-4o-mini",
            "deepseek-v3"
        )
        
        assert config["strategy"]["mode"] == "loadbalance"
        # Find challenger target
        challenger_target = next(
            t for t in config["targets"] 
            if t.get("override_params", {}).get("model") == "deepseek-v3"
        )
        assert challenger_target["weight"] == 0


# =============================================================================
# AutomatedRolloutController Tests
# =============================================================================

class TestAutomatedRolloutController:
    """Tests for AutomatedRolloutController."""
    
    def test_initialization(self, controller):
        """Test controller initializes correctly."""
        assert controller.experiment_id == "test-exp-001"
        assert controller.challenger_model == "deepseek-v3"
        assert controller.production_model == "gpt-4o-mini"
        assert controller.stage == ExperimentStage.SHADOW
        assert len(controller.history) == 1  # Initial "started" entry
    
    def test_update_metrics(
        self, controller, passing_challenger_metrics, production_metrics
    ):
        """Test metrics update."""
        controller.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=passing_challenger_metrics
        )
        
        assert controller.production_metrics == production_metrics
        assert controller.challenger_metrics == passing_challenger_metrics
    
    @pytest.mark.asyncio
    async def test_check_promotion_insufficient_duration(self, controller):
        """Test promotion check fails when duration not met."""
        # Stage just started, duration not met
        result = await controller.check_promotion()
        
        assert result.decision == PromotionDecision.HOLD
        assert "duration" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_check_promotion_insufficient_samples(
        self, controller, production_metrics
    ):
        """Test promotion check fails when samples not met."""
        # Fake enough duration
        controller.stage_start_time = datetime.utcnow() - timedelta(hours=48)
        
        # But not enough samples
        controller.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=ExperimentMetrics(total_requests=10)
        )
        
        result = await controller.check_promotion()
        
        assert result.decision == PromotionDecision.HOLD
        assert "samples" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_check_promotion_quality_gate_failure(
        self, controller, production_metrics, failing_challenger_metrics
    ):
        """Test promotion check fails on quality gates."""
        controller.stage_start_time = datetime.utcnow() - timedelta(hours=48)
        controller.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=failing_challenger_metrics
        )
        
        result = await controller.check_promotion()
        
        assert result.decision in [PromotionDecision.HOLD, PromotionDecision.ROLLBACK]
        assert "quality" in result.reason.lower() or "gate" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_check_promotion_success(
        self, controller, production_metrics, passing_challenger_metrics
    ):
        """Test successful promotion check."""
        controller.stage_start_time = datetime.utcnow() - timedelta(hours=48)
        controller.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=passing_challenger_metrics
        )
        
        result = await controller.check_promotion()
        
        assert result.decision == PromotionDecision.PROMOTE
        assert result.target_stage == ExperimentStage.CANARY
    
    @pytest.mark.asyncio
    async def test_check_promotion_already_full(self, controller):
        """Test promotion check when already at full rollout."""
        controller.stage = ExperimentStage.FULL
        
        result = await controller.check_promotion()
        
        assert result.decision == PromotionDecision.HOLD
        assert "already" in result.reason.lower()
    
    @pytest.mark.asyncio
    async def test_promote(self, controller):
        """Test stage promotion."""
        initial_stage = controller.stage
        
        new_stage = await controller.promote()
        
        assert new_stage == ExperimentStage.CANARY
        assert controller.stage == ExperimentStage.CANARY
        assert controller.stage != initial_stage
    
    @pytest.mark.asyncio
    async def test_promote_sequence(self, controller):
        """Test full promotion sequence."""
        expected_sequence = [
            ExperimentStage.CANARY,
            ExperimentStage.GRADUATED,
            ExperimentStage.RAMPING,
            ExperimentStage.MAJORITY,
            ExperimentStage.FULL,
        ]
        
        for expected in expected_sequence:
            new_stage = await controller.promote()
            assert new_stage == expected
        
        # Should stay at FULL
        new_stage = await controller.promote()
        assert new_stage == ExperimentStage.FULL
    
    @pytest.mark.asyncio
    async def test_rollback(self, controller):
        """Test rollback functionality."""
        # First promote to canary
        await controller.promote()
        assert controller.stage == ExperimentStage.CANARY
        
        # Then rollback
        new_stage = await controller.rollback("High error rate detected")
        
        assert new_stage == ExperimentStage.SHADOW
        assert controller.stage == ExperimentStage.SHADOW
        
        # Check history recorded rollback
        rollback_entry = next(
            h for h in controller.history if h.action == "rollback"
        )
        assert "error rate" in rollback_entry.reason.lower()
    
    def test_get_portkey_config(self, controller):
        """Test Portkey config generation."""
        config = controller.get_portkey_config()
        
        assert "strategy" in config
        assert "targets" in config
    
    def test_get_dashboard_data(
        self, controller, production_metrics, passing_challenger_metrics
    ):
        """Test dashboard data generation."""
        controller.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=passing_challenger_metrics
        )
        
        data = controller.get_dashboard_data()
        
        assert data["experiment_id"] == "test-exp-001"
        assert data["challenger_model"] == "deepseek-v3"
        assert data["production_model"] == "gpt-4o-mini"
        assert "challenger_metrics" in data
        assert "production_metrics" in data
        assert "stage_progress" in data
        assert "history" in data
    
    def test_get_history(self, controller):
        """Test history retrieval."""
        history = controller.get_history()
        
        assert len(history) >= 1
        assert history[0]["action"] == "started"
    
    @pytest.mark.asyncio
    async def test_baseline_capture_on_first_promotion(
        self, controller, passing_challenger_metrics
    ):
        """Test baseline metrics captured on first promotion."""
        controller.challenger_metrics = passing_challenger_metrics
        
        await controller.promote()
        
        assert controller.baseline_metrics is not None
        assert controller.baseline_metrics.total_requests == passing_challenger_metrics.total_requests


# =============================================================================
# ExperimentManager Tests
# =============================================================================

class TestExperimentManager:
    """Tests for ExperimentManager."""
    
    def test_create_experiment(self, experiment_manager):
        """Test experiment creation."""
        controller = experiment_manager.create_experiment(
            experiment_id="exp-001",
            challenger_model="deepseek-v3",
            production_model="gpt-4o-mini"
        )
        
        assert controller is not None
        assert controller.experiment_id == "exp-001"
        assert "exp-001" in experiment_manager.experiments
    
    def test_create_duplicate_experiment(self, experiment_manager):
        """Test that duplicate experiment IDs raise error."""
        experiment_manager.create_experiment(
            "exp-001", "deepseek-v3", "gpt-4o-mini"
        )
        
        with pytest.raises(ValueError, match="already exists"):
            experiment_manager.create_experiment(
                "exp-001", "deepseek-v3", "gpt-4o-mini"
            )
    
    def test_get_experiment(self, experiment_manager):
        """Test experiment retrieval."""
        experiment_manager.create_experiment(
            "exp-001", "deepseek-v3", "gpt-4o-mini"
        )
        
        controller = experiment_manager.get_experiment("exp-001")
        assert controller is not None
        
        nonexistent = experiment_manager.get_experiment("nonexistent")
        assert nonexistent is None
    
    def test_list_experiments(self, experiment_manager):
        """Test experiment listing."""
        experiment_manager.create_experiment(
            "exp-001", "deepseek-v3", "gpt-4o-mini"
        )
        experiment_manager.create_experiment(
            "exp-002", "llama-4-scout", "gpt-4o-mini"
        )
        
        experiments = experiment_manager.list_experiments()
        
        assert len(experiments) == 2
        assert any(e["experiment_id"] == "exp-001" for e in experiments)
        assert any(e["experiment_id"] == "exp-002" for e in experiments)
    
    def test_delete_experiment(self, experiment_manager):
        """Test experiment deletion."""
        experiment_manager.create_experiment(
            "exp-001", "deepseek-v3", "gpt-4o-mini"
        )
        
        result = experiment_manager.delete_experiment("exp-001")
        assert result is True
        assert experiment_manager.get_experiment("exp-001") is None
        
        result = experiment_manager.delete_experiment("nonexistent")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_all_promotions(
        self, experiment_manager, passing_challenger_metrics, production_metrics
    ):
        """Test bulk promotion checking."""
        exp1 = experiment_manager.create_experiment(
            "exp-001", "deepseek-v3", "gpt-4o-mini"
        )
        exp2 = experiment_manager.create_experiment(
            "exp-002", "llama-4-scout", "gpt-4o-mini"
        )
        
        # Setup exp1 for promotion
        exp1.stage_start_time = datetime.utcnow() - timedelta(hours=48)
        exp1.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=passing_challenger_metrics
        )
        
        results = await experiment_manager.check_all_promotions()
        
        assert "exp-001" in results
        assert "exp-002" in results
        assert results["exp-001"].decision == PromotionDecision.PROMOTE


# =============================================================================
# Integration Tests
# =============================================================================

class TestABGraduationIntegration:
    """Integration tests for A/B graduation system."""
    
    @pytest.mark.asyncio
    async def test_full_graduation_workflow(self):
        """Test full workflow from shadow to full rollout."""
        manager = ExperimentManager()
        
        # 1. Create experiment
        controller = manager.create_experiment(
            "test-graduation",
            "deepseek-v3",
            "gpt-4o-mini"
        )
        
        # 2. Simulate passing metrics that will pass all quality gates
        good_metrics = ExperimentMetrics(
            total_requests=100000,
            successful_responses=99900,  # 99.9% success rate
            refusals=50,                 # 0.05% refusal rate
            errors=50,                   # 0.05% error rate (< 0.001)
            avg_latency_ms=160,          # Within 10% of production 180ms
            avg_quality_score=0.92,
            avg_faithfulness=0.94,
            cost_per_request=0.0002
        )
        
        production_metrics = ExperimentMetrics(
            total_requests=1000000,
            successful_responses=995000,
            refusals=3000,
            errors=2000,
            avg_latency_ms=180,
            avg_quality_score=0.90,
            avg_faithfulness=0.95,
            cost_per_request=0.002
        )
        
        controller.update_metrics(
            production_metrics=production_metrics,
            challenger_metrics=good_metrics
        )
        
        # 3. Simulate time passing and promote through stages
        stages_visited = [controller.stage]
        
        for _ in range(5):  # Max 5 promotions to reach FULL
            # Fast-forward time
            controller.stage_start_time = datetime.utcnow() - timedelta(hours=72)
            
            result = await controller.check_promotion()
            
            if result.decision == PromotionDecision.PROMOTE:
                await controller.promote()
                stages_visited.append(controller.stage)
            
            if controller.stage == ExperimentStage.FULL:
                break
        
        # 4. Verify we reached full rollout
        assert controller.stage == ExperimentStage.FULL
        assert stages_visited == [
            ExperimentStage.SHADOW,
            ExperimentStage.CANARY,
            ExperimentStage.GRADUATED,
            ExperimentStage.RAMPING,
            ExperimentStage.MAJORITY,
            ExperimentStage.FULL,
        ]
        
        # 5. Verify history
        history = controller.get_history()
        promote_count = sum(1 for h in history if h["action"] == "promote")
        assert promote_count == 5
    
    @pytest.mark.asyncio
    async def test_rollback_on_degradation(self):
        """Test automatic rollback when quality degrades."""
        controller = AutomatedRolloutController(
            "test-rollback",
            "deepseek-v3",
            "gpt-4o-mini"
        )
        
        # Start with good metrics and promote to canary
        controller.stage_start_time = datetime.utcnow() - timedelta(hours=48)
        controller.update_metrics(
            production_metrics=ExperimentMetrics(total_requests=100000, successful_responses=99500),
            challenger_metrics=ExperimentMetrics(
                total_requests=1000,
                successful_responses=990,
                avg_quality_score=0.92,
                avg_faithfulness=0.94
            )
        )
        
        await controller.promote()
        assert controller.stage == ExperimentStage.CANARY
        
        # Now simulate degradation
        controller.stage_start_time = datetime.utcnow() - timedelta(hours=8)
        controller.update_metrics(
            challenger_metrics=ExperimentMetrics(
                total_requests=5000,
                successful_responses=4000,
                errors=500,  # High error rate
                avg_quality_score=0.60,  # Low quality
                avg_faithfulness=0.65
            )
        )
        
        result = await controller.check_promotion()
        
        # Should either HOLD or ROLLBACK
        assert result.decision in [PromotionDecision.HOLD, PromotionDecision.ROLLBACK]
    
    def test_portkey_config_progression(self):
        """Test that Portkey configs change correctly through stages."""
        config_manager = PortkeyConfigManager()
        
        # Track traffic percentages through stages
        stages_and_traffic = [
            (ExperimentStage.SHADOW, 0),
            (ExperimentStage.CANARY, 1),
            (ExperimentStage.GRADUATED, 10),
            (ExperimentStage.RAMPING, 50),
            (ExperimentStage.MAJORITY, 90),
            (ExperimentStage.FULL, 100),
        ]
        
        for stage, expected_challenger_pct in stages_and_traffic:
            config = config_manager.get_stage_config(
                stage,
                "gpt-4o-mini",
                "deepseek-v3"
            )
            
            # Find challenger weight
            if stage == ExperimentStage.FULL:
                # In FULL, challenger is primary with 100%
                assert config["targets"][0]["weight"] == 100
            else:
                challenger_target = next(
                    (t for t in config["targets"] 
                     if t.get("override_params", {}).get("model") == "deepseek-v3"),
                    None
                )
                if challenger_target:
                    assert challenger_target["weight"] == expected_challenger_pct
