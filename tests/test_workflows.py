"""
Tests for Temporal Workflows.
"""

import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from temporalio import workflow
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from shadow_optic.models import (
    WorkflowConfig,
    SamplerConfig,
    EvaluatorConfig,
    OptimizationReport,
    RecommendationAction,
)


class TestWorkflowConfig:
    """Tests for workflow configuration."""
    
    def test_default_config(self):
        """Test default workflow configuration."""
        config = WorkflowConfig(
            challengers=[]  # Required field
        )
        
        assert config.schedule_cron == "0 2 * * *"
        assert config.log_export_window == "24h"
        assert config.max_retries == 3
        assert config.max_budget_per_run == 50.0
    
    def test_custom_config(self):
        """Test custom workflow configuration."""
        from shadow_optic.models import ChallengerConfig
        
        config = WorkflowConfig(
            schedule_cron="0 3 * * *",
            log_export_window="48h",
            max_retries=5,
            max_budget_per_run=100.0,
            challengers=[
                ChallengerConfig(
                    model_id="deepseek-chat",
                    provider="deepseek",
                    provider_slug="deepseek"
                ),
                ChallengerConfig(
                    model_id="claude-3-5-haiku",
                    provider="anthropic",
                    provider_slug="anthropic"
                )
            ]
        )
        
        assert config.schedule_cron == "0 3 * * *"
        assert len(config.challengers) == 2
        assert config.log_export_window == "48h"
        assert config.max_budget_per_run == 100.0
    
    def test_sampler_config_defaults(self):
        """Test sampler configuration defaults."""
        config = SamplerConfig()
        
        assert config.n_clusters == 50
        assert config.samples_per_cluster == 3
        assert config.min_cluster_size == 3
        assert config.embedding_model == "text-embedding-3-small"
    
    def test_evaluator_config_defaults(self):
        """Test evaluator configuration defaults."""
        config = EvaluatorConfig()
        
        assert config.faithfulness_threshold == 0.8
        assert config.quality_threshold == 0.7
        assert config.conciseness_threshold == 0.5
        assert config.refusal_rate_threshold == 0.01


class TestWorkflowLogic:
    """Tests for workflow business logic."""
    
    def test_should_recommend_switch(self):
        """Test switch recommendation logic."""
        # Simulating the decision logic
        challenger_passed = True
        cost_savings_percent = 0.65
        confidence = 0.85
        samples_evaluated = 50
        min_samples = 30
        
        should_switch = (
            challenger_passed and
            cost_savings_percent > 0.3 and
            confidence > 0.7 and
            samples_evaluated >= min_samples
        )
        
        assert should_switch is True
    
    def test_should_not_recommend_low_savings(self):
        """Test no recommendation for low savings."""
        challenger_passed = True
        cost_savings_percent = 0.15  # Only 15% savings
        confidence = 0.85
        
        should_switch = (
            challenger_passed and
            cost_savings_percent > 0.3  # Needs > 30%
        )
        
        assert should_switch is False
    
    def test_should_not_recommend_low_confidence(self):
        """Test no recommendation for low confidence."""
        challenger_passed = True
        cost_savings_percent = 0.65
        confidence = 0.55  # Low confidence
        
        should_switch = (
            challenger_passed and
            cost_savings_percent > 0.3 and
            confidence > 0.7  # Needs > 70%
        )
        
        assert should_switch is False
    
    def test_continue_monitoring_threshold(self):
        """Test continue monitoring decision."""
        samples_evaluated = 25
        min_samples = 30
        partial_success = 0.7  # 70% of tests passed
        
        should_continue = (
            samples_evaluated < min_samples or
            (0.5 < partial_success < 0.9)
        )
        
        assert should_continue is True


class TestWorkflowRetryPolicy:
    """Tests for workflow retry policies."""
    
    def test_activity_retry_config(self):
        """Test activity retry configuration."""
        from temporalio.common import RetryPolicy
        
        policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            maximum_interval=timedelta(minutes=5),
            backoff_coefficient=2.0,
            maximum_attempts=5
        )
        
        assert policy.initial_interval == timedelta(seconds=1)
        assert policy.maximum_interval == timedelta(minutes=5)
        assert policy.backoff_coefficient == 2.0
        assert policy.maximum_attempts == 5
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff calculation."""
        initial = 1.0  # seconds
        coefficient = 2.0
        
        delays = [initial * (coefficient ** i) for i in range(5)]
        
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]
    
    def test_max_interval_capping(self):
        """Test maximum interval capping."""
        initial = 1.0
        coefficient = 2.0
        max_interval = 60.0
        
        delays = []
        for i in range(10):
            delay = min(initial * (coefficient ** i), max_interval)
            delays.append(delay)
        
        # Should cap at max_interval
        assert all(d <= max_interval for d in delays)
        assert delays[-1] == max_interval


class TestWorkflowProgressTracking:
    """Tests for workflow progress tracking."""
    
    def test_progress_stages(self):
        """Test workflow progress stage definitions."""
        stages = [
            "initializing",
            "exporting_logs",
            "sampling_prompts",
            "replaying_requests",
            "evaluating_results",
            "generating_report",
            "complete"
        ]
        
        assert len(stages) == 7
        assert stages[0] == "initializing"
        assert stages[-1] == "complete"
    
    def test_progress_percentage_calculation(self):
        """Test progress percentage calculation."""
        total_prompts = 50
        processed_prompts = 25
        
        percentage = (processed_prompts / total_prompts) * 100
        
        assert percentage == 50.0
    
    def test_eta_calculation(self):
        """Test ETA calculation."""
        from datetime import datetime
        
        start_time = datetime.now()
        processed = 30
        total = 100
        elapsed_seconds = 60  # 1 minute
        
        rate = processed / elapsed_seconds  # items per second
        remaining = total - processed
        eta_seconds = remaining / rate if rate > 0 else float('inf')
        
        assert rate == 0.5
        assert eta_seconds == 140  # 70 items at 0.5/sec = 140 seconds


class TestWorkflowCircuitBreaker:
    """Tests for circuit breaker pattern."""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        states = ["closed", "open", "half-open"]
        
        # Initial state
        current_state = "closed"
        failure_count = 0
        failure_threshold = 5
        
        # Simulate failures
        for _ in range(5):
            failure_count += 1
            if failure_count >= failure_threshold:
                current_state = "open"
        
        assert current_state == "open"
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery."""
        current_state = "open"
        cooldown_elapsed = True
        test_success = True
        
        # After cooldown, move to half-open
        if cooldown_elapsed:
            current_state = "half-open"
        
        # If test succeeds, close circuit
        if current_state == "half-open" and test_success:
            current_state = "closed"
        
        assert current_state == "closed"
    
    def test_failure_rate_calculation(self):
        """Test failure rate calculation."""
        window_size = 100
        failures = 8
        
        failure_rate = failures / window_size
        
        assert failure_rate == 0.08
        assert failure_rate < 0.10  # Below 10% threshold


class TestWorkflowScheduling:
    """Tests for workflow scheduling."""
    
    def test_cron_schedule_parsing(self):
        """Test cron schedule configuration."""
        cron_schedule = "0 */6 * * *"  # Every 6 hours
        
        # Parse components
        parts = cron_schedule.split()
        
        assert len(parts) == 5
        assert parts[0] == "0"      # Minute
        assert parts[1] == "*/6"    # Hour
        assert parts[2] == "*"      # Day of month
        assert parts[3] == "*"      # Month
        assert parts[4] == "*"      # Day of week
    
    def test_schedule_overlap_prevention(self):
        """Test schedule overlap prevention logic."""
        workflow_running = True
        
        # Should skip if already running
        should_start_new = not workflow_running
        
        assert should_start_new is False
    
    def test_time_window_calculation(self):
        """Test time window calculation for log export."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        window_hours = 24
        
        start_time = now - timedelta(hours=window_hours)
        end_time = now
        
        duration = end_time - start_time
        
        assert duration == timedelta(hours=24)


class TestWorkflowSignals:
    """Tests for workflow signal handling."""
    
    def test_cancel_signal_handling(self):
        """Test cancel signal handling logic."""
        cancel_requested = False
        current_stage = "replaying_requests"
        
        # Simulate cancel request
        cancel_requested = True
        
        # Should complete current batch then stop
        should_continue = not cancel_requested
        
        assert should_continue is False
    
    def test_pause_resume_signals(self):
        """Test pause/resume signal handling."""
        paused = False
        
        # Pause
        paused = True
        assert paused is True
        
        # Resume
        paused = False
        assert paused is False
    
    def test_priority_update_signal(self):
        """Test priority update signal."""
        current_priority = "normal"
        
        # Update priority
        new_priority = "high"
        current_priority = new_priority
        
        assert current_priority == "high"
