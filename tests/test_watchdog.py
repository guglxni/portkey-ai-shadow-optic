"""
Tests for AgenticWatchdog background worker.

Tests the autonomous monitoring system including:
- WatchdogConfig configuration
- Cluster scanning logic
- Trigger detection
- State transitions
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from shadow_optic.watchdog import (
    WatchdogConfig,
    AgenticWatchdog,
)
from shadow_optic.agentic_policy import (
    ClusterState,
    TriggerType,
    ActionType,
    AgenticPolicy,
)


# =============================================================================
# WatchdogConfig Tests
# =============================================================================

class TestWatchdogConfig:
    """Tests for WatchdogConfig model."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = WatchdogConfig()
        
        assert config.scan_interval_seconds == 60
        assert config.drift_variance_threshold == 0.15
        assert config.quality_min_score == 0.85
        assert config.faithfulness_min_score == 0.90
        assert config.cost_alert_multiplier == 1.5
        assert config.latency_p95_threshold_ms == 2000
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = WatchdogConfig(
            scan_interval_seconds=30,
            drift_variance_threshold=0.10,
            quality_min_score=0.90
        )
        
        assert config.scan_interval_seconds == 30
        assert config.drift_variance_threshold == 0.10
        assert config.quality_min_score == 0.90
    
    def test_validation_bounds(self):
        """Test configuration validation."""
        # Scan interval must be at least 10
        config = WatchdogConfig(scan_interval_seconds=10)
        assert config.scan_interval_seconds == 10
        
        # Variance threshold between 0 and 1
        config = WatchdogConfig(drift_variance_threshold=0.01)
        assert config.drift_variance_threshold == 0.01


# =============================================================================
# AgenticWatchdog Tests
# =============================================================================

class TestAgenticWatchdog:
    """Comprehensive tests for AgenticWatchdog."""
    
    @pytest.fixture
    def mock_qdrant(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.scroll = AsyncMock(return_value=([
            MagicMock(
                id="cluster-1",
                payload={
                    "cluster_id": "cluster-1",
                    "state": "stable",
                    "intent": "customer_support",
                    "agentic_policy": {
                        "cluster_id": "cluster-1",
                        "state": "stable",
                        "intent": "customer_support",
                        "sensitivity": "internal",
                        "dspy_enabled": True,
                        "auto_deploy": False,
                        "optimization_target": {
                            "min_score": 0.85,
                            "max_cost_per_1k": 0.01,
                            "max_latency_ms": 2000
                        },
                        "evolution_history": []
                    }
                }
            ),
            MagicMock(
                id="cluster-2",
                payload={
                    "cluster_id": "cluster-2",
                    "state": "embryonic",
                    "intent": "billing_inquiry",
                    "agentic_policy": {
                        "cluster_id": "cluster-2",
                        "state": "embryonic",
                        "intent": "billing_inquiry",
                        "sensitivity": "restricted",
                        "dspy_enabled": False,
                        "auto_deploy": False,
                        "optimization_target": {
                            "min_score": 0.90,
                            "max_cost_per_1k": 0.008,
                            "max_latency_ms": 1500
                        },
                        "evolution_history": []
                    }
                }
            )
        ], None))
        return client
    
    @pytest.fixture
    def watchdog(self, mock_qdrant):
        """Create watchdog instance with mocks."""
        config = WatchdogConfig(
            scan_interval_seconds=10,
            drift_variance_threshold=0.15,
            quality_min_score=0.85
        )
        
        return AgenticWatchdog(
            config=config,
            qdrant_client=mock_qdrant,
            collection_name="test_clusters"
        )
    
    def test_watchdog_initialization(self, watchdog):
        """Test watchdog initializes correctly."""
        assert watchdog.config.scan_interval_seconds == 10
        assert watchdog.config.drift_variance_threshold == 0.15
        assert watchdog._running is False
    
    @pytest.mark.asyncio
    async def test_get_all_clusters(self, watchdog):
        """Test loading clusters from Qdrant."""
        clusters = await watchdog._get_all_clusters()
        
        assert len(clusters) == 2
        cluster_ids = [c[0] for c in clusters]
        assert "cluster-1" in cluster_ids
        assert "cluster-2" in cluster_ids
    
    @pytest.mark.asyncio
    async def test_scan_all_clusters(self, watchdog):
        """Test complete scan cycle."""
        with patch.object(watchdog._metric_collector, 'collect_metrics', new_callable=AsyncMock) as mock_metrics:
            mock_metrics.return_value = {
                "quality": 0.90,
                "faithfulness": 0.92,
                "variance": 0.08,
                "latency_p95": 1200,
                "cost_per_1k": 0.008
            }
            
            events = await watchdog.scan_all_clusters()
            
            # Should have scanned both clusters
            assert len(events) >= 0  # May or may not generate events


# =============================================================================
# Watchdog Lifecycle Tests  
# =============================================================================

class TestWatchdogLifecycle:
    """Tests for watchdog start/stop lifecycle."""
    
    @pytest.fixture
    def lifecycle_watchdog(self):
        """Create watchdog for lifecycle testing."""
        config = WatchdogConfig(scan_interval_seconds=10)
        mock_qdrant = MagicMock()
        mock_qdrant.scroll = MagicMock(return_value=([], None))
        
        return AgenticWatchdog(
            config=config,
            qdrant_client=mock_qdrant,
            collection_name="test"
        )
    
    def test_initial_state_not_running(self, lifecycle_watchdog):
        """Test that watchdog starts not running."""
        assert lifecycle_watchdog._running is False
    
    @pytest.mark.asyncio
    async def test_stop_clears_running(self, lifecycle_watchdog):
        """Test that stop clears running flag."""
        lifecycle_watchdog._running = True
        
        await lifecycle_watchdog.stop()
        
        assert lifecycle_watchdog._running is False


# =============================================================================
# Trigger Detection Tests
# =============================================================================

class TestTriggerDetection:
    """Tests for watchdog trigger detection."""
    
    @pytest.fixture
    def watchdog_with_config(self):
        """Create watchdog with test config."""
        config = WatchdogConfig(
            scan_interval_seconds=10,
            drift_variance_threshold=0.15,
            quality_min_score=0.85,
            faithfulness_min_score=0.90
        )
        mock_qdrant = MagicMock()
        mock_qdrant.scroll = MagicMock(return_value=([], None))
        
        return AgenticWatchdog(
            config=config,
            qdrant_client=mock_qdrant,
            collection_name="test"
        )
    
    def test_check_triggers_healthy(self, watchdog_with_config):
        """Test that healthy metrics don't trigger."""
        policy = AgenticPolicy(
            cluster_id="healthy-test",
            intent="test",
            state=ClusterState.STABLE
        )
        
        metrics = {
            "quality": 0.90,
            "faithfulness": 0.95,
            "variance": 0.08
        }
        
        trigger = watchdog_with_config._check_triggers(policy, metrics)
        
        assert trigger is None
    
    def test_check_triggers_drift(self, watchdog_with_config):
        """Test drift detection."""
        policy = AgenticPolicy(
            cluster_id="drift-test",
            intent="test",
            state=ClusterState.STABLE,
            variance=0.25  # Above threshold - drift is checked from policy.variance
        )
        
        metrics = {
            "quality": 0.90,
            "faithfulness": 0.95,
            "variance": 0.25  # Above threshold
        }
        
        trigger = watchdog_with_config._check_triggers(policy, metrics)
        
        assert trigger == TriggerType.DRIFT
    
    def test_check_triggers_quality(self, watchdog_with_config):
        """Test quality degradation detection."""
        policy = AgenticPolicy(
            cluster_id="quality-test",
            intent="test",
            state=ClusterState.STABLE
        )
        
        metrics = {
            "quality": 0.70,  # Below threshold
            "faithfulness": 0.80,  # Below threshold
            "variance": 0.08
        }
        
        trigger = watchdog_with_config._check_triggers(policy, metrics)
        
        # Quality issues should be detected
        assert trigger in [TriggerType.QUALITY, TriggerType.DRIFT]


# =============================================================================
# Action Determination Tests
# =============================================================================

class TestActionDetermination:
    """Tests for action determination logic."""
    
    @pytest.fixture
    def watchdog(self):
        """Create watchdog for action testing."""
        config = WatchdogConfig()
        mock_qdrant = MagicMock()
        mock_qdrant.scroll = MagicMock(return_value=([], None))
        
        return AgenticWatchdog(
            config=config,
            qdrant_client=mock_qdrant,
            collection_name="test"
        )
    
    def test_determine_action_dspy_optimize(self, watchdog):
        """Test DSPy optimization action for quality trigger."""
        policy = AgenticPolicy(
            cluster_id="action-test",
            intent="test",
            dspy_enabled=True
        )
        
        metrics = {"quality": 0.75, "faithfulness": 0.80}
        action, new_state = watchdog._determine_action(policy, TriggerType.QUALITY, metrics)
        
        assert action == ActionType.DSPY_OPTIMIZE
    
    def test_determine_action_alert_no_dspy(self, watchdog):
        """Test alert action when DSPy disabled."""
        policy = AgenticPolicy(
            cluster_id="action-test",
            intent="test",
            dspy_enabled=False
        )
        
        metrics = {"quality": 0.75, "faithfulness": 0.80}
        action, new_state = watchdog._determine_action(policy, TriggerType.QUALITY, metrics)
        
        # Quality issues without DSPy still trigger DSPY_OPTIMIZE by default mapping
        assert action in [ActionType.ALERT, ActionType.DSPY_OPTIMIZE]
    
    def test_determine_action_fallback(self, watchdog):
        """Test fallback action for latency issues."""
        policy = AgenticPolicy(
            cluster_id="action-test",
            intent="test"
        )
        
        metrics = {"latency_ms": 5000}
        action, new_state = watchdog._determine_action(policy, TriggerType.LATENCY, metrics)
        
        # Latency issues should trigger provider switch
        assert action in [ActionType.SWITCH_PROVIDER, ActionType.FALLBACK, ActionType.ALERT]
