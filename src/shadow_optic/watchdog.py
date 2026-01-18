"""
Agentic Watchdog - Background worker for autonomous prompt governance.

Implements the "Watchdog Loop" that continuously monitors Qdrant clusters
for triggers and initiates appropriate actions:

1. Drift Detection - Scans for semantic/statistical drift
2. Cost Monitoring - Tracks cost against targets
3. Quality Enforcement - Ensures evaluation scores meet thresholds
4. Latency Tracking - Monitors response times against SLAs

When triggers are detected, the Watchdog:
- Updates cluster state (STABLE -> DRIFTING -> UNSTABLE)
- Invokes DSPy optimization for failing clusters
- Applies routing rules for immediate mitigation
- Records evolution history for audit

Integration Points:
- Arize Phoenix: Drift signals feed into state updates
- Evidently AI: Statistical drift metrics
- Shadow-Optic Evaluator: Quality scores
- DSPy Optimizer: Prompt rewriting for challenger models
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from shadow_optic.agentic_policy import (
    ActionType,
    AgenticPolicy,
    ClusterState,
    TriggerType,
    policy_from_qdrant_payload,
    policy_to_qdrant_payload,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Watchdog Configuration
# =============================================================================

class WatchdogConfig(BaseModel):
    """Configuration for the AgenticWatchdog."""
    
    # Scan intervals
    scan_interval_seconds: int = Field(
        default=60,
        ge=10,
        description="Interval between cluster scans"
    )
    
    # Drift thresholds
    drift_variance_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Variance threshold for drift detection"
    )
    drift_window_hours: int = Field(
        default=24,
        ge=1,
        description="Time window for drift calculation"
    )
    
    # Quality thresholds
    quality_min_score: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable quality score"
    )
    faithfulness_min_score: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable faithfulness score"
    )
    
    # Cost thresholds
    cost_alert_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        description="Alert when cost exceeds target by this multiplier"
    )
    
    # Latency thresholds
    latency_p95_threshold_ms: int = Field(
        default=2000,
        ge=100,
        description="P95 latency threshold in milliseconds"
    )
    
    # Action configuration
    auto_optimize: bool = Field(
        default=False,
        description="Automatically trigger DSPy optimization"
    )
    auto_fallback: bool = Field(
        default=True,
        description="Automatically apply fallback routing"
    )
    
    # Stability requirements
    min_samples_for_stable: int = Field(
        default=100,
        ge=10,
        description="Minimum samples before a cluster can be STABLE"
    )
    stable_window_hours: int = Field(
        default=48,
        ge=1,
        description="Hours of stability required before STABLE state"
    )


# =============================================================================
# Watchdog Events
# =============================================================================

@dataclass
class WatchdogEvent:
    """Event emitted by the watchdog."""
    timestamp: datetime
    cluster_id: str
    trigger_type: TriggerType
    old_state: ClusterState
    new_state: ClusterState
    action_taken: Optional[ActionType]
    metrics: Dict[str, float]
    message: str


class WatchdogEventHandler:
    """Base class for handling watchdog events."""
    
    async def on_event(self, event: WatchdogEvent) -> None:
        """Handle a watchdog event."""
        logger.info(
            f"[{event.trigger_type.value}] Cluster {event.cluster_id}: "
            f"{event.old_state.value} -> {event.new_state.value} | {event.message}"
        )


class AlertingEventHandler(WatchdogEventHandler):
    """Event handler that sends alerts."""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
    
    async def on_event(self, event: WatchdogEvent) -> None:
        await super().on_event(event)
        
        # In production, send to Slack/PagerDuty/etc.
        if event.new_state == ClusterState.UNSTABLE:
            logger.warning(
                f"ALERT: Cluster {event.cluster_id} is UNSTABLE. "
                f"Action required: {event.action_taken}"
            )


# =============================================================================
# Metric Collectors
# =============================================================================

class MetricCollector:
    """Collects metrics for a cluster from various sources."""
    
    def __init__(
        self,
        phoenix_observer: Optional[Any] = None,
        evaluator: Optional[Any] = None,
    ):
        self._phoenix = phoenix_observer
        self._evaluator = evaluator
    
    async def collect_metrics(
        self,
        cluster_id: str,
        window_hours: int = 24
    ) -> Dict[str, float]:
        """Collect current metrics for a cluster.
        
        Returns:
            Dict with keys: quality, faithfulness, latency_ms, cost_per_1k, variance
        """
        metrics = {
            "quality": 1.0,
            "faithfulness": 1.0,
            "latency_ms": 100.0,
            "cost_per_1k": 0.001,
            "variance": 0.0,
            "sample_count": 0,
        }
        
        # In production, query Phoenix/Qdrant for actual metrics
        # For now, return placeholder metrics
        
        try:
            if self._phoenix:
                # Query Phoenix for traces in this cluster
                # traces = await self._phoenix.get_cluster_traces(cluster_id, window_hours)
                # metrics = self._calculate_metrics(traces)
                pass
        except Exception as e:
            logger.warning(f"Failed to collect metrics for {cluster_id}: {e}")
        
        return metrics
    
    def calculate_variance(
        self,
        embeddings: List[List[float]]
    ) -> float:
        """Calculate variance of embeddings in a cluster.
        
        Higher variance indicates potential drift.
        """
        if not embeddings or len(embeddings) < 2:
            return 0.0
        
        try:
            import numpy as np
            arr = np.array(embeddings)
            # Use mean pairwise distance as variance proxy
            centroid = arr.mean(axis=0)
            distances = np.linalg.norm(arr - centroid, axis=1)
            return float(np.std(distances))
        except ImportError:
            return 0.0


# =============================================================================
# Agentic Watchdog
# =============================================================================

class AgenticWatchdog:
    """
    Background worker for autonomous prompt governance.
    
    The Watchdog continuously scans Qdrant clusters and:
    1. Detects drift, cost overruns, quality degradation
    2. Updates cluster state machine
    3. Triggers appropriate actions (DSPy, fallback, alert)
    4. Records evolution history
    
    Example:
        ```python
        watchdog = AgenticWatchdog(
            qdrant_client=qdrant,
            config=WatchdogConfig(auto_optimize=True),
        )
        
        # Register handlers
        watchdog.add_event_handler(AlertingEventHandler())
        
        # Start monitoring
        await watchdog.start()
        ```
    """
    
    def __init__(
        self,
        qdrant_client: Any,
        config: Optional[WatchdogConfig] = None,
        collection_name: str = "prompt_clusters",
        dspy_optimizer: Optional[Any] = None,
    ):
        self.qdrant = qdrant_client
        self.config = config or WatchdogConfig()
        self.collection_name = collection_name
        self.dspy_optimizer = dspy_optimizer
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._event_handlers: List[WatchdogEventHandler] = []
        self._metric_collector = MetricCollector()
        
        # Cache of policies to detect changes
        self._policy_cache: Dict[str, AgenticPolicy] = {}
    
    def add_event_handler(self, handler: WatchdogEventHandler) -> None:
        """Register an event handler."""
        self._event_handlers.append(handler)
    
    async def start(self) -> None:
        """Start the watchdog loop."""
        if self._running:
            logger.warning("Watchdog already running")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"AgenticWatchdog started "
            f"(interval: {self.config.scan_interval_seconds}s)"
        )
    
    async def stop(self) -> None:
        """Stop the watchdog loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("AgenticWatchdog stopped")
    
    async def _run_loop(self) -> None:
        """Main watchdog loop."""
        while self._running:
            try:
                await self.scan_all_clusters()
            except Exception as e:
                logger.error(f"Watchdog scan failed: {e}")
            
            await asyncio.sleep(self.config.scan_interval_seconds)
    
    async def scan_all_clusters(self) -> List[WatchdogEvent]:
        """Scan all clusters and process triggers.
        
        Returns:
            List of events generated during the scan
        """
        events = []
        
        try:
            # Scroll through all clusters
            clusters = await self._get_all_clusters()
            
            logger.debug(f"Scanning {len(clusters)} clusters")
            
            for cluster_id, payload in clusters:
                try:
                    event = await self._process_cluster(cluster_id, payload)
                    if event:
                        events.append(event)
                        await self._emit_event(event)
                except Exception as e:
                    logger.error(f"Failed to process cluster {cluster_id}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to scan clusters: {e}")
        
        return events
    
    async def _get_all_clusters(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all cluster IDs and their payloads from Qdrant."""
        clusters = []
        
        try:
            # Scroll through collection
            from qdrant_client.models import ScrollRequest
            
            offset = None
            while True:
                result = await self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                
                points, next_offset = result
                
                for point in points:
                    cluster_id = point.payload.get("cluster_id", str(point.id))
                    clusters.append((cluster_id, point.payload))
                
                if next_offset is None:
                    break
                offset = next_offset
                
        except Exception as e:
            logger.error(f"Failed to scroll clusters: {e}")
        
        return clusters
    
    async def _process_cluster(
        self,
        cluster_id: str,
        payload: Dict[str, Any]
    ) -> Optional[WatchdogEvent]:
        """Process a single cluster and check for triggers.
        
        Returns:
            WatchdogEvent if action was taken, None otherwise
        """
        # Load or create policy
        policy = self._get_or_create_policy(cluster_id, payload)
        old_state = policy.state
        
        # Collect current metrics
        metrics = await self._metric_collector.collect_metrics(
            cluster_id,
            window_hours=self.config.drift_window_hours
        )
        policy.current_metrics = metrics
        policy.variance = metrics.get("variance", 0.0)
        
        # Check for triggers
        trigger = self._check_triggers(policy, metrics)
        
        if trigger is None:
            # No trigger - check if we should stabilize
            await self._check_stabilization(policy, metrics)
            return None
        
        # Determine action
        action, new_state = self._determine_action(policy, trigger, metrics)
        
        # Apply state transition
        if new_state != old_state:
            policy.state = new_state
        
        # Execute action
        await self._execute_action(policy, action, trigger, metrics)
        
        # Update policy in Qdrant
        await self._update_policy(cluster_id, policy)
        
        # Create event
        event = WatchdogEvent(
            timestamp=datetime.now(timezone.utc),
            cluster_id=cluster_id,
            trigger_type=trigger,
            old_state=old_state,
            new_state=new_state,
            action_taken=action,
            metrics=metrics,
            message=self._format_event_message(trigger, metrics, action)
        )
        
        return event
    
    def _get_or_create_policy(
        self,
        cluster_id: str,
        payload: Dict[str, Any]
    ) -> AgenticPolicy:
        """Get cached policy or create from payload."""
        if cluster_id in self._policy_cache:
            return self._policy_cache[cluster_id]
        
        try:
            policy = policy_from_qdrant_payload(payload)
        except Exception:
            from shadow_optic.agentic_policy import create_default_policy
            policy = create_default_policy(
                cluster_id=cluster_id,
                intent=payload.get("intent", "general"),
                sensitivity=payload.get("sensitivity", "internal")
            )
        
        self._policy_cache[cluster_id] = policy
        return policy
    
    def _check_triggers(
        self,
        policy: AgenticPolicy,
        metrics: Dict[str, float]
    ) -> Optional[TriggerType]:
        """Check all trigger conditions."""
        
        # 1. Drift check
        if policy.variance > self.config.drift_variance_threshold:
            return TriggerType.DRIFT
        
        # 2. Quality check
        quality = metrics.get("quality", 1.0)
        if quality < self.config.quality_min_score:
            return TriggerType.QUALITY
        
        faithfulness = metrics.get("faithfulness", 1.0)
        if faithfulness < self.config.faithfulness_min_score:
            return TriggerType.QUALITY
        
        # 3. Cost check
        cost = metrics.get("cost_per_1k", 0.0)
        target_cost = policy.optimization_target.max_cost_per_1k
        if cost > target_cost * self.config.cost_alert_multiplier:
            return TriggerType.COST
        
        # 4. Latency check
        latency = metrics.get("latency_ms", 0)
        if latency > self.config.latency_p95_threshold_ms:
            return TriggerType.LATENCY
        
        return None
    
    def _determine_action(
        self,
        policy: AgenticPolicy,
        trigger: TriggerType,
        metrics: Dict[str, float]
    ) -> Tuple[ActionType, ClusterState]:
        """Determine action and new state based on trigger."""
        
        # Check routing rules first
        rule = policy.get_applicable_rule(metrics)
        if rule:
            action_str = rule.action.lower()
            if "fallback" in action_str:
                return ActionType.FALLBACK, ClusterState.DRIFTING
            elif "switch" in action_str:
                return ActionType.SWITCH_PROVIDER, ClusterState.DRIFTING
        
        # Default action based on trigger type
        action_map = {
            TriggerType.DRIFT: (ActionType.ALERT, ClusterState.DRIFTING),
            TriggerType.QUALITY: (ActionType.DSPY_OPTIMIZE, ClusterState.UNSTABLE),
            TriggerType.COST: (ActionType.ALERT, ClusterState.DRIFTING),
            TriggerType.LATENCY: (ActionType.SWITCH_PROVIDER, ClusterState.DRIFTING),
        }
        
        action, new_state = action_map.get(
            trigger, 
            (ActionType.ALERT, ClusterState.DRIFTING)
        )
        
        # Override to DSPy if configured
        if (
            self.config.auto_optimize and 
            trigger in {TriggerType.QUALITY, TriggerType.DRIFT} and
            policy.dspy_enabled
        ):
            action = ActionType.DSPY_OPTIMIZE
        
        return action, new_state
    
    async def _execute_action(
        self,
        policy: AgenticPolicy,
        action: ActionType,
        trigger: TriggerType,
        metrics: Dict[str, float]
    ) -> None:
        """Execute the determined action."""
        
        if action == ActionType.DSPY_OPTIMIZE:
            if self.dspy_optimizer:
                try:
                    await self.dspy_optimizer.optimize_cluster(
                        cluster_id=policy.cluster_id,
                        policy=policy,
                        metrics=metrics
                    )
                    policy.add_evolution_record(
                        action=action,
                        change_description="DSPy optimization triggered",
                        triggered_by=trigger
                    )
                except Exception as e:
                    logger.error(f"DSPy optimization failed: {e}")
        
        elif action == ActionType.FALLBACK:
            if self.config.auto_fallback and policy.fallback_model:
                policy.add_evolution_record(
                    action=action,
                    change_description=f"Fallback to {policy.fallback_model}",
                    triggered_by=trigger
                )
        
        elif action == ActionType.ALERT:
            policy.add_evolution_record(
                action=action,
                change_description=f"Alert triggered: {trigger.value}",
                triggered_by=trigger
            )
    
    async def _check_stabilization(
        self,
        policy: AgenticPolicy,
        metrics: Dict[str, float]
    ) -> None:
        """Check if a drifting cluster can be stabilized."""
        if policy.state != ClusterState.DRIFTING:
            return
        
        # Check if metrics are healthy
        quality = metrics.get("quality", 0)
        faithfulness = metrics.get("faithfulness", 0)
        sample_count = metrics.get("sample_count", 0)
        
        if (
            quality >= self.config.quality_min_score and
            faithfulness >= self.config.faithfulness_min_score and
            sample_count >= self.config.min_samples_for_stable and
            policy.variance < self.config.drift_variance_threshold
        ):
            policy.state = ClusterState.STABLE
            policy.add_evolution_record(
                action=ActionType.ALERT,
                change_description="Cluster stabilized after recovery"
            )
    
    async def _update_policy(
        self,
        cluster_id: str,
        policy: AgenticPolicy
    ) -> None:
        """Update policy in Qdrant."""
        try:
            from qdrant_client.models import PointStruct
            
            payload = policy_to_qdrant_payload(policy)
            
            # Update the point's payload
            await self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[cluster_id],  # Assuming cluster_id is used as point ID
            )
            
            # Update cache
            self._policy_cache[cluster_id] = policy
            
        except Exception as e:
            logger.error(f"Failed to update policy for {cluster_id}: {e}")
    
    async def _emit_event(self, event: WatchdogEvent) -> None:
        """Emit event to all handlers."""
        for handler in self._event_handlers:
            try:
                await handler.on_event(event)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")
    
    def _format_event_message(
        self,
        trigger: TriggerType,
        metrics: Dict[str, float],
        action: ActionType
    ) -> str:
        """Format a human-readable event message."""
        if trigger == TriggerType.DRIFT:
            return f"Drift detected (variance: {metrics.get('variance', 0):.3f})"
        elif trigger == TriggerType.QUALITY:
            return (
                f"Quality degradation "
                f"(quality: {metrics.get('quality', 0):.2f}, "
                f"faithfulness: {metrics.get('faithfulness', 0):.2f})"
            )
        elif trigger == TriggerType.COST:
            return f"Cost overrun (${metrics.get('cost_per_1k', 0):.4f}/1k tokens)"
        elif trigger == TriggerType.LATENCY:
            return f"Latency spike ({metrics.get('latency_ms', 0):.0f}ms)"
        else:
            return f"Trigger: {trigger.value}, Action: {action.value}"


# =============================================================================
# Factory Functions
# =============================================================================

async def create_watchdog(
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "prompt_clusters",
    config: Optional[WatchdogConfig] = None,
) -> AgenticWatchdog:
    """Create and initialize an AgenticWatchdog.
    
    Args:
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        collection_name: Qdrant collection to monitor
        config: Watchdog configuration
        
    Returns:
        Configured AgenticWatchdog instance
    """
    from qdrant_client import AsyncQdrantClient
    
    qdrant = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
    
    watchdog = AgenticWatchdog(
        qdrant_client=qdrant,
        config=config or WatchdogConfig(),
        collection_name=collection_name,
    )
    
    # Add default handlers
    watchdog.add_event_handler(AlertingEventHandler())
    
    return watchdog
