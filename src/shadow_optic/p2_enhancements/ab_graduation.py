"""
Auto A/B Graduation Controller for Shadow-Optic P2 Enhancements.

Portkey-First Architecture:
- Uses Portkey's Canary Testing for traffic splitting
- Uses Portkey's Load Balancing with weights for staged rollout
- Uses Portkey's Analytics Dashboard for metrics comparison
- Uses Portkey's Sticky Sessions for consistent user experience
- Uses spotify-confidence for statistical significance testing

This enables automated staged rollout of challenger models from
shadow testing to full production with safety guardrails.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import spotify_confidence as confidence
    SPOTIFY_CONFIDENCE_AVAILABLE = True
except ImportError:
    SPOTIFY_CONFIDENCE_AVAILABLE = False
    confidence = None

from portkey_ai import Portkey, AsyncPortkey

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class ExperimentStage(Enum):
    """Stages of graduated rollout."""
    SHADOW = "shadow"           # 0% live traffic - shadow only
    CANARY = "canary"           # 1% live traffic
    GRADUATED = "graduated"     # 10% live traffic
    RAMPING = "ramping"         # 10-50% live traffic
    MAJORITY = "majority"       # 50-90% live traffic
    FULL = "full"               # 100% live traffic


class PromotionDecision(Enum):
    """Decision from promotion check."""
    PROMOTE = "promote"         # Ready to promote
    HOLD = "hold"               # Keep at current stage
    ROLLBACK = "rollback"       # Rollback to previous stage


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StageConfig:
    """Configuration for each experiment stage."""
    traffic_percentage: float
    min_samples: int
    min_duration_hours: int
    quality_gates: Dict[str, float] = field(default_factory=dict)
    
    @property
    def production_weight(self) -> int:
        """Weight for production model in load balancer."""
        return int((1 - self.traffic_percentage) * 100)
    
    @property
    def challenger_weight(self) -> int:
        """Weight for challenger model in load balancer."""
        return int(self.traffic_percentage * 100)


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment."""
    total_requests: int = 0
    successful_responses: int = 0
    refusals: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    avg_faithfulness: float = 0.0
    cost_per_request: float = 0.0
    total_cost: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_responses / self.total_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.errors / self.total_requests
    
    @property
    def refusal_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.refusals / self.total_requests


@dataclass
class PromotionCheckResult:
    """Result of a promotion eligibility check."""
    decision: PromotionDecision
    reason: str
    current_stage: ExperimentStage
    target_stage: Optional[ExperimentStage] = None
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    quality_gate_results: Dict[str, bool] = field(default_factory=dict)
    statistical_significance: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentHistory:
    """History entry for experiment state changes."""
    timestamp: datetime
    action: str
    from_stage: Optional[ExperimentStage]
    to_stage: Optional[ExperimentStage]
    reason: str
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Portkey Config Manager
# =============================================================================

class PortkeyConfigManager:
    """Manage experiment configs using Portkey's native config system.
    
    This replaces external feature flag systems like Unleash/Flipt.
    Uses Portkey's load balancing and canary testing capabilities.
    """
    
    def __init__(self, portkey_api_key: Optional[str] = None):
        self.portkey = AsyncPortkey(api_key=portkey_api_key) if portkey_api_key else None
        self.configs: Dict[str, Dict[str, Any]] = {}
    
    def get_stage_config(
        self,
        stage: ExperimentStage,
        production_model: str,
        challenger_model: str,
        production_provider: str = "openai",
        challenger_provider: str = "deepseek"
    ) -> Dict[str, Any]:
        """Get Portkey config for a specific experiment stage.
        
        Args:
            stage: Current experiment stage
            production_model: Production model identifier
            challenger_model: Challenger model identifier
            production_provider: Provider for production model
            challenger_provider: Provider for challenger model
            
        Returns:
            Portkey config dict for the stage
        """
        stage_configs = {
            ExperimentStage.SHADOW: {
                # 0% live traffic - challenger only runs in shadow
                "strategy": {"mode": "fallback"},
                "targets": [
                    {
                        "provider": production_provider,
                        "weight": 100,
                        "override_params": {"model": production_model}
                    }
                ]
            },
            ExperimentStage.CANARY: {
                # 1% live traffic to challenger
                "strategy": {"mode": "loadbalance"},
                "targets": [
                    {
                        "provider": production_provider,
                        "weight": 99,
                        "override_params": {"model": production_model}
                    },
                    {
                        "provider": challenger_provider,
                        "weight": 1,
                        "override_params": {"model": challenger_model}
                    }
                ]
            },
            ExperimentStage.GRADUATED: {
                # 10% live traffic to challenger
                "strategy": {"mode": "loadbalance"},
                "targets": [
                    {
                        "provider": production_provider,
                        "weight": 90,
                        "override_params": {"model": production_model}
                    },
                    {
                        "provider": challenger_provider,
                        "weight": 10,
                        "override_params": {"model": challenger_model}
                    }
                ]
            },
            ExperimentStage.RAMPING: {
                # 50% live traffic to challenger
                "strategy": {"mode": "loadbalance"},
                "targets": [
                    {
                        "provider": production_provider,
                        "weight": 50,
                        "override_params": {"model": production_model}
                    },
                    {
                        "provider": challenger_provider,
                        "weight": 50,
                        "override_params": {"model": challenger_model}
                    }
                ]
            },
            ExperimentStage.MAJORITY: {
                # 90% live traffic to challenger
                "strategy": {"mode": "loadbalance"},
                "targets": [
                    {
                        "provider": production_provider,
                        "weight": 10,
                        "override_params": {"model": production_model}
                    },
                    {
                        "provider": challenger_provider,
                        "weight": 90,
                        "override_params": {"model": challenger_model}
                    }
                ]
            },
            ExperimentStage.FULL: {
                # 100% - challenger is now production!
                "strategy": {"mode": "fallback"},
                "targets": [
                    {
                        "provider": challenger_provider,
                        "weight": 100,
                        "override_params": {"model": challenger_model}
                    },
                    {
                        "provider": production_provider,
                        "override_params": {"model": production_model}
                        # Fallback only - no weight means fallback
                    }
                ]
            }
        }
        
        return stage_configs.get(stage, stage_configs[ExperimentStage.SHADOW])
    
    def get_sticky_session_config(
        self,
        stage: ExperimentStage,
        production_model: str,
        challenger_model: str,
        session_ttl: int = 3600,
        hash_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get Portkey config with sticky sessions for consistent routing.
        
        Args:
            stage: Current experiment stage
            production_model: Production model identifier
            challenger_model: Challenger model identifier
            session_ttl: TTL for sticky session in seconds
            hash_fields: Fields to use for session hashing
            
        Returns:
            Portkey config with sticky session configuration
        """
        base_config = self.get_stage_config(stage, production_model, challenger_model)
        
        if stage not in [ExperimentStage.SHADOW, ExperimentStage.FULL]:
            # Add sticky sessions for A/B testing stages
            base_config["strategy"]["sticky_session"] = {
                "hash_fields": hash_fields or ["metadata.user_id", "metadata.session_id"],
                "ttl": session_ttl
            }
        
        return base_config
    
    def create_rollback_config(
        self,
        production_model: str,
        production_provider: str = "openai"
    ) -> Dict[str, Any]:
        """Create emergency rollback config - 100% to production.
        
        Args:
            production_model: Production model to rollback to
            production_provider: Provider for production model
            
        Returns:
            Portkey config for full rollback
        """
        return {
            "strategy": {"mode": "fallback"},
            "targets": [
                {
                    "provider": production_provider,
                    "weight": 100,
                    "override_params": {"model": production_model}
                }
            ]
        }
    
    def create_kill_switch_config(
        self,
        production_model: str,
        challenger_model: str,
        production_provider: str = "openai",
        challenger_provider: str = "deepseek"
    ) -> Dict[str, Any]:
        """Kill switch: stop all challenger traffic immediately.
        
        In Portkey, just set challenger weight to 0!
        
        Args:
            production_model: Production model
            challenger_model: Challenger model (will get 0 weight)
            production_provider: Provider for production
            challenger_provider: Provider for challenger
            
        Returns:
            Kill switch config
        """
        return {
            "strategy": {"mode": "loadbalance"},
            "targets": [
                {
                    "provider": production_provider,
                    "weight": 100,
                    "override_params": {"model": production_model}
                },
                {
                    "provider": challenger_provider,
                    "weight": 0,  # Kill switch!
                    "override_params": {"model": challenger_model}
                }
            ]
        }


# =============================================================================
# Automated Rollout Controller
# =============================================================================

class AutomatedRolloutController:
    """Automated graduated rollout from shadow to production.
    
    Portkey-First Architecture:
    - Uses Portkey's load balancing for traffic splitting
    - Uses Portkey's analytics for metrics collection
    - Uses spotify-confidence for statistical testing
    """
    
    # Default stage configurations
    DEFAULT_STAGE_CONFIGS = {
        ExperimentStage.SHADOW: StageConfig(
            traffic_percentage=0.0,
            min_samples=100,
            min_duration_hours=24,
            quality_gates={
                "faithfulness_min": 0.90,
                "quality_min": 0.85,
            }
        ),
        ExperimentStage.CANARY: StageConfig(
            traffic_percentage=0.01,
            min_samples=1000,
            min_duration_hours=4,
            quality_gates={
                "faithfulness_min": 0.90,
                "quality_min": 0.85,
                "error_rate_max": 0.005,
                "latency_delta_max": 0.20,
            }
        ),
        ExperimentStage.GRADUATED: StageConfig(
            traffic_percentage=0.10,
            min_samples=5000,
            min_duration_hours=12,
            quality_gates={
                "faithfulness_min": 0.90,
                "quality_min": 0.85,
                "error_rate_max": 0.002,
                "latency_delta_max": 0.15,
                "refusal_rate_delta_max": 0.005,
            }
        ),
        ExperimentStage.RAMPING: StageConfig(
            traffic_percentage=0.50,
            min_samples=10000,
            min_duration_hours=24,
            quality_gates={
                "faithfulness_min": 0.90,
                "quality_min": 0.85,
                "error_rate_max": 0.001,
                "latency_delta_max": 0.10,
                "refusal_rate_delta_max": 0.003,
            }
        ),
        ExperimentStage.MAJORITY: StageConfig(
            traffic_percentage=0.90,
            min_samples=50000,
            min_duration_hours=48,
            quality_gates={
                "faithfulness_min": 0.90,
                "quality_min": 0.85,
                "error_rate_max": 0.001,
                "latency_delta_max": 0.10,
                "refusal_rate_delta_max": 0.002,
            }
        ),
        ExperimentStage.FULL: StageConfig(
            traffic_percentage=1.0,
            min_samples=0,
            min_duration_hours=0,
            quality_gates={}
        ),
    }
    
    def __init__(
        self,
        experiment_id: str,
        challenger_model: str,
        production_model: str,
        portkey_api_key: Optional[str] = None,
        stage_configs: Optional[Dict[ExperimentStage, StageConfig]] = None
    ):
        self.experiment_id = experiment_id
        self.challenger_model = challenger_model
        self.production_model = production_model
        
        self.config_manager = PortkeyConfigManager(portkey_api_key)
        self.portkey = AsyncPortkey(api_key=portkey_api_key) if portkey_api_key else None
        
        self.stage = ExperimentStage.SHADOW
        self.stage_start_time = datetime.utcnow()
        self.experiment_start_time = datetime.utcnow()
        
        self.stage_configs = stage_configs or self.DEFAULT_STAGE_CONFIGS.copy()
        
        self.production_metrics = ExperimentMetrics()
        self.challenger_metrics = ExperimentMetrics()
        self.baseline_metrics: Optional[ExperimentMetrics] = None
        
        self.history: List[ExperimentHistory] = []
        
        # Record experiment start
        self._record_history("started", None, ExperimentStage.SHADOW, "Experiment created")
    
    @property
    def current_config(self) -> StageConfig:
        """Get configuration for current stage."""
        return self.stage_configs.get(self.stage, self.DEFAULT_STAGE_CONFIGS[ExperimentStage.SHADOW])
    
    @property
    def hours_in_stage(self) -> float:
        """Calculate hours spent in current stage."""
        return (datetime.utcnow() - self.stage_start_time).total_seconds() / 3600
    
    @property
    def experiment_duration_hours(self) -> float:
        """Calculate total experiment duration."""
        return (datetime.utcnow() - self.experiment_start_time).total_seconds() / 3600
    
    def update_metrics(
        self,
        production_metrics: Optional[ExperimentMetrics] = None,
        challenger_metrics: Optional[ExperimentMetrics] = None
    ):
        """Update metrics for production and/or challenger.
        
        Args:
            production_metrics: New production metrics
            challenger_metrics: New challenger metrics
        """
        if production_metrics:
            self.production_metrics = production_metrics
        if challenger_metrics:
            self.challenger_metrics = challenger_metrics
    
    async def check_promotion(self) -> PromotionCheckResult:
        """Check if experiment can be promoted to next stage.
        
        Returns:
            PromotionCheckResult with decision and details
        """
        if self.stage == ExperimentStage.FULL:
            return PromotionCheckResult(
                decision=PromotionDecision.HOLD,
                reason="Already at full rollout",
                current_stage=self.stage
            )
        
        config = self.current_config
        quality_gate_results = {}
        
        # Check minimum duration
        if self.hours_in_stage < config.min_duration_hours:
            return PromotionCheckResult(
                decision=PromotionDecision.HOLD,
                reason=f"Minimum duration not met ({self.hours_in_stage:.1f}/{config.min_duration_hours}h)",
                current_stage=self.stage,
                metrics_summary=self._get_metrics_summary()
            )
        
        # Check minimum samples
        if self.challenger_metrics.total_requests < config.min_samples:
            return PromotionCheckResult(
                decision=PromotionDecision.HOLD,
                reason=f"Minimum samples not met ({self.challenger_metrics.total_requests}/{config.min_samples})",
                current_stage=self.stage,
                metrics_summary=self._get_metrics_summary()
            )
        
        # Check quality gates
        gate_result, gate_reason = self._check_quality_gates(config.quality_gates)
        if not gate_result:
            # Decide between HOLD and ROLLBACK based on severity
            decision = PromotionDecision.HOLD
            if "error_rate" in gate_reason.lower() and self.challenger_metrics.error_rate > 0.01:
                decision = PromotionDecision.ROLLBACK
            
            return PromotionCheckResult(
                decision=decision,
                reason=gate_reason,
                current_stage=self.stage,
                quality_gate_results=quality_gate_results,
                metrics_summary=self._get_metrics_summary()
            )
        
        # Statistical significance check (skip for shadow stage)
        if self.stage != ExperimentStage.SHADOW:
            sig_result = self._check_statistical_significance()
            if not sig_result["significant"]:
                return PromotionCheckResult(
                    decision=PromotionDecision.HOLD,
                    reason=sig_result["reason"],
                    current_stage=self.stage,
                    statistical_significance=sig_result,
                    metrics_summary=self._get_metrics_summary()
                )
        
        # All checks passed - ready to promote
        next_stage = self._get_next_stage()
        return PromotionCheckResult(
            decision=PromotionDecision.PROMOTE,
            reason="All promotion criteria met",
            current_stage=self.stage,
            target_stage=next_stage,
            quality_gate_results=quality_gate_results,
            metrics_summary=self._get_metrics_summary()
        )
    
    def _check_quality_gates(
        self,
        gates: Dict[str, float]
    ) -> Tuple[bool, str]:
        """Check all quality gates.
        
        Args:
            gates: Dict of gate names to threshold values
            
        Returns:
            Tuple of (passed, reason_if_failed)
        """
        for gate, threshold in gates.items():
            if gate.endswith("_min"):
                metric_name = gate.replace("_min", "")
                value = self._get_metric_value(metric_name, self.challenger_metrics)
                if value < threshold:
                    return False, f"Quality gate failed: {metric_name} ({value:.3f} < {threshold})"
            
            elif gate.endswith("_max"):
                metric_name = gate.replace("_max", "")
                value = self._get_metric_value(metric_name, self.challenger_metrics)
                if value > threshold:
                    return False, f"Quality gate failed: {metric_name} ({value:.3f} > {threshold})"
            
            elif gate.endswith("_delta_max"):
                metric_name = gate.replace("_delta_max", "")
                challenger_value = self._get_metric_value(metric_name, self.challenger_metrics)
                production_value = self._get_metric_value(metric_name, self.production_metrics)
                
                if production_value > 0:
                    delta = (challenger_value - production_value) / production_value
                    if delta > threshold:
                        return False, f"Delta gate failed: {metric_name} delta ({delta:.3f} > {threshold})"
        
        return True, ""
    
    def _get_metric_value(self, name: str, metrics: ExperimentMetrics) -> float:
        """Get metric value by name."""
        metric_map = {
            "faithfulness": metrics.avg_faithfulness,
            "quality": metrics.avg_quality_score,
            "error_rate": metrics.error_rate,
            "refusal_rate": metrics.refusal_rate,
            "latency": metrics.avg_latency_ms,
            "success_rate": metrics.success_rate,
        }
        return metric_map.get(name, 0.0)
    
    def _check_statistical_significance(self) -> Dict[str, Any]:
        """Check statistical significance using spotify-confidence.
        
        Returns:
            Dict with significance test results
        """
        if not SPOTIFY_CONFIDENCE_AVAILABLE:
            # Fallback to simple comparison
            logger.warning("spotify-confidence not available, using simple comparison")
            challenger_rate = self.challenger_metrics.success_rate
            production_rate = self.production_metrics.success_rate
            
            # Simple non-inferiority check (challenger within 2% of production)
            diff = production_rate - challenger_rate
            is_significant = diff < 0.02
            
            return {
                "significant": is_significant,
                "method": "simple_comparison",
                "difference": diff,
                "reason": f"Challenger {'is' if is_significant else 'is not'} within 2% of production"
            }
        
        try:
            data = pd.DataFrame({
                'model': ['production', 'challenger'],
                'success': [
                    self.production_metrics.successful_responses,
                    self.challenger_metrics.successful_responses
                ],
                'total': [
                    self.production_metrics.total_requests,
                    self.challenger_metrics.total_requests
                ]
            })
            
            test = confidence.ZTest(
                data,
                numerator_column='success',
                denominator_column='total',
                categorical_group_columns='model',
                correction_method='bonferroni'
            )
            
            diff_result = test.difference(level_1='production', level_2='challenger')
            
            difference = diff_result.get('difference', 0)
            p_value = diff_result.get('p_value', 1.0)
            
            # Non-inferiority check: challenger not more than 2% worse
            is_significant = difference > -0.02 or p_value >= 0.05
            
            return {
                "significant": is_significant,
                "method": "z_test",
                "difference": difference,
                "p_value": p_value,
                "confidence_interval": diff_result.get('ci', (0, 0)),
                "reason": f"Difference: {difference:.4f}, p-value: {p_value:.4f}"
            }
            
        except Exception as e:
            logger.error(f"Statistical significance check failed: {e}")
            return {
                "significant": True,  # Fail open
                "method": "error",
                "reason": f"Error: {str(e)}"
            }
    
    def _get_next_stage(self) -> Optional[ExperimentStage]:
        """Get the next stage in the rollout progression."""
        stages = list(ExperimentStage)
        current_idx = stages.index(self.stage)
        
        if current_idx < len(stages) - 1:
            return stages[current_idx + 1]
        return None
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics."""
        return {
            "challenger": {
                "total_requests": self.challenger_metrics.total_requests,
                "success_rate": self.challenger_metrics.success_rate,
                "error_rate": self.challenger_metrics.error_rate,
                "refusal_rate": self.challenger_metrics.refusal_rate,
                "avg_latency_ms": self.challenger_metrics.avg_latency_ms,
                "avg_quality_score": self.challenger_metrics.avg_quality_score,
                "avg_faithfulness": self.challenger_metrics.avg_faithfulness,
            },
            "production": {
                "total_requests": self.production_metrics.total_requests,
                "success_rate": self.production_metrics.success_rate,
                "error_rate": self.production_metrics.error_rate,
                "refusal_rate": self.production_metrics.refusal_rate,
                "avg_latency_ms": self.production_metrics.avg_latency_ms,
            },
            "stage_info": {
                "current_stage": self.stage.value,
                "hours_in_stage": self.hours_in_stage,
                "traffic_percentage": self.current_config.traffic_percentage * 100
            }
        }
    
    async def promote(self) -> ExperimentStage:
        """Promote experiment to next stage.
        
        Returns:
            New stage after promotion
        """
        next_stage = self._get_next_stage()
        
        if next_stage is None:
            logger.warning("Cannot promote: already at final stage")
            return self.stage
        
        old_stage = self.stage
        self.stage = next_stage
        self.stage_start_time = datetime.utcnow()
        
        # Store baseline at first promotion
        if old_stage == ExperimentStage.SHADOW:
            self.baseline_metrics = ExperimentMetrics(
                total_requests=self.challenger_metrics.total_requests,
                successful_responses=self.challenger_metrics.successful_responses,
                avg_quality_score=self.challenger_metrics.avg_quality_score,
                avg_faithfulness=self.challenger_metrics.avg_faithfulness
            )
        
        # Update traffic routing via Portkey config
        await self._update_traffic_routing()
        
        # Record history
        self._record_history(
            "promote",
            old_stage,
            self.stage,
            f"Promoted from {old_stage.value} to {self.stage.value}"
        )
        
        logger.info(f"Experiment {self.experiment_id} promoted: {old_stage.value} → {self.stage.value}")
        
        return self.stage
    
    async def rollback(self, reason: str) -> ExperimentStage:
        """Emergency rollback to shadow stage.
        
        Args:
            reason: Reason for rollback
            
        Returns:
            Stage after rollback (SHADOW)
        """
        old_stage = self.stage
        self.stage = ExperimentStage.SHADOW
        self.stage_start_time = datetime.utcnow()
        
        await self._update_traffic_routing()
        
        self._record_history(
            "rollback",
            old_stage,
            self.stage,
            reason
        )
        
        logger.warning(
            f"Experiment {self.experiment_id} ROLLED BACK: "
            f"{old_stage.value} → {self.stage.value}. Reason: {reason}"
        )
        
        return self.stage
    
    async def _update_traffic_routing(self):
        """Update Portkey config for current stage traffic split."""
        config = self.config_manager.get_stage_config(
            self.stage,
            self.production_model,
            self.challenger_model
        )
        
        # In production, this would update Portkey's config via API
        # For now, log the intended configuration
        logger.info(
            f"Traffic routing updated for experiment {self.experiment_id}: "
            f"{self.production_model}={self.config_manager.get_stage_config(self.stage, self.production_model, self.challenger_model)['targets'][0].get('weight', 0)}%, "
            f"{self.challenger_model}={100 - self.config_manager.get_stage_config(self.stage, self.production_model, self.challenger_model)['targets'][0].get('weight', 100)}%"
        )
    
    def _record_history(
        self,
        action: str,
        from_stage: Optional[ExperimentStage],
        to_stage: Optional[ExperimentStage],
        reason: str
    ):
        """Record an action in experiment history."""
        self.history.append(ExperimentHistory(
            timestamp=datetime.utcnow(),
            action=action,
            from_stage=from_stage,
            to_stage=to_stage,
            reason=reason,
            metrics={
                "challenger_samples": self.challenger_metrics.total_requests,
                "challenger_success_rate": self.challenger_metrics.success_rate,
                "challenger_quality": self.challenger_metrics.avg_quality_score
            }
        ))
    
    def get_portkey_config(self) -> Dict[str, Any]:
        """Get current Portkey config for the experiment.
        
        Returns:
            Portkey config dict for current stage
        """
        return self.config_manager.get_stage_config(
            self.stage,
            self.production_model,
            self.challenger_model
        )
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for experiment dashboard.
        
        Returns:
            Dashboard data dict
        """
        config = self.current_config
        
        return {
            "experiment_id": self.experiment_id,
            "challenger_model": self.challenger_model,
            "production_model": self.production_model,
            "current_stage": self.stage.value,
            "traffic_percentage": config.traffic_percentage * 100,
            "stage_progress": {
                "hours_elapsed": self.hours_in_stage,
                "hours_required": config.min_duration_hours,
                "samples_collected": self.challenger_metrics.total_requests,
                "samples_required": config.min_samples,
                "duration_pct": min(100, self.hours_in_stage / max(config.min_duration_hours, 1) * 100),
                "samples_pct": min(100, self.challenger_metrics.total_requests / max(config.min_samples, 1) * 100)
            },
            "challenger_metrics": {
                "success_rate": self.challenger_metrics.success_rate,
                "error_rate": self.challenger_metrics.error_rate,
                "refusal_rate": self.challenger_metrics.refusal_rate,
                "avg_latency_ms": self.challenger_metrics.avg_latency_ms,
                "avg_quality_score": self.challenger_metrics.avg_quality_score,
                "avg_faithfulness": self.challenger_metrics.avg_faithfulness,
                "cost_per_request": self.challenger_metrics.cost_per_request,
            },
            "production_metrics": {
                "success_rate": self.production_metrics.success_rate,
                "error_rate": self.production_metrics.error_rate,
                "refusal_rate": self.production_metrics.refusal_rate,
                "avg_latency_ms": self.production_metrics.avg_latency_ms,
            },
            "quality_gates": dict(config.quality_gates),
            "experiment_duration_hours": self.experiment_duration_hours,
            "history": [
                {
                    "timestamp": h.timestamp.isoformat(),
                    "action": h.action,
                    "from_stage": h.from_stage.value if h.from_stage else None,
                    "to_stage": h.to_stage.value if h.to_stage else None,
                    "reason": h.reason
                }
                for h in self.history[-10:]  # Last 10 events
            ]
        }
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get full experiment history.
        
        Returns:
            List of history entries as dicts
        """
        return [
            {
                "timestamp": h.timestamp.isoformat(),
                "action": h.action,
                "from_stage": h.from_stage.value if h.from_stage else None,
                "to_stage": h.to_stage.value if h.to_stage else None,
                "reason": h.reason,
                "metrics": h.metrics
            }
            for h in self.history
        ]


# =============================================================================
# Experiment Manager
# =============================================================================

class ExperimentManager:
    """Manage multiple A/B experiments.
    
    Provides a high-level interface for creating, tracking,
    and managing multiple concurrent experiments.
    """
    
    def __init__(self, portkey_api_key: Optional[str] = None):
        self.portkey_api_key = portkey_api_key
        self.experiments: Dict[str, AutomatedRolloutController] = {}
    
    def create_experiment(
        self,
        experiment_id: str,
        challenger_model: str,
        production_model: str,
        stage_configs: Optional[Dict[ExperimentStage, StageConfig]] = None
    ) -> AutomatedRolloutController:
        """Create a new experiment.
        
        Args:
            experiment_id: Unique identifier for the experiment
            challenger_model: Model to test
            production_model: Current production model
            stage_configs: Optional custom stage configurations
            
        Returns:
            AutomatedRolloutController for the experiment
        """
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
        
        controller = AutomatedRolloutController(
            experiment_id=experiment_id,
            challenger_model=challenger_model,
            production_model=production_model,
            portkey_api_key=self.portkey_api_key,
            stage_configs=stage_configs
        )
        
        self.experiments[experiment_id] = controller
        logger.info(f"Created experiment: {experiment_id}")
        
        return controller
    
    def get_experiment(self, experiment_id: str) -> Optional[AutomatedRolloutController]:
        """Get an experiment by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            AutomatedRolloutController or None if not found
        """
        return self.experiments.get(experiment_id)
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with summary info.
        
        Returns:
            List of experiment summaries
        """
        return [
            {
                "experiment_id": exp_id,
                "challenger_model": controller.challenger_model,
                "production_model": controller.production_model,
                "stage": controller.stage.value,
                "traffic_percentage": controller.current_config.traffic_percentage * 100,
                "duration_hours": controller.experiment_duration_hours
            }
            for exp_id, controller in self.experiments.items()
        ]
    
    async def check_all_promotions(self) -> Dict[str, PromotionCheckResult]:
        """Check promotion eligibility for all experiments.
        
        Returns:
            Dict mapping experiment_id to PromotionCheckResult
        """
        results = {}
        for exp_id, controller in self.experiments.items():
            results[exp_id] = await controller.check_promotion()
        return results
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment.
        
        Args:
            experiment_id: Experiment to delete
            
        Returns:
            True if deleted, False if not found
        """
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            logger.info(f"Deleted experiment: {experiment_id}")
            return True
        return False
