"""
Quality Trend Detection and Analytics.

Implements statistical process control for model quality monitoring.
Detects degradation, anomalies, and drifts in model performance over time.
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np
from scipy import stats

from shadow_optic.models import EvaluationResult, QualityAlertEvent

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """Alert structure for quality events."""
    alert_type: str  # "degradation", "improvement", "anomaly"
    model: str
    metric: str
    current_value: float
    expected_value: float
    deviation: float
    significance: float  # p-value
    timestamp: datetime
    recommendation: str


class QualityTrendDetector:
    """
    Statistical process control for quality monitoring.
    
    Detects:
    1. Gradual degradation (trend) using Mann-Kendall / t-test logic
    2. Sudden drops (anomaly) using Z-score
    """
    
    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.05  # p-value for significance
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        # Stores rolling history of metric scores per model
        # history[model][metric] -> Deque[float]
        self.history: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=window_size))
        )
    
    def record_observation(
        self,
        model: str,
        metrics: Dict[str, float]
    ):
        """Record a new set of quality metrics for a model."""
        for metric, value in metrics.items():
            self.history[model][metric].append(value)
    
    def detect_degradation(
        self,
        model: str,
        metric: str
    ) -> Optional[QualityAlert]:
        """
        Detect statistically significant degradation using t-test.
        Compares the first half of the window (baseline) with the second half (recent).
        """
        values = list(self.history[model][metric])
        
        # Need enough data points for statistical significance
        if len(values) < 20:
            return None
        
        # Split into baseline (older) and recent (newer)
        mid = len(values) // 2
        baseline = values[:mid]
        recent = values[mid:]
        
        # Two-sample t-test (independent)
        # Null hypothesis: means are equal
        # Alternative hypothesis: means are different
        t_stat, p_value = stats.ttest_ind(baseline, recent)
        
        baseline_mean = np.mean(baseline)
        recent_mean = np.mean(recent)
        
        # Check if difference is significant AND it's a degradation (recent < baseline)
        if p_value < self.alert_threshold and recent_mean < baseline_mean:
            deviation = (baseline_mean - recent_mean) / baseline_mean if baseline_mean != 0 else 0
            
            return QualityAlert(
                alert_type="degradation",
                model=model,
                metric=metric,
                current_value=float(recent_mean),
                expected_value=float(baseline_mean),
                deviation=float(deviation),
                significance=float(p_value),
                timestamp=datetime.utcnow(),
                recommendation=self._generate_recommendation(model, metric, deviation)
            )
        
        return None
    
    def detect_anomaly(
        self,
        model: str,
        metric: str,
        new_value: float
    ) -> Optional[QualityAlert]:
        """
        Detect anomalous single observation using Z-score.
        Flags values more than 3 standard deviations from the mean.
        """
        values = list(self.history[model][metric])
        
        if len(values) < 10:
            return None
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return None
        
        z_score = abs((new_value - mean) / std)
        
        # 3 sigma rule (99.7% confidence)
        if z_score > 3:
            return QualityAlert(
                alert_type="anomaly",
                model=model,
                metric=metric,
                current_value=new_value,
                expected_value=float(mean),
                deviation=float(z_score),
                significance=float(2 * (1 - stats.norm.cdf(z_score))),
                timestamp=datetime.utcnow(),
                recommendation=f"Investigate unusual {metric} value ({new_value:.2f}) for {model}"
            )
        
        return None
    
    def _generate_recommendation(
        self,
        model: str,
        metric: str,
        deviation: float
    ) -> str:
        """Generate actionable recommendation based on the alert."""
        
        if metric == "faithfulness":
            if deviation > 0.1:
                return f"CRITICAL: {model} factual accuracy degraded by {deviation:.1%}. Consider removing from challenger pool."
            else:
                return f"Monitor {model} - minor faithfulness degradation ({deviation:.1%})"
        
        elif metric == "refusal_rate":
            # Note: For refusal rate, "degradation" usually means rate went UP (worse) or DOWN (better)?
            # Here we assume 'metrics' passed in are scores where higher is better.
            # If refusal_rate is passed as a raw rate (0.05), higher is worse.
            # Usually we invert negative metrics (like 1 - refusal_rate) or handle specifically.
            # Assuming the caller handles normalization or we just detect change.
            return f"Check {model} guardrail changes - metric '{metric}' changed significantly."
            
        elif metric == "latency":
             return f"Possible capacity issues with {model} - {metric} degraded by {deviation:.1%}"
             
        return f"Investigate {metric} degradation for {model}"

    def analyze_batch(
        self,
        evaluations: List[EvaluationResult],
        model: str
    ) -> List[QualityAlert]:
        """
        Process a batch of evaluations and return any alerts.
        """
        alerts = []
        
        for eval_result in evaluations:
            # Extract metrics
            metrics = {
                "faithfulness": eval_result.faithfulness,
                "quality": eval_result.quality,
                "conciseness": eval_result.conciseness,
                "composite_score": eval_result.composite_score
            }
            
            # Check for anomalies on individual points
            for metric, value in metrics.items():
                anomaly = self.detect_anomaly(model, metric, value)
                if anomaly:
                    alerts.append(anomaly)
            
            # Record history
            self.record_observation(model, metrics)
        
        # Check for trends after processing the batch
        for metric in ["faithfulness", "quality", "conciseness", "composite_score"]:
            degradation = self.detect_degradation(model, metric)
            if degradation:
                alerts.append(degradation)
                
        return alerts
