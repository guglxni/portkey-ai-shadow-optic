"""
Decision Engine for Shadow-Optic.

Analyzes evaluation results and generates actionable recommendations
for model switching. Implements multi-criteria decision making with:

- Quality thresholds (faithfulness, quality, conciseness)
- Refusal rate limits
- Cost-benefit analysis
- Confidence scoring
- Explainable recommendations
- Intelligent model selection from 250+ Portkey models

The engine outputs clear recommendations with supporting evidence
for stakeholder decision-making.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from shadow_optic.models import (
    ChallengerResult,
    CostAnalysis,
    EvaluationResult,
    EvaluatorConfig,
    OptimizationReport,
    Recommendation,
    RecommendationAction,
)

logger = logging.getLogger(__name__)


# Import model registry for pricing (lazy import to avoid circular deps)
def _get_model_pricing() -> Dict[str, Dict[str, float]]:
    """Get model pricing from registry or use fallback."""
    try:
        from shadow_optic.model_registry import MODEL_REGISTRY
        return {
            model_id: {
                "input": spec.input_cost_per_1m,
                "output": spec.output_cost_per_1m
            }
            for model_id, spec in MODEL_REGISTRY.items()
        }
    except ImportError:
        # Fallback pricing if registry not available
        return MODEL_PRICING_FALLBACK


# Fallback pricing per 1M tokens (USD) - January 17, 2026
# Used only if model_registry import fails
MODEL_PRICING_FALLBACK = {
    # OpenAI GPT-5.2 series
    "gpt-5.2": {"input": 1.75, "output": 14.00},
    "gpt-5.2-pro": {"input": 21.00, "output": 168.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1": {"input": 3.00, "output": 12.00},
    "gpt-4.1-mini": {"input": 0.80, "output": 3.20},
    "gpt-4.1-nano": {"input": 0.20, "output": 0.80},
    # Anthropic Claude 4.5 series
    "claude-opus-4.5": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4.5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    # Google Gemini series
    "gemini-3-pro": {"input": 2.00, "output": 12.00},
    "gemini-3-flash": {"input": 0.50, "output": 3.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    # Meta Llama 4 series
    "llama-4-405b": {"input": 0.50, "output": 0.50},
    "llama-4-70b": {"input": 0.25, "output": 0.25},
    "llama-4-8b": {"input": 0.05, "output": 0.05},
    # Mistral series
    "mistral-large-3": {"input": 2.00, "output": 6.00},
    "mistral-medium-3": {"input": 1.00, "output": 3.00},
    # DeepSeek series
    "deepseek-v3": {"input": 0.14, "output": 0.28},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Grok series
    "grok-3": {"input": 2.00, "output": 10.00},
    "grok-3-mini": {"input": 0.50, "output": 2.00},
}

# Lazy-loaded pricing
MODEL_PRICING: Dict[str, Dict[str, float]] = {}


def get_model_pricing(model_id: str) -> Dict[str, float]:
    """
    Get pricing for a model, with lazy loading from registry.
    
    Falls back to default pricing if model not found.
    """
    global MODEL_PRICING
    
    # Lazy load from registry
    if not MODEL_PRICING:
        MODEL_PRICING = _get_model_pricing()
    
    # Strip @provider/ prefix if present
    clean_id = model_id.split("/")[-1] if "/" in model_id else model_id
    
    # Try exact match
    if clean_id in MODEL_PRICING:
        return MODEL_PRICING[clean_id]
    
    # Try without version suffix
    base_id = clean_id.rsplit("-", 1)[0] if "-" in clean_id else clean_id
    if base_id in MODEL_PRICING:
        return MODEL_PRICING[base_id]
    
    # Default fallback
    logger.warning(f"Unknown model pricing for {model_id}, using defaults")
    return {"input": 1.0, "output": 1.0}


class DecisionEngine:
    """
    Analyzes evaluation results and recommends model switching actions.
    
    Decision Process:
        1. Aggregate results per challenger model
        2. Apply quality thresholds
        3. Calculate cost savings
        4. Compute confidence score
        5. Generate recommendation with reasoning
    
    Example:
        ```python
        engine = DecisionEngine(config=EvaluatorConfig())
        
        report = engine.generate_report(
            challenger_results=results,
            production_model="gpt-5.2-turbo",
            traces_analyzed=3000,
            samples_evaluated=150
        )
        
        print(report.recommendation.action)  # SWITCH_RECOMMENDED
        print(report.recommendation.reasoning)  # Detailed explanation
        ```
    """
    
    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        monthly_request_volume: int = 100_000
    ):
        self.config = config or EvaluatorConfig()
        self.monthly_volume = monthly_request_volume
    
    def aggregate_results(
        self,
        evaluations: List[EvaluationResult],
        challenger_model: str
    ) -> ChallengerResult:
        """
        Aggregate individual evaluations into challenger summary.
        
        Calculates:
            - Average scores across all metrics
            - Refusal rate
            - Success rate
            - Threshold pass/fail
        """
        if not evaluations:
            return ChallengerResult(
                model=challenger_model,
                samples_evaluated=0,
                avg_faithfulness=0.0,
                avg_quality=0.0,
                avg_conciseness=0.0,
                refusal_rate=1.0,
                avg_latency_ms=0.0,
                avg_cost_per_request=0.0,
                success_rate=0.0,
                passed_thresholds=False,
                failure_reasons=["No evaluations available"],
                individual_results=evaluations
            )
        
        # Filter successful evaluations (non-refusals)
        successful = [e for e in evaluations if not e.is_refusal]
        refusals = [e for e in evaluations if e.is_refusal]
        
        n_total = len(evaluations)
        n_successful = len(successful)
        
        # Calculate averages
        avg_faithfulness = (
            sum(e.faithfulness for e in successful) / n_successful
            if n_successful > 0 else 0.0
        )
        avg_quality = (
            sum(e.quality for e in successful) / n_successful
            if n_successful > 0 else 0.0
        )
        avg_conciseness = (
            sum(e.conciseness for e in successful) / n_successful
            if n_successful > 0 else 0.0
        )
        
        refusal_rate = len(refusals) / n_total
        success_rate = n_successful / n_total
        
        # Check thresholds
        failure_reasons = []
        
        if avg_faithfulness < self.config.faithfulness_threshold:
            failure_reasons.append(
                f"Faithfulness {avg_faithfulness:.2f} < {self.config.faithfulness_threshold}"
            )
        
        if avg_quality < self.config.quality_threshold:
            failure_reasons.append(
                f"Quality {avg_quality:.2f} < {self.config.quality_threshold}"
            )
        
        if avg_conciseness < self.config.conciseness_threshold:
            failure_reasons.append(
                f"Conciseness {avg_conciseness:.2f} < {self.config.conciseness_threshold}"
            )
        
        if refusal_rate > self.config.refusal_rate_threshold:
            failure_reasons.append(
                f"Refusal rate {refusal_rate:.2%} > {self.config.refusal_rate_threshold:.2%}"
            )
        
        passed = len(failure_reasons) == 0
        
        # Estimate cost (using model pricing from registry)
        pricing = get_model_pricing(challenger_model)
        # Rough estimate: 500 input + 500 output tokens per request
        avg_cost = (500 * pricing["input"] + 500 * pricing["output"]) / 1_000_000
        
        return ChallengerResult(
            model=challenger_model,
            samples_evaluated=n_total,
            avg_faithfulness=avg_faithfulness,
            avg_quality=avg_quality,
            avg_conciseness=avg_conciseness,
            refusal_rate=refusal_rate,
            avg_latency_ms=0.0,  # Would need to track from shadow results
            avg_cost_per_request=avg_cost,
            success_rate=success_rate,
            passed_thresholds=passed,
            failure_reasons=failure_reasons,
            individual_results=evaluations
        )
    
    def calculate_cost_analysis(
        self,
        production_model: str,
        challenger_model: str,
        challenger_result: ChallengerResult
    ) -> CostAnalysis:
        """
        Calculate cost comparison between production and challenger.
        
        Returns projected savings based on current request volume.
        """
        # Get pricing from model registry
        prod_pricing = get_model_pricing(production_model)
        chal_pricing = get_model_pricing(challenger_model)
        
        # Cost per 1000 requests (estimate 500 input + 500 output tokens)
        prod_cost_per_1k = (
            (500 * prod_pricing["input"] + 500 * prod_pricing["output"]) / 1_000_000
        ) * 1000
        
        chal_cost_per_1k = (
            (500 * chal_pricing["input"] + 500 * chal_pricing["output"]) / 1_000_000
        ) * 1000
        
        savings_percent = (prod_cost_per_1k - chal_cost_per_1k) / prod_cost_per_1k
        
        # Monthly savings
        monthly_savings = (
            (prod_cost_per_1k - chal_cost_per_1k) *
            (self.monthly_volume / 1000)
        )
        
        # Breakeven quality threshold
        # At what quality level do we stop caring about cost savings?
        # Simple model: 1% quality drop = 10% cost savings acceptable
        breakeven = 1.0 - (savings_percent / 10)
        
        return CostAnalysis(
            production_cost_per_1k=prod_cost_per_1k,
            challenger_cost_per_1k=chal_cost_per_1k,
            savings_percent=savings_percent,
            estimated_monthly_savings=monthly_savings,
            breakeven_quality_threshold=breakeven
        )
    
    def compute_confidence(
        self,
        challenger_result: ChallengerResult
    ) -> float:
        """
        Compute confidence score for the recommendation.
        
        Factors:
            - Sample size (more samples = higher confidence)
            - Variance in scores (lower variance = higher confidence)
            - Margin above thresholds (bigger margin = higher confidence)
        """
        confidence = 0.0
        
        # Sample size factor (0-0.4)
        # Confidence increases with log of samples
        import math
        sample_factor = min(0.4, math.log10(challenger_result.samples_evaluated + 1) / 5)
        confidence += sample_factor
        
        # Margin factor (0-0.4)
        # How far above thresholds?
        faith_margin = challenger_result.avg_faithfulness - self.config.faithfulness_threshold
        qual_margin = challenger_result.avg_quality - self.config.quality_threshold
        
        if faith_margin > 0 and qual_margin > 0:
            margin_factor = min(0.4, (faith_margin + qual_margin) / 2)
            confidence += margin_factor
        
        # Consistency factor (0-0.2)
        # Lower variance = higher consistency
        if challenger_result.individual_results:
            scores = [r.composite_score for r in challenger_result.individual_results]
            if len(scores) > 1:
                import statistics
                variance = statistics.variance(scores)
                consistency = max(0, 0.2 - variance * 2)
                confidence += consistency
        
        return min(1.0, confidence)
    
    def select_best_challenger(
        self,
        challenger_results: Dict[str, ChallengerResult]
    ) -> Optional[Tuple[str, ChallengerResult]]:
        """
        Select the best challenger model that passes all thresholds.
        
        Selection criteria (in order):
            1. Passes all quality thresholds
            2. Lowest cost
            3. Highest composite score (tie-breaker)
        """
        passing = [
            (model, result)
            for model, result in challenger_results.items()
            if result.passed_thresholds
        ]
        
        if not passing:
            return None
        
        # Sort by cost (ascending), then by composite score (descending)
        passing.sort(key=lambda x: (
            x[1].avg_cost_per_request,
            -x[1].composite_score
        ))
        
        return passing[0]
    
    def generate_recommendation(
        self,
        challenger_results: Dict[str, ChallengerResult],
        production_model: str
    ) -> Recommendation:
        """
        Generate final recommendation based on all challenger results.
        
        Actions:
            - SWITCH_RECOMMENDED: A challenger meets all criteria
            - CONTINUE_MONITORING: Some challengers are close, need more data
            - NO_VIABLE_CHALLENGER: All challengers failed significantly
            - INSUFFICIENT_DATA: Not enough samples to decide
        """
        # Check for insufficient data
        total_samples = sum(r.samples_evaluated for r in challenger_results.values())
        if total_samples < 50:
            return Recommendation(
                action=RecommendationAction.INSUFFICIENT_DATA,
                confidence=0.2,
                reasoning=f"Only {total_samples} samples evaluated. "
                         "Need at least 50 for reliable recommendation.",
                suggested_actions=[
                    "Wait for more production traffic",
                    "Manually add test cases to golden set",
                    "Reduce sampling thresholds temporarily"
                ]
            )
        
        # Find best passing challenger
        best = self.select_best_challenger(challenger_results)
        
        if best:
            model, result = best
            cost_analysis = self.calculate_cost_analysis(
                production_model, model, result
            )
            confidence = self.compute_confidence(result)
            
            return Recommendation(
                action=RecommendationAction.SWITCH_RECOMMENDED,
                challenger=model,
                confidence=confidence,
                reasoning=self._generate_switch_reasoning(model, result, cost_analysis),
                cost_analysis=cost_analysis,
                metrics_summary={
                    "faithfulness": result.avg_faithfulness,
                    "quality": result.avg_quality,
                    "conciseness": result.avg_conciseness,
                    "refusal_rate": result.refusal_rate,
                    "composite_score": result.composite_score
                },
                suggested_actions=[
                    f"Begin canary rollout with 1% traffic to {model}",
                    "Set up quality monitoring alerts",
                    "Configure automatic rollback thresholds",
                    "Schedule review meeting after 48 hours"
                ],
                rollout_config={
                    "initial_percentage": 1,
                    "ramp_schedule": [1, 5, 10, 25, 50, 100],
                    "quality_gates": {
                        "faithfulness_min": 0.85,
                        "quality_min": 0.75,
                        "error_rate_max": 0.01
                    }
                }
            )
        
        # Check if any challenger is close
        close_challengers = [
            (model, result)
            for model, result in challenger_results.items()
            if len(result.failure_reasons) <= 1 and result.avg_faithfulness >= 0.7
        ]
        
        if close_challengers:
            return Recommendation(
                action=RecommendationAction.CONTINUE_MONITORING,
                confidence=0.5,
                reasoning=self._generate_monitoring_reasoning(close_challengers),
                suggested_actions=[
                    "Investigate failure reasons for near-passing models",
                    "Consider relaxing thresholds if acceptable",
                    "Collect more samples for borderline cases",
                    "Review evaluation criteria for potential bias"
                ]
            )
        
        # No viable challenger
        return Recommendation(
            action=RecommendationAction.NO_VIABLE_CHALLENGER,
            confidence=0.8,
            reasoning=self._generate_no_viable_reasoning(challenger_results),
            suggested_actions=[
                "Evaluate different challenger models",
                "Check if production model can be optimized instead",
                "Review if use case is suitable for cheaper models",
                "Consider prompt engineering to improve challenger performance"
            ]
        )
    
    def _generate_switch_reasoning(
        self,
        model: str,
        result: ChallengerResult,
        cost_analysis: CostAnalysis
    ) -> str:
        """Generate detailed reasoning for switch recommendation."""
        return f"""
Based on evaluation of {result.samples_evaluated} production requests, 
**{model}** is recommended as a production replacement.

**Quality Metrics:**
- Faithfulness: {result.avg_faithfulness:.2f} (threshold: {self.config.faithfulness_threshold})
- Quality: {result.avg_quality:.2f} (threshold: {self.config.quality_threshold})
- Conciseness: {result.avg_conciseness:.2f} (threshold: {self.config.conciseness_threshold})
- Refusal Rate: {result.refusal_rate:.2%} (threshold: {self.config.refusal_rate_threshold:.2%})

**Cost Impact:**
- Current cost: ${cost_analysis.production_cost_per_1k:.2f}/1K requests
- With {model}: ${cost_analysis.challenger_cost_per_1k:.2f}/1K requests
- Savings: {cost_analysis.savings_percent:.0%} ({cost_analysis.estimated_monthly_savings:,.0f}/month)

**Confidence:** {self.compute_confidence(result):.0%}

The model passed all quality thresholds while offering significant cost reduction.
A phased rollout is recommended to validate production behavior.
""".strip()
    
    def _generate_monitoring_reasoning(
        self,
        close_challengers: List[Tuple[str, ChallengerResult]]
    ) -> str:
        """Generate reasoning for continue monitoring recommendation."""
        details = []
        for model, result in close_challengers:
            details.append(
                f"- {model}: {result.failure_reasons[0] if result.failure_reasons else 'Close to threshold'}"
            )
        
        return f"""
No challenger currently meets all quality thresholds, but some are close:

{chr(10).join(details)}

These models may become viable with:
- Additional training data
- Prompt optimization
- Threshold adjustments (if business acceptable)
- More evaluation samples to reduce variance

Continue monitoring recommended.
""".strip()
    
    def _generate_no_viable_reasoning(
        self,
        results: Dict[str, ChallengerResult]
    ) -> str:
        """Generate reasoning for no viable challenger recommendation."""
        details = []
        for model, result in results.items():
            reasons = "; ".join(result.failure_reasons[:2])
            details.append(f"- {model}: {reasons}")
        
        return f"""
All evaluated challengers failed to meet quality thresholds:

{chr(10).join(details)}

The current production model appears necessary for this use case.
Consider evaluating different model families or optimizing the production model.
""".strip()
    
    def generate_report(
        self,
        challenger_results: Dict[str, ChallengerResult],
        production_model: str,
        traces_analyzed: int,
        samples_evaluated: int,
        time_window: str = "24h"
    ) -> OptimizationReport:
        """
        Generate complete optimization report.
        
        This is the main entry point for report generation.
        """
        recommendation = self.generate_recommendation(
            challenger_results, production_model
        )
        
        best_challenger = None
        if recommendation.action == RecommendationAction.SWITCH_RECOMMENDED:
            best_challenger = recommendation.challenger
        
        return OptimizationReport(
            time_window=time_window,
            traces_analyzed=traces_analyzed,
            samples_evaluated=samples_evaluated,
            challenger_results=challenger_results,
            best_challenger=best_challenger,
            recommendation=recommendation,
            execution_stats={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "production_model": production_model,
                "challengers_evaluated": list(challenger_results.keys())
            }
        )


class ThompsonSamplingModelSelector:
    """
    Multi-Armed Bandit for intelligent model selection.
    
    Instead of testing all challengers equally, uses Thompson Sampling to:
    1. Explore promising models more often
    2. Exploit known good performers
    3. Automatically adapt as model quality changes
    
    This reduces shadow testing budget by focusing on likely winners.
    
    Example:
        ```python
        selector = ThompsonSamplingModelSelector(["deepseek-v3", "llama-4"])
        
        # Select model to test
        model = selector.select_model()
        
        # After evaluation, update beliefs
        selector.update(model, passed=True, quality_score=0.92)
        
        # Over time, selection biases toward better models
        ```
    """
    
    def __init__(self, challenger_models: List[str]):
        self.models = challenger_models
        # Beta distribution parameters (successes, failures)
        self.alpha = {model: 1.0 for model in challenger_models}
        self.beta = {model: 1.0 for model in challenger_models}
        self.history = []
    
    def select_model(self) -> str:
        """Select next model to test using Thompson Sampling."""
        import numpy as np
        from scipy.stats import beta
        
        samples = {}
        for model in self.models:
            samples[model] = beta.rvs(self.alpha[model], self.beta[model])
        
        selected = max(samples, key=samples.get)
        
        self.history.append({
            "selected": selected,
            "samples": samples,
            "timestamp": datetime.now(timezone.utc)
        })
        
        return selected
    
    def update(self, model: str, passed: bool, quality_score: float):
        """Update beliefs based on evaluation result."""
        if quality_score >= 0.9:
            self.alpha[model] += 1.0
        elif quality_score >= 0.7:
            self.alpha[model] += 0.5
            self.beta[model] += 0.5
        else:
            self.beta[model] += 1.0
    
    def get_model_rankings(self) -> List[Tuple[str, float, float]]:
        """Get current model rankings with confidence."""
        rankings = []
        for model in self.models:
            expected = self.alpha[model] / (self.alpha[model] + self.beta[model])
            n_obs = self.alpha[model] + self.beta[model] - 2
            confidence = min(1.0, n_obs / 100)
            rankings.append((model, expected, confidence))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def explain_selection(self) -> str:
        """Generate human-readable explanation."""
        rankings = self.get_model_rankings()
        
        lines = ["Current Model Rankings (Thompson Sampling):", "-" * 50]
        
        for rank, (model, expected, confidence) in enumerate(rankings, 1):
            lines.append(f"{rank}. {model}")
            lines.append(f"   Expected Quality: {expected:.2%}")
            lines.append(f"   Confidence: {confidence:.1%}")
            lines.append(f"   α={self.alpha[model]:.1f}, β={self.beta[model]:.1f}")
            lines.append("")
        
        return "\n".join(lines)
