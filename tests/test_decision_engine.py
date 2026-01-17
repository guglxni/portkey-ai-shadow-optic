"""
Tests for the Decision Engine.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from shadow_optic.decision_engine import (
    DecisionEngine,
    ThompsonSamplingModelSelector,
    get_model_pricing,
)
from shadow_optic.model_registry import MODEL_REGISTRY
from shadow_optic.models import (
    ChallengerResult,
    CostAnalysis,
    EvaluationResult,
    EvaluatorConfig,
    OptimizationReport,
    Recommendation,
    RecommendationAction,
)


class TestDecisionEngine:
    """Tests for DecisionEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create decision engine with default config."""
        return DecisionEngine(
            config=EvaluatorConfig(
                faithfulness_threshold=0.8,
                quality_threshold=0.7,
                conciseness_threshold=0.5,
                refusal_rate_threshold=0.01
            )
        )
    
    @pytest.fixture
    def passing_evaluations(self):
        """Create evaluations that pass all thresholds."""
        return [
            EvaluationResult(
                shadow_result_id=f"result_{i}",
                faithfulness=0.9,
                quality=0.85,
                conciseness=0.7,
                is_refusal=False,
                composite_score=0.87,
                passed_thresholds=True
            )
            for i in range(50)
        ]
    
    @pytest.fixture
    def failing_evaluations(self):
        """Create evaluations that fail thresholds."""
        return [
            EvaluationResult(
                shadow_result_id=f"result_{i}",
                faithfulness=0.6,  # Below threshold
                quality=0.5,  # Below threshold
                conciseness=0.4,
                is_refusal=False,
                composite_score=0.55,
                passed_thresholds=False
            )
            for i in range(50)
        ]
    
    def test_aggregate_passing_results(self, engine, passing_evaluations):
        """Test aggregation of passing evaluations."""
        result = engine.aggregate_results(
            evaluations=passing_evaluations,
            challenger_model="deepseek-v3"
        )
        
        assert result.model == "deepseek-v3"
        assert result.samples_evaluated == 50
        assert result.avg_faithfulness >= 0.8
        assert result.avg_quality >= 0.7
        assert result.passed_thresholds is True
        assert len(result.failure_reasons) == 0
    
    def test_aggregate_failing_results(self, engine, failing_evaluations):
        """Test aggregation of failing evaluations."""
        result = engine.aggregate_results(
            evaluations=failing_evaluations,
            challenger_model="deepseek-v3"
        )
        
        assert result.passed_thresholds is False
        assert len(result.failure_reasons) > 0
        assert any("Faithfulness" in r for r in result.failure_reasons)
    
    def test_aggregate_with_refusals(self, engine):
        """Test aggregation handling refusals."""
        evaluations = [
            EvaluationResult(
                shadow_result_id=f"result_{i}",
                faithfulness=0.0,
                quality=0.0,
                conciseness=0.0,
                is_refusal=True,
                composite_score=0.0,
                passed_thresholds=False
            )
            for i in range(5)  # 5 refusals
        ] + [
            EvaluationResult(
                shadow_result_id=f"result_{i+5}",
                faithfulness=0.9,
                quality=0.85,
                conciseness=0.7,
                is_refusal=False,
                composite_score=0.87,
                passed_thresholds=True
            )
            for i in range(95)  # 95 successes
        ]
        
        result = engine.aggregate_results(evaluations, "deepseek-v3")
        
        # 5% refusal rate should fail the 1% threshold
        assert result.refusal_rate == 0.05
        assert result.passed_thresholds is False
        assert any("Refusal rate" in r for r in result.failure_reasons)
    
    def test_calculate_cost_analysis(self, engine):
        """Test cost analysis calculation."""
        challenger_result = ChallengerResult(
            model="deepseek-v3",
            samples_evaluated=100,
            avg_faithfulness=0.9,
            avg_quality=0.85,
            avg_conciseness=0.7,
            refusal_rate=0.0,
            avg_latency_ms=500,
            avg_cost_per_request=0.0001,
            success_rate=1.0,
            passed_thresholds=True
        )
        
        analysis = engine.calculate_cost_analysis(
            production_model="gpt-5.2-turbo",
            challenger_model="deepseek-v3",
            challenger_result=challenger_result
        )
        
        # DeepSeek should be much cheaper
        assert analysis.savings_percent > 0.5
        assert analysis.challenger_cost_per_1k < analysis.production_cost_per_1k
        assert analysis.estimated_monthly_savings > 0
    
    def test_generate_switch_recommendation(self, engine, passing_evaluations):
        """Test recommendation generation when switch is viable."""
        results = {
            "deepseek-v3": engine.aggregate_results(passing_evaluations, "deepseek-v3")
        }
        
        recommendation = engine.generate_recommendation(
            challenger_results=results,
            production_model="gpt-5.2-turbo"
        )
        
        assert recommendation.action == RecommendationAction.SWITCH_RECOMMENDED
        assert recommendation.challenger == "deepseek-v3"
        assert recommendation.confidence > 0.5
        assert len(recommendation.suggested_actions) > 0
    
    def test_generate_no_viable_recommendation(self, engine, failing_evaluations):
        """Test recommendation when no challenger is viable."""
        results = {
            "deepseek-v3": engine.aggregate_results(failing_evaluations, "deepseek-v3"),
            "llama-4-scout": engine.aggregate_results(failing_evaluations, "llama-4-scout")
        }
        
        recommendation = engine.generate_recommendation(
            challenger_results=results,
            production_model="gpt-5.2-turbo"
        )
        
        assert recommendation.action == RecommendationAction.NO_VIABLE_CHALLENGER
        assert recommendation.challenger is None
    
    def test_generate_insufficient_data_recommendation(self, engine):
        """Test recommendation with insufficient data."""
        evaluations = [
            EvaluationResult(
                shadow_result_id="result_0",
                faithfulness=0.9,
                quality=0.85,
                conciseness=0.7,
                is_refusal=False,
                composite_score=0.87,
                passed_thresholds=True
            )
        ]  # Only 1 evaluation
        
        results = {"deepseek-v3": engine.aggregate_results(evaluations, "deepseek-v3")}
        
        recommendation = engine.generate_recommendation(
            challenger_results=results,
            production_model="gpt-5.2-turbo"
        )
        
        assert recommendation.action == RecommendationAction.INSUFFICIENT_DATA
    
    def test_select_best_challenger_by_cost(self, engine):
        """Test best challenger selection prefers lower cost."""
        results = {
            "deepseek-v3": ChallengerResult(
                model="deepseek-v3",
                samples_evaluated=100,
                avg_faithfulness=0.9,
                avg_quality=0.85,
                avg_conciseness=0.7,
                refusal_rate=0.0,
                avg_latency_ms=500,
                avg_cost_per_request=0.0001,  # Cheaper
                success_rate=1.0,
                passed_thresholds=True
            ),
            "gpt-4o-mini": ChallengerResult(
                model="gpt-4o-mini",
                samples_evaluated=100,
                avg_faithfulness=0.95,  # Slightly better
                avg_quality=0.90,
                avg_conciseness=0.75,
                refusal_rate=0.0,
                avg_latency_ms=300,
                avg_cost_per_request=0.001,  # More expensive
                success_rate=1.0,
                passed_thresholds=True
            )
        }
        
        best = engine.select_best_challenger(results)
        
        # Should select cheaper option when both pass
        assert best is not None
        assert best[0] == "deepseek-v3"
    
    def test_generate_complete_report(self, engine, passing_evaluations):
        """Test complete report generation."""
        results = {
            "deepseek-v3": engine.aggregate_results(passing_evaluations, "deepseek-v3")
        }
        
        report = engine.generate_report(
            challenger_results=results,
            production_model="gpt-5.2-turbo",
            traces_analyzed=1000,
            samples_evaluated=50,
            time_window="24h"
        )
        
        assert report.traces_analyzed == 1000
        assert report.samples_evaluated == 50
        assert report.time_window == "24h"
        assert "deepseek-v3" in report.challenger_results
        assert report.recommendation is not None


class TestThompsonSamplingModelSelector:
    """Tests for Thompson Sampling model selector."""
    
    @pytest.fixture
    def selector(self):
        """Create selector with test models."""
        return ThompsonSamplingModelSelector([
            "deepseek-v3", "llama-4-scout", "mixtral-next"
        ])
    
    def test_initial_state(self, selector):
        """Test initial Thompson Sampling state."""
        assert len(selector.models) == 3
        
        for model in selector.models:
            assert selector.alpha[model] == 1.0
            assert selector.beta[model] == 1.0
    
    def test_select_model(self, selector):
        """Test model selection returns valid model."""
        selected = selector.select_model()
        
        assert selected in selector.models
        assert len(selector.history) == 1
    
    def test_update_on_success(self, selector):
        """Test belief update on successful evaluation."""
        model = "deepseek-v3"
        initial_alpha = selector.alpha[model]
        
        selector.update(model, passed=True, quality_score=0.95)
        
        # Alpha should increase
        assert selector.alpha[model] > initial_alpha
    
    def test_update_on_failure(self, selector):
        """Test belief update on failed evaluation."""
        model = "deepseek-v3"
        initial_beta = selector.beta[model]
        
        selector.update(model, passed=False, quality_score=0.5)
        
        # Beta should increase
        assert selector.beta[model] > initial_beta
    
    def test_convergence_to_best(self, selector):
        """Test that selection converges to best performing model."""
        # Simulate many evaluations where deepseek-v3 is best
        for _ in range(100):
            selector.update("deepseek-v3", True, 0.95)
            selector.update("llama-4-scout", True, 0.75)
            selector.update("mixtral-next", True, 0.60)
        
        # DeepSeek should have highest expected value
        rankings = selector.get_model_rankings()
        
        assert rankings[0][0] == "deepseek-v3"
        assert rankings[0][1] > rankings[1][1]
    
    def test_explain_selection(self, selector):
        """Test explanation generation."""
        # Add some history
        selector.update("deepseek-v3", True, 0.9)
        selector.update("llama-4-scout", False, 0.5)
        
        explanation = selector.explain_selection()
        
        assert "deepseek-v3" in explanation
        assert "llama-4-scout" in explanation
        assert "Expected Quality" in explanation


class TestModelPricing:
    """Tests for model pricing configuration."""
    
    def test_pricing_exists_for_main_models(self):
        """Test pricing data exists for main models (Model Catalog format - Jan 17, 2026)."""
        expected_models = [
            "gpt-5.2",
            "gpt-5-mini",
            "claude-haiku-4.5",
            "gemini-2.5-flash"
        ]
        
        for model in expected_models:
            assert model in MODEL_REGISTRY, f"{model} should be in MODEL_REGISTRY"
            spec = MODEL_REGISTRY[model]
            assert spec.input_cost_per_1m > 0
            assert spec.output_cost_per_1m > 0
    
    def test_challenger_cheaper_than_production(self):
        """Test challengers are cheaper than production models (Latest Jan 17, 2026 models)."""
        production_spec = MODEL_REGISTRY["gpt-5.2"]
        production_cost = (
            production_spec.input_cost_per_1m +
            production_spec.output_cost_per_1m
        )
        
        for challenger in ["gpt-5-mini", "claude-haiku-4.5", "gemini-2.5-flash"]:
            challenger_spec = MODEL_REGISTRY[challenger]
            challenger_cost = (
                challenger_spec.input_cost_per_1m +
                challenger_spec.output_cost_per_1m
            )
            assert challenger_cost < production_cost, f"{challenger} should be cheaper than gpt-5.2"
