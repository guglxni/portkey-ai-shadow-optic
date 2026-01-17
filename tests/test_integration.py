#!/usr/bin/env python3
"""Integration tests for Shadow-Optic core functionality."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from shadow_optic.models import (
    ProductionTrace, GoldenPrompt, ShadowResult, EvaluationResult,
    ChallengerResult, CostAnalysis, Recommendation, RecommendationAction,
    OptimizationReport, WorkflowConfig, SamplerConfig, EvaluatorConfig,
    ChallengerConfig, TraceStatus, SampleType
)
from shadow_optic.decision_engine import DecisionEngine, ThompsonSamplingModelSelector


class TestModelsIntegration:
    """Test Pydantic models work correctly."""
    
    def test_production_trace_creation(self):
        """Test ProductionTrace model."""
        trace = ProductionTrace(
            trace_id='test-1',
            request_timestamp=datetime.now(),
            prompt='What is Python?',
            response='Python is a programming language.',
            model='gpt-5.2-turbo',
            latency_ms=250,
            tokens_prompt=10,
            tokens_completion=20,
            cost=0.001
        )
        assert trace.trace_id == 'test-1'
        assert trace.status == TraceStatus.SUCCESS
    
    def test_golden_prompt_creation(self):
        """Test GoldenPrompt model."""
        golden = GoldenPrompt(
            original_trace_id='test-1',
            prompt='What is Python?',
            production_response='Python is a programming language.'
        )
        assert golden.original_trace_id == 'test-1'
        assert golden.sample_type == SampleType.CENTROID
    
    def test_shadow_result_creation(self):
        """Test ShadowResult model."""
        shadow = ShadowResult(
            golden_prompt_id='test-1',
            challenger_model='deepseek-v3',
            shadow_response='Python is a versatile programming language.',
            latency_ms=180,
            tokens_prompt=10,
            tokens_completion=25,
            cost=0.00005,
            success=True
        )
        assert shadow.challenger_model == 'deepseek-v3'
        assert shadow.success is True
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult model with composite score."""
        eval_result = EvaluationResult(
            shadow_result_id='test-1_deepseek-v3',
            faithfulness=0.92,
            quality=0.88,
            conciseness=0.75,
            composite_score=0.87,
            passed_thresholds=True
        )
        assert eval_result.passed_thresholds is True
        assert 0 <= eval_result.composite_score <= 1
    
    def test_config_models(self):
        """Test configuration models."""
        sampler_config = SamplerConfig()
        assert sampler_config.n_clusters == 50
        
        evaluator_config = EvaluatorConfig()
        assert evaluator_config.faithfulness_threshold == 0.8
        
        challenger_config = ChallengerConfig(
            model_id='deepseek-chat',
            provider='deepseek',
            provider_slug='deepseek'
        )
        assert challenger_config.model_id == 'deepseek-chat'


class TestDecisionEngineIntegration:
    """Test DecisionEngine functionality."""
    
    @pytest.fixture
    def engine(self):
        """Create decision engine."""
        return DecisionEngine(
            config=EvaluatorConfig(),
            monthly_request_volume=100_000
        )
    
    @pytest.fixture
    def passing_evaluations(self):
        """Create passing evaluation results."""
        return [
            EvaluationResult(
                shadow_result_id=f'result-{i}',
                faithfulness=0.90,
                quality=0.85,
                conciseness=0.72,
                composite_score=0.86,
                passed_thresholds=True
            )
            for i in range(50)
        ]
    
    @pytest.fixture
    def failing_evaluations(self):
        """Create failing evaluation results."""
        return [
            EvaluationResult(
                shadow_result_id=f'result-{i}',
                faithfulness=0.65,  # Below threshold
                quality=0.55,
                conciseness=0.40,
                composite_score=0.55,
                passed_thresholds=False
            )
            for i in range(50)
        ]
    
    def test_aggregate_passing_results(self, engine, passing_evaluations):
        """Test aggregation of passing results."""
        result = engine.aggregate_results(passing_evaluations, 'deepseek-v3')
        
        assert result.model == 'deepseek-v3'
        assert result.samples_evaluated == 50
        assert result.passed_thresholds is True
        assert result.avg_faithfulness >= 0.8
        assert len(result.failure_reasons) == 0
    
    def test_aggregate_failing_results(self, engine, failing_evaluations):
        """Test aggregation of failing results."""
        result = engine.aggregate_results(failing_evaluations, 'deepseek-v3')
        
        assert result.passed_thresholds is False
        assert len(result.failure_reasons) > 0
    
    def test_cost_analysis(self, engine, passing_evaluations):
        """Test cost analysis calculation."""
        challenger_result = engine.aggregate_results(passing_evaluations, 'deepseek-v3')
        
        cost_analysis = engine.calculate_cost_analysis(
            production_model='gpt-5.2-turbo',
            challenger_model='deepseek-v3',
            challenger_result=challenger_result
        )
        
        # DeepSeek should be much cheaper
        assert cost_analysis.savings_percent > 0.5
        assert cost_analysis.estimated_monthly_savings > 0
    
    def test_switch_recommendation(self, engine, passing_evaluations):
        """Test that engine recommends switch for passing challengers."""
        challenger_result = engine.aggregate_results(passing_evaluations, 'deepseek-v3')
        
        recommendation = engine.generate_recommendation(
            challenger_results={'deepseek-v3': challenger_result},
            production_model='gpt-5.2-turbo'
        )
        
        assert recommendation.action == RecommendationAction.SWITCH_RECOMMENDED
        assert recommendation.challenger == 'deepseek-v3'
        assert recommendation.confidence > 0.5
    
    def test_no_viable_recommendation(self, engine, failing_evaluations):
        """Test that engine returns no viable when all fail."""
        challenger_result = engine.aggregate_results(failing_evaluations, 'deepseek-v3')
        
        recommendation = engine.generate_recommendation(
            challenger_results={'deepseek-v3': challenger_result},
            production_model='gpt-5.2-turbo'
        )
        
        assert recommendation.action == RecommendationAction.NO_VIABLE_CHALLENGER
    
    def test_insufficient_data_recommendation(self, engine):
        """Test insufficient data when too few samples."""
        few_evals = [
            EvaluationResult(
                shadow_result_id='result-0',
                faithfulness=0.92,
                quality=0.88,
                conciseness=0.75,
                composite_score=0.87,
                passed_thresholds=True
            )
        ]
        
        challenger_result = engine.aggregate_results(few_evals, 'deepseek-v3')
        
        recommendation = engine.generate_recommendation(
            challenger_results={'deepseek-v3': challenger_result},
            production_model='gpt-5.2-turbo'
        )
        
        assert recommendation.action == RecommendationAction.INSUFFICIENT_DATA
    
    def test_full_report_generation(self, engine, passing_evaluations):
        """Test complete report generation."""
        challenger_result = engine.aggregate_results(passing_evaluations, 'deepseek-v3')
        
        report = engine.generate_report(
            challenger_results={'deepseek-v3': challenger_result},
            production_model='gpt-5.2-turbo',
            traces_analyzed=1000,
            samples_evaluated=50
        )
        
        assert report.traces_analyzed == 1000
        assert report.samples_evaluated == 50
        assert 'deepseek-v3' in report.challenger_results
        assert report.recommendation is not None


class TestThompsonSampling:
    """Test Thompson Sampling model selector."""
    
    def test_initialization(self):
        """Test selector initialization."""
        selector = ThompsonSamplingModelSelector(['deepseek-v3', 'llama-4-scout'])
        
        assert len(selector.models) == 2
        assert selector.alpha['deepseek-v3'] == 1.0
        assert selector.beta['deepseek-v3'] == 1.0
    
    def test_selection(self):
        """Test model selection."""
        selector = ThompsonSamplingModelSelector(['deepseek-v3', 'llama-4-scout'])
        
        selected = selector.select_model()
        
        assert selected in ['deepseek-v3', 'llama-4-scout']
        assert len(selector.history) == 1
    
    def test_update_beliefs(self):
        """Test belief updates."""
        selector = ThompsonSamplingModelSelector(['deepseek-v3'])
        
        initial_alpha = selector.alpha['deepseek-v3']
        
        # Good quality update
        selector.update('deepseek-v3', passed=True, quality_score=0.95)
        
        assert selector.alpha['deepseek-v3'] > initial_alpha
    
    def test_rankings(self):
        """Test model rankings."""
        selector = ThompsonSamplingModelSelector(['deepseek-v3', 'llama-4-scout'])
        
        # Make deepseek-v3 much better
        for _ in range(20):
            selector.update('deepseek-v3', True, 0.95)
            selector.update('llama-4-scout', False, 0.55)
        
        rankings = selector.get_model_rankings()
        
        assert rankings[0][0] == 'deepseek-v3'


class TestEvaluatorIntegration:
    """Test ShadowEvaluator with mocked dependencies."""
    
    @pytest.fixture
    def mock_portkey(self):
        """Create mock Portkey client."""
        mock = AsyncMock()
        return mock
    
    def test_refusal_detection(self):
        """Test refusal pattern detection."""
        from shadow_optic.evaluator import RefusalDetector
        
        detector = RefusalDetector()
        
        # Should detect refusal
        assert detector.is_refusal("I cannot assist with that request.") is True
        assert detector.is_refusal("I'm sorry, but I can't help with this.") is True
        
        # Should not detect as refusal
        assert detector.is_refusal("Python is a programming language.") is False


class TestSamplerIntegration:
    """Test SemanticSampler with mocked dependencies."""
    
    @pytest.mark.asyncio
    async def test_empty_traces_handling(self):
        """Test sampler handles empty traces."""
        from shadow_optic.sampler import SemanticSampler
        
        mock_qdrant = AsyncMock()
        mock_portkey = AsyncMock()
        
        sampler = SemanticSampler(
            qdrant_client=mock_qdrant,
            portkey_client=mock_portkey
        )
        
        result = await sampler.sample([])
        
        assert result == []


class TestWorkflowConfigValidation:
    """Test workflow configuration validation."""
    
    def test_valid_workflow_config(self):
        """Test valid workflow configuration."""
        config = WorkflowConfig(
            challengers=[
                ChallengerConfig(
                    model_id='deepseek-chat',
                    provider='deepseek',
                    provider_slug='deepseek'
                )
            ]
        )
        
        assert config.log_export_window == '24h'
        assert config.max_budget_per_run == 50.0
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        config = WorkflowConfig(
            challengers=[
                ChallengerConfig(
                    model_id='deepseek-chat',
                    provider='deepseek',
                    provider_slug='deepseek'
                )
            ],
            evaluator=EvaluatorConfig(
                faithfulness_threshold=0.9,
                quality_threshold=0.85,
                conciseness_threshold=0.6
            )
        )
        
        assert config.evaluator.faithfulness_threshold == 0.9


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
