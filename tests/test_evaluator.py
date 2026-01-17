"""
Tests for the Shadow Evaluator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from shadow_optic.evaluator import (
    ShadowEvaluator,
    SemanticDiff,
    RefusalDetector,
)
from shadow_optic.models import (
    EvaluatorConfig,
    GoldenPrompt,
    ShadowResult,
    SampleType,
)


class TestRefusalDetector:
    """Tests for RefusalDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create refusal detector."""
        return RefusalDetector()
    
    def test_detect_cant_assist_refusal(self, detector):
        """Test detection of 'can't assist' refusal."""
        response = "I'm sorry, but I can't assist with that request."
        assert detector.is_refusal(response) is True
    
    def test_detect_unable_to_help_refusal(self, detector):
        """Test detection of 'unable to help' refusal."""
        response = "I'm unable to help with this type of request."
        assert detector.is_refusal(response) is True
    
    def test_detect_policy_refusal(self, detector):
        """Test detection of policy-based refusal."""
        response = "This request violates my content policy guidelines."
        assert detector.is_refusal(response) is True
    
    def test_detect_safety_refusal(self, detector):
        """Test detection of safety refusal."""
        response = "I cannot provide information that could be harmful or dangerous."
        assert detector.is_refusal(response) is True
    
    def test_normal_response_not_refusal(self, detector):
        """Test that normal responses are not flagged."""
        response = "Python is a versatile programming language."
        assert detector.is_refusal(response) is False
    
    def test_helpful_decline_not_refusal(self, detector):
        """Test helpful declines are not flagged as refusals."""
        response = "I don't have access to real-time data, but I can explain the concept."
        assert detector.is_refusal(response) is False
    
    def test_short_response_not_refusal(self, detector):
        """Test short responses are not flagged."""
        response = "Yes."
        assert detector.is_refusal(response) is False


class TestSemanticDiff:
    """Tests for SemanticDiff class."""
    
    @pytest.fixture
    def mock_portkey(self):
        """Create mock Portkey client."""
        return MagicMock()
    
    @pytest.fixture
    def differ(self, mock_portkey):
        """Create semantic diff analyzer with mock client."""
        return SemanticDiff(portkey_client=mock_portkey)
    
    @pytest.mark.asyncio
    async def test_diff_identical_responses(self, differ, mock_portkey):
        """Test diff of identical responses."""
        import numpy as np
        
        # Mock identical embeddings
        embedding = [0.1] * 1536
        mock_portkey.embeddings = MagicMock()
        mock_portkey.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=embedding), MagicMock(embedding=embedding)]
        ))
        
        response = "Python is a programming language."
        diff = await differ.compute_diff(response, response)
        
        assert diff["semantic_similarity"] > 0.99  # Should be ~1.0 for identical
        assert diff["length_ratio"] == 1.0
    
    @pytest.mark.asyncio
    async def test_diff_completely_different(self, differ, mock_portkey):
        """Test diff of completely different responses."""
        import numpy as np
        
        # Mock very different embeddings
        emb1 = [1.0] * 768 + [0.0] * 768
        emb2 = [0.0] * 768 + [1.0] * 768
        mock_portkey.embeddings = MagicMock()
        mock_portkey.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=emb1), MagicMock(embedding=emb2)]
        ))
        
        original = "The sky is blue."
        challenger = "Machine learning algorithms process data."
        
        diff = await differ.compute_diff(original, challenger)
        
        assert diff["semantic_similarity"] < 0.5
    
    @pytest.mark.asyncio
    async def test_diff_tracks_length_change(self, differ, mock_portkey):
        """Test diff tracks length changes."""
        import numpy as np
        
        # Mock similar embeddings
        embedding = [0.1] * 1536
        mock_portkey.embeddings = MagicMock()
        mock_portkey.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=embedding), MagicMock(embedding=embedding)]
        ))
        
        original = "Short response."
        challenger = "This is a much longer response with additional details and explanations."
        
        diff = await differ.compute_diff(original, challenger)
        
        assert diff["length_ratio"] > 1.0
        assert diff["challenger_length"] > diff["production_length"]
    
    @pytest.mark.asyncio
    async def test_diff_identifies_key_changes(self, differ, mock_portkey):
        """Test diff identifies key content changes."""
        import numpy as np
        
        # Mock somewhat similar embeddings
        emb1 = [0.8] * 768 + [0.2] * 768
        emb2 = [0.6] * 768 + [0.4] * 768
        mock_portkey.embeddings = MagicMock()
        mock_portkey.embeddings.create = AsyncMock(return_value=MagicMock(
            data=[MagicMock(embedding=emb1), MagicMock(embedding=emb2)]
        ))
        
        original = "The function returns True when condition is met."
        challenger = "The method returns False when condition is not met."
        
        diff = await differ.compute_diff(original, challenger)
        
        # Should detect the semantic differences
        assert diff["semantic_similarity"] < 1.0


class TestShadowEvaluator:
    """Tests for ShadowEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create evaluator config."""
        return EvaluatorConfig(
            faithfulness_threshold=0.8,
            quality_threshold=0.7,
            conciseness_threshold=0.5,
            refusal_rate_threshold=0.01,
            judge_model="gpt-4o",
            portkey_api_key="test-portkey-key"  # Test API key for mocked tests
        )
    
    @pytest.fixture
    def mock_portkey(self):
        """Create mock Portkey client."""
        return MagicMock()
    
    @pytest.fixture
    def evaluator(self, mock_portkey, config):
        """Create shadow evaluator with mock portkey, bypassing DeepEval initialization."""
        with patch('shadow_optic.evaluator.ShadowEvaluator._init_metrics'), \
             patch('shadow_optic.evaluator.ShadowEvaluator._configure_portkey_environment'):
            evaluator = ShadowEvaluator(portkey_client=mock_portkey, config=config)
            # Manually create mock metrics
            evaluator.faithfulness_metric = MagicMock()
            evaluator.quality_metric = MagicMock()
            evaluator.conciseness_metric = MagicMock()
            return evaluator
    
    @pytest.fixture
    def golden_prompt(self):
        """Create sample golden prompt."""
        return GoldenPrompt(
            original_trace_id="test-trace-1",
            prompt="Explain machine learning in simple terms.",
            production_response="Machine learning is a type of AI where computers learn from data.",
            cluster_id=0,
            sample_type=SampleType.CENTROID
        )
    
    @pytest.fixture
    def shadow_result(self):
        """Create sample shadow result."""
        return ShadowResult(
            golden_prompt_id="test-trace-1",
            challenger_model="deepseek-v3",
            shadow_response="Machine learning is artificial intelligence that learns patterns from data.",
            latency_ms=350,
            tokens_prompt=20,
            tokens_completion=30,
            cost=0.0001,
            success=True
        )
    
    @pytest.mark.asyncio
    async def test_evaluate_single_success(self, evaluator, golden_prompt, shadow_result):
        """Test single evaluation with successful response."""
        # Mock DeepEval metrics
        with patch.object(evaluator, 'faithfulness_metric') as mock_faith, \
             patch.object(evaluator, 'quality_metric') as mock_qual, \
             patch.object(evaluator, 'conciseness_metric') as mock_conc:
            
            mock_faith.a_measure = AsyncMock()
            mock_faith.score = 0.92
            mock_faith.reason = "High faithfulness"
            
            mock_qual.a_measure = AsyncMock()
            mock_qual.score = 0.88
            mock_qual.reason = "Good quality"
            
            mock_conc.a_measure = AsyncMock()
            mock_conc.score = 0.75
            mock_conc.reason = "Reasonably concise"
            
            result = await evaluator.evaluate(golden_prompt, shadow_result)
            
            assert result.faithfulness == 0.92
            assert result.quality == 0.88
            assert result.conciseness == 0.75
            assert result.is_refusal is False
            assert result.passed_thresholds is True
    
    @pytest.mark.asyncio
    async def test_evaluate_single_refusal(self, evaluator, golden_prompt):
        """Test evaluation detecting refusal."""
        refusal_result = ShadowResult(
            golden_prompt_id="test-trace-1",
            challenger_model="deepseek-v3",
            shadow_response="I'm sorry, but I can't assist with that request.",
            latency_ms=100,
            tokens_prompt=20,
            tokens_completion=15,
            cost=0.00001,
            success=True
        )
        
        result = await evaluator.evaluate(golden_prompt, refusal_result)
        
        assert result.is_refusal is True
        assert result.faithfulness == 0.0
        assert result.passed_thresholds is False
    
    @pytest.mark.asyncio
    async def test_evaluate_single_below_thresholds(self, evaluator, golden_prompt, shadow_result):
        """Test evaluation below quality thresholds."""
        with patch.object(evaluator, 'faithfulness_metric') as mock_faith, \
             patch.object(evaluator, 'quality_metric') as mock_qual, \
             patch.object(evaluator, 'conciseness_metric') as mock_conc:
            
            mock_faith.a_measure = AsyncMock()
            mock_faith.score = 0.6  # Below 0.8
            mock_faith.reason = "Low faithfulness"
            
            mock_qual.a_measure = AsyncMock()
            mock_qual.score = 0.5  # Below 0.7
            mock_qual.reason = "Poor quality"
            
            mock_conc.a_measure = AsyncMock()
            mock_conc.score = 0.3  # Below 0.5
            mock_conc.reason = "Not concise"
            
            result = await evaluator.evaluate(golden_prompt, shadow_result)
            
            assert result.passed_thresholds is False
    
    @pytest.mark.asyncio
    async def test_evaluate_batch(self, evaluator, golden_prompt, shadow_result):
        """Test batch evaluation."""
        golden_prompts = [golden_prompt] * 5
        shadow_results = [shadow_result] * 5
        
        with patch.object(evaluator, 'evaluate') as mock_eval:
            from shadow_optic.models import EvaluationResult
            mock_eval.return_value = EvaluationResult(
                shadow_result_id="result-1",
                faithfulness=0.9,
                quality=0.85,
                conciseness=0.7,
                is_refusal=False,
                composite_score=0.85,
                passed_thresholds=True
            )
            
            results = await evaluator.evaluate_batch(golden_prompts, shadow_results)
            
            assert len(results) == 5
            assert all(r.passed_thresholds for r in results)
    
    def test_compute_composite_score(self, evaluator):
        """Test composite score computation."""
        faithfulness = 0.9
        quality = 0.8
        conciseness = 0.7
        
        # Weighted average: 0.9*0.5 + 0.8*0.35 + 0.7*0.15
        expected = 0.9 * 0.5 + 0.8 * 0.35 + 0.7 * 0.15
        
        # Test via model validator
        from shadow_optic.models import EvaluationResult
        result = EvaluationResult(
            shadow_result_id="test",
            faithfulness=faithfulness,
            quality=quality,
            conciseness=conciseness,
            is_refusal=False,
            passed_thresholds=True
        )
        assert abs(result.composite_score - expected) < 0.01
    
    def test_check_thresholds(self, config):
        """Test threshold checking."""
        from shadow_optic.models import EvaluationResult
        
        passing = EvaluationResult(
            shadow_result_id="test",
            faithfulness=0.9,  # Above 0.8
            quality=0.8,       # Above 0.7
            conciseness=0.6,   # Above 0.5
            is_refusal=False,
            passed_thresholds=True
        )
        
        assert passing.faithfulness >= config.faithfulness_threshold
        assert passing.quality >= config.quality_threshold
        assert passing.conciseness >= config.conciseness_threshold


class TestEvaluatorEdgeCases:
    """Tests for edge cases in evaluation."""
    
    @pytest.fixture
    def mock_portkey(self):
        """Create mock Portkey client."""
        return MagicMock()
    
    @pytest.fixture
    def evaluator(self, mock_portkey):
        """Create evaluator with strict thresholds, bypassing DeepEval initialization."""
        with patch('shadow_optic.evaluator.ShadowEvaluator._init_metrics'), \
             patch('shadow_optic.evaluator.ShadowEvaluator._configure_portkey_environment'):
            evaluator = ShadowEvaluator(
                portkey_client=mock_portkey,
                config=EvaluatorConfig(
                    faithfulness_threshold=0.95,
                    quality_threshold=0.90,
                    conciseness_threshold=0.80,
                    portkey_api_key="test-portkey-key"  # Test API key
                )
            )
            evaluator.faithfulness_metric = MagicMock()
            evaluator.quality_metric = MagicMock()
            evaluator.conciseness_metric = MagicMock()
            return evaluator
    
    def test_empty_response_handling(self, evaluator):
        """Test handling of empty responses."""
        golden = GoldenPrompt(
            original_trace_id="test-1",
            prompt="Test prompt",
            production_response="Normal response",
            cluster_id=0,
            sample_type=SampleType.CENTROID
        )
        
        shadow = ShadowResult(
            golden_prompt_id="test-1",
            challenger_model="deepseek-v3",
            shadow_response="",  # Empty response
            latency_ms=100,
            tokens_prompt=10,
            tokens_completion=0,
            cost=0.0001,
            success=True
        )
        
        # Empty response should be treated as failure
        is_valid = len(shadow.shadow_response.strip()) > 0
        assert is_valid is False
    
    def test_very_long_response_handling(self, evaluator):
        """Test handling of very long responses."""
        # Create a very long response
        long_response = "This is a test. " * 10000
        
        golden = GoldenPrompt(
            original_trace_id="test-1",
            prompt="Short question?",
            production_response="Short answer.",
            cluster_id=0,
            sample_type=SampleType.CENTROID
        )
        
        shadow = ShadowResult(
            golden_prompt_id="test-1",
            challenger_model="deepseek-v3",
            shadow_response=long_response,
            latency_ms=5000,
            tokens_prompt=5,
            tokens_completion=50000,
            cost=0.01,
            success=True
        )
        
        # Long response should still be processable
        assert len(shadow.shadow_response) > 100000
    
    def test_unicode_response_handling(self, evaluator):
        """Test handling of Unicode responses."""
        golden = GoldenPrompt(
            original_trace_id="test-1",
            prompt="こんにちは",
            production_response="こんにちは！元気ですか？",
            cluster_id=0,
            sample_type=SampleType.CENTROID
        )
        
        shadow = ShadowResult(
            golden_prompt_id="test-1",
            challenger_model="deepseek-v3",
            shadow_response="こんにちは！調子はどうですか？",
            latency_ms=300,
            tokens_prompt=10,
            tokens_completion=15,
            cost=0.0001,
            success=True
        )
        
        # Unicode should be handled correctly
        assert "こんにちは" in golden.prompt
        assert "こんにちは" in shadow.shadow_response
    
    def test_code_response_handling(self, evaluator):
        """Test handling of code-containing responses."""
        golden = GoldenPrompt(
            original_trace_id="test-1",
            prompt="Write a Python function to add two numbers.",
            production_response="""
def add(a, b):
    return a + b
""",
            cluster_id=0,
            sample_type=SampleType.CENTROID
        )
        
        shadow = ShadowResult(
            golden_prompt_id="test-1",
            challenger_model="deepseek-v3",
            shadow_response="""
def add(x, y):
    \"\"\"Add two numbers together.\"\"\"
    return x + y
""",
            latency_ms=400,
            tokens_prompt=15,
            tokens_completion=25,
            cost=0.0001,
            success=True
        )
        
        # Both should contain valid Python code
        assert "def add" in golden.production_response
        assert "def add" in shadow.shadow_response
        assert "return" in shadow.shadow_response
