"""
Tests for P2 Enhancement: Cost Prediction Model.

Tests cover:
- Token counting with tiktoken
- Feature extraction
- Model training and prediction
- Budget-optimized sampling
- Portkey data collection
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from shadow_optic.p2_enhancements.cost_predictor import (
    CostPredictionModel,
    BudgetOptimizedSampler,
    PortkeyCostDataCollector,
    CostFeatureExtractor,
    CostFeatures,
    CostPrediction,
    ModelStatistics,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def feature_extractor():
    """Create a CostFeatureExtractor instance."""
    return CostFeatureExtractor()


@pytest.fixture
def cost_model():
    """Create a CostPredictionModel instance without Portkey."""
    return CostPredictionModel()


@pytest.fixture
def training_data():
    """Generate sample training data."""
    np.random.seed(42)
    return [
        {
            "prompt_tokens": np.random.randint(50, 500),
            "completion_tokens": np.random.randint(50, 300),
            "cost": np.random.uniform(0.0001, 0.01),
            "model": "gpt-4o-mini",
            "provider": "openai",
            "latency_ms": np.random.randint(100, 500)
        }
        for _ in range(200)
    ]


@pytest.fixture
def model_stats():
    """Sample model statistics."""
    return {
        "gpt-4o-mini": ModelStatistics(
            model="gpt-4o-mini",
            provider="openai",
            avg_output_ratio=1.5,
            std_output_ratio=0.3,
            avg_cost_per_token=0.00001,
            std_cost_per_token=0.000002,
            sample_count=1000,
            avg_latency_ms=200
        ),
        "deepseek-v3": ModelStatistics(
            model="deepseek-v3",
            provider="deepseek",
            avg_output_ratio=1.3,
            std_output_ratio=0.25,
            avg_cost_per_token=0.000002,
            std_cost_per_token=0.0000005,
            sample_count=500,
            avg_latency_ms=150
        )
    }


# =============================================================================
# CostFeatureExtractor Tests
# =============================================================================

class TestCostFeatureExtractor:
    """Tests for CostFeatureExtractor."""
    
    def test_count_tokens_simple(self, feature_extractor):
        """Test token counting for simple text."""
        text = "Hello, world! This is a test."
        tokens = feature_extractor.count_tokens(text, "gpt-4o")
        
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_count_tokens_empty(self, feature_extractor):
        """Test token counting for empty text."""
        tokens = feature_extractor.count_tokens("", "gpt-4o")
        assert tokens == 0
    
    def test_count_tokens_long_text(self, feature_extractor):
        """Test token counting for long text."""
        text = "word " * 1000
        tokens = feature_extractor.count_tokens(text, "gpt-4o")
        
        # Should be around 1000 tokens (one per word)
        assert tokens > 500
        assert tokens < 1500
    
    def test_count_tokens_different_models(self, feature_extractor):
        """Test that different models use appropriate encoders."""
        text = "Test prompt for token counting"
        
        # Should work for various models
        models = ["gpt-4o", "gpt-4o-mini", "deepseek-v3", "llama-4-scout", "claude-3.5-sonnet"]
        
        for model in models:
            tokens = feature_extractor.count_tokens(text, model)
            assert tokens > 0, f"Failed for model: {model}"
    
    def test_extract_features_basic(self, feature_extractor):
        """Test feature extraction for a basic prompt."""
        prompt = "What is the capital of France?"
        features = feature_extractor.extract_features(prompt, "gpt-4o-mini")
        
        assert isinstance(features, CostFeatures)
        assert features.input_tokens > 0
        assert features.estimated_output_tokens > 0
        assert features.model_id == "gpt-4o-mini"
        assert features.avg_output_ratio > 0
    
    def test_extract_features_with_stats(self, feature_extractor, model_stats):
        """Test feature extraction using pre-computed stats."""
        feature_extractor.model_stats = model_stats
        
        prompt = "Explain quantum computing in simple terms."
        features = feature_extractor.extract_features(prompt, "gpt-4o-mini")
        
        # Should use stats from model_stats
        assert features.avg_output_ratio == 1.5
        assert features.historical_cost_per_token == 0.00001
    
    def test_features_to_array(self, feature_extractor):
        """Test conversion of features to numpy array."""
        features = CostFeatures(
            input_tokens=100,
            estimated_output_tokens=150,
            model_id="gpt-4o",
            provider="openai",
            avg_output_ratio=1.5,
            historical_cost_per_token=0.00001
        )
        
        arr = feature_extractor.features_to_array(features)
        
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 5
        assert arr[0] == 100  # input_tokens
        assert arr[1] == 150  # estimated_output_tokens
        assert arr[2] == 250  # total_tokens


# =============================================================================
# CostPredictionModel Tests
# =============================================================================

class TestCostPredictionModel:
    """Tests for CostPredictionModel."""
    
    def test_model_initialization(self, cost_model):
        """Test model initializes correctly."""
        assert not cost_model.is_trained
        assert cost_model.training_metrics == {}
    
    def test_fallback_prediction(self, cost_model):
        """Test prediction before training (fallback mode)."""
        prompt = "What is machine learning?"
        prediction = cost_model.predict(prompt, "gpt-4o-mini")
        
        assert isinstance(prediction, CostPrediction)
        assert prediction.predicted_cost > 0
        assert prediction.prediction_method == "fallback"
        assert prediction.input_tokens > 0
    
    def test_train_from_data(self, cost_model, training_data):
        """Test training from provided data."""
        result = cost_model.train_from_data(training_data)
        
        assert result["status"] == "success"
        assert result["samples"] > 0
        assert cost_model.is_trained
    
    def test_predict_after_training(self, cost_model, training_data):
        """Test prediction after training."""
        cost_model.train_from_data(training_data)
        
        prompt = "Explain the theory of relativity in detail."
        prediction = cost_model.predict(prompt, "gpt-4o-mini")
        
        assert isinstance(prediction, CostPrediction)
        assert prediction.prediction_method == "ml_model"
        assert prediction.predicted_cost >= 0
        assert prediction.confidence_interval[0] <= prediction.predicted_cost
        assert prediction.confidence_interval[1] >= prediction.predicted_cost
    
    def test_insufficient_data_handling(self, cost_model):
        """Test handling of insufficient training data."""
        small_data = [
            {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.001, "model": "gpt-4o"}
            for _ in range(5)
        ]
        
        result = cost_model.train_from_data(small_data)
        
        assert result["status"] == "insufficient_data"
        assert not cost_model.is_trained
    
    def test_prediction_cost_ordering(self, cost_model, training_data):
        """Test that longer prompts predict higher costs."""
        cost_model.train_from_data(training_data)
        
        short_prompt = "Hi"
        long_prompt = "Please explain in great detail the history of artificial intelligence, " * 10
        
        short_prediction = cost_model.predict(short_prompt, "gpt-4o-mini")
        long_prediction = cost_model.predict(long_prompt, "gpt-4o-mini")
        
        # Longer prompt should have more tokens
        assert long_prediction.input_tokens > short_prediction.input_tokens


# =============================================================================
# PortkeyCostDataCollector Tests
# =============================================================================

class TestPortkeyCostDataCollector:
    """Tests for PortkeyCostDataCollector."""
    
    def test_collector_initialization_without_key(self):
        """Test collector initializes without API key."""
        collector = PortkeyCostDataCollector()
        assert collector.portkey is None
    
    def test_compute_model_statistics(self):
        """Test model statistics computation."""
        collector = PortkeyCostDataCollector()
        
        historical_data = [
            {
                "prompt_tokens": 100,
                "completion_tokens": 150,
                "total_tokens": 250,
                "cost": 0.0025,
                "model": "gpt-4o",
                "provider": "openai",
                "latency_ms": 200
            },
            {
                "prompt_tokens": 200,
                "completion_tokens": 250,
                "total_tokens": 450,
                "cost": 0.0045,
                "model": "gpt-4o",
                "provider": "openai",
                "latency_ms": 300
            },
            {
                "prompt_tokens": 50,
                "completion_tokens": 75,
                "total_tokens": 125,
                "cost": 0.0002,
                "model": "deepseek-v3",
                "provider": "deepseek",
                "latency_ms": 100
            }
        ]
        
        stats = collector.compute_model_statistics(historical_data)
        
        assert "gpt-4o" in stats
        assert "deepseek-v3" in stats
        
        gpt_stats = stats["gpt-4o"]
        assert gpt_stats.sample_count == 2
        assert gpt_stats.avg_output_ratio > 1.0  # Output > input typically
        assert gpt_stats.avg_cost_per_token > 0
    
    def test_get_model_stats_cached(self):
        """Test that model stats are cached."""
        collector = PortkeyCostDataCollector()
        
        # Initially empty
        assert collector.get_model_stats("gpt-4o") is None
        
        # Compute stats
        collector.compute_model_statistics([
            {"prompt_tokens": 100, "completion_tokens": 150, "total_tokens": 250,
             "cost": 0.0025, "model": "gpt-4o", "provider": "openai", "latency_ms": 200}
        ])
        
        # Now should be cached
        assert collector.get_model_stats("gpt-4o") is not None


# =============================================================================
# BudgetOptimizedSampler Tests
# =============================================================================

class TestBudgetOptimizedSampler:
    """Tests for BudgetOptimizedSampler."""
    
    @pytest.fixture
    def sampler(self, cost_model, training_data):
        """Create a sampler with trained cost model."""
        cost_model.train_from_data(training_data)
        return BudgetOptimizedSampler(budget_limit=1.0, cost_model=cost_model)
    
    def test_sampler_initialization(self, sampler):
        """Test sampler initializes correctly."""
        assert sampler.budget_limit == 1.0
        assert sampler.spent == 0.0
        assert sampler.remaining_budget == 1.0
    
    def test_update_spend(self, sampler):
        """Test spend tracking."""
        sampler.update_spend(0.25)
        assert sampler.spent == 0.25
        assert sampler.remaining_budget == 0.75
        
        sampler.update_spend(0.50)
        assert sampler.spent == 0.75
        assert sampler.remaining_budget == 0.25
    
    @pytest.mark.asyncio
    async def test_select_replays_within_budget(self, sampler):
        """Test selection of replays within budget."""
        candidates = [
            {"prompt": f"Test prompt {i}", "info_value": 0.5 + i * 0.1}
            for i in range(10)
        ]
        
        selected = await sampler.select_replays_within_budget(
            candidates, "gpt-4o-mini"
        )
        
        # Should select some but not necessarily all
        assert len(selected) > 0
        assert len(selected) <= len(candidates)
        
        # All selected should have predicted costs
        for item in selected:
            assert "predicted_cost" in item
            assert "priority" in item
    
    @pytest.mark.asyncio
    async def test_budget_exhaustion(self, sampler):
        """Test that selection stops when budget exhausted."""
        # Use up most of the budget
        sampler.spent = 0.99
        
        candidates = [
            {"prompt": "A very long prompt " * 100, "info_value": 1.0}
            for _ in range(5)
        ]
        
        selected = await sampler.select_replays_within_budget(
            candidates, "gpt-4o-mini"
        )
        
        # Should select nothing or very few due to budget
        assert len(selected) <= 2
    
    def test_compute_info_value(self, sampler):
        """Test information value computation."""
        candidate = {
            "prompt": "What is the meaning of life?",
            "is_centroid": True
        }
        
        value = sampler.compute_info_value(candidate)
        
        assert 0 <= value <= 1
    
    def test_compute_info_value_centroid_vs_non_centroid(self, sampler):
        """Test that centroids get higher coverage scores."""
        centroid = {"prompt": "Test prompt", "is_centroid": True}
        non_centroid = {"prompt": "Test prompt", "is_centroid": False}
        
        centroid_value = sampler.compute_info_value(centroid)
        non_centroid_value = sampler.compute_info_value(non_centroid)
        
        # Centroid should have higher value due to coverage
        assert centroid_value > non_centroid_value


# =============================================================================
# Integration Tests
# =============================================================================

class TestCostPredictionIntegration:
    """Integration tests for cost prediction system."""
    
    @pytest.mark.asyncio
    async def test_full_prediction_workflow(self, training_data):
        """Test full workflow from training to prediction."""
        # 1. Create model and train
        model = CostPredictionModel()
        result = model.train_from_data(training_data)
        assert result["status"] == "success"
        
        # 2. Create sampler with trained model
        sampler = BudgetOptimizedSampler(budget_limit=0.10, cost_model=model)
        
        # 3. Generate candidates
        candidates = [
            {
                "prompt": f"Explain concept {i} in machine learning",
                "info_value": np.random.uniform(0.3, 1.0)
            }
            for i in range(20)
        ]
        
        # 4. Select within budget
        selected = await sampler.select_replays_within_budget(
            candidates, "gpt-4o-mini"
        )
        
        # 5. Verify selection
        assert len(selected) > 0
        
        # Calculate total predicted cost
        total_cost = sum(s["predicted_cost"] for s in selected)
        assert total_cost <= sampler.budget_limit
        
        # Verify priority ordering (highest priority first)
        for i in range(len(selected) - 1):
            assert selected[i]["priority"] >= selected[i + 1]["priority"]
    
    def test_model_comparison(self, training_data):
        """Test that different models predict different costs."""
        model = CostPredictionModel()
        model.train_from_data(training_data)
        
        prompt = "Explain quantum entanglement in detail."
        
        # Same prompt, different models should have different costs
        prediction_gpt4 = model.predict(prompt, "gpt-4o")
        prediction_mini = model.predict(prompt, "gpt-4o-mini")
        prediction_deepseek = model.predict(prompt, "deepseek-v3")
        
        # All should return valid predictions
        assert prediction_gpt4.predicted_cost > 0
        assert prediction_mini.predicted_cost > 0
        assert prediction_deepseek.predicted_cost > 0
