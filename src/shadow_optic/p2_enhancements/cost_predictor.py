"""
Cost Prediction Model for Shadow-Optic P2 Enhancements.

Portkey-First Architecture:
- Uses Portkey Analytics API for historical cost data
- Uses tiktoken for pre-request token estimation
- Uses scikit-learn for ML model training (GradientBoostingRegressor)

This enables intelligent budget allocation by predicting replay costs
before execution, maximizing information value within budget constraints.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import tiktoken
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from portkey_ai import Portkey, AsyncPortkey

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CostFeatures:
    """Features for cost prediction model - extracted from Portkey analytics."""
    
    # Token-based features (pre-computed with tiktoken)
    input_tokens: int
    estimated_output_tokens: int
    
    # Model-specific features (from Portkey Model Catalog)
    model_id: str
    provider: str
    
    # Historical features (from Portkey Analytics API)
    avg_output_ratio: float = 1.5  # Historical output_tokens / input_tokens
    historical_cost_per_token: float = 0.00001  # From Portkey cost tracking


@dataclass
class CostPrediction:
    """Result of a cost prediction."""
    predicted_cost: float
    confidence_interval: Tuple[float, float]
    input_tokens: int
    estimated_output_tokens: int
    model: str
    prediction_method: str  # "ml_model" or "fallback"
    
    @property
    def estimated_total_tokens(self) -> int:
        return self.input_tokens + self.estimated_output_tokens


@dataclass
class ModelStatistics:
    """Per-model statistics computed from Portkey data."""
    model: str
    provider: str
    avg_output_ratio: float
    std_output_ratio: float
    avg_cost_per_token: float
    std_cost_per_token: float
    sample_count: int
    avg_latency_ms: float
    
    @property
    def is_reliable(self) -> bool:
        """Check if we have enough samples for reliable predictions."""
        return self.sample_count >= 100


# =============================================================================
# Portkey Data Collector
# =============================================================================

class PortkeyCostDataCollector:
    """Collect training data from Portkey's Analytics API.
    
    Portkey automatically tracks for each request:
    - tokens (input + output)
    - cost (in USD)
    - model used
    - latency
    - metadata
    """
    
    def __init__(
        self, 
        portkey_api_key: Optional[str] = None,
        base_url: str = "https://api.portkey.ai/v1"
    ):
        self.portkey = AsyncPortkey(
            api_key=portkey_api_key,
            base_url=base_url
        ) if portkey_api_key else None
        self._model_stats_cache: Dict[str, ModelStatistics] = {}
    
    async def fetch_historical_costs(
        self,
        days: int = 30,
        metadata_filter: Optional[Dict[str, Any]] = None,
        limit: int = 10000
    ) -> List[Dict[str, Any]]:
        """Fetch historical cost data from Portkey logs.
        
        This data is automatically tracked by Portkey for every request.
        
        Args:
            days: Number of days of historical data to fetch
            metadata_filter: Optional filter for specific experiments/users
            limit: Maximum number of records to fetch
            
        Returns:
            List of cost records with tokens, cost, model, latency, etc.
        """
        if not self.portkey:
            logger.warning("No Portkey API key configured, returning empty data")
            return []
        
        try:
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Fetch logs from Portkey API
            # Note: The actual API structure may vary based on Portkey's SDK version
            logs = []
            
            # In production, this would use Portkey's logs API:
            # logs = await self.portkey.logs.list(
            #     start_date=start_date.isoformat(),
            #     end_date=end_date.isoformat(),
            #     limit=limit,
            #     filter=metadata_filter
            # )
            
            # For now, we'll implement a simulation mode for testing
            logger.info(f"Fetching {days} days of historical data (limit: {limit})")
            
            return [
                {
                    "prompt_tokens": log.get("request", {}).get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": log.get("request", {}).get("usage", {}).get("completion_tokens", 0),
                    "total_tokens": log.get("request", {}).get("usage", {}).get("total_tokens", 0),
                    "cost": log.get("cost"),
                    "model": log.get("model"),
                    "provider": log.get("provider"),
                    "latency_ms": log.get("latency"),
                    "metadata": log.get("metadata", {}),
                    "timestamp": log.get("timestamp")
                }
                for log in logs
                if log.get("cost") is not None
            ]
            
        except Exception as e:
            logger.error(f"Error fetching Portkey data: {e}")
            return []
    
    def compute_model_statistics(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, ModelStatistics]:
        """Compute per-model statistics from Portkey data.
        
        Args:
            historical_data: List of historical cost records
            
        Returns:
            Dict mapping model name to ModelStatistics
        """
        stats: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {"costs": [], "ratios": [], "latencies": [], "provider": None}
        )
        
        for record in historical_data:
            model = record.get("model")
            if not model:
                continue
                
            prompt_tokens = record.get("prompt_tokens", 0)
            completion_tokens = record.get("completion_tokens", 0)
            total_tokens = record.get("total_tokens", prompt_tokens + completion_tokens)
            cost = record.get("cost", 0)
            
            if prompt_tokens > 0 and total_tokens > 0 and cost:
                ratio = completion_tokens / prompt_tokens if prompt_tokens > 0 else 1.0
                cost_per_token = cost / total_tokens
                
                stats[model]["ratios"].append(ratio)
                stats[model]["costs"].append(cost_per_token)
                stats[model]["latencies"].append(record.get("latency_ms", 0))
                stats[model]["provider"] = record.get("provider", "unknown")
        
        result = {}
        for model, data in stats.items():
            if len(data["costs"]) > 0:
                result[model] = ModelStatistics(
                    model=model,
                    provider=data["provider"] or "unknown",
                    avg_output_ratio=float(np.mean(data["ratios"])),
                    std_output_ratio=float(np.std(data["ratios"])),
                    avg_cost_per_token=float(np.mean(data["costs"])),
                    std_cost_per_token=float(np.std(data["costs"])),
                    sample_count=len(data["costs"]),
                    avg_latency_ms=float(np.mean(data["latencies"]))
                )
        
        self._model_stats_cache = result
        return result
    
    def get_model_stats(self, model: str) -> Optional[ModelStatistics]:
        """Get cached statistics for a specific model."""
        return self._model_stats_cache.get(model)


# =============================================================================
# Feature Extractor
# =============================================================================

class CostFeatureExtractor:
    """Extract features using tiktoken for pre-request estimation.
    
    Uses tiktoken for token counting before the request hits Portkey,
    enabling accurate cost prediction.
    """
    
    # Default model pricing (fallback when Portkey data unavailable)
    DEFAULT_PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "deepseek-v3": {"input": 0.00014, "output": 0.00028},
        "deepseek-chat": {"input": 0.00014, "output": 0.00028},
        "llama-4-scout": {"input": 0.0001, "output": 0.0002},
        "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    def __init__(self, model_stats: Optional[Dict[str, ModelStatistics]] = None):
        self.encoders: Dict[str, tiktoken.Encoding] = {}
        self.model_stats = model_stats or {}
        self.label_encoder = LabelEncoder()
    
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create encoder for model.
        
        Uses the appropriate tokenizer based on model family.
        """
        if model not in self.encoders:
            try:
                # Try to get model-specific encoding
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback encodings based on model family
                if "gpt-4" in model.lower() or "gpt-3" in model.lower():
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
                elif "deepseek" in model.lower():
                    # DeepSeek uses similar tokenization to GPT-4
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
                elif "llama" in model.lower():
                    # Llama uses a different tokenizer, but cl100k is a reasonable approximation
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
                elif "claude" in model.lower():
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
                else:
                    # Universal fallback
                    self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens in text for a specific model.
        
        Args:
            text: Input text to tokenize
            model: Model name for tokenizer selection
            
        Returns:
            Number of tokens
        """
        enc = self._get_encoder(model)
        return len(enc.encode(text))
    
    def extract_features(
        self,
        prompt: str,
        model: str,
        provider: str = "openai"
    ) -> CostFeatures:
        """Extract features for cost prediction.
        
        Args:
            prompt: The prompt text
            model: Target model name
            provider: Model provider
            
        Returns:
            CostFeatures with extracted values
        """
        enc = self._get_encoder(model)
        input_tokens = len(enc.encode(prompt))
        
        # Use historical stats from Portkey data if available
        stats = self.model_stats.get(model)
        
        if stats:
            avg_ratio = stats.avg_output_ratio
            cost_per_token = stats.avg_cost_per_token
        else:
            # Fallback to default estimates
            avg_ratio = 1.5  # Typical output/input ratio
            pricing = self.DEFAULT_PRICING.get(model, {"input": 0.001, "output": 0.002})
            # Average of input and output pricing
            cost_per_token = (pricing["input"] + pricing["output"]) / 2 / 1000
        
        return CostFeatures(
            input_tokens=input_tokens,
            estimated_output_tokens=int(input_tokens * avg_ratio),
            model_id=model,
            provider=provider,
            avg_output_ratio=avg_ratio,
            historical_cost_per_token=cost_per_token
        )
    
    def features_to_array(self, features: CostFeatures) -> np.ndarray:
        """Convert CostFeatures to numpy array for ML model."""
        return np.array([
            features.input_tokens,
            features.estimated_output_tokens,
            features.input_tokens + features.estimated_output_tokens,
            features.avg_output_ratio,
            features.historical_cost_per_token
        ])


# =============================================================================
# Cost Prediction Model
# =============================================================================

class CostPredictionModel:
    """ML model trained on Portkey analytics data.
    
    Uses GradientBoostingRegressor for accurate cost predictions
    based on historical Portkey data.
    """
    
    def __init__(
        self,
        portkey_api_key: Optional[str] = None,
        model_path: Optional[str] = None
    ):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.data_collector = PortkeyCostDataCollector(portkey_api_key)
        self.feature_extractor = CostFeatureExtractor()
        self.is_trained = False
        self.training_metrics: Dict[str, Any] = {}
        self.model_path = model_path
        
        # Load pre-trained model if path provided
        if model_path:
            self.load(model_path)
    
    async def train_from_portkey(
        self,
        days: int = 30,
        min_samples: int = 100
    ) -> Dict[str, Any]:
        """Train model using Portkey's historical data.
        
        Portkey automatically tracks costs - we just need to fetch and train!
        
        Args:
            days: Number of days of historical data
            min_samples: Minimum samples required to train
            
        Returns:
            Training metrics dict
        """
        logger.info(f"Training cost prediction model from {days} days of Portkey data")
        
        # Fetch historical data from Portkey
        historical_data = await self.data_collector.fetch_historical_costs(days=days)
        
        if len(historical_data) < min_samples:
            logger.warning(
                f"Insufficient data for training: {len(historical_data)} < {min_samples}"
            )
            return {
                "status": "insufficient_data",
                "samples": len(historical_data),
                "min_required": min_samples
            }
        
        # Compute model statistics
        model_stats = self.data_collector.compute_model_statistics(historical_data)
        self.feature_extractor.model_stats = model_stats
        
        # Prepare training data
        X, y = [], []
        for record in historical_data:
            if all(key in record for key in ["prompt_tokens", "completion_tokens", "cost"]):
                model = record.get("model", "unknown")
                stats = model_stats.get(model, ModelStatistics(
                    model=model,
                    provider="unknown",
                    avg_output_ratio=1.5,
                    std_output_ratio=0.5,
                    avg_cost_per_token=0.00001,
                    std_cost_per_token=0.000005,
                    sample_count=1,
                    avg_latency_ms=500
                ))
                
                X.append([
                    record["prompt_tokens"],
                    record["completion_tokens"],
                    record["prompt_tokens"] + record["completion_tokens"],
                    stats.avg_output_ratio,
                    stats.avg_cost_per_token
                ])
                y.append(record["cost"])
        
        if len(X) < min_samples:
            return {
                "status": "insufficient_valid_data",
                "samples": len(X),
                "min_required": min_samples
            }
        
        X = np.array(X)
        y = np.array(y)
        
        # Train with cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=min(5, len(X) // 20 + 1),  # Adjust CV folds based on data size
            scoring='neg_mean_absolute_error'
        )
        
        self.model.fit(X, y)
        self.is_trained = True
        
        self.training_metrics = {
            "status": "success",
            "samples": len(X),
            "cv_mae": float(-cv_scores.mean()),
            "cv_mae_std": float(cv_scores.std()),
            "models_tracked": len(model_stats),
            "feature_importances": dict(zip(
                ["input_tokens", "output_tokens", "total_tokens", "output_ratio", "cost_per_token"],
                self.model.feature_importances_.tolist()
            ))
        }
        
        logger.info(f"Model trained successfully: {self.training_metrics}")
        return self.training_metrics
    
    def train_from_data(
        self,
        training_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train model from provided data (for testing).
        
        Args:
            training_data: List of dicts with prompt_tokens, completion_tokens, cost, model
            
        Returns:
            Training metrics dict
        """
        if len(training_data) < 10:
            return {"status": "insufficient_data", "samples": len(training_data)}
        
        # Compute model statistics
        model_stats = self.data_collector.compute_model_statistics(training_data)
        self.feature_extractor.model_stats = model_stats
        
        # Prepare training data
        X, y = [], []
        for record in training_data:
            model = record.get("model", "unknown")
            stats = model_stats.get(model)
            
            if stats is None:
                continue
            
            X.append([
                record["prompt_tokens"],
                record["completion_tokens"],
                record["prompt_tokens"] + record["completion_tokens"],
                stats.avg_output_ratio,
                stats.avg_cost_per_token
            ])
            y.append(record["cost"])
        
        X = np.array(X)
        y = np.array(y)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        return {"status": "success", "samples": len(X)}
    
    def predict(
        self,
        prompt: str,
        model: str,
        provider: str = "openai"
    ) -> CostPrediction:
        """Predict cost for a replay.
        
        Args:
            prompt: The prompt text
            model: Target model name
            provider: Model provider
            
        Returns:
            CostPrediction with predicted cost and confidence
        """
        features = self.feature_extractor.extract_features(prompt, model, provider)
        
        if not self.is_trained:
            return self._fallback_estimate(features)
        
        X = self.feature_extractor.features_to_array(features).reshape(1, -1)
        predicted_cost = float(self.model.predict(X)[0])
        
        # Estimate confidence interval using training error
        mae = self.training_metrics.get("cv_mae", predicted_cost * 0.1)
        
        return CostPrediction(
            predicted_cost=max(0, predicted_cost),
            confidence_interval=(
                max(0, predicted_cost - 2 * mae),
                predicted_cost + 2 * mae
            ),
            input_tokens=features.input_tokens,
            estimated_output_tokens=features.estimated_output_tokens,
            model=model,
            prediction_method="ml_model"
        )
    
    def _fallback_estimate(self, features: CostFeatures) -> CostPrediction:
        """Simple fallback using default pricing data.
        
        Used when ML model is not trained.
        """
        total_tokens = features.input_tokens + features.estimated_output_tokens
        estimated_cost = total_tokens * features.historical_cost_per_token
        
        return CostPrediction(
            predicted_cost=estimated_cost,
            confidence_interval=(estimated_cost * 0.5, estimated_cost * 2.0),
            input_tokens=features.input_tokens,
            estimated_output_tokens=features.estimated_output_tokens,
            model=features.model_id,
            prediction_method="fallback"
        )
    
    def save(self, path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump({
            "model": self.model,
            "feature_extractor_stats": self.feature_extractor.model_stats,
            "training_metrics": self.training_metrics,
            "version": "1.0.0"
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load trained model from disk."""
        try:
            data = joblib.load(path)
            self.model = data["model"]
            self.feature_extractor.model_stats = data.get("feature_extractor_stats", {})
            self.training_metrics = data.get("training_metrics", {})
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
            self.is_trained = False


# =============================================================================
# Budget-Optimized Sampler
# =============================================================================

class BudgetOptimizedSampler:
    """Use Portkey's budget tracking + ML prediction for optimal sampling.
    
    Integrates with Portkey's Budget Limits feature and uses ML predictions
    to maximize information value within budget constraints.
    """
    
    def __init__(
        self,
        portkey_api_key: Optional[str] = None,
        budget_limit: float = 10.0,
        cost_model: Optional[CostPredictionModel] = None
    ):
        self.portkey = AsyncPortkey(api_key=portkey_api_key) if portkey_api_key else None
        self.cost_model = cost_model or CostPredictionModel(portkey_api_key)
        self.budget_limit = budget_limit
        self.spent = 0.0
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget."""
        return max(0, self.budget_limit - self.spent)
    
    async def get_current_spend(self, provider_slug: Optional[str] = None) -> float:
        """Get current spend from Portkey's budget tracking.
        
        Portkey tracks this automatically via Budget Limits feature.
        """
        if not self.portkey:
            return self.spent
        
        try:
            # In production, use Portkey's analytics API
            # analytics = await self.portkey.analytics.get(
            #     metrics=["cost"],
            #     group_by="provider",
            #     filter={"provider": provider_slug} if provider_slug else None
            # )
            # return analytics.total_cost
            return self.spent
        except Exception as e:
            logger.warning(f"Failed to get spend from Portkey: {e}")
            return self.spent
    
    def update_spend(self, cost: float):
        """Update local spend tracking."""
        self.spent += cost
    
    async def select_replays_within_budget(
        self,
        candidates: List[Dict[str, Any]],
        model: str,
        info_value_key: str = "info_value"
    ) -> List[Dict[str, Any]]:
        """Select replays that fit within remaining budget.
        
        Uses a greedy algorithm optimizing for information value per cost.
        
        Args:
            candidates: List of candidate replays with prompts and info values
            model: Target model for predictions
            info_value_key: Key in candidate dict for information value score
            
        Returns:
            Selected candidates that fit within budget
        """
        remaining = self.remaining_budget
        
        if remaining <= 0:
            logger.warning("No remaining budget for replays")
            return []
        
        # Score and sort by information value / cost
        scored = []
        for candidate in candidates:
            prompt = candidate.get("prompt", "")
            prediction = self.cost_model.predict(prompt, model)
            
            info_value = candidate.get(info_value_key, 1.0)
            priority = info_value / max(prediction.predicted_cost, 0.0001)
            
            scored.append({
                **candidate,
                "predicted_cost": prediction.predicted_cost,
                "priority": priority,
                "prediction": prediction
            })
        
        # Sort by priority (highest first)
        scored.sort(key=lambda x: x["priority"], reverse=True)
        
        # Greedy selection within budget
        selected = []
        total = 0.0
        
        for item in scored:
            if total + item["predicted_cost"] <= remaining:
                selected.append(item)
                total += item["predicted_cost"]
        
        logger.info(
            f"Selected {len(selected)}/{len(candidates)} candidates "
            f"within budget (${total:.4f}/${remaining:.4f})"
        )
        
        return selected
    
    def compute_info_value(
        self,
        candidate: Dict[str, Any],
        novelty_weight: float = 0.4,
        complexity_weight: float = 0.3,
        coverage_weight: float = 0.3
    ) -> float:
        """Compute information value for a candidate.
        
        Considers:
        - Novelty: How different from already tested prompts
        - Complexity: Token count and structure complexity
        - Coverage: Semantic cluster coverage contribution
        
        Args:
            candidate: Candidate dict with prompt and metadata
            novelty_weight: Weight for novelty score
            complexity_weight: Weight for complexity score
            coverage_weight: Weight for coverage score
            
        Returns:
            Combined information value score (0-1)
        """
        prompt = candidate.get("prompt", "")
        
        # Novelty (simple proxy: length variance from mean)
        novelty = min(1.0, len(prompt) / 1000)
        
        # Complexity (token count normalized)
        tokens = self.cost_model.feature_extractor.count_tokens(prompt, "gpt-4")
        complexity = min(1.0, tokens / 500)
        
        # Coverage (from cluster info if available)
        is_centroid = candidate.get("is_centroid", False)
        coverage = 1.0 if is_centroid else 0.5
        
        return (
            novelty * novelty_weight +
            complexity * complexity_weight +
            coverage * coverage_weight
        )
