# Shadow-Optic P2 Enhancement Implementation Roadmap

**Document Type:** Implementation Roadmap - Portkey-First Architecture  
**Date:** January 17, 2026  
**Status:** Research & Planning  
**Target:** Phase 2 Enhancements  
**Hackathon:** Portkey AI Hackathon 2026

---

## Executive Summary

This document provides a comprehensive roadmap for implementing the **P2 (Priority 2) enhancements** identified in `ENHANCEMENTS.md`. **As a Portkey Hackathon project, this roadmap prioritizes Portkey's native features** and only supplements with FOSS libraries where Portkey doesn't provide coverage.

### Portkey-First Design Philosophy

> **Maximize Portkey → Supplement with FOSS only where necessary**

Portkey provides extensive built-in capabilities that eliminate the need for many external libraries. This approach:
- Reduces dependency complexity
- Leverages battle-tested production infrastructure
- Maximizes integration with the Portkey ecosystem
- Demonstrates deep Portkey platform expertise for the hackathon

### P2 Enhancement Overview

| Enhancement | Description | Portkey Coverage | FOSS Supplement |
|-------------|-------------|------------------|-----------------|
| **Cost Prediction Model** | ML model to predict replay costs | **90%** - Analytics, Budget Limits, Token Tracking | tiktoken, scikit-learn |
| **Automatic Guardrail Tuning** | LLM-powered prompt adjustment | **80%** - Guardrails, Feedback API, DSPy Integration | DSPy (via Portkey) |
| **Auto A/B Graduation** | Automated staged rollout | **95%** - Canary Testing, Load Balancing, Analytics | spotify-confidence |

---

## Portkey vs External Libraries Analysis

### Why NOT LiteLLM?

**LiteLLM is redundant** when building on Portkey. Here's the feature comparison:

| Feature | LiteLLM | Portkey Native | Decision |
|---------|---------|----------------|----------|
| Cost Tracking | `completion_cost()` function | Built-in per-request cost in logs & analytics |  **Use Portkey** |
| Token Counting | Uses tiktoken internally | Automatic token tracking in every request |  **Use Portkey** |
| Budget Limits | Not available | Cost-based and token-based limits on providers |  **Use Portkey** |
| Multi-Provider | 100+ LLMs | 250+ LLMs via unified API |  **Use Portkey** |
| Analytics Dashboard | Not available | 40+ metrics dashboard |  **Use Portkey** |
| Guardrails | Not available | 20+ built-in + partner integrations |  **Use Portkey** |
| A/B Testing | Not available | Canary testing + load balancing |  **Use Portkey** |

### Why NOT Unleash/Flipt/Flagsmith?

**Portkey's Load Balancing + Canary Testing replaces feature flags** for A/B model experiments:

| Feature | External Feature Flags | Portkey Native | Decision |
|---------|----------------------|----------------|----------|
| Percentage Rollouts | Unleash gradual rollouts | `weight` in loadbalance config |  **Use Portkey** |
| Canary Testing | Manual integration | Native canary testing with analytics |  **Use Portkey** |
| Kill Switch | Flag toggle | Remove target from config or set `weight: 0` |  **Use Portkey** |
| A/B Metrics | External analytics | Built-in analytics dashboard |  **Use Portkey** |
| Sticky Sessions | Session affinity | `sticky_session` with hash_fields |  **Use Portkey** |

### Why NOT NeMo Guardrails / LLM Guard for Core Features?

**Portkey Guardrails provides comprehensive coverage:**

| Guardrail Type | External Library | Portkey Native | Decision |
|----------------|------------------|----------------|----------|
| Content Moderation | LLM Guard Toxicity | `Moderate Content` PRO check |  **Use Portkey** |
| PII Detection | LLM Guard PII | `Detect PII` with categories |  **Use Portkey** |
| Prompt Injection | LLM Guard PromptInjection | Partner: Azure/Pillar/Lasso |  **Use Portkey** |
| JSON Schema | Guardrails AI | `JSON Schema` validation |  **Use Portkey** |
| Regex Matching | Custom code | `Regex Match` guardrail |  **Use Portkey** |
| Gibberish Detection | NeMo Guardrails | `Detect Gibberish` PRO check |  **Use Portkey** |
| Custom Guardrails | Custom webhook | `Webhook` guardrail |  **Use Portkey** |

### What FOSS We Still Need (Portkey Doesn't Cover)

| Capability | FOSS Library | Reason |
|------------|--------------|--------|
| Token counting for prediction | **tiktoken** | Pre-request estimation (before Portkey sees it) |
| ML model training | **scikit-learn** | GradientBoostingRegressor for cost prediction |
| Statistical significance | **spotify-confidence** | A/B test statistical analysis (Portkey has data, we need stats) |
| Prompt optimization | **DSPy** | Via Portkey's native DSPy integration |

---

## Part 1: Cost Prediction Model

### 1.1 Overview

The Cost Prediction Model enables intelligent budget allocation by predicting the cost of replaying prompts before execution. This allows Shadow-Optic to maximize information value within budget constraints.

### 1.2 Portkey-First Architecture

**What Portkey Provides:**
-  **Real-time cost tracking** - Every request logs cost automatically
-  **Token usage tracking** - Input/output tokens per request
-  **Budget limits** - Cost-based and token-based limits on providers
-  **Analytics API** - Historical cost data for ML training
-  **Metadata filtering** - Group costs by experiment, user, model

**What We Need FOSS For:**
-  **tiktoken** - Pre-request token estimation (before hitting Portkey)
-  **scikit-learn** - ML model training (GradientBoostingRegressor)

### 1.3 Minimal FOSS Dependencies

| Library | Stars | License | Purpose | Why Needed |
|---------|-------|---------|---------|------------|
| **tiktoken** | 17k | MIT | Token counting | Pre-request estimation before Portkey API call |
| **scikit-learn** | 60k | BSD-3 | ML model | Train cost prediction model on Portkey analytics data |

> **Note:** LiteLLM is **NOT needed** - Portkey provides the same cost/token tracking natively.

### 1.4 Implementation Using Portkey Data

```python
from dataclasses import dataclass
from typing import List, Optional
import tiktoken
import numpy as np
from portkey_ai import Portkey

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
    avg_output_ratio: float  # Historical output_tokens / input_tokens
    historical_cost_per_token: float  # From Portkey cost tracking

class PortkeyCostDataCollector:
    """Collect training data from Portkey's Analytics API."""
    
    def __init__(self, portkey_api_key: str):
        self.portkey = Portkey(api_key=portkey_api_key)
    
    async def fetch_historical_costs(
        self,
        days: int = 30,
        metadata_filter: Optional[dict] = None
    ) -> List[dict]:
        """Fetch historical cost data from Portkey logs.
        
        Portkey tracks for each request:
        - tokens (input + output)
        - cost (in USD)
        - model used
        - latency
        - metadata
        """
        # Use Portkey's logs API to fetch historical data
        # This data is automatically tracked by Portkey for every request
        logs = await self.portkey.logs.list(
            start_date=f"-{days}d",
            limit=10000
        )
        
        return [
            {
                "prompt_tokens": log.request.usage.prompt_tokens,
                "completion_tokens": log.request.usage.completion_tokens,
                "total_tokens": log.request.usage.total_tokens,
                "cost": log.cost,  # Portkey tracks this automatically!
                "model": log.model,
                "provider": log.provider,
                "latency_ms": log.latency,
                "metadata": log.metadata
            }
            for log in logs
            if log.cost is not None  # Filter out models without pricing
        ]
    
    def compute_model_statistics(
        self,
        historical_data: List[dict]
    ) -> dict:
        """Compute per-model statistics from Portkey data."""
        
        from collections import defaultdict
        stats = defaultdict(lambda: {"costs": [], "ratios": []})
        
        for record in historical_data:
            model = record["model"]
            if record["prompt_tokens"] > 0:
                ratio = record["completion_tokens"] / record["prompt_tokens"]
                cost_per_token = record["cost"] / record["total_tokens"]
                
                stats[model]["ratios"].append(ratio)
                stats[model]["costs"].append(cost_per_token)
        
        return {
            model: {
                "avg_output_ratio": np.mean(data["ratios"]),
                "std_output_ratio": np.std(data["ratios"]),
                "avg_cost_per_token": np.mean(data["costs"]),
                "sample_count": len(data["costs"])
            }
            for model, data in stats.items()
        }


class CostFeatureExtractor:
    """Extract features using tiktoken for pre-request estimation."""
    
    def __init__(self, model_stats: dict = None):
        self.encoders = {}
        self.model_stats = model_stats or {}
    
    def _get_encoder(self, model: str) -> tiktoken.Encoding:
        """Get or create encoder for model."""
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback for non-OpenAI models
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")
        return self.encoders[model]
    
    def extract_features(
        self,
        prompt: str,
        model: str,
        provider: str = "openai"
    ) -> CostFeatures:
        """Extract features for cost prediction."""
        
        enc = self._get_encoder(model)
        input_tokens = len(enc.encode(prompt))
        
        # Use historical stats from Portkey data
        stats = self.model_stats.get(model, {})
        avg_ratio = stats.get("avg_output_ratio", 1.5)
        cost_per_token = stats.get("avg_cost_per_token", 0.00001)
        
        return CostFeatures(
            input_tokens=input_tokens,
            estimated_output_tokens=int(input_tokens * avg_ratio),
            model_id=model,
            provider=provider,
            avg_output_ratio=avg_ratio,
            historical_cost_per_token=cost_per_token
        )
```

### 1.5 Cost Prediction Model (Using Portkey Data)

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib

class CostPredictionModel:
    """ML model trained on Portkey analytics data."""
    
    def __init__(self, portkey_api_key: str):
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.data_collector = PortkeyCostDataCollector(portkey_api_key)
        self.feature_extractor = CostFeatureExtractor()
        self.is_trained = False
    
    async def train_from_portkey(self, days: int = 30) -> dict:
        """Train model using Portkey's historical data.
        
        Portkey automatically tracks costs - we just need to fetch and train!
        """
        # Fetch historical data from Portkey
        historical_data = await self.data_collector.fetch_historical_costs(days=days)
        
        # Compute model statistics
        model_stats = self.data_collector.compute_model_statistics(historical_data)
        self.feature_extractor.model_stats = model_stats
        
        # Prepare training data
        X, y = [], []
        for record in historical_data:
            X.append([
                record["prompt_tokens"],
                record["completion_tokens"],
                record["total_tokens"],
                model_stats.get(record["model"], {}).get("avg_output_ratio", 1.5),
                model_stats.get(record["model"], {}).get("avg_cost_per_token", 0.00001)
            ])
            y.append(record["cost"])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train with cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        self.model.fit(X, y)
        self.is_trained = True
        
        return {
            "samples": len(X),
            "cv_mae": -cv_scores.mean(),
            "models_tracked": len(model_stats)
        }
    
    def predict(self, prompt: str, model: str) -> float:
        """Predict cost for a replay."""
        if not self.is_trained:
            return self._fallback_estimate(prompt, model)
        
        features = self.feature_extractor.extract_features(prompt, model)
        X = np.array([[
            features.input_tokens,
            features.estimated_output_tokens,
            features.input_tokens + features.estimated_output_tokens,
            features.avg_output_ratio,
            features.historical_cost_per_token
        ]])
        return self.model.predict(X)[0]
    
    def _fallback_estimate(self, prompt: str, model: str) -> float:
        """Simple fallback using Portkey's pricing data."""
        features = self.feature_extractor.extract_features(prompt, model)
        total_tokens = features.input_tokens + features.estimated_output_tokens
        return total_tokens * features.historical_cost_per_token
```

### 1.6 Integration with Portkey Budget Limits

```python
class BudgetOptimizedSampler:
    """Use Portkey's budget tracking + ML prediction for optimal sampling."""
    
    def __init__(
        self,
        portkey_api_key: str,
        budget_limit: float
    ):
        self.portkey = Portkey(api_key=portkey_api_key)
        self.cost_model = CostPredictionModel(portkey_api_key)
        self.budget_limit = budget_limit
        self.spent = 0.0
    
    async def get_current_spend(self, provider_slug: str) -> float:
        """Get current spend from Portkey's budget tracking.
        
        Portkey tracks this automatically via Budget Limits feature!
        """
        # Portkey's analytics API provides current spend
        analytics = await self.portkey.analytics.get(
            metrics=["cost"],
            group_by="provider",
            filter={"provider": provider_slug}
        )
        return analytics.total_cost
    
    async def select_replays_within_budget(
        self,
        candidates: List[dict],
        model: str
    ) -> List[dict]:
        """Select replays that fit within remaining budget."""
        
        remaining = self.budget_limit - self.spent
        selected = []
        
        # Score and sort by information value / cost
        scored = []
        for candidate in candidates:
            cost = self.cost_model.predict(candidate["prompt"], model)
            info_value = self._compute_info_value(candidate)
            scored.append({
                **candidate,
                "predicted_cost": cost,
                "priority": info_value / max(cost, 0.0001)
            })
        
        scored.sort(key=lambda x: x["priority"], reverse=True)
        
        # Greedy selection within budget
        total = 0.0
        for item in scored:
            if total + item["predicted_cost"] <= remaining:
                selected.append(item)
                total += item["predicted_cost"]
        
        return selected
```

---

## Part 2: Automatic Guardrail Tuning

### 2.1 Overview

Automatic Guardrail Tuning uses LLM-powered analysis to detect refusal patterns and automatically adjust system prompts to reduce false refusals while maintaining safety.

### 2.2 Portkey-First Architecture

**What Portkey Provides:**
-  **20+ Built-in Guardrails** - Regex, JSON Schema, Contains Code, etc.
-  **LLM-based PRO Guardrails** - Moderate Content, Detect PII, Detect Gibberish
-  **Partner Guardrails** - Aporia, Azure, Pillar, Lasso, Patronus, etc.
-  **Custom Webhook Guardrails** - Bring your own guardrail logic
-  **Feedback API** - Track guardrail pass/fail with feedback values
-  **Status Codes 246/446** - Programmatic handling of guardrail failures
-  **DSPy Integration** - Native Portkey + DSPy for prompt optimization

**What We Need FOSS For:**
-  **DSPy** - Prompt optimization (but via Portkey's native integration!)

> **Key Insight:** Portkey's Guardrails + DSPy integration means we can build the entire guardrail tuning system on Portkey's stack.

### 2.3 Minimal FOSS Dependencies

| Library | Stars | License | Purpose | Notes |
|---------|-------|---------|---------|-------|
| **DSPy** | 31.6k | MIT | Prompt optimization | **Use via Portkey's DSPy integration** |

> **Note:** NeMo Guardrails, Guardrails AI, and LLM Guard are **NOT needed** - Portkey provides equivalent functionality natively.

### 2.3 Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Automatic Guardrail Tuning                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Refusal      │───▶│ Pattern      │───▶│ Prompt           │  │
│  │ Detector     │    │ Analyzer     │    │ Optimizer (DSPy) │  │
│  │ (LLM Guard)  │    │              │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│         │                   │                      │            │
│         ▼                   ▼                      ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ Refusal      │    │ False Refusal│    │ System Prompt    │  │
│  │ Patterns DB  │    │ Classifier   │    │ Generator        │  │
│  │ (Qdrant)     │    │              │    │                  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ NeMo         │    │ Safety       │    │ A/B Testing      │  │
│  │ Guardrails   │    │ Validator    │    │ Framework        │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Core Implementation

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import dspy
from qdrant_client import QdrantClient

class RefusalType(Enum):
    """Types of model refusals."""
    SAFETY = "safety"           # Legitimate safety refusal
    OVERCAUTIOUS = "overcautious"  # False positive refusal
    CAPABILITY = "capability"    # "I can't do that" type
    POLICY = "policy"           # Policy-based refusal
    AMBIGUOUS = "ambiguous"     # Unclear refusal reason

@dataclass
class RefusalPattern:
    """Detected refusal pattern."""
    prompt: str
    response: str
    refusal_type: RefusalType
    confidence: float
    trigger_phrases: List[str]
    suggested_mitigation: Optional[str] = None

class RefusalDetector:
    """Enhanced refusal detection using LLM Guard + embeddings."""
    
    REFUSAL_PATTERNS = [
        r"I (?:cannot|can't|won't|will not|am not able to)",
        r"I'm (?:sorry|afraid|unable)",
        r"(?:Unfortunately|Regrettably),? I",
        r"(?:This|That) (?:is|would be) (?:against|outside|beyond)",
        r"I (?:don't|do not) (?:have|feel comfortable)",
        r"(?:As an AI|As a language model|As an assistant)",
    ]
    
    def __init__(self, qdrant_url: str = "localhost:6333"):
        from llm_guard.output_scanners import NoRefusal
        self.refusal_scanner = NoRefusal()
        self.qdrant = QdrantClient(url=qdrant_url)
        self.pattern_collection = "refusal_patterns"
    
    async def detect_refusal(
        self,
        prompt: str,
        response: str
    ) -> Tuple[bool, Optional[RefusalPattern]]:
        """Detect if response is a refusal and classify it."""
        
        # Quick check with LLM Guard
        _, is_refusal, confidence = self.refusal_scanner.scan("", response)
        
        if not is_refusal and confidence < 0.3:
            return False, None
        
        # Detailed analysis
        refusal_type = await self._classify_refusal_type(prompt, response)
        trigger_phrases = self._extract_trigger_phrases(response)
        
        pattern = RefusalPattern(
            prompt=prompt,
            response=response,
            refusal_type=refusal_type,
            confidence=confidence,
            trigger_phrases=trigger_phrases
        )
        
        # Store in Qdrant for pattern analysis
        await self._store_pattern(pattern)
        
        return True, pattern
    
    async def _classify_refusal_type(
        self,
        prompt: str,
        response: str
    ) -> RefusalType:
        """Classify the type of refusal using LLM."""
        
        classification_prompt = f"""Classify this model refusal into one of these categories:
        
1. SAFETY - Legitimate safety concern (harmful content, illegal activity)
2. OVERCAUTIOUS - False positive, request seems benign but model refused
3. CAPABILITY - Model claims it cannot perform the task
4. POLICY - Citing policy reasons
5. AMBIGUOUS - Unclear reason for refusal

User Prompt: {prompt[:500]}
Model Response: {response[:500]}

Classification (respond with just the category name):"""
        
        # Use DSPy for classification
        classify = dspy.Predict("prompt -> classification")
        result = classify(prompt=classification_prompt)
        
        type_map = {
            "SAFETY": RefusalType.SAFETY,
            "OVERCAUTIOUS": RefusalType.OVERCAUTIOUS,
            "CAPABILITY": RefusalType.CAPABILITY,
            "POLICY": RefusalType.POLICY,
            "AMBIGUOUS": RefusalType.AMBIGUOUS
        }
        
        return type_map.get(result.classification.upper(), RefusalType.AMBIGUOUS)
    
    def _extract_trigger_phrases(self, response: str) -> List[str]:
        """Extract phrases that indicate refusal."""
        import re
        triggers = []
        
        for pattern in self.REFUSAL_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            triggers.extend(matches)
        
        return list(set(triggers))


class GuardrailTuner:
    """Automatic guardrail tuning using DSPy optimization."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.lm = dspy.LM(model)
        dspy.configure(lm=self.lm)
        
        self.refusal_detector = RefusalDetector()
        self.system_prompt_history = []
    
    async def analyze_refusal_patterns(
        self,
        refusals: List[RefusalPattern]
    ) -> dict:
        """Analyze patterns in refusals to identify improvement areas."""
        
        # Group by type
        by_type = {}
        for r in refusals:
            if r.refusal_type not in by_type:
                by_type[r.refusal_type] = []
            by_type[r.refusal_type].append(r)
        
        # Find overcautious patterns (our optimization target)
        overcautious = by_type.get(RefusalType.OVERCAUTIOUS, [])
        
        if not overcautious:
            return {"status": "no_optimization_needed", "overcautious_rate": 0}
        
        # Cluster similar overcautious refusals
        clusters = await self._cluster_refusals(overcautious)
        
        return {
            "status": "optimization_available",
            "overcautious_rate": len(overcautious) / len(refusals),
            "clusters": clusters,
            "total_refusals": len(refusals),
            "by_type": {k.value: len(v) for k, v in by_type.items()}
        }
    
    async def generate_improved_prompt(
        self,
        current_system_prompt: str,
        overcautious_examples: List[RefusalPattern]
    ) -> str:
        """Generate improved system prompt to reduce false refusals."""
        
        # DSPy signature for prompt optimization
        class PromptOptimizer(dspy.Signature):
            """Optimize system prompt to reduce false refusals while maintaining safety."""
            
            current_prompt: str = dspy.InputField(desc="Current system prompt")
            false_refusal_examples: str = dspy.InputField(
                desc="Examples of benign prompts that were incorrectly refused"
            )
            optimized_prompt: str = dspy.OutputField(
                desc="Improved system prompt that handles these cases while staying safe"
            )
            reasoning: str = dspy.OutputField(
                desc="Explanation of changes made"
            )
        
        # Format examples
        examples_text = "\n\n".join([
            f"User: {r.prompt[:200]}\nRefused with: {r.response[:200]}"
            for r in overcautious_examples[:5]
        ])
        
        optimizer = dspy.ChainOfThought(PromptOptimizer)
        result = optimizer(
            current_prompt=current_system_prompt,
            false_refusal_examples=examples_text
        )
        
        return result.optimized_prompt
    
    async def validate_prompt_safety(
        self,
        optimized_prompt: str,
        safety_test_suite: List[dict]
    ) -> Tuple[bool, dict]:
        """Validate that optimized prompt doesn't reduce safety."""
        
        results = {"passed": 0, "failed": 0, "failures": []}
        
        for test in safety_test_suite:
            # Test should still refuse
            if test["expected"] == "refuse":
                response = await self._test_prompt(
                    system_prompt=optimized_prompt,
                    user_prompt=test["prompt"]
                )
                
                is_refusal, _ = await self.refusal_detector.detect_refusal(
                    test["prompt"], response
                )
                
                if is_refusal:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "test": test,
                        "response": response[:200]
                    })
            
            # Test should answer
            elif test["expected"] == "answer":
                response = await self._test_prompt(
                    system_prompt=optimized_prompt,
                    user_prompt=test["prompt"]
                )
                
                is_refusal, _ = await self.refusal_detector.detect_refusal(
                    test["prompt"], response
                )
                
                if not is_refusal:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                    results["failures"].append({
                        "test": test,
                        "response": response[:200]
                    })
        
        pass_rate = results["passed"] / (results["passed"] + results["failed"])
        is_safe = pass_rate >= 0.95  # 95% threshold
        
        return is_safe, results
    
    async def _cluster_refusals(
        self,
        refusals: List[RefusalPattern]
    ) -> List[dict]:
        """Cluster similar refusals for pattern analysis."""
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        # Get embeddings
        embeddings = await self._get_embeddings([r.prompt for r in refusals])
        
        # Cluster
        clustering = DBSCAN(eps=0.3, min_samples=2).fit(embeddings)
        
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(refusals[i])
        
        return [
            {
                "cluster_id": k,
                "count": len(v),
                "representative_prompts": [r.prompt[:100] for r in v[:3]],
                "common_triggers": self._find_common_triggers(v)
            }
            for k, v in clusters.items()
            if k != -1  # Exclude noise cluster
        ]
```

### 2.5 NeMo Guardrails Integration

```yaml
# configs/guardrails/config.yml
models:
  - type: main
    engine: openai
    model: gpt-4o-mini

rails:
  input:
    flows:
      - check jailbreak
      - mask sensitive data on input
  
  output:
    flows:
      - self check facts
      - self check hallucination
      - adaptive refusal check  # Our custom flow

  config:
    sensitive_data_detection:
      input:
        entities:
          - PERSON
          - EMAIL_ADDRESS
          - PHONE_NUMBER
```

```colang
# configs/guardrails/adaptive_refusal.co

define flow adaptive refusal check
  """Check if response is an overcautious refusal."""
  
  $refusal_check = execute check_overcautious_refusal
  
  if $refusal_check.is_overcautious
    # Retry with tuned prompt
    $retry_response = execute retry_with_tuned_prompt
    bot $retry_response
  else
    # Allow original response
    pass
```

### 2.6 Dependencies (Minimal - Portkey-First)

```toml
# In pyproject.toml

[project.dependencies]
# DSPy via Portkey's native integration
dspy = ">=3.1.0"

# NOTE: These are NOT needed - Portkey provides:
# - nemoguardrails: Use Portkey's 20+ built-in guardrails
# - llm-guard: Use Portkey's Detect PII, Moderate Content, etc.
# - guardrails-ai: Use Portkey's JSON Schema, Webhook guardrails
```

---

## Part 3: Auto A/B Graduation

### 3.1 Overview

Auto A/B Graduation automates the staged rollout of challenger models from shadow testing to full production, with safety guardrails at each stage.

### 3.2 Portkey-First Architecture

**What Portkey Provides:**
-  **Canary Testing** - Native support via load balancing with weights
-  **Load Balancing** - Weighted distribution across models/providers
-  **Sticky Sessions** - Consistent routing with `hash_fields` and TTL
-  **Analytics Dashboard** - Compare model performance (cost, latency, errors)
-  **Fallbacks** - Automatic failover on errors
-  **Kill Switch** - Set `weight: 0` to stop traffic instantly
-  **Feedback API** - Track success/failure for quality gates
-  **Metadata Grouping** - Group analytics by experiment_id

**What We Need FOSS For:**
-  **spotify-confidence** - Statistical significance testing (Portkey has data, we need stats)

### 3.3 Minimal FOSS Dependencies

| Library | Stars | License | Purpose | Notes |
|---------|-------|---------|---------|-------|
| **spotify-confidence** | 279 | Apache-2.0 | A/B test statistics | Production-tested at Spotify |

> **Note:** Unleash, Flipt, Flagsmith are **NOT needed** - Portkey provides equivalent functionality via load balancing + canary testing.

### 3.4 Portkey Config for Staged Rollout

```python
# Portkey configs for each experiment stage

EXPERIMENT_CONFIGS = {
    "shadow": {
        # 0% live traffic - challenger only runs in shadow
        "strategy": {"mode": "fallback"},
        "targets": [
            {"provider": "@production-model", "weight": 1.0}
        ],
        # Shadow replay happens separately via Temporal workflow
    },
    
    "canary": {
        # 1% live traffic to challenger
        "strategy": {"mode": "loadbalance"},
        "targets": [
            {"provider": "@production-model", "weight": 0.99},
            {"provider": "@challenger-model", "weight": 0.01}
        ]
    },
    
    "graduated": {
        # 10% live traffic to challenger
        "strategy": {"mode": "loadbalance"},
        "targets": [
            {"provider": "@production-model", "weight": 0.90},
            {"provider": "@challenger-model", "weight": 0.10}
        ]
    },
    
    "ramping": {
        # 50% live traffic to challenger
        "strategy": {"mode": "loadbalance"},
        "targets": [
            {"provider": "@production-model", "weight": 0.50},
            {"provider": "@challenger-model", "weight": 0.50}
        ]
    },
    
    "majority": {
        # 90% live traffic to challenger (about to graduate!)
        "strategy": {"mode": "loadbalance"},
        "targets": [
            {"provider": "@production-model", "weight": 0.10},
            {"provider": "@challenger-model", "weight": 0.90}
        ]
    },
    
    "full": {
        # 100% - challenger is now production!
        "strategy": {"mode": "fallback"},
        "targets": [
            {"provider": "@challenger-model", "weight": 1.0},
            {"provider": "@production-model"}  # Fallback only
        ]
    }
}

# With sticky sessions for consistent user experience
EXPERIMENT_CONFIG_WITH_STICKY = {
    "strategy": {
        "mode": "loadbalance",
        "sticky_session": {
            "hash_fields": ["metadata.user_id", "metadata.experiment_id"],
            "ttl": 3600  # 1 hour
        }
    },
    "targets": [
        {"provider": "@production-model", "weight": 0.90},
        {"provider": "@challenger-model", "weight": 0.10}
    ]
}
```

### 3.5 Implementation Using Portkey Analytics

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import spotify_confidence as confidence
import pandas as pd
import asyncio

class ExperimentStage(Enum):
    """Stages of graduated rollout."""
    SHADOW = "shadow"           # 0% live traffic
    CANARY = "canary"           # 1% live traffic
    GRADUATED = "graduated"     # 10% live traffic
    RAMPING = "ramping"         # 10-50% live traffic
    MAJORITY = "majority"       # 50-90% live traffic
    FULL = "full"               # 100% live traffic

@dataclass
class StageConfig:
    """Configuration for each experiment stage."""
    traffic_percentage: float
    min_samples: int
    min_duration_hours: int
    quality_gates: Dict[str, float]

@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment."""
    total_requests: int = 0
    successful_responses: int = 0
    refusals: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_quality_score: float = 0.0
    cost_per_request: float = 0.0
    
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

class AutomatedRolloutController:
    """Automated graduated rollout from shadow to production."""
    
    STAGE_CONFIGS = {
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
        production_model: str
    ):
        self.experiment_id = experiment_id
        self.challenger_model = challenger_model
        self.production_model = production_model
        
        self.stage = ExperimentStage.SHADOW
        self.stage_start_time = datetime.utcnow()
        
        self.production_metrics = ExperimentMetrics()
        self.challenger_metrics = ExperimentMetrics()
        self.baseline_metrics: Optional[ExperimentMetrics] = None
        
        self.history: List[dict] = []
    
    async def check_promotion(self) -> Tuple[bool, Optional[str]]:
        """Check if experiment can be promoted to next stage."""
        
        if self.stage == ExperimentStage.FULL:
            return False, "Already at full rollout"
        
        config = self.STAGE_CONFIGS[self.stage]
        
        # Check minimum duration
        hours_in_stage = (datetime.utcnow() - self.stage_start_time).total_seconds() / 3600
        if hours_in_stage < config.min_duration_hours:
            return False, f"Minimum duration not met ({hours_in_stage:.1f}/{config.min_duration_hours}h)"
        
        # Check minimum samples
        if self.challenger_metrics.total_requests < config.min_samples:
            return False, f"Minimum samples not met ({self.challenger_metrics.total_requests}/{config.min_samples})"
        
        # Check quality gates
        gate_result = await self._check_quality_gates(config.quality_gates)
        if not gate_result[0]:
            return False, gate_result[1]
        
        # Statistical significance check
        if self.stage != ExperimentStage.SHADOW:
            sig_result = await self._check_statistical_significance()
            if not sig_result[0]:
                return False, sig_result[1]
        
        return True, None
    
    async def _check_quality_gates(
        self,
        gates: Dict[str, float]
    ) -> Tuple[bool, Optional[str]]:
        """Check all quality gates."""
        
        for gate, threshold in gates.items():
            if gate.endswith("_min"):
                metric_name = gate.replace("_min", "")
                value = getattr(self.challenger_metrics, f"avg_{metric_name}", 
                               getattr(self.challenger_metrics, metric_name, 0))
                if value < threshold:
                    return False, f"Quality gate failed: {metric_name} ({value:.3f} < {threshold})"
            
            elif gate.endswith("_max"):
                metric_name = gate.replace("_max", "")
                value = getattr(self.challenger_metrics, metric_name, 0)
                if value > threshold:
                    return False, f"Quality gate failed: {metric_name} ({value:.3f} > {threshold})"
            
            elif gate.endswith("_delta_max"):
                metric_name = gate.replace("_delta_max", "")
                challenger_value = getattr(self.challenger_metrics, metric_name, 0)
                production_value = getattr(self.production_metrics, metric_name, 0)
                
                if production_value > 0:
                    delta = (challenger_value - production_value) / production_value
                    if delta > threshold:
                        return False, f"Delta gate failed: {metric_name} delta ({delta:.3f} > {threshold})"
        
        return True, None
    
    async def _check_statistical_significance(self) -> Tuple[bool, Optional[str]]:
        """Check statistical significance using spotify-confidence."""
        
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
        
        diff = test.difference(level_1='production', level_2='challenger')
        
        # We want challenger to be at least as good (non-inferiority)
        if diff['difference'] < -0.02:  # More than 2% worse
            if diff['p-value'] < 0.05:
                return False, f"Challenger significantly worse: {diff['difference']:.3f} (p={diff['p-value']:.4f})"
        
        return True, None
    
    async def promote(self) -> ExperimentStage:
        """Promote experiment to next stage."""
        
        stages = list(ExperimentStage)
        current_idx = stages.index(self.stage)
        
        if current_idx < len(stages) - 1:
            old_stage = self.stage
            self.stage = stages[current_idx + 1]
            self.stage_start_time = datetime.utcnow()
            
            # Store baseline at first promotion
            if old_stage == ExperimentStage.SHADOW:
                self.baseline_metrics = ExperimentMetrics(
                    total_requests=self.challenger_metrics.total_requests,
                    successful_responses=self.challenger_metrics.successful_responses,
                    avg_quality_score=self.challenger_metrics.avg_quality_score
                )
            
            # Update traffic routing
            await self._update_traffic_routing()
            
            # Log transition
            self.history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "promote",
                "from_stage": old_stage.value,
                "to_stage": self.stage.value,
                "challenger_metrics": {
                    "total_requests": self.challenger_metrics.total_requests,
                    "success_rate": self.challenger_metrics.success_rate,
                    "avg_quality_score": self.challenger_metrics.avg_quality_score
                }
            })
            
            # Send notification
            await self._send_notification(
                level="info",
                title=f"Experiment {self.experiment_id} promoted",
                message=f"Stage: {old_stage.value} → {self.stage.value}"
            )
        
        return self.stage
    
    async def rollback(self, reason: str):
        """Emergency rollback to shadow."""
        
        old_stage = self.stage
        self.stage = ExperimentStage.SHADOW
        self.stage_start_time = datetime.utcnow()
        
        await self._update_traffic_routing()
        
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": "rollback",
            "from_stage": old_stage.value,
            "to_stage": self.stage.value,
            "reason": reason
        })
        
        await self._send_notification(
            level="critical",
            title=f"Experiment {self.experiment_id} ROLLED BACK",
            message=f"Reason: {reason}\nStage: {old_stage.value} → {self.stage.value}"
        )
    
    async def _update_traffic_routing(self):
        """Update Portkey config for current stage traffic split."""
        
        traffic = self.STAGE_CONFIGS[self.stage].traffic_percentage
        
        # Create Portkey config
        config = {
            "strategy": {"mode": "loadbalance"},
            "targets": [
                {
                    "provider": "openai",  # Or appropriate provider
                    "weight": int((1 - traffic) * 100),
                    "override_params": {"model": self.production_model}
                },
                {
                    "provider": "deepseek",  # Or appropriate provider
                    "weight": int(traffic * 100),
                    "override_params": {"model": self.challenger_model}
                }
            ]
        }
        
        # Update via Portkey API
        from portkey_ai import Portkey
        portkey = Portkey()
        
        # Note: Actual implementation would use Portkey's config API
        # This is a placeholder for the integration pattern
        print(f"Updated traffic routing: {self.production_model}={100-int(traffic*100)}%, {self.challenger_model}={int(traffic*100)}%")
    
    async def _send_notification(self, level: str, title: str, message: str):
        """Send notification about experiment status."""
        # Integration with alerting system (Slack, PagerDuty, etc.)
        print(f"[{level.upper()}] {title}: {message}")
    
    def get_dashboard_data(self) -> dict:
        """Get data for experiment dashboard."""
        
        config = self.STAGE_CONFIGS[self.stage]
        hours_in_stage = (datetime.utcnow() - self.stage_start_time).total_seconds() / 3600
        
        return {
            "experiment_id": self.experiment_id,
            "challenger_model": self.challenger_model,
            "production_model": self.production_model,
            "current_stage": self.stage.value,
            "traffic_percentage": config.traffic_percentage * 100,
            "stage_progress": {
                "hours_elapsed": hours_in_stage,
                "hours_required": config.min_duration_hours,
                "samples_collected": self.challenger_metrics.total_requests,
                "samples_required": config.min_samples,
                "duration_pct": min(100, hours_in_stage / max(config.min_duration_hours, 1) * 100),
                "samples_pct": min(100, self.challenger_metrics.total_requests / max(config.min_samples, 1) * 100)
            },
            "challenger_metrics": {
                "success_rate": self.challenger_metrics.success_rate,
                "error_rate": self.challenger_metrics.error_rate,
                "refusal_rate": self.challenger_metrics.refusal_rate,
                "avg_latency_ms": self.challenger_metrics.avg_latency_ms,
                "avg_quality_score": self.challenger_metrics.avg_quality_score,
            },
            "production_metrics": {
                "success_rate": self.production_metrics.success_rate,
                "error_rate": self.production_metrics.error_rate,
                "refusal_rate": self.production_metrics.refusal_rate,
                "avg_latency_ms": self.production_metrics.avg_latency_ms,
            },
            "quality_gates": config.quality_gates,
            "history": self.history[-10:]  # Last 10 events
        }
```

### 3.5 Portkey Config Manager (Replaces Feature Flags!)

```python
# Use Portkey configs instead of external feature flags

from portkey_ai import Portkey
from typing import Optional

class PortkeyConfigManager:
    """Manage experiment configs using Portkey's native config system.
    
    This replaces external feature flag systems like Unleash/Flipt!
    """
    
    def __init__(self, portkey_api_key: str):
        self.portkey = Portkey(api_key=portkey_api_key)
        self.configs = {}
    
    def get_stage_config(self, stage: ExperimentStage) -> dict:
        """Get Portkey config for a specific experiment stage."""
        return EXPERIMENT_CONFIGS.get(stage.value, EXPERIMENT_CONFIGS["shadow"])
    
    async def update_experiment_stage(
        self,
        experiment_id: str,
        new_stage: ExperimentStage,
        production_model: str,
        challenger_model: str
    ) -> str:
        """Update experiment to new stage by creating a new Portkey config.
        
        Returns the config ID to use in requests.
        """
        stage_config = self.get_stage_config(new_stage)
        
        # Build the actual config with model names
        config = {
            **stage_config,
            "targets": [
                {
                    **target,
                    "override_params": {"model": production_model if "production" in str(target.get("provider", "")) else challenger_model}
                }
                for target in stage_config.get("targets", [])
            ],
            "metadata": {
                "experiment_id": experiment_id,
                "stage": new_stage.value
            }
        }
        
        # Save config via Portkey API (or use pre-created configs)
        # In production, you'd create these configs in Portkey dashboard
        # and reference by config_id
        
        return f"pc-{experiment_id}-{new_stage.value}"
    
    def create_rollback_config(
        self,
        production_model: str
    ) -> dict:
        """Create emergency rollback config - 100% to production."""
        return {
            "strategy": {"mode": "fallback"},
            "targets": [
                {"provider": "@production-model", "weight": 1.0}
            ]
        }
    
    def create_kill_switch_config(
        self,
        experiment_id: str
    ) -> dict:
        """Kill switch: stop all challenger traffic immediately.
        
        In Portkey, just set challenger weight to 0!
        """
        return {
            "strategy": {"mode": "loadbalance"},
            "targets": [
                {"provider": "@production-model", "weight": 1.0},
                {"provider": "@challenger-model", "weight": 0}  # Kill switch!
            ]
        }
```

### 3.6 Dependencies (Minimal - Portkey-First)

```toml
# In pyproject.toml

[project.dependencies]
# Statistical analysis (Portkey has data, we need stats)
spotify-confidence = ">=4.0.0"

# NOTE: These are NOT needed - Portkey provides:
# - UnleashClient: Use Portkey load balancing + configs
# - flipt: Use Portkey canary testing
# - flagsmith: Use Portkey analytics for A/B comparison
```

---

## Part 4: Integration Patterns

### 4.1 Unified Enhancement Module (Portkey-First)

```python
# src/shadow_optic/p2_enhancements/__init__.py

from .cost_predictor import CostPredictionModel, BudgetOptimizedSampler, PortkeyCostDataCollector
from .guardrail_tuner import PortkeyGuardrailTuner, PortkeyRefusalDetector
from .ab_graduation import PortkeyAutomatedRollout, PortkeyConfigManager

__all__ = [
    # Cost Prediction (Portkey Analytics + tiktoken + sklearn)
    "CostPredictionModel",
    "BudgetOptimizedSampler",
    "PortkeyCostDataCollector",
    
    # Guardrail Tuning (Portkey Guardrails + DSPy via Portkey)
    "GuardrailTuner", 
    "RefusalDetector",
    
    # A/B Graduation
    "AutomatedRolloutController",
    "ExperimentStage",
]
```

### 4.2 Temporal Workflow Integration

```python
from temporalio import workflow, activity

@workflow.defn
class P2EnhancedOptimizationWorkflow:
    """Workflow with all P2 enhancements integrated."""
    
    @workflow.run
    async def run(self, config: dict):
        # 1. Cost-optimized sampling
        samples = await workflow.execute_activity(
            collect_budget_optimized_samples,
            args=[config["budget"], config["challenger_model"]],
            start_to_close_timeout=timedelta(minutes=30)
        )
        
        # 2. Shadow replay with guardrail monitoring
        results = await workflow.execute_activity(
            replay_with_guardrail_tuning,
            args=[samples, config["challenger_model"]],
            start_to_close_timeout=timedelta(hours=2)
        )
        
        # 3. Evaluate with DeepEval
        evaluation = await workflow.execute_activity(
            evaluate_results,
            args=[results],
            start_to_close_timeout=timedelta(minutes=30)
        )
        
        # 4. Auto graduation check
        if evaluation.passed:
            graduation = await workflow.execute_activity(
                check_and_promote_experiment,
                args=[config["experiment_id"], evaluation],
                start_to_close_timeout=timedelta(minutes=5)
            )
            
            return {
                "status": "promoted" if graduation.promoted else "holding",
                "stage": graduation.stage,
                "metrics": evaluation.metrics
            }
        
        return {
            "status": "failed",
            "reason": evaluation.failure_reason,
            "metrics": evaluation.metrics
        }
```

---

## Part 5: Implementation Timeline

### Phase 1: Foundation (Week 1-2)

- [ ] Install and configure tiktoken, LiteLLM
- [ ] Implement CostFeatureExtractor
- [ ] Create initial CostPredictionModel with fallback estimation
- [ ] Add cost prediction to existing sampler

### Phase 2: Guardrail Tuning (Week 2-4)

- [ ] Install NeMo Guardrails and LLM Guard
- [ ] Implement RefusalDetector with pattern analysis
- [ ] Integrate DSPy for prompt optimization
- [ ] Create safety validation test suite
- [ ] Add Colang flows for adaptive refusal handling

### Phase 3: A/B Graduation (Week 4-6)

- [ ] Install spotify-confidence and Unleash
- [ ] Implement AutomatedRolloutController
- [ ] Create quality gate evaluation pipeline
- [ ] Integrate with Portkey for traffic routing
- [ ] Build experiment dashboard API

### Phase 4: Integration & Testing (Week 6-8)

- [ ] Create unified P2 enhancement module
- [ ] Integrate with Temporal workflows
- [ ] Add comprehensive test coverage
- [ ] Performance benchmarking
- [ ] Documentation and runbooks

---

## Part 6: Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cost prediction accuracy | Medium | Medium | Start with fallback, iterate with real data |
| Guardrail tuning reduces safety | Low | High | Comprehensive safety test suite, gradual rollout |
| A/B graduation creates incidents | Medium | High | Conservative gates, instant rollback capability |
| Integration complexity | Medium | Medium | Modular design, feature flags for each component |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Training data quality | Medium | Medium | Data validation, outlier detection |
| Statistical significance errors | Low | Medium | Conservative p-value thresholds |
| Traffic routing failures | Low | High | Health checks, circuit breakers |

---

## Part 7: Success Metrics

### Cost Prediction Model

- **MAE < $0.001** per request on validation set
- **Budget utilization > 95%** (no underspend)
- **Information value improvement > 20%** vs random sampling

### Guardrail Tuning

- **Overcautious refusal reduction > 30%**
- **Safety test suite pass rate > 99%**
- **No increase in legitimate safety blocks**

### A/B Graduation

- **Time to full rollout < 7 days** for qualified models
- **Zero regression incidents** from automated promotion
- **Statistical power > 80%** for promotion decisions

---

## Appendix A: Complete Dependencies (Portkey-First - Minimal!)

```toml
# pyproject.toml additions for P2 enhancements
# DRAMATICALLY REDUCED by using Portkey's native features!

[project.dependencies]
# ============================================
# REQUIRED - Portkey doesn't cover these
# ============================================

# Cost Prediction - Pre-request token estimation
tiktoken = ">=0.12.0"

# Cost Prediction - ML model training
scikit-learn = ">=1.5.0"
joblib = ">=1.4.0"

# Guardrail Tuning - Via Portkey's native DSPy integration
dspy = ">=3.1.0"

# A/B Graduation - Statistical significance testing
spotify-confidence = ">=4.0.0"

# Supporting libraries
pandas = ">=2.2.0"
numpy = ">=1.26.0"

# ============================================
# NOT NEEDED - Portkey provides these natively
# ============================================
# litellm - Portkey has cost tracking, token counting, 250+ LLMs
# nemoguardrails - Portkey has 20+ guardrails, partner integrations
# llm-guard - Portkey has Detect PII, Moderate Content, etc.
# guardrails-ai - Portkey has JSON Schema, Webhook guardrails
# UnleashClient - Portkey has load balancing, canary testing
# flipt - Portkey has weighted routing, kill switches
# flagsmith - Portkey has analytics dashboard for A/B comparison
```

---

## Appendix B: FOSS Repository Summary (Portkey-First Analysis)

###  Libraries We're Using

| Library | GitHub | Stars | License | Purpose | Why Needed |
|---------|--------|-------|---------|---------|------------|
| **tiktoken** | openai/tiktoken | 17k | MIT | Token counting | Pre-request estimation before Portkey API |
| **scikit-learn** | scikit-learn/scikit-learn | 60k | BSD-3 | ML training | GradientBoostingRegressor for cost model |
| **DSPy** | stanfordnlp/dspy | 31.6k | MIT | Prompt optimization | Via Portkey's native integration |
| **spotify-confidence** | spotify/confidence | 279 | Apache-2.0 | A/B statistics | Statistical significance testing |

###  Libraries NOT Needed (Portkey Provides)

| Library | Stars | Portkey Equivalent | Decision |
|---------|-------|-------------------|----------|
| LiteLLM | 34k | Cost tracking, token counting, 250+ LLMs |  Skip |
| NeMo Guardrails | 5.5k | 20+ built-in guardrails |  Skip |
| Guardrails AI | 6.3k | JSON Schema, Webhook guardrails |  Skip |
| LLM Guard | 2.4k | Detect PII, Moderate Content |  Skip |
| Unleash | 13.1k | Load balancing, canary testing |  Skip |
| Flipt | 4.7k | Weighted routing, configs |  Skip |
| Flagsmith | 6.2k | Analytics dashboard |  Skip |
| MLflow | 23.7k | Portkey observability |  Skip |

### Dependency Reduction Summary

| Category | Original Plan | Portkey-First | Reduction |
|----------|---------------|---------------|-----------|
| Cost Prediction | 4 deps | 3 deps | **-25%** |
| Guardrail Tuning | 3 deps | 1 dep | **-67%** |
| A/B Graduation | 2 deps | 1 dep | **-50%** |
| **Total** | **9 deps** | **5 deps** | **-44%** |

---

## Appendix C: Portkey Feature Mapping

| P2 Enhancement | Portkey Features Used |
|----------------|----------------------|
| **Cost Prediction** | Analytics API, Budget Limits, Token Tracking, Logs API |
| **Guardrail Tuning** | Guardrails (20+), Webhook Guardrails, DSPy Integration, Feedback API, Status 246/446 |
| **A/B Graduation** | Canary Testing, Load Balancing, Sticky Sessions, Analytics Dashboard, Metadata Filtering |

---

*Document generated: January 16, 2026*  
*Last updated: January 17, 2026 - Revised for Portkey-First Architecture*  
*Author: Shadow-Optic Development Team*  
*Hackathon: Portkey AI Hackathon 2026*
