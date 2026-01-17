# Shadow-Optic: Autonomous Cost-Quality Optimization Engine

## Technical Architecture Documentation

**Version:** 1.0.0  
**Date:** January 17, 2026  
**Challenge Track:** Track 4 - Cost-Quality Optimization via Historical Replay  
**Status:** Production-Ready Architecture

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Component Deep Dive](#component-deep-dive)
4. [Data Flow & Processing Pipeline](#data-flow--processing-pipeline)
5. [Portkey Integration Strategy](#portkey-integration-strategy)
6. [Evaluation Framework](#evaluation-framework)
7. [State Management & Orchestration](#state-management--orchestration)
8. [Observability & Explainability](#observability--explainability)
9. [Failure Handling & Resilience](#failure-handling--resilience)
10. [Security & Cost Controls](#security--cost-controls)
11. [Deployment Architecture](#deployment-architecture)
12. [API Reference](#api-reference)

---

## 1. Executive Summary

### What is Shadow-Optic?

Shadow-Optic is a **production-grade, autonomous optimization engine** that continuously analyzes your LLM production traffic and validates whether cheaper/faster models can deliver equivalent quality. It operates in "shadow mode" - replaying historical requests against challenger models asynchronously, measuring quality metrics, and providing actionable recommendations.

### Core Value Proposition

```
"Switching from GPT-5.2 to DeepSeek-V3 reduces cost by 99.1% 
 with a 3% quality impact and 3.7% increase in refusal rate.
 RECOMMENDATION: Do not switch - refusal rate exceeds safety threshold."
```

### Key Differentiators

| Feature | Shadow-Optic | Traditional A/B Testing |
|---------|--------------|------------------------|
| **User Impact** | Zero - fully asynchronous | Direct - affects users |
| **Cost Control** | Segregated R&D budget via Virtual Keys | Shared production budget |
| **Evaluation** | LLM-as-Judge with multi-metric scoring | Manual review |
| **State Management** | Durable workflows with Temporal | Stateless scripts |
| **Sampling** | Semantic clustering (95% compute savings) | Random sampling |

---

## 2. System Architecture Overview

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SHADOW-OPTIC ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐         LANE A: PRODUCTION (ZERO LATENCY)                  │
│  │    User     │──────────────────────────────────────────────────────────┐ │
│  └─────────────┘                                                          │ │
│         │                                                                 │ │
│         ▼                                                                 │ │
│  ┌─────────────────────────────────────────────────┐                      │ │
│  │              PORTKEY AI GATEWAY                 │                      │ │
│  │  ┌─────────────────────────────────────────┐   │                      │ │
│  │  │  production-config (GPT-5.2/Claude 4.5) │   │                      │ │
│  │  │  virtual-key: prod-key                  │   │                      │ │
│  │  │  metadata: {environment: production}    │   │                      │ │
│  │  └─────────────────────────────────────────┘   │                      │ │
│  │                      │                         │                      │ │
│  │                      ▼                         │                      │ │
│  │  ┌─────────────────────────────────────────┐   │      ┌───────────┐   │ │
│  │  │         LOGGING & TRACING               │   │      │  Response │   │ │
│  │  │  - Full request/response capture        │   │──────│  to User  │───┘ │
│  │  │  - Cost metrics                         │   │      └───────────┘     │
│  │  │  - Latency metrics                      │   │                        │
│  │  │  - Trace IDs                            │   │                        │
│  │  └─────────────────────────────────────────┘   │                        │
│  └─────────────────────────────────────────────────┘                        │
│                           │                                                 │
│                           │ Async Log Export                                │
│                           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    LANE B: SHADOW OPTIMIZATION                          ││
│  │                                                                         ││
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐   ││
│  │  │   TEMPORAL    │    │    QDRANT     │    │     REPLAY ENGINE     │   ││
│  │  │  ORCHESTRATOR │───▶│   SAMPLER     │───▶│                       │   ││
│  │  │               │    │               │    │  ┌─────────────────┐  │   ││
│  │  │  Workflows:   │    │  - Embedding  │    │  │ Challenger #1   │  │   ││
│  │  │  - Fetch Logs │    │  - Clustering │    │  │ (DeepSeek-V3)   │  │   ││
│  │  │  - Sample     │    │  - Centroid   │    │  ├─────────────────┤  │   ││
│  │  │  - Replay     │    │    Selection  │    │  │ Challenger #2   │  │   ││
│  │  │  - Evaluate   │    │               │    │  │ (Llama-4-Scout) │  │   ││
│  │  │               │    │  95% compute  │    │  ├─────────────────┤  │   ││
│  │  │  Retries:     │    │  savings      │    │  │ Challenger #3   │  │   ││
│  │  │  Exponential  │    │               │    │  │ (Mixtral-Next)  │  │   ││
│  │  │  Backoff      │    │               │    │  └─────────────────┘  │   ││
│  │  └───────────────┘    └───────────────┘    └───────────────────────┘   ││
│  │                                                      │                  ││
│  │                                                      ▼                  ││
│  │  ┌───────────────────────────────────────────────────────────────────┐ ││
│  │  │                        DEEPEVAL JUDGE                             │ ││
│  │  │                                                                   │ ││
│  │  │   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │ ││
│  │  │   │  Faithfulness  │  │   Conciseness  │  │  Refusal Rate  │     │ ││
│  │  │   │    (0.0-1.0)   │  │    (tokens)    │  │   (percentage) │     │ ││
│  │  │   └────────────────┘  └────────────────┘  └────────────────┘     │ ││
│  │  │                                                                   │ ││
│  │  │   Judge Model: GPT-5.2 (Gold Standard)                           │ ││
│  │  │   Metrics: G-Eval, Custom Rubrics                                │ ││
│  │  └───────────────────────────────────────────────────────────────────┘ ││
│  │                                    │                                    ││
│  │                                    ▼                                    ││
│  │  ┌───────────────────────────────────────────────────────────────────┐ ││
│  │  │                    FEEDBACK & ANALYTICS                           │ ││
│  │  │                                                                   │ ││
│  │  │   POST /feedback                                                  │ ││
│  │  │   {                                                               │ ││
│  │  │     trace_id: "shadow-{original_trace_id}",                       │ ││
│  │  │     value: 9,  // Quality Score                                   │ ││
│  │  │     metadata: {                                                   │ ││
│  │  │       roi_savings: "99.1%",                                       │ ││
│  │  │       challenger: "deepseek-v3",                                  │ ││
│  │  │       faithfulness: 0.96,                                         │ ││
│  │  │       refusal_delta: "+3.7%"                                      │ ││
│  │  │     }                                                             │ ││
│  │  │   }                                                               │ ││
│  │  └───────────────────────────────────────────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        ARIZE PHOENIX (OBSERVABILITY)                    ││
│  │   - Trace visualization for evaluation steps                            ││
│  │   - Failure case analysis                                               ││
│  │   - Quality trend dashboards                                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack (January 2026)

| Layer | Technology | Version | Purpose | Status |
|-------|------------|---------|---------|--------|
| **Gateway** | Portkey AI | Latest | Model Catalog, Logs Export, Feedback API | ✅ Implemented |
| **Orchestration** | Temporal.io | 1.25+ | Durable workflow execution | ✅ Implemented |
| **Vector DB** | Qdrant | 1.16+ | Semantic clustering & sampling | ✅ Implemented |
| **Evaluation** | DeepEval | 3.7+ | LLM-as-Judge (Faithfulness, Quality, Conciseness) | ✅ Implemented |
| **Observability** | Arize Phoenix | 4.0+ | OTLP trace visualization | ✅ Implemented |
| **Metrics** | Prometheus | 2.54+ | API metrics collection | ✅ Implemented |
| **Analytics** | QualityTrendDetector | - | T-test degradation, Z-score anomaly | ✅ Implemented |
| **Model Selection** | Thompson Sampling | - | Multi-Armed Bandit for challenger selection | ✅ Implemented |
| **Model Registry** | Pareto-Optimal | - | 36+ models with intelligent selection | ✅ Implemented |
| **Runtime** | Python | 3.11+ | Application runtime | ✅ Implemented |
| **LLM (Production)** | GPT-5.2 / Claude 4.5 | Latest | High-quality baseline | ✅ Supported |
| **LLM (Challenger)** | 250+ via Portkey | Latest | DeepSeek, Llama, Gemini, Mistral, etc. | ✅ Supported |

---

## 3. Component Deep Dive

### 3.1 Portkey AI Gateway - The Central Nervous System

Portkey serves as the **single point of control** for all LLM traffic, providing:

#### Virtual Keys (Billing Firewall)

```python
# Configuration for segregated billing
VIRTUAL_KEYS = {
    "production": {
        "key_id": "vk-prod-xxx",
        "budget_limit": None,  # Production: unlimited
        "rate_limit": "10000/min",
        "providers": ["openai", "anthropic"]
    },
    "shadow": {
        "key_id": "vk-shadow-xxx", 
        "budget_limit": 50.00,  # R&D: $50/month cap
        "rate_limit": "1000/min",
        "providers": ["deepseek", "together", "groq"]
    }
}
```

#### Config Objects (Routing Logic)

```python
# Production Config - Maximum quality
PRODUCTION_CONFIG = {
    "strategy": {"mode": "single"},
    "targets": [{
        "provider": "openai",
        "api_key": "{{virtual_key.production}}",
        "override_params": {
            "model": "gpt-5.2-turbo"
        }
    }],
    "cache": {
        "mode": "semantic",
        "max_age": 3600
    }
}

# Shadow Challenger Config - Cost optimization testing
SHADOW_CHALLENGER_CONFIG = {
    "strategy": {"mode": "fallback"},
    "targets": [
        {
            "provider": "deepseek",
            "api_key": "{{virtual_key.shadow}}",
            "override_params": {"model": "deepseek-v3"}
        },
        {
            "provider": "together",
            "api_key": "{{virtual_key.shadow}}",
            "override_params": {"model": "meta-llama/Llama-4-Scout-70B"}
        },
        {
            "provider": "groq",
            "api_key": "{{virtual_key.shadow}}",
            "override_params": {"model": "mixtral-next-8x22b"}
        }
    ],
    "retry": {
        "attempts": 3,
        "on_status_codes": [429, 500, 502, 503]
    }
}
```

#### Feedback API Integration

```python
from portkey_ai import Portkey

portkey = Portkey(api_key="PORTKEY_API_KEY")

# Push evaluation results back to Portkey for analytics
def push_feedback(trace_id: str, quality_score: float, metadata: dict):
    portkey.feedback.create(
        trace_id=f"shadow-{trace_id}",
        value=int(quality_score * 10),  # Scale 0-10
        weight=1.0,
        metadata={
            "_experiment": "shadow-optic-v1",
            "_challenger_model": metadata.get("challenger"),
            "faithfulness": metadata.get("faithfulness"),
            "cost_savings": metadata.get("cost_savings"),
            "refusal_rate": metadata.get("refusal_rate"),
            "latency_delta": metadata.get("latency_delta")
        }
    )
```

### 3.2 Temporal Orchestrator - Durable Workflow Engine

#### Why Temporal Over Simple Scripts?

| Scenario | Simple Script | Temporal Workflow |
|----------|---------------|-------------------|
| API timeout | ❌ Crashes, loses state | ✅ Retries automatically |
| Partial failure | ❌ Must restart from beginning | ✅ Resumes from last checkpoint |
| Rate limiting | ❌ Manual handling | ✅ Built-in backoff |
| Long-running (hours) | ❌ Process may be killed | ✅ Durable across restarts |
| Observability | ❌ Custom logging | ✅ Full history in Temporal UI |

#### Workflow Architecture

```python
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

# Activity retry configuration
ACTIVITY_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=10,
    non_retryable_error_types=["InvalidConfigError", "AuthenticationError"]
)

@workflow.defn
class OptimizeModelSelectionWorkflow:
    """
    Main Shadow-Optic workflow that runs continuously.
    Orchestrates: Log Fetch → Sampling → Replay → Evaluation → Feedback
    """
    
    @workflow.run
    async def run(self, config: OptimizationConfig) -> OptimizationReport:
        # Step 1: Fetch production logs from Portkey
        logs = await workflow.execute_activity(
            fetch_portkey_logs,
            FetchLogsInput(
                time_window=config.time_window,
                limit=config.sample_size,
                filters={"environment": "production"}
            ),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=ACTIVITY_RETRY_POLICY
        )
        
        # Step 2: Semantic clustering and sampling
        golden_prompts = await workflow.execute_activity(
            cluster_and_sample_prompts,
            ClusterInput(traces=logs, max_samples=config.max_golden_prompts),
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=ACTIVITY_RETRY_POLICY
        )
        
        # Step 3: Fan-out to multiple challenger models
        replay_tasks = []
        for challenger in config.challenger_models:
            task = workflow.execute_activity(
                replay_shadow_requests,
                ReplayInput(prompts=golden_prompts, challenger=challenger),
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=ACTIVITY_RETRY_POLICY
            )
            replay_tasks.append(task)
        
        shadow_results = await asyncio.gather(*replay_tasks)
        
        # Step 4: Evaluate quality with DeepEval
        evaluation_results = await workflow.execute_activity(
            evaluate_with_deepeval,
            EvaluationInput(
                production_responses=logs,
                shadow_responses=shadow_results,
                metrics=config.evaluation_metrics
            ),
            start_to_close_timeout=timedelta(minutes=15),
            retry_policy=ACTIVITY_RETRY_POLICY
        )
        
        # Step 5: Push feedback to Portkey
        await workflow.execute_activity(
            push_feedback_to_portkey,
            FeedbackInput(results=evaluation_results),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=ACTIVITY_RETRY_POLICY
        )
        
        # Step 6: Generate report
        return await workflow.execute_activity(
            generate_optimization_report,
            ReportInput(results=evaluation_results),
            start_to_close_timeout=timedelta(minutes=2)
        )
```

#### Scheduled Execution

```python
from temporalio.client import Client, Schedule, ScheduleSpec, ScheduleIntervalSpec

async def setup_scheduled_optimization():
    client = await Client.connect("localhost:7233")
    
    # Run optimization every 5 minutes
    await client.create_schedule(
        "shadow-optic-schedule",
        Schedule(
            action=ScheduleActionStartWorkflow(
                OptimizeModelSelectionWorkflow.run,
                OptimizationConfig.default(),
                id="shadow-optic-{timestamp}",
                task_queue="shadow-optic-queue"
            ),
            spec=ScheduleSpec(
                intervals=[ScheduleIntervalSpec(every=timedelta(minutes=5))]
            )
        )
    )
```

### 3.3 Qdrant - Semantic Stratified Sampling

#### The Problem with Random Sampling

If you randomly sample 50 prompts from 1000 production logs:
- You might get 40 "Hello, how are you?" variations
- And miss critical edge cases like complex coding questions

#### Semantic Stratified Sampling Algorithm

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
from sklearn.cluster import KMeans

class SemanticSampler:
    def __init__(self):
        self.qdrant = QdrantClient(":memory:")  # Or connect to Qdrant Cloud
        self.collection_name = "prompt_embeddings"
        
    async def initialize_collection(self, dimension: int = 1536):
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE
            )
        )
    
    async def sample_golden_prompts(
        self,
        traces: List[ProductionTrace],
        embeddings: List[np.ndarray],
        n_clusters: int = 50,
        samples_per_cluster: int = 3  # Centroid + 2 outliers
    ) -> List[GoldenPrompt]:
        """
        Semantic stratified sampling algorithm:
        1. Cluster prompts by semantic similarity
        2. For each cluster, select:
           - The centroid (most representative)
           - 2 outliers (edge cases)
        3. Return golden prompts for replay
        """
        
        # Store embeddings in Qdrant
        points = [
            PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "trace_id": trace.trace_id,
                    "prompt": trace.prompt,
                    "response": trace.response,
                    "cost": trace.cost,
                    "latency": trace.latency
                }
            )
            for i, (trace, embedding) in enumerate(zip(traces, embeddings))
        ]
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        
        # Cluster using K-Means
        embeddings_matrix = np.array(embeddings)
        kmeans = KMeans(n_clusters=min(n_clusters, len(traces) // 3))
        cluster_labels = kmeans.fit_predict(embeddings_matrix)
        
        golden_prompts = []
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Get cluster centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            
            # Find point closest to centroid (most representative)
            distances_to_centroid = [
                np.linalg.norm(embeddings_matrix[i] - centroid)
                for i in cluster_indices
            ]
            centroid_idx = cluster_indices[np.argmin(distances_to_centroid)]
            golden_prompts.append(self._create_golden_prompt(
                traces[centroid_idx], 
                sample_type="centroid",
                cluster_id=cluster_id
            ))
            
            # Find 2 outliers (furthest from centroid)
            if len(cluster_indices) > 2:
                outlier_indices = np.argsort(distances_to_centroid)[-2:]
                for oi in outlier_indices:
                    golden_prompts.append(self._create_golden_prompt(
                        traces[cluster_indices[oi]],
                        sample_type="outlier", 
                        cluster_id=cluster_id
                    ))
        
        return golden_prompts
    
    def _create_golden_prompt(
        self, 
        trace: ProductionTrace, 
        sample_type: str,
        cluster_id: int
    ) -> GoldenPrompt:
        return GoldenPrompt(
            original_trace_id=trace.trace_id,
            prompt=trace.prompt,
            production_response=trace.response,
            production_cost=trace.cost,
            production_latency=trace.latency,
            sample_type=sample_type,
            cluster_id=cluster_id
        )
```

### 3.4 DeepEval - The LLM Judge

#### Evaluation Metrics Suite

```python
from deepeval import evaluate
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

class ShadowEvaluator:
    """
    Multi-metric evaluation suite for shadow replay results.
    Uses LLM-as-Judge pattern with GPT-5.2 as the gold standard.
    """
    
    def __init__(self, judge_model: str = "gpt-5.2-turbo"):
        self.judge_model = judge_model
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        # Metric 1: Factual Consistency (Faithfulness)
        self.faithfulness_metric = FaithfulnessMetric(
            model=self.judge_model,
            threshold=0.8
        )
        
        # Metric 2: Answer Quality (G-Eval)
        self.quality_metric = GEval(
            name="Answer Quality",
            model=self.judge_model,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            criteria="""
            Evaluate the quality of the 'actual output' compared to the 'expected output':
            1. Completeness: Does the actual output address all aspects of the input?
            2. Accuracy: Is the information factually correct?
            3. Coherence: Is the response well-structured and easy to understand?
            4. Relevance: Does the response stay on topic?
            
            The expected output is from a production model (GPT-5.2).
            The actual output is from a challenger model (being tested for cost savings).
            """,
            threshold=0.7
        )
        
        # Metric 3: Conciseness (Token Efficiency)
        self.conciseness_metric = GEval(
            name="Conciseness",
            model=self.judge_model,
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            criteria="""
            Compare the token efficiency of 'actual output' vs 'expected output':
            - If actual output is more concise while preserving meaning: Score 0.8-1.0
            - If similar length: Score 0.5-0.7
            - If actual output is unnecessarily verbose: Score 0.0-0.4
            """,
            threshold=0.5
        )
    
    async def evaluate_single(
        self,
        prompt: str,
        production_response: str,
        shadow_response: str,
        context: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate a single shadow response against production baseline."""
        
        test_case = LLMTestCase(
            input=prompt,
            actual_output=shadow_response,
            expected_output=production_response,
            retrieval_context=[context] if context else None
        )
        
        # Run all metrics
        metrics_results = {}
        
        try:
            self.faithfulness_metric.measure(test_case)
            metrics_results["faithfulness"] = {
                "score": self.faithfulness_metric.score,
                "reason": self.faithfulness_metric.reason,
                "passed": self.faithfulness_metric.score >= 0.8
            }
        except Exception as e:
            metrics_results["faithfulness"] = {"error": str(e), "score": 0.0}
        
        try:
            self.quality_metric.measure(test_case)
            metrics_results["quality"] = {
                "score": self.quality_metric.score,
                "reason": self.quality_metric.reason,
                "passed": self.quality_metric.score >= 0.7
            }
        except Exception as e:
            metrics_results["quality"] = {"error": str(e), "score": 0.0}
        
        try:
            self.conciseness_metric.measure(test_case)
            metrics_results["conciseness"] = {
                "score": self.conciseness_metric.score,
                "reason": self.conciseness_metric.reason,
                "passed": self.conciseness_metric.score >= 0.5
            }
        except Exception as e:
            metrics_results["conciseness"] = {"error": str(e), "score": 0.0}
        
        # Calculate composite score
        scores = [m.get("score", 0) for m in metrics_results.values() if "score" in m]
        composite_score = sum(scores) / len(scores) if scores else 0.0
        
        return EvaluationResult(
            prompt=prompt,
            production_response=production_response,
            shadow_response=shadow_response,
            metrics=metrics_results,
            composite_score=composite_score,
            passed=all(m.get("passed", False) for m in metrics_results.values())
        )
    
    async def detect_refusals(
        self,
        production_responses: List[str],
        shadow_responses: List[str]
    ) -> RefusalAnalysis:
        """
        Detect and analyze refusal rate differences.
        A refusal is when the model declines to answer a prompt.
        """
        
        refusal_patterns = [
            "I cannot", "I'm unable to", "I can't help with",
            "I apologize, but", "As an AI", "I'm not able to",
            "I won't be able to", "I must decline"
        ]
        
        def is_refusal(response: str) -> bool:
            response_lower = response.lower()
            return any(pattern.lower() in response_lower for pattern in refusal_patterns)
        
        prod_refusals = sum(1 for r in production_responses if is_refusal(r))
        shadow_refusals = sum(1 for r in shadow_responses if is_refusal(r))
        
        prod_rate = prod_refusals / len(production_responses) if production_responses else 0
        shadow_rate = shadow_refusals / len(shadow_responses) if shadow_responses else 0
        
        return RefusalAnalysis(
            production_refusal_count=prod_refusals,
            shadow_refusal_count=shadow_refusals,
            production_refusal_rate=prod_rate,
            shadow_refusal_rate=shadow_rate,
            delta=shadow_rate - prod_rate,
            exceeds_threshold=abs(shadow_rate - prod_rate) > 0.01  # 1% threshold
        )
```

---

## 4. Data Flow & Processing Pipeline

### Complete Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHADOW-OPTIC DATA FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 1: PRODUCTION LOGGING                                                │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  User Request                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Portkey Gateway                                                      │   │
│  │ Headers:                                                             │   │
│  │   x-portkey-config: production-config                                │   │
│  │   x-portkey-virtual-key: prod-key                                    │   │
│  │   x-portkey-metadata: {"environment": "production", "_user": "xyz"}  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Portkey Logs (Stored)                                                │   │
│  │ {                                                                    │   │
│  │   "trace_id": "tr-abc123",                                           │   │
│  │   "request": {"messages": [...], "model": "gpt-5.2"},                │   │
│  │   "response": {"choices": [...], "usage": {...}},                    │   │
│  │   "cost": 0.015,                                                     │   │
│  │   "latency_ms": 1250,                                                │   │
│  │   "timestamp": "2026-01-17T10:30:00Z"                                │   │
│  │ }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 2: LOG INGESTION & SAMPLING (Every 5 minutes)                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  Temporal Workflow Trigger                                                  │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Activity: fetch_portkey_logs                                         │   │
│  │ - Query: Last 5 minutes, environment=production                      │   │
│  │ - Output: 1,000 production traces                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Activity: generate_embeddings                                        │   │
│  │ - Model: text-embedding-3-small                                      │   │
│  │ - Input: User prompts from traces                                    │   │
│  │ - Output: 1,000 x 1536-dim vectors                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Activity: cluster_and_sample                                         │   │
│  │ - Algorithm: K-Means clustering (k=50)                               │   │
│  │ - Selection: Centroid + 2 outliers per cluster                       │   │
│  │ - Output: 50 "Golden Prompts"                                        │   │
│  │ - Savings: 95% compute reduction                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 3: SHADOW REPLAY (Fan-out to Challengers)                            │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  50 Golden Prompts                                                          │
│       │                                                                     │
│       ├───────────────────────────────────────────────┐                     │
│       │                                               │                     │
│       ▼                                               ▼                     │
│  ┌────────────────────────┐                 ┌────────────────────────┐     │
│  │ Challenger: DeepSeek-V3│                 │ Challenger: Llama-4    │     │
│  │ Headers:               │                 │ Headers:               │     │
│  │  x-portkey-config:     │                 │  x-portkey-config:     │     │
│  │   shadow-deepseek      │                 │   shadow-llama4        │     │
│  │  x-portkey-virtual-key:│                 │  x-portkey-virtual-key:│     │
│  │   shadow-ops-key       │                 │   shadow-ops-key       │     │
│  │  x-portkey-trace-id:   │                 │  x-portkey-trace-id:   │     │
│  │   shadow-{orig_id}     │                 │   shadow-{orig_id}     │     │
│  └────────────────────────┘                 └────────────────────────┘     │
│       │                                               │                     │
│       ▼                                               ▼                     │
│  ┌────────────────────────┐                 ┌────────────────────────┐     │
│  │ Shadow Response        │                 │ Shadow Response        │     │
│  │ + Cost: $0.00014       │                 │ + Cost: $0.0003        │     │
│  │ + Latency: 800ms       │                 │ + Latency: 650ms       │     │
│  └────────────────────────┘                 └────────────────────────┘     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 4: EVALUATION & SCORING                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ DeepEval Judge (GPT-5.2)                                             │   │
│  │                                                                      │   │
│  │ For each (Production, Shadow) pair:                                  │   │
│  │                                                                      │   │
│  │ 1. Faithfulness Score    →  0.96  (Does shadow contradict prod?)    │   │
│  │ 2. Quality Score         →  0.89  (Overall answer quality)          │   │
│  │ 3. Conciseness Score     →  0.72  (Token efficiency)                │   │
│  │ 4. Refusal Detection     →  false (Did shadow refuse?)              │   │
│  │                                                                      │   │
│  │ Composite Score = (0.96 + 0.89 + 0.72) / 3 = 0.857                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STAGE 5: FEEDBACK & REPORTING                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Portkey Feedback API                                                 │   │
│  │                                                                      │   │
│  │ POST /v1/feedback                                                    │   │
│  │ {                                                                    │   │
│  │   "trace_id": "shadow-tr-abc123",                                    │   │
│  │   "value": 9,                                                        │   │
│  │   "metadata": {                                                      │   │
│  │     "challenger": "deepseek-v3",                                     │   │
│  │     "faithfulness": 0.96,                                            │   │
│  │     "quality": 0.89,                                                 │   │
│  │     "conciseness": 0.72,                                             │   │
│  │     "composite": 0.857,                                              │   │
│  │     "cost_savings_pct": 99.1,                                        │   │
│  │     "latency_delta_ms": -450                                         │   │
│  │   }                                                                  │   │
│  │ }                                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Optimization Report                                                  │   │
│  │                                                                      │   │
│  │ ┌─────────────────────────────────────────────────────────────────┐ │   │
│  │ │ SHADOW-OPTIC OPTIMIZATION REPORT                                │ │   │
│  │ │ Generated: 2026-01-17T10:35:00Z                                 │ │   │
│  │ │                                                                 │ │   │
│  │ │ CHALLENGER: DeepSeek-V3                                         │ │   │
│  │ │ ─────────────────────────────────────────────────────────────── │ │   │
│  │ │ Metric          │ Production  │ Shadow   │ Delta   │ Status    │ │   │
│  │ │ ─────────────────────────────────────────────────────────────── │ │   │
│  │ │ Cost/1k req     │ $15.00      │ $0.14    │ -99.1%  │ ✅ WIN    │ │   │
│  │ │ Faithfulness    │ 0.99        │ 0.96     │ -3.0%   │ ⚠️ OK     │ │   │
│  │ │ Quality         │ 0.95        │ 0.89     │ -6.3%   │ ⚠️ OK     │ │   │
│  │ │ Refusal Rate    │ 0.5%        │ 4.2%     │ +3.7%   │ ❌ FAIL   │ │   │
│  │ │ Latency (p50)   │ 1250ms      │ 800ms    │ -36%    │ ✅ WIN    │ │   │
│  │ │                                                                 │ │   │
│  │ │ RECOMMENDATION: DO NOT SWITCH                                   │ │   │
│  │ │ Reason: Refusal rate (4.2%) exceeds threshold (1.0%)            │ │   │
│  │ │ Suggested Action: Fine-tune guardrails or test Llama-4 Scout    │ │   │
│  │ └─────────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Portkey Integration Strategy

### 5.1 Headers Reference

| Header | Purpose | Example |
|--------|---------|---------|
| `x-portkey-api-key` | Authentication | `pk-xxx` |
| `x-portkey-config` | Routing configuration | `pc-shadow-config` or JSON |
| `x-portkey-virtual-key` | Billing segregation | `vk-shadow-ops` |
| `x-portkey-trace-id` | Request tracing | `shadow-{original_id}` |
| `x-portkey-metadata` | Custom metadata | `{"experiment": "v1"}` |
| `x-portkey-cache` | Cache control | `semantic` or `simple` |
| `x-portkey-retry` | Retry configuration | `{"attempts": 3}` |

### 5.2 Log Export API

```python
import httpx
from datetime import datetime, timedelta

class PortkeyLogExporter:
    def __init__(self, api_key: str):
        self.base_url = "https://api.portkey.ai/v1"
        self.headers = {"x-portkey-api-key": api_key}
    
    async def fetch_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        filters: dict = None,
        limit: int = 1000
    ) -> List[dict]:
        """
        Fetch production logs from Portkey.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            filters: Optional filters (environment, model, etc.)
            limit: Maximum logs to fetch
        
        Returns:
            List of log entries with full request/response data
        """
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/logs",
                headers=self.headers,
                params={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "limit": limit,
                    **(filters or {})
                }
            )
            response.raise_for_status()
            return response.json()["logs"]
```

### 5.3 Feedback API Integration

```python
from portkey_ai import Portkey

class FeedbackPusher:
    def __init__(self, api_key: str):
        self.portkey = Portkey(api_key=api_key)
    
    async def push_evaluation_result(
        self,
        original_trace_id: str,
        evaluation: EvaluationResult,
        challenger_model: str,
        cost_savings: float
    ):
        """Push evaluation results to Portkey for analytics."""
        
        self.portkey.feedback.create(
            trace_id=f"shadow-{original_trace_id}",
            value=int(evaluation.composite_score * 10),  # 0-10 scale
            weight=1.0,
            metadata={
                # Indexed fields (start with _)
                "_experiment": "shadow-optic",
                "_challenger": challenger_model,
                "_environment": "shadow",
                
                # Metric details
                "faithfulness_score": evaluation.metrics.get("faithfulness", {}).get("score"),
                "quality_score": evaluation.metrics.get("quality", {}).get("score"),
                "conciseness_score": evaluation.metrics.get("conciseness", {}).get("score"),
                "composite_score": evaluation.composite_score,
                
                # Cost analysis
                "cost_savings_percent": cost_savings,
                "production_cost": evaluation.production_cost,
                "shadow_cost": evaluation.shadow_cost,
                
                # Decision
                "recommendation": "switch" if evaluation.passed else "do_not_switch",
                "failure_reasons": evaluation.failure_reasons
            }
        )
```

---

## 6. Evaluation Framework

### 6.1 Metric Definitions

| Metric | Description | Threshold | Weight |
|--------|-------------|-----------|--------|
| **Faithfulness** | Does shadow response contradict production? | ≥ 0.80 | 0.35 |
| **Quality** | Overall answer correctness and completeness | ≥ 0.70 | 0.35 |
| **Conciseness** | Token efficiency (shorter = better cost) | ≥ 0.50 | 0.15 |
| **Refusal Rate** | % of prompts refused by shadow vs prod | ≤ +1.0% | 0.15 |

### 6.2 Decision Matrix

```python
class OptimizationDecisionEngine:
    """
    Decision engine for model switch recommendations.
    Implements clear trade-off logic.
    """
    
    THRESHOLDS = {
        "faithfulness_min": 0.80,
        "quality_min": 0.70,
        "refusal_rate_max_delta": 0.01,  # 1%
        "cost_savings_min": 0.20,  # 20%
        "latency_increase_max": 0.50,  # 50%
    }
    
    def make_recommendation(
        self,
        metrics: dict,
        cost_analysis: CostAnalysis
    ) -> Recommendation:
        """
        Generate switch recommendation based on metrics.
        
        Decision Logic:
        1. Check hard blockers (refusal rate, safety)
        2. Check quality thresholds
        3. Calculate ROI if quality passes
        4. Generate recommendation with reasoning
        """
        
        blockers = []
        warnings = []
        
        # Check hard blockers
        if metrics.get("refusal_rate_delta", 0) > self.THRESHOLDS["refusal_rate_max_delta"]:
            blockers.append(
                f"Refusal rate increase ({metrics['refusal_rate_delta']:.1%}) "
                f"exceeds threshold ({self.THRESHOLDS['refusal_rate_max_delta']:.1%})"
            )
        
        if metrics.get("faithfulness", 1.0) < self.THRESHOLDS["faithfulness_min"]:
            blockers.append(
                f"Faithfulness ({metrics['faithfulness']:.2f}) "
                f"below minimum ({self.THRESHOLDS['faithfulness_min']})"
            )
        
        # Check warnings
        if metrics.get("quality", 1.0) < self.THRESHOLDS["quality_min"]:
            warnings.append(
                f"Quality ({metrics['quality']:.2f}) "
                f"below recommended ({self.THRESHOLDS['quality_min']})"
            )
        
        # Calculate ROI
        cost_savings = cost_analysis.savings_percent
        quality_impact = 1.0 - metrics.get("composite_score", 0)
        roi_score = cost_savings * (1 - quality_impact)
        
        # Generate recommendation
        if blockers:
            return Recommendation(
                action="DO_NOT_SWITCH",
                confidence=0.95,
                reasoning=f"Blocked by: {'; '.join(blockers)}",
                suggested_actions=[
                    "Fine-tune model guardrails",
                    "Test alternative challenger models",
                    "Review refusal patterns"
                ],
                metrics_summary=metrics,
                cost_analysis=cost_analysis
            )
        elif warnings and cost_savings < self.THRESHOLDS["cost_savings_min"]:
            return Recommendation(
                action="NOT_RECOMMENDED",
                confidence=0.70,
                reasoning=f"Warnings: {'; '.join(warnings)}. Cost savings insufficient.",
                suggested_actions=[
                    "Monitor for quality improvements in challenger model",
                    "Consider partial traffic migration"
                ],
                metrics_summary=metrics,
                cost_analysis=cost_analysis
            )
        else:
            return Recommendation(
                action="SWITCH_RECOMMENDED",
                confidence=0.85 if not warnings else 0.65,
                reasoning=f"ROI Score: {roi_score:.2f}. Cost savings: {cost_savings:.1%}",
                suggested_actions=[
                    f"Migrate {min(10, int(cost_savings * 100))}% traffic initially",
                    "Set up A/B monitoring",
                    "Configure automatic rollback"
                ],
                metrics_summary=metrics,
                cost_analysis=cost_analysis
            )
```

---

## 7. State Management & Orchestration

### 7.1 Temporal Workflow States

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKFLOW STATE MACHINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│    ┌──────────┐                                                 │
│    │ PENDING  │ ◄──── Workflow scheduled                        │
│    └────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│    ┌──────────┐                                                 │
│    │ FETCHING │ ◄──── Fetching logs from Portkey                │
│    │  LOGS    │                                                 │
│    └────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│    ┌──────────┐                                                 │
│    │ SAMPLING │ ◄──── Clustering & selecting golden prompts     │
│    └────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│    ┌──────────┐                                                 │
│    │ REPLAYING│ ◄──── Fan-out to challenger models              │
│    └────┬─────┘       (Parallel execution)                      │
│         │                                                       │
│         ▼                                                       │
│    ┌──────────┐                                                 │
│    │EVALUATING│ ◄──── Running DeepEval metrics                  │
│    └────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│    ┌──────────┐                                                 │
│    │ FEEDBACK │ ◄──── Pushing to Portkey                        │
│    └────┬─────┘                                                 │
│         │                                                       │
│         ▼                                                       │
│    ┌──────────┐                                                 │
│    │COMPLETED │ ◄──── Report generated                          │
│    └──────────┘                                                 │
│                                                                 │
│    Error States:                                                │
│    ┌──────────┐                                                 │
│    │ RETRYING │ ◄──── Activity failed, retrying with backoff    │
│    └──────────┘                                                 │
│    ┌──────────┐                                                 │
│    │  FAILED  │ ◄──── Max retries exceeded                      │
│    └──────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Activity Definitions

```python
from temporalio import activity
from dataclasses import dataclass
from typing import List

@dataclass
class FetchLogsInput:
    time_window: timedelta
    limit: int
    filters: dict

@dataclass
class ProductionTrace:
    trace_id: str
    prompt: str
    response: str
    cost: float
    latency: float
    model: str
    timestamp: datetime

@activity.defn
async def fetch_portkey_logs(input: FetchLogsInput) -> List[ProductionTrace]:
    """Fetch production logs from Portkey API."""
    exporter = PortkeyLogExporter(os.environ["PORTKEY_API_KEY"])
    
    end_time = datetime.utcnow()
    start_time = end_time - input.time_window
    
    raw_logs = await exporter.fetch_logs(
        start_time=start_time,
        end_time=end_time,
        filters=input.filters,
        limit=input.limit
    )
    
    return [
        ProductionTrace(
            trace_id=log["trace_id"],
            prompt=extract_prompt(log["request"]),
            response=extract_response(log["response"]),
            cost=log["cost"],
            latency=log["latency_ms"],
            model=log["model"],
            timestamp=datetime.fromisoformat(log["timestamp"])
        )
        for log in raw_logs
    ]

@activity.defn
async def cluster_and_sample_prompts(input: ClusterInput) -> List[GoldenPrompt]:
    """Cluster prompts semantically and select representatives."""
    sampler = SemanticSampler()
    
    # Generate embeddings
    embeddings = await generate_embeddings(
        [trace.prompt for trace in input.traces]
    )
    
    # Cluster and sample
    return await sampler.sample_golden_prompts(
        traces=input.traces,
        embeddings=embeddings,
        n_clusters=input.max_samples
    )

@activity.defn
async def replay_shadow_requests(input: ReplayInput) -> List[ShadowResult]:
    """Replay golden prompts against a challenger model."""
    from portkey_ai import Portkey
    
    portkey = Portkey(
        api_key=os.environ["PORTKEY_API_KEY"],
        config=f"shadow-{input.challenger}-config",
        virtual_key="shadow-ops-key"
    )
    
    results = []
    for prompt in input.prompts:
        try:
            start_time = time.time()
            response = await portkey.chat.completions.create(
                messages=[{"role": "user", "content": prompt.prompt}],
                model=input.challenger,
                extra_headers={
                    "x-portkey-trace-id": f"shadow-{prompt.original_trace_id}"
                }
            )
            latency = (time.time() - start_time) * 1000
            
            results.append(ShadowResult(
                original_trace_id=prompt.original_trace_id,
                challenger_model=input.challenger,
                response=response.choices[0].message.content,
                cost=calculate_cost(response.usage, input.challenger),
                latency=latency,
                success=True
            ))
        except Exception as e:
            activity.heartbeat(f"Error replaying: {str(e)}")
            results.append(ShadowResult(
                original_trace_id=prompt.original_trace_id,
                challenger_model=input.challenger,
                response=None,
                cost=0,
                latency=0,
                success=False,
                error=str(e)
            ))
    
    return results

@activity.defn
async def evaluate_with_deepeval(input: EvaluationInput) -> EvaluationResults:
    """Evaluate shadow responses against production using DeepEval."""
    evaluator = ShadowEvaluator()
    
    results = []
    for prod, shadow in zip(input.production_responses, input.shadow_responses):
        if shadow.success:
            result = await evaluator.evaluate_single(
                prompt=prod.prompt,
                production_response=prod.response,
                shadow_response=shadow.response
            )
            results.append(result)
    
    # Calculate aggregate metrics
    refusal_analysis = await evaluator.detect_refusals(
        [r.production_response for r in results],
        [r.shadow_response for r in results]
    )
    
    return EvaluationResults(
        individual_results=results,
        refusal_analysis=refusal_analysis,
        aggregate_metrics=calculate_aggregates(results)
    )
```

---

## 8. Observability & Explainability

### 8.1 Arize Phoenix Integration

```python
import phoenix as px
from phoenix.trace import SpanEvaluations

class PhoenixObserver:
    """
    Arize Phoenix integration for deep-dive trace visualization.
    Complements Portkey's high-level analytics with detailed evaluation traces.
    """
    
    def __init__(self):
        # Launch Phoenix UI
        self.session = px.launch_app()
        print(f"Phoenix UI available at: {self.session.url}")
    
    def log_evaluation_trace(
        self,
        evaluation: EvaluationResult,
        shadow_result: ShadowResult
    ):
        """Log detailed evaluation trace to Phoenix."""
        
        # Create trace with all context
        with px.trace("shadow_evaluation") as trace:
            trace.set_attribute("original_trace_id", evaluation.original_trace_id)
            trace.set_attribute("challenger_model", shadow_result.challenger_model)
            
            # Log the prompt
            with trace.span("prompt") as span:
                span.set_attribute("content", evaluation.prompt)
            
            # Log production response
            with trace.span("production_response") as span:
                span.set_attribute("content", evaluation.production_response)
                span.set_attribute("cost", shadow_result.production_cost)
            
            # Log shadow response
            with trace.span("shadow_response") as span:
                span.set_attribute("content", evaluation.shadow_response)
                span.set_attribute("cost", shadow_result.shadow_cost)
                span.set_attribute("latency", shadow_result.latency)
            
            # Log each metric evaluation
            for metric_name, metric_result in evaluation.metrics.items():
                with trace.span(f"metric_{metric_name}") as span:
                    span.set_attribute("score", metric_result.get("score", 0))
                    span.set_attribute("reason", metric_result.get("reason", ""))
                    span.set_attribute("passed", metric_result.get("passed", False))
            
            # Log final decision
            with trace.span("decision") as span:
                span.set_attribute("composite_score", evaluation.composite_score)
                span.set_attribute("passed", evaluation.passed)
                span.set_attribute("failure_reasons", evaluation.failure_reasons)
```

### 8.2 Metrics & Dashboards

#### Key Performance Indicators (KPIs)

| KPI | Description | Target |
|-----|-------------|--------|
| **Shadow Coverage** | % of production traces replayed | > 5% |
| **Evaluation Success Rate** | % of replays successfully evaluated | > 95% |
| **Average Quality Score** | Mean composite score across challengers | > 0.75 |
| **Cost Savings Identified** | Potential savings from best challenger | Report only |
| **Refusal Rate Delta** | Difference in refusal rates | < 1% |

#### Portkey Dashboard Queries

```sql
-- Query for Portkey Analytics Dashboard

-- Average quality score by challenger model
SELECT 
    metadata->>'_challenger' as challenger,
    AVG(value) as avg_quality_score,
    COUNT(*) as sample_count
FROM feedback
WHERE metadata->>'_experiment' = 'shadow-optic'
GROUP BY metadata->>'_challenger'
ORDER BY avg_quality_score DESC;

-- Cost savings trend over time
SELECT 
    DATE_TRUNC('hour', created_at) as hour,
    metadata->>'_challenger' as challenger,
    AVG((metadata->>'cost_savings_percent')::float) as avg_savings
FROM feedback
WHERE metadata->>'_experiment' = 'shadow-optic'
GROUP BY hour, challenger
ORDER BY hour DESC;

-- Failure analysis
SELECT 
    metadata->>'failure_reasons' as failure_reason,
    COUNT(*) as occurrence_count
FROM feedback
WHERE 
    metadata->>'_experiment' = 'shadow-optic'
    AND metadata->>'recommendation' = 'do_not_switch'
GROUP BY metadata->>'failure_reasons'
ORDER BY occurrence_count DESC;
```

---

## 9. Failure Handling & Resilience

### 9.1 Retry Policies

```python
from temporalio.common import RetryPolicy
from datetime import timedelta

# Default retry policy for API calls
DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=10,
    non_retryable_error_types=[
        "AuthenticationError",
        "InvalidConfigError",
        "BudgetExceededError"
    ]
)

# Aggressive retry for critical operations
CRITICAL_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(milliseconds=500),
    backoff_coefficient=1.5,
    maximum_interval=timedelta(minutes=2),
    maximum_attempts=20
)

# Conservative retry for expensive operations
EXPENSIVE_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=3.0,
    maximum_interval=timedelta(minutes=10),
    maximum_attempts=5
)
```

### 9.2 Error Classification

| Error Type | Handling | Retry? |
|------------|----------|--------|
| **Rate Limit (429)** | Exponential backoff | Yes |
| **Server Error (5xx)** | Immediate retry | Yes |
| **Authentication (401)** | Alert, stop workflow | No |
| **Budget Exceeded** | Alert, pause shadow ops | No |
| **Invalid Request (400)** | Log, skip sample | No |
| **Timeout** | Retry with longer timeout | Yes |
| **Model Not Available** | Fallback to next model | Automatic |

### 9.3 Circuit Breaker Pattern

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, stop requests
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreaker:
    """
    Circuit breaker for challenger model APIs.
    Prevents cascade failures when a provider is down.
    """
    
    failure_threshold: int = 5
    recovery_timeout: timedelta = timedelta(minutes=5)
    half_open_requests: int = 3
    
    def __init__(self, name: str):
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count_in_half_open = 0
    
    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.success_count_in_half_open += 1
            if self.success_count_in_half_open >= self.half_open_requests:
                self._close()
        else:
            self.failure_count = 0
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self._open()
    
    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if datetime.utcnow() - self.last_failure_time > self.recovery_timeout:
                self._half_open()
                return True
            return False
        
        # HALF_OPEN: Allow limited requests
        return True
    
    def _open(self):
        self.state = CircuitState.OPEN
        activity.logger.warning(f"Circuit breaker OPENED for {self.name}")
    
    def _close(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count_in_half_open = 0
        activity.logger.info(f"Circuit breaker CLOSED for {self.name}")
    
    def _half_open(self):
        self.state = CircuitState.HALF_OPEN
        self.success_count_in_half_open = 0
        activity.logger.info(f"Circuit breaker HALF_OPEN for {self.name}")
```

---

## 10. Security & Cost Controls

### 10.1 Virtual Key Budget Limits

```python
# Portkey Virtual Key Configuration
VIRTUAL_KEY_CONFIG = {
    "production": {
        "name": "Production Operations",
        "budget": {
            "type": "unlimited",
            "alert_threshold": None
        },
        "rate_limits": {
            "requests_per_minute": 10000,
            "tokens_per_minute": 1000000
        },
        "allowed_models": ["gpt-5.2-turbo", "claude-4.5-sonnet"]
    },
    "shadow-ops": {
        "name": "Shadow Operations R&D",
        "budget": {
            "type": "monthly",
            "limit_usd": 50.00,
            "alert_threshold": 0.80,  # Alert at 80%
            "hard_stop": True  # Stop when budget exhausted
        },
        "rate_limits": {
            "requests_per_minute": 1000,
            "tokens_per_minute": 100000
        },
        "allowed_models": ["deepseek-v3", "llama-4-scout-70b", "mixtral-next"]
    }
}
```

### 10.2 Data Privacy

```python
class DataPrivacyHandler:
    """
    Handles PII and sensitive data in shadow replays.
    Ensures compliance with data protection requirements.
    """
    
    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{16}\b',  # Credit card
        r'\b[\w.-]+@[\w.-]+\.\w+\b',  # Email
        r'\b\d{10}\b',  # Phone
    ]
    
    def __init__(self, redaction_enabled: bool = True):
        self.redaction_enabled = redaction_enabled
        self.pii_regex = [re.compile(p) for p in self.PII_PATTERNS]
    
    def sanitize_for_shadow(self, trace: ProductionTrace) -> ProductionTrace:
        """
        Sanitize trace before shadow replay.
        - Redact PII from prompts
        - Remove user identifiers
        - Hash any remaining sensitive fields
        """
        if not self.redaction_enabled:
            return trace
        
        sanitized_prompt = trace.prompt
        for pattern in self.pii_regex:
            sanitized_prompt = pattern.sub("[REDACTED]", sanitized_prompt)
        
        return ProductionTrace(
            trace_id=trace.trace_id,
            prompt=sanitized_prompt,
            response=trace.response,  # Production response stays for comparison
            cost=trace.cost,
            latency=trace.latency,
            model=trace.model,
            timestamp=trace.timestamp
        )
    
    def validate_no_pii_in_feedback(self, metadata: dict) -> dict:
        """Ensure no PII leaks into feedback metadata."""
        clean_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                for pattern in self.pii_regex:
                    if pattern.search(value):
                        value = "[CONTAINS_PII]"
                        break
            clean_metadata[key] = value
        return clean_metadata
```

---

## 11. Deployment Architecture

### 11.1 Docker Compose (Development)

```yaml
version: '3.8'

services:
  # Temporal Server
  temporal:
    image: temporalio/auto-setup:1.21
    ports:
      - "7233:7233"
    environment:
      - DB=postgresql
      - DB_PORT=5432
      - POSTGRES_USER=temporal
      - POSTGRES_PWD=temporal
      - POSTGRES_SEEDS=postgres
    depends_on:
      - postgres

  # Temporal Web UI
  temporal-ui:
    image: temporalio/ui:2.21
    ports:
      - "8080:8080"
    environment:
      - TEMPORAL_ADDRESS=temporal:7233

  # PostgreSQL (Temporal backend)
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=temporal
      - POSTGRES_PASSWORD=temporal
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:v1.16.3
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  # Shadow-Optic Worker
  shadow-optic-worker:
    build: .
    environment:
      - PORTKEY_API_KEY=${PORTKEY_API_KEY}
      - TEMPORAL_ADDRESS=temporal:7233
      - QDRANT_URL=http://qdrant:6333
      - OPENAI_API_KEY=${OPENAI_API_KEY}  # For DeepEval judge
    depends_on:
      - temporal
      - qdrant
    command: python -m shadow_optic.worker

  # Phoenix Observability
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"

volumes:
  postgres_data:
  qdrant_data:
```

### 11.2 Kubernetes (Production)

```yaml
# shadow-optic-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shadow-optic-worker
  labels:
    app: shadow-optic
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shadow-optic
  template:
    metadata:
      labels:
        app: shadow-optic
    spec:
      containers:
      - name: worker
        image: shadow-optic:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: PORTKEY_API_KEY
          valueFrom:
            secretKeyRef:
              name: shadow-optic-secrets
              key: portkey-api-key
        - name: TEMPORAL_ADDRESS
          value: "temporal.temporal.svc.cluster.local:7233"
        - name: QDRANT_URL
          value: "http://qdrant.qdrant.svc.cluster.local:6333"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: shadow-optic
spec:
  selector:
    app: shadow-optic
  ports:
  - port: 8000
    targetPort: 8000
```

---

## 12. API Reference

### 12.1 Configuration API

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import timedelta

class ChallengerConfig(BaseModel):
    """Configuration for a challenger model."""
    model_id: str = Field(..., description="Model identifier (e.g., 'deepseek-v3')")
    provider: str = Field(..., description="Provider name (e.g., 'deepseek')")
    portkey_config_id: str = Field(..., description="Portkey config ID for routing")
    max_tokens: int = Field(default=4096, description="Max tokens for generation")
    temperature: float = Field(default=0.7, ge=0, le=2)

class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""
    faithfulness_threshold: float = Field(default=0.8, ge=0, le=1)
    quality_threshold: float = Field(default=0.7, ge=0, le=1)
    conciseness_threshold: float = Field(default=0.5, ge=0, le=1)
    max_refusal_rate_delta: float = Field(default=0.01, ge=0, le=1)
    judge_model: str = Field(default="gpt-5.2-turbo")

class SamplingConfig(BaseModel):
    """Configuration for semantic sampling."""
    max_logs_to_fetch: int = Field(default=1000, ge=100, le=10000)
    n_clusters: int = Field(default=50, ge=10, le=200)
    samples_per_cluster: int = Field(default=3, ge=1, le=10)
    embedding_model: str = Field(default="text-embedding-3-small")

class OptimizationConfig(BaseModel):
    """Main configuration for Shadow-Optic optimization."""
    
    # Scheduling
    run_interval: timedelta = Field(default=timedelta(minutes=5))
    time_window: timedelta = Field(default=timedelta(minutes=5))
    
    # Challengers
    challenger_models: List[ChallengerConfig] = Field(
        default_factory=lambda: [
            ChallengerConfig(
                model_id="deepseek-v3",
                provider="deepseek",
                portkey_config_id="shadow-deepseek-config"
            ),
            ChallengerConfig(
                model_id="llama-4-scout-70b",
                provider="together",
                portkey_config_id="shadow-llama4-config"
            )
        ]
    )
    
    # Sampling
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    
    # Evaluation
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Budget
    shadow_budget_limit_usd: float = Field(default=50.0, ge=1.0)
    
    @classmethod
    def default(cls) -> "OptimizationConfig":
        return cls()
```

### 12.2 Output Models

```python
from enum import Enum
from datetime import datetime

class RecommendationAction(str, Enum):
    SWITCH_RECOMMENDED = "switch_recommended"
    NOT_RECOMMENDED = "not_recommended"
    DO_NOT_SWITCH = "do_not_switch"

class Recommendation(BaseModel):
    """Final recommendation from Shadow-Optic."""
    action: RecommendationAction
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    suggested_actions: List[str]
    metrics_summary: dict
    cost_analysis: "CostAnalysis"

class CostAnalysis(BaseModel):
    """Cost comparison between production and shadow."""
    production_cost_per_1k: float
    shadow_cost_per_1k: float
    savings_percent: float
    estimated_monthly_savings: float

class OptimizationReport(BaseModel):
    """Complete optimization report."""
    report_id: str
    generated_at: datetime
    time_window: timedelta
    samples_evaluated: int
    
    challenger_results: dict[str, "ChallengerResult"]
    best_challenger: Optional[str]
    recommendation: Recommendation
    
    execution_metrics: dict

class ChallengerResult(BaseModel):
    """Results for a single challenger model."""
    model_id: str
    samples_processed: int
    success_rate: float
    
    avg_faithfulness: float
    avg_quality: float
    avg_conciseness: float
    refusal_rate: float
    
    avg_latency_ms: float
    avg_cost_per_request: float
    
    passed_thresholds: bool
    failure_reasons: List[str]
```

---

## Appendix A: Quick Start Guide

### Prerequisites

1. Python 3.11+
2. Docker & Docker Compose
3. Portkey API Key
4. OpenAI API Key (for DeepEval judge)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/shadow-optic.git
cd shadow-optic

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export PORTKEY_API_KEY="your-portkey-key"
export OPENAI_API_KEY="your-openai-key"

# Start infrastructure
docker-compose up -d

# Run worker
python -m shadow_optic.worker
```

### First Run

```python
from shadow_optic import ShadowOptic

# Initialize
so = ShadowOptic.from_env()

# Run single optimization cycle
report = await so.run_optimization()

# Print results
print(f"Best challenger: {report.best_challenger}")
print(f"Recommendation: {report.recommendation.action}")
print(f"Cost savings: {report.recommendation.cost_analysis.savings_percent:.1%}")
```

---

## Appendix B: Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Workflow stuck in RETRYING | API rate limit | Increase retry intervals |
| Low faithfulness scores | Model quality mismatch | Try different challenger |
| High refusal rate | Model guardrails | Fine-tune system prompts |
| Budget exceeded | High replay volume | Reduce sampling rate |
| Qdrant connection failed | Container not ready | Wait or restart container |

---

## Appendix C: Model Pricing Reference (Jan 2026)

| Model | Provider | Input $/1M | Output $/1M | Notes |
|-------|----------|------------|-------------|-------|
| GPT-5.2-turbo | OpenAI | $5.00 | $15.00 | Production default |
| Claude 4.5 Sonnet | Anthropic | $3.00 | $15.00 | High reasoning |
| DeepSeek-V3 | DeepSeek | $0.07 | $0.14 | 99% cost reduction |
| Llama-4-Scout-70B | Together | $0.50 | $0.80 | Open source |
| Mixtral-Next-8x22B | Groq | $0.30 | $0.50 | Fast inference |

---

*Document Version: 1.0.0*  
*Last Updated: January 17, 2026*  
*Author: Shadow-Optic Team*
