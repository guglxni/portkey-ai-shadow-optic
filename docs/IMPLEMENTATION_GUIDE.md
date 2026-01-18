# Shadow-Optic Implementation Guide

**A Step-by-Step Guide to Building and Deploying Shadow-Optic**

---

## Table of Contents

1. [Prerequisites & Setup](#prerequisites--setup)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Phase 1: Core Infrastructure](#phase-1-core-infrastructure)
4. [Phase 2: Portkey Integration](#phase-2-portkey-integration)
5. [Phase 3: Semantic Sampling](#phase-3-semantic-sampling)
6. [Phase 4: Shadow Replay](#phase-4-shadow-replay)
7. [Phase 5: LLM-as-Judge Evaluation](#phase-5-llm-as-judge-evaluation)
8. [Phase 6: Decision Engine](#phase-6-decision-engine)
9. [Phase 7: Workflow Orchestration](#phase-7-workflow-orchestration)
10. [Phase 8: Production Deployment](#phase-8-production-deployment)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites & Setup

### Required Accounts

1. **Portkey Account** (https://portkey.ai)
   - Sign up for free
   - Get your API key from Dashboard → Settings → API Keys
   - Create Virtual Keys for production and shadow testing

2. **OpenAI Account** (https://platform.openai.com)
   - Required for DeepEval's LLM-as-Judge
   - Get API key from API Keys page

3. **DeepSeek/Together/Other Challenger Providers**
   - Sign up with your preferred challenger model providers
   - Get API keys for each

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/shadow-optic/shadow-optic.git
cd shadow-optic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Edit environment variables
# Required: PORTKEY_API_KEY, OPENAI_API_KEY
nano .env
```

### Infrastructure Services

Start the required services using Docker Compose:

```bash
# Start all infrastructure
docker-compose up -d

# Verify services are healthy
docker-compose ps

# Expected output:
# shadow-optic-temporal      running (healthy)
# shadow-optic-temporal-db   running (healthy)
# shadow-optic-temporal-ui   running (healthy)
# shadow-optic-qdrant        running (healthy)
# shadow-optic-phoenix       running (healthy)
```

Access the UIs:
- Temporal UI: http://localhost:8080
- Phoenix UI: http://localhost:6006
- Qdrant Dashboard: http://localhost:6333/dashboard

---

## Understanding the Architecture

### Data Flow

```
Production Traffic
      │
      ▼
┌──────────────┐
│   Portkey    │─────────► Logs Export API
│   Gateway    │
└──────────────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │  Temporal Workflow  │
                    │  (Scheduled Daily)  │
                    └─────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Activity │       │ Activity │       │ Activity │
    │  Export  │       │  Sample  │       │  Replay  │
    └────┬─────┘       └────┬─────┘       └────┬─────┘
         │                  │                   │
         ▼                  ▼                   ▼
    Production         Qdrant             Challenger
    Traces           Clustering            Models
         │                  │                   │
         └──────────────────┴───────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Activity: Evaluate  │
                    │   (DeepEval/GPT)    │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Decision Engine     │
                    │ Recommendation      │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Portkey Feedback    │
                    │ API                 │
                    └─────────────────────┘
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Portkey** | Gateway, logging, feedback | Portkey SDK |
| **Temporal** | Workflow orchestration | temporalio |
| **Qdrant** | Vector storage, clustering | qdrant-client |
| **DeepEval** | LLM evaluation | deepeval |
| **Phoenix** | Observability | arize-phoenix |

---

## Phase 1: Core Infrastructure

### Step 1.1: Verify Temporal Connection

```python
# test_temporal.py
import asyncio
from temporalio.client import Client

async def test_connection():
    client = await Client.connect("localhost:7233")
    print(f"Connected to Temporal!")
    
    # List workflows to verify
    async for wf in client.list_workflows(page_size=1):
        print(f"Found workflow: {wf.id}")
        break
    else:
        print("No workflows yet (this is expected)")

asyncio.run(test_connection())
```

### Step 1.2: Verify Qdrant Connection

```python
# test_qdrant.py
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Check collections
collections = client.get_collections()
print(f"Qdrant collections: {collections}")

# Create test collection
client.create_collection(
    collection_name="test",
    vectors_config={"size": 1536, "distance": "Cosine"}
)
print("Test collection created!")

# Cleanup
client.delete_collection("test")
print("Test collection deleted!")
```

### Step 1.3: Verify Portkey Connection

```python
# test_portkey.py
import os
from portkey_ai import Portkey

client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])

# Test with a simple request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello!"}]
)
print(f"Response: {response.choices[0].message.content}")
```

---

## Phase 2: Portkey Integration

### Step 2.1: Configure Virtual Keys

Virtual Keys enable billing segregation. Create two keys:

1. **Production Virtual Key**
   - Dashboard → Virtual Keys → Create
   - Name: `production-vk`
   - Budget: Unlimited (or your production budget)
   - Providers: OpenAI, Anthropic

2. **Shadow Virtual Key**
   - Dashboard → Virtual Keys → Create
   - Name: `shadow-vk`
   - Budget: $50/month (limit shadow testing costs)
   - Providers: DeepSeek, Together, Mistral

### Step 2.2: Set Up Configs

Create Portkey configs for routing:

```python
# Production config (saved in Portkey dashboard)
production_config = {
    "strategy": {"mode": "single"},
    "targets": [{
        "provider": "openai",
        "virtual_key": "production-vk",
        "override_params": {"model": "gpt-5.2-turbo"}
    }]
}

# Shadow config with fallback
shadow_config = {
    "strategy": {"mode": "fallback"},
    "targets": [
        {
            "provider": "deepseek",
            "virtual_key": "shadow-vk",
            "override_params": {"model": "deepseek-v3"}
        },
        {
            "provider": "together",
            "virtual_key": "shadow-vk",
            "override_params": {"model": "llama-4-scout"}
        }
    ]
}
```

### Step 2.3: Implement Log Export

```python
# Example: Fetching logs from Portkey
async def export_logs(time_window: str = "24h"):
    portkey = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    
    # Parse time window
    hours = int(time_window.replace("h", ""))
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    logs = []
    page = 1
    
    while True:
        response = await portkey.logs.list(
            start_timestamp=start_time.isoformat(),
            page=page,
            page_size=100
        )
        
        if not response.data:
            break
            
        logs.extend(response.data)
        page += 1
    
    return logs
```

---

## Phase 3: Semantic Sampling

### Step 3.1: Understanding the Strategy

Instead of replaying ALL production traffic (expensive!), we use **semantic stratified sampling**:

1. **Embed all prompts** using text-embedding-3-small
2. **Cluster embeddings** using K-Means
3. **Sample strategically** from each cluster:
   - 1 centroid (most representative)
   - 2 outliers (edge cases)

This achieves 95%+ reduction in replay volume while maintaining coverage.

### Step 3.2: Implement Embedding

```python
async def compute_embeddings(texts: List[str]) -> np.ndarray:
    portkey = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    
    all_embeddings = []
    batch_size = 100
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        response = await portkey.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        
        embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings)
```

### Step 3.3: Implement Clustering

```python
from sklearn.cluster import KMeans

def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 50):
    # Adjust clusters based on data size
    n_clusters = min(n_clusters, len(embeddings) // 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    return labels, kmeans.cluster_centers_
```

### Step 3.4: Implement Sampling

```python
def sample_from_cluster(
    traces: List[ProductionTrace],
    embeddings: np.ndarray,
    cluster_indices: List[int],
    centroid: np.ndarray
) -> List[GoldenPrompt]:
    
    samples = []
    
    # Get cluster embeddings
    cluster_embeddings = embeddings[cluster_indices]
    
    # Find centroid (closest to center)
    distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    centroid_idx = cluster_indices[np.argmin(distances)]
    
    samples.append(GoldenPrompt(
        original_trace_id=traces[centroid_idx].trace_id,
        prompt=traces[centroid_idx].prompt,
        production_response=traces[centroid_idx].response,
        sample_type=SampleType.CENTROID
    ))
    
    # Find outliers (furthest from center)
    outlier_indices = np.argsort(distances)[-2:]
    for idx in outlier_indices:
        global_idx = cluster_indices[idx]
        if global_idx != centroid_idx:
            samples.append(GoldenPrompt(
                original_trace_id=traces[global_idx].trace_id,
                prompt=traces[global_idx].prompt,
                production_response=traces[global_idx].response,
                sample_type=SampleType.OUTLIER
            ))
    
    return samples
```

---

## Phase 4: Shadow Replay

### Step 4.1: Implement Replay Logic

```python
async def replay_prompt(
    golden: GoldenPrompt,
    challenger_config: ChallengerConfig
) -> ShadowResult:
    
    portkey = Portkey(api_key=os.environ["PORTKEY_API_KEY"])
    
    start_time = datetime.utcnow()
    
    try:
        response = await portkey.chat.completions.create(
            model=challenger_config.model_id,
            messages=[{"role": "user", "content": golden.prompt}],
            max_tokens=challenger_config.max_tokens,
            temperature=challenger_config.temperature,
            headers={
                "x-portkey-virtual-key": challenger_config.virtual_key_alias
            }
        )
        
        latency = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ShadowResult(
            golden_prompt_id=golden.original_trace_id,
            challenger_model=challenger_config.model_id,
            shadow_response=response.choices[0].message.content,
            latency_ms=latency,
            tokens_prompt=response.usage.prompt_tokens,
            tokens_completion=response.usage.completion_tokens,
            success=True
        )
        
    except Exception as e:
        return ShadowResult(
            golden_prompt_id=golden.original_trace_id,
            challenger_model=challenger_config.model_id,
            shadow_response="",
            latency_ms=0,
            tokens_prompt=0,
            tokens_completion=0,
            success=False,
            error=str(e)
        )
```

### Step 4.2: Implement Circuit Breaker

```python
class CircuitBreaker:
    def __init__(self, max_failures: int = 5, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure = None
        self.state = "closed"  # closed, open, half-open
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = datetime.utcnow()
        
        if self.failures >= self.max_failures:
            self.state = "open"
    
    def record_success(self):
        self.failures = 0
        self.state = "closed"
    
    def can_execute(self) -> bool:
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if we should try again
            if (datetime.utcnow() - self.last_failure).seconds > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        
        return True  # half-open
```

---

## Phase 5: LLM-as-Judge Evaluation

### Step 5.1: Configure DeepEval

```python
from deepeval.metrics import GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Faithfulness metric
faithfulness = FaithfulnessMetric(
    threshold=0.8,
    model="gpt-5.2-turbo",
    include_reason=True
)

# Quality metric via G-Eval
quality = GEval(
    name="Quality",
    criteria="""
    Evaluate the quality of the actual output compared to expected.
    Consider completeness, accuracy, helpfulness, and appropriateness.
    Score 1.0 for semantically equivalent, 0.0 for completely wrong.
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-5.2-turbo"
)
```

### Step 5.2: Implement Evaluation

```python
async def evaluate_response(
    golden: GoldenPrompt,
    shadow: ShadowResult
) -> EvaluationResult:
    
    # Check for refusal
    if is_refusal(shadow.shadow_response):
        return EvaluationResult(
            shadow_result_id=f"{shadow.golden_prompt_id}_{shadow.challenger_model}",
            faithfulness=0.0,
            quality=0.0,
            conciseness=0.0,
            is_refusal=True,
            composite_score=0.0,
            passed_thresholds=False
        )
    
    # Create test case
    test_case = LLMTestCase(
        input=golden.prompt,
        actual_output=shadow.shadow_response,
        expected_output=golden.production_response,
        retrieval_context=[golden.production_response]
    )
    
    # Run metrics
    faithfulness.measure(test_case)
    quality.measure(test_case)
    
    composite = (
        faithfulness.score * 0.5 +
        quality.score * 0.35 +
        0.7 * 0.15  # Default conciseness
    )
    
    return EvaluationResult(
        shadow_result_id=f"{shadow.golden_prompt_id}_{shadow.challenger_model}",
        faithfulness=faithfulness.score,
        quality=quality.score,
        conciseness=0.7,
        is_refusal=False,
        composite_score=composite,
        passed_thresholds=(
            faithfulness.score >= 0.8 and
            quality.score >= 0.7
        ),
        reasoning=f"{faithfulness.reason}\n{quality.reason}"
    )
```

### Step 5.3: Detect Refusals

```python
REFUSAL_PATTERNS = [
    "I cannot", "I can't", "I'm unable to",
    "I apologize, but", "As an AI, I cannot",
    "This request violates", "I must decline"
]

def is_refusal(response: str) -> bool:
    response_lower = response.lower()
    
    for pattern in REFUSAL_PATTERNS:
        if pattern.lower() in response_lower[:200]:
            return True
    
    return False
```

---

## Phase 6: Decision Engine

### Step 6.1: Aggregate Results

```python
def aggregate_challenger_results(
    evaluations: List[EvaluationResult],
    model: str
) -> ChallengerResult:
    
    successful = [e for e in evaluations if not e.is_refusal]
    refusals = [e for e in evaluations if e.is_refusal]
    
    avg_faithfulness = sum(e.faithfulness for e in successful) / len(successful) if successful else 0
    avg_quality = sum(e.quality for e in successful) / len(successful) if successful else 0
    
    failure_reasons = []
    if avg_faithfulness < 0.8:
        failure_reasons.append(f"Faithfulness {avg_faithfulness:.2f} < 0.80")
    if avg_quality < 0.7:
        failure_reasons.append(f"Quality {avg_quality:.2f} < 0.70")
    if len(refusals) / len(evaluations) > 0.01:
        failure_reasons.append(f"Refusal rate {len(refusals)/len(evaluations):.2%} > 1%")
    
    return ChallengerResult(
        model=model,
        samples_evaluated=len(evaluations),
        avg_faithfulness=avg_faithfulness,
        avg_quality=avg_quality,
        avg_conciseness=0.7,
        refusal_rate=len(refusals) / len(evaluations),
        passed_thresholds=len(failure_reasons) == 0,
        failure_reasons=failure_reasons
    )
```

### Step 6.2: Generate Recommendation

```python
def generate_recommendation(
    results: Dict[str, ChallengerResult],
    production_model: str
) -> Recommendation:
    
    # Find passing challengers
    passing = [
        (model, result)
        for model, result in results.items()
        if result.passed_thresholds
    ]
    
    if not passing:
        return Recommendation(
            action=RecommendationAction.NO_VIABLE_CHALLENGER,
            confidence=0.8,
            reasoning="No challenger met all quality thresholds."
        )
    
    # Sort by cost (cheapest first)
    passing.sort(key=lambda x: MODEL_PRICING.get(x[0], {}).get("output", 999))
    
    best_model, best_result = passing[0]
    
    return Recommendation(
        action=RecommendationAction.SWITCH_RECOMMENDED,
        challenger=best_model,
        confidence=compute_confidence(best_result),
        reasoning=f"{best_model} passed all thresholds with composite score {best_result.composite_score:.2f}",
        suggested_actions=[
            f"Begin canary rollout with 1% traffic to {best_model}",
            "Monitor quality metrics for 48 hours",
            "Review before scaling to 10%"
        ]
    )
```

---

## Phase 7: Workflow Orchestration

### Step 7.1: Define Workflow

```python
from temporalio import workflow
from datetime import timedelta

@workflow.defn
class OptimizeModelSelectionWorkflow:
    
    @workflow.run
    async def run(self, config: WorkflowConfig) -> OptimizationReport:
        
        # Step 1: Export logs
        traces = await workflow.execute_activity(
            "export_portkey_logs",
            args=[config.log_export_window],
            start_to_close_timeout=timedelta(minutes=10)
        )
        
        # Step 2: Sample
        golden_prompts = await workflow.execute_activity(
            "sample_golden_prompts",
            args=[traces, config.sampler],
            start_to_close_timeout=timedelta(minutes=15)
        )
        
        # Step 3: Replay (parallel)
        shadow_results = {}
        for challenger in config.challengers:
            results = await workflow.execute_activity(
                "replay_shadow_requests",
                args=[golden_prompts, challenger],
                start_to_close_timeout=timedelta(minutes=30)
            )
            shadow_results[challenger.model_id] = results
        
        # Step 4: Evaluate
        evaluations = await workflow.execute_activity(
            "evaluate_shadow_results",
            args=[golden_prompts, shadow_results, config.evaluator],
            start_to_close_timeout=timedelta(minutes=45)
        )
        
        # Step 5: Generate report
        report = await workflow.execute_activity(
            "generate_optimization_report",
            args=[traces, golden_prompts, evaluations, config],
            start_to_close_timeout=timedelta(minutes=5)
        )
        
        # Step 6: Submit feedback
        await workflow.execute_activity(
            "submit_portkey_feedback",
            args=[report],
            start_to_close_timeout=timedelta(minutes=5)
        )
        
        return report
```

### Step 7.2: Run Worker

```bash
# Start the worker
python -m shadow_optic.worker
```

### Step 7.3: Trigger Workflow

```python
from temporalio.client import Client

async def trigger():
    client = await Client.connect("localhost:7233")
    
    result = await client.execute_workflow(
        OptimizeModelSelectionWorkflow.run,
        config,
        id="optimize-2024-01-15",
        task_queue="shadow-optic"
    )
    
    print(f"Recommendation: {result.recommendation.action}")
```

---

## Phase 8: Production Deployment

### Step 8.1: Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shadow-optic-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: shadow-optic-worker
  template:
    spec:
      containers:
      - name: worker
        image: shadow-optic:latest
        command: ["python", "-m", "shadow_optic.worker"]
        env:
        - name: TEMPORAL_ADDRESS
          value: "temporal.temporal:7233"
        - name: PORTKEY_API_KEY
          valueFrom:
            secretKeyRef:
              name: shadow-optic-secrets
              key: portkey-api-key
```

### Step 8.2: Schedule Daily Runs

```python
# Schedule via Temporal
await client.start_workflow(
    ScheduledOptimizationWorkflow.run,
    config,
    id="scheduled-optimization",
    task_queue="shadow-optic",
    cron_schedule="0 2 * * *"  # 2 AM daily
)
```

### Step 8.3: Set Up Monitoring

1. **Temporal UI**: Monitor workflow executions
2. **Phoenix**: View LLM traces and evaluations
3. **Portkey Dashboard**: Track costs and feedback
4. **Grafana**: Custom dashboards (optional)

---

## Troubleshooting

### Common Issues

**1. "Temporal connection refused"**
```bash
# Check if Temporal is running
docker-compose ps temporal

# Check logs
docker-compose logs temporal
```

**2. "Qdrant connection error"**
```bash
# Verify Qdrant is healthy
curl http://localhost:6333/health
```

**3. "Portkey API error"**
- Verify API key is correct
- Check Virtual Key budget
- Ensure model access is enabled

**4. "DeepEval timeout"**
- Increase evaluation timeout
- Reduce batch size
- Check OpenAI API status

**5. "Workflow stuck"**
```bash
# Check workflow status in Temporal UI
# Or via CLI:
temporal workflow describe -w <workflow_id>
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m shadow_optic.worker
```

### Getting Help

- GitHub Issues: https://github.com/shadow-optic/shadow-optic/issues
- Portkey Discord: https://discord.gg/portkey
- Temporal Slack: https://temporal.io/slack

---

*Good luck with your Shadow-Optic implementation! *
