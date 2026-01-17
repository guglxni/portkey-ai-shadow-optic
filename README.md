# 🔮 Shadow-Optic

**The Autonomous Cost-Quality Optimization Engine for LLMs**

[![Portkey AI Builders Challenge](https://img.shields.io/badge/Portkey-Track%204-blue?style=for-the-badge)](https://portkey.ai/builders-challenge)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Track 4: Cost-Quality Optimization via Historical Replay**
> 
> *"What if your LLM costs could optimize themselves?"*

---

## 🎯 What is Shadow-Optic?

Shadow-Optic is an **autonomous optimization engine** that continuously evaluates whether cheaper LLM models can replace your production model without sacrificing quality. It operates in **shadow mode**—completely invisible to end users—replaying a sample of your production traffic against challenger models and using LLM-as-Judge to evaluate quality.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SHADOW-OPTIC FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Production                                                                  │
│  Traffic ──────► Portkey ──────► Logs Export ──────► Shadow-Optic           │
│      │           Gateway              │                    │                 │
│      │              │                 │                    ▼                 │
│      │              │                 │           ┌─────────────────┐        │
│      │              │                 │           │ Semantic Sample │        │
│      │              │                 │           │  (5% of logs)   │        │
│      │              │                 │           └────────┬────────┘        │
│      │              │                 │                    │                 │
│      │              │                 │                    ▼                 │
│      ▼              ▼                 │           ┌─────────────────┐        │
│   Users          Live                 │           │ Shadow Replay   │        │
│   Happy!       Responses              │           │ DeepSeek/Llama  │        │
│                                       │           └────────┬────────┘        │
│                                       │                    │                 │
│                                       │                    ▼                 │
│                                       │           ┌─────────────────┐        │
│                                       │           │ LLM-as-Judge    │        │
│                                       │           │ GPT-5.2 Eval    │        │
│                                       │           └────────┬────────┘        │
│                                       │                    │                 │
│                                       │                    ▼                 │
│                                       │           ┌─────────────────┐        │
│                                       ◄───────────│ Recommendation  │        │
│                            Feedback API           │ SWITCH/MONITOR  │        │
│                                                   └─────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🔒 True Shadow Mode
- **Zero impact** on production traffic
- Billing segregation via Portkey Virtual Keys
- Invisible to end users

### 🧠 Intelligent Sampling
- **Semantic stratified sampling** using K-Means clustering
- Tests both typical cases (centroids) and edge cases (outliers)
- **95% reduction** in evaluation costs

### ⚖️ LLM-as-Judge Evaluation
- **Faithfulness**: Does the response contain the same facts?
- **Quality**: Is it helpful and complete?
- **Conciseness**: Is it efficient?
- Powered by DeepEval with GPT-5.2

### 🔄 Durable Workflow Orchestration
- Temporal.io for fault-tolerant execution
- Automatic retries with exponential backoff
- State persistence across failures

### 📊 Actionable Insights
- Clear SWITCH/MONITOR/WAIT recommendations
- Cost-benefit analysis with ROI projections
- Confidence scores with explainability

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Portkey API Key ([get one here](https://portkey.ai))
- OpenAI API Key (for DeepEval)

### 1. Clone & Setup

```bash
git clone https://github.com/shadow-optic/shadow-optic.git
cd shadow-optic

# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

### 2. Start Infrastructure

```bash
# Start Temporal, Qdrant, and Phoenix
docker-compose up -d temporal qdrant phoenix

# Wait for services to be healthy
docker-compose ps
```

### 3. Run the Worker

```bash
# Install dependencies
pip install -e ".[dev]"

# Start the worker
python -m shadow_optic.worker
```

### 4. Trigger an Optimization Run

```python
from temporalio.client import Client
from shadow_optic.workflows import OptimizeModelSelectionWorkflow
from shadow_optic.models import WorkflowConfig, ChallengerConfig

# Connect to Temporal
client = await Client.connect("localhost:7233")

# Configure optimization
config = WorkflowConfig(
    log_export_window="24h",
    challengers=[
        ChallengerConfig(
            model_id="deepseek-v3",
            provider="deepseek",
            virtual_key_alias="shadow-vk"
        ),
        ChallengerConfig(
            model_id="llama-4-scout",
            provider="together",
            virtual_key_alias="shadow-vk"
        )
    ]
)

# Execute workflow
result = await client.execute_workflow(
    OptimizeModelSelectionWorkflow.run,
    config,
    id="optimize-2024-01-15",
    task_queue="shadow-optic"
)

print(f"Recommendation: {result.recommendation.action}")
print(f"Best Challenger: {result.best_challenger}")
```

---

## 📁 Project Structure

```
shadow-optic/
├── src/shadow_optic/
│   ├── __init__.py          # Package initialization (32 exports)
│   ├── models.py             # Pydantic data models
│   ├── sampler.py            # Semantic stratified sampling (Qdrant)
│   ├── evaluator.py          # LLM-as-Judge with DeepEval
│   ├── decision_engine.py    # Recommendation & Thompson Sampling
│   ├── model_registry.py     # 36+ models, Pareto-optimal selection
│   ├── analytics.py          # Quality trend detection
│   ├── observability.py      # Arize Phoenix integration
│   ├── templates.py          # Prompt template extraction
│   ├── workflows.py          # Temporal workflow definitions
│   ├── activities.py         # Temporal activity implementations
│   ├── worker.py             # Temporal worker runner
│   ├── api.py                # FastAPI REST server
│   └── cli.py                # Command-line interface
├── docs/
│   ├── ARCHITECTURE.md       # Detailed system architecture
│   ├── IMPLEMENTATION_GUIDE.md # Step-by-step implementation guide
│   └── ENHANCEMENTS.md       # Future enhancement proposals
├── configs/                  # Configuration files
├── monitoring/               # Prometheus/Grafana dashboards
├── tests/                    # Test suite (89 tests)
├── docker-compose.yml        # Infrastructure services
├── Dockerfile               # Container build
├── requirements.txt         # Dependencies
└── pyproject.toml           # Project configuration
```

---

## 🔧 Configuration

### Portkey Virtual Keys

Shadow-Optic uses Portkey Virtual Keys to segregate shadow testing costs from production:

1. Go to [Portkey Dashboard](https://portkey.ai/dashboard)
2. Create a Virtual Key for shadow testing with a budget cap
3. Set `PORTKEY_SHADOW_VIRTUAL_KEY` in your `.env`

### Quality Thresholds

Configure in `.env` or workflow config:

| Threshold | Default | Description |
|-----------|---------|-------------|
| `FAITHFULNESS_THRESHOLD` | 0.80 | Minimum factual consistency |
| `QUALITY_THRESHOLD` | 0.70 | Minimum overall quality |
| `CONCISENESS_THRESHOLD` | 0.50 | Minimum response efficiency |
| `REFUSAL_RATE_THRESHOLD` | 0.01 | Maximum refusal increase |

### Scheduling

By default, optimization runs daily at 2 AM:

```python
# Start scheduled workflow
await client.start_workflow(
    ScheduledOptimizationWorkflow.run,
    config,
    id="scheduled-optimization",
    task_queue="shadow-optic",
    cron_schedule="0 2 * * *"  # 2 AM daily
)
```

---

## 📊 Dashboard & Observability

### Temporal UI
Access workflow execution details at `http://localhost:8080`

### Arize Phoenix
View LLM traces and evaluation details at `http://localhost:6006`

### Portkey Dashboard
See feedback scores and cost analytics at [app.portkey.ai](https://app.portkey.ai)

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=shadow_optic --cov-report=html

# Run specific test file
pytest tests/test_sampler.py -v
```

---

## 🏆 Portkey AI Builders Challenge

This project is built for **Track 4: Cost-Quality Optimization via Historical Replay**.

### Judging Criteria Addressed

| Criteria | Implementation |
|----------|---------------|
| **Production Readiness** | Docker, health checks, graceful shutdown |
| **Thoughtful AI Use** | LLM-as-Judge, semantic sampling, Thompson Sampling |
| **System Design** | Temporal workflows, event sourcing, fan-out/fan-in |
| **Correctness** | Type safety (Pydantic), threshold validation |
| **Engineering Quality** | Clean architecture, comprehensive docs |
| **Failure Handling** | Retries, circuit breakers, dead letter queues |
| **Explainability** | Reasoning chains, confidence scores, reports |

---

## 🛣️ Implementation Status

### ✅ Core Features (Complete)
- [x] Core workflow implementation (Temporal)
- [x] Semantic sampling with Qdrant & K-Means clustering
- [x] DeepEval LLM-as-Judge integration (Faithfulness, Quality, Conciseness)
- [x] Portkey Model Catalog integration (250+ models)
- [x] Portkey Logs Export & Feedback API
- [x] Thompson Sampling model selection (Multi-Armed Bandit)
- [x] Quality trend detection (T-test, Z-score anomaly)
- [x] Prompt template extraction
- [x] Intelligent model registry (36+ models, Pareto-optimal)
- [x] Arize Phoenix observability (OTLP traces)
- [x] FastAPI REST server with Prometheus metrics
- [x] CLI interface
- [x] Docker infrastructure (Temporal, Qdrant, Phoenix, Grafana)

### 🚧 Future Enhancements
- [ ] Automatic A/B rollout graduation
- [ ] Cost prediction ML model
- [ ] Automatic guardrail tuning
- [ ] Slack/PagerDuty alerting integration

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Portkey](https://portkey.ai) - AI Gateway and observability
- [Temporal](https://temporal.io) - Workflow orchestration
- [DeepEval](https://github.com/confident-ai/deepeval) - LLM evaluation
- [Qdrant](https://qdrant.tech) - Vector database
- [Arize Phoenix](https://phoenix.arize.com) - LLM observability

---

<p align="center">
  <strong>Built with ❤️ for the Portkey AI Builders Challenge</strong>
</p>
