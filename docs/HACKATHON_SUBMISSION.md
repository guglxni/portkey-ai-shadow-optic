#  Shadow-Optic: Enterprise LLM Cost & Safety Optimization

> **"Reduce LLM costs by 79% while maintaining safety compliance"**

## Hackathon Demo Results (Jan 17, 2026)

###  Key Metrics

| Model | Cost/Request | vs GPT-5.2 | Safety Score | Recommendation |
|-------|-------------|------------|--------------|----------------|
| **GPT-5.2** (Production) | $0.00294 | Baseline | 96.8% | Current standard |
| **GPT-5 Mini** | $0.00061 | **↓ 79.3%** | 100% |  **RECOMMENDED** |
| Claude Haiku 4.5 | $0.00131 | ↓ 55.4% | 80.6% | ALERT: More refusals |
| Grok 4 Fast | $0.00212 | ↓ 28.0% | 83.9% | ALERT: More refusals |

###  ROI Analysis (at 1M requests/year)

| Metric | Current (GPT-5.2) | Optimized (GPT-5 Mini) | Savings |
|--------|-------------------|------------------------|---------|
| Annual Cost | $2,940.52 | $608.68 | **$2,331.84** |
| Cost Reduction | - | **79.3%** | - |

---

##  What Shadow-Optic Does

Shadow-Optic is an enterprise platform that helps organizations:

1. **Compare LLM costs** across providers (OpenAI, Anthropic, xAI) with statistical rigor
2. **Measure safety compliance** through adversarial prompt testing
3. **Make data-driven decisions** on model selection with A/B testing
4. **Track everything** with Arize Phoenix observability integration

### Core Value Proposition

```
"Switching from GPT-5.2 to GPT-5 Mini reduces costs by 79.3% 
while maintaining equivalent safety compliance."
```

---

##  Quick Start

### Prerequisites

- Python 3.11+
- Portkey API Key (for LLM gateway)
- Docker (optional, for Phoenix)

### Installation

```bash
# Clone and setup
git clone <repo>
cd shadow-optic
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set API key
export PORTKEY_API_KEY=your_key_here
```

### Run the Demo

```bash
# Run comprehensive hackathon demo
python scripts/hackathon_demo.py --prompts 45

# View traces in Phoenix
# Open http://localhost:6006
```

---

## 📁 Project Structure

```
shadow-optic/
├── scripts/
│   ├── hackathon_demo.py       #  Main hackathon demo
│   └── production_ready_demo.py # Production comparison script
├── src/shadow_optic/
│   ├── decision_engine.py       # A/B testing & recommendations
│   ├── evaluator.py             # Quality & safety evaluation
│   ├── sampler.py               # Semantic prompt sampling
│   └── p2_enhancements/
│       ├── ab_graduation.py     # Automated rollout controller
│       ├── cost_predictor.py    # ML-based cost prediction
│       └── guardrail_tuner.py   # DSPy-powered tuning
├── tests/                       # 181 unit tests
└── monitoring/                  # Grafana/Prometheus configs
```

---

## 🔬 Features

### 1. Multi-Provider Comparison
- Real-time cost tracking across OpenAI, Anthropic, xAI
- Statistical confidence intervals (95% CI)
- Token usage analysis

### 2. Safety & Guardrail Testing
- Adversarial prompt injection tests
- Refusal rate tracking
- Automated safety scoring

### 3. A/B Testing Framework
- Thompson Sampling for model selection
- Automated graduation/rollback
- Quality gates at each stage

### 4. Observability
- Arize Phoenix integration
- Full request tracing
- Latency and error tracking

---

##  Test Results

```
========================= test session starts =========================
collected 181 items

tests/test_ab_graduation.py ................ [24%]
tests/test_cost_predictor.py ............... [40%]
tests/test_decision_engine.py .............. [55%]
tests/test_evaluator.py .................... [70%]
tests/test_guardrail_tuner.py .............. [82%]
tests/test_sampler.py ...................... [92%]
tests/test_workflows.py .................... [100%]

========================= 181 passed =========================
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| LLM Gateway | Portkey AI |
| Observability | Arize Phoenix |
| ML Framework | DSPy, scikit-learn |
| Workflow Engine | Temporal |
| Statistics | spotify-confidence |
| API | FastAPI |

### 🧑‍⚖️ LLM-as-Judge Configuration

For quality evaluation, we use **Claude Opus 4.5** via AWS Bedrock as the judge model:

```python
# Default judge model for highest quality evaluation
judge_model = "@bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0"

# Alternatives available:
# - @open-ai/o3-pro (advanced reasoning)
# - @open-ai/gpt-5.2-pro (strong general evaluation)
```

Claude Opus 4.5 is selected because:
- **Superior reasoning** for nuanced quality comparisons
- **Safety-aware** evaluation aligned with enterprise requirements
- **Multi-modal** support for future expansion

---

## 🎖️ Hackathon Highlights

1. **Real Production Models**: GPT-5.2, GPT-5 Mini, Claude Haiku 4.5, Grok 4 Fast
2. **Real Pricing Data**: January 2026 pricing from Portkey Model Catalog
3. **Statistical Rigor**: Confidence intervals, A/B testing, p-values
4. **Enterprise Ready**: 181 unit tests, full observability, automated rollouts
5. **Actionable Insights**: Clear recommendations with quantified ROI

---

##  Sample Output

```
╭──────────────────────────────────────────────────────────╮
│     SHADOW-OPTIC                                       │
│    Enterprise LLM Cost & Safety Optimization Platform    │
╰──────────────────────────────────────────────────────────╯

 Cost Analysis
┌────────────────┬────────┬───────────┬────────────┐
│ Model          │ Tier   │ Total     │ vs Prod    │
├────────────────┼────────┼───────────┼────────────┤
│ GPT-5.2        │ prem   │ $0.0912   │ Baseline   │
│ GPT-5 Mini     │ std    │ $0.0189   │ ↓ 79.3%    │
│ Claude Haiku   │ std    │ $0.0406   │ ↓ 55.4%    │
│ Grok 4 Fast    │ econ   │ $0.0656   │ ↓ 28.0%    │
└────────────────┴────────┴───────────┴────────────┘

 A/B Test: GPT-5.2 vs GPT-5 Mini
   💵 Cost Savings: +79.3%
    Safety Delta: -3.2%
    SWITCH to GPT-5 Mini: Save 79.3% with acceptable safety
```

---

## 👥 Team

Built for the Portkey AI Hackathon - January 2026

---

## 📜 License

MIT License
