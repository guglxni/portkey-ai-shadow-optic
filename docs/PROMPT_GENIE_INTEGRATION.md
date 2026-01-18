# PromptGenie Integration Plan

**Status:** Implemented  
**Completion Date:** January 2026  
**Dependencies:** `shadow-optic`, `prompt_genie`

---

## 1. Executive Summary

This document outlines the technical strategy for integrating **PromptGenie** (The Evolutionary Prompt Registry) into **Shadow-Optic** (The Cost-Quality Optimization Engine).

**Previous State:**
`shadow-optic` used a brittle, regex-based `PromptTemplateExtractor` that failed on complex natural language prompts and lacked semantic understanding.

**Current State (Implemented):**
`shadow-optic` now utilizes `prompt_genie` as its intelligent backend for:
1.  **Semantic Clustering:** Grouping traces by intent (embeddings) rather than just syntax.
2.  **Agentic Extraction:** Using LLMs (via Portkey) to reverse-engineer canonical Jinja2 templates.
3.  **Drift Detection:** Identifying when a prompt family has evolved.

---

## 2. Architecture: The "Genie Bridge"

We implemented an **Adapter Pattern** within `shadow-optic` that wraps `prompt_genie`'s core components.

### 2.1 Component Mapping

| Shadow-Optic Component | Replaced By (PromptGenie) | Integration Point |
| :--- | :--- | :--- |
| `templates.PromptTemplateExtractor` | `GenieTemplateExtractor` (Implemented) | `src/shadow_optic/templates.py` |
| `templates.cluster_into_template_families` | `IncrementalClusterManager` | Via PromptGeniePipeline |
| `WorkflowConfig.genie` | `GenieConfig` (Implemented) | `src/shadow_optic/models.py` |

### 2.2 Data Flow

1.  **Ingest:** `OptimizeModelSelectionWorkflow` fetches raw traces from Portkey.
2.  **Branch:** Workflow checks `config.genie.enabled` flag.
3.  **If Genie Enabled:**
    *   Calls `sample_golden_prompts_genie` activity.
    *   Traces converted to `AgenticPrompt` objects.
    *   `PromptGeniePipeline` processes traces (embedding, clustering, extraction).
4.  **Return:** `GoldenPrompt` objects enriched with Genie fields.

---

## 3. Implementation Details (Completed)

### 3.1 Files Modified

| File | Changes |
| :--- | :--- |
| `src/shadow_optic/models.py` | Added `GenieConfig` class, updated `GoldenPrompt` with 5 Genie fields |
| `src/shadow_optic/templates.py` | Added `GenieTemplateExtractor` adapter (~300 lines) |
| `src/shadow_optic/activities.py` | Added `sample_golden_prompts_genie` activity |
| `src/shadow_optic/workflows.py` | Updated `_sample_golden_prompts` to conditionally use Genie |
| `pyproject.toml` | Added `sentence-transformers`, `aiosqlite`, `jinja2` dependencies |

### 3.2 New Model Fields (`GoldenPrompt`)

```python
# PromptGenie Integration Fields
genie_cluster_id: Optional[str]       # Semantic cluster ID from Qdrant
genie_confidence: Optional[float]     # Template extraction confidence (0-1)
genie_template: Optional[str]         # Canonical Jinja2 template
genie_routing_decision: Optional[str] # Routing strategy from agentic metadata
genie_intent_tags: Optional[List[str]]# Semantic intent labels
```

### 3.3 GenieConfig Class

```python
class GenieConfig(BaseModel):
    enabled: bool = False
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "prompt_clusters"
    similarity_threshold: float = 0.85
    min_cluster_size: int = 5
    extraction_model: str = "gpt-5-mini"
    fallback_model: str = "claude-haiku-4-5"
    sqlite_path: str = "./data/prompt_genie.db"
```
        1. Initialize Genie Pipeline.
        2. Convert traces -> AgenticPrompts.
        3. Run pipeline.process_single() for all traces (parallelized).
        4. Collect results and map back to GoldenPrompts.
        """
```

### 3.4 Workflow Integration

The `_sample_golden_prompts` method in `OptimizeModelSelectionWorkflow` now branches based on `config.genie.enabled`:

```python
async def _sample_golden_prompts(self, traces, config) -> List[GoldenPrompt]:
    if config.genie.enabled:
        return await workflow.execute_activity(
            "sample_golden_prompts_genie",
            args=[traces, config.genie, config.sampler.samples_per_cluster],
            ...
        )
    else:
        return await workflow.execute_activity(
            "sample_golden_prompts",  # Legacy
            args=[traces, config.sampler],
            ...
        )
```

### 3.5 Shared Infrastructure

Both systems access the same persistence layer:

*   **Qdrant:** Same collection (`prompt_clusters`) for vector clustering.
*   **Portkey:** Shared API Key for LLM template extraction.

---

## 4. Usage Guide

### Step 1: Environment Setup
```bash
# Install PromptGenie as local package
pip install -e ./prompt_genie

# Start Qdrant
docker-compose up -d qdrant

# Set environment variables
export PORTKEY_API_KEY="your-key"
```

### Step 2: Enable Genie in Config
```python
from shadow_optic.models import WorkflowConfig, GenieConfig

config = WorkflowConfig(
    challengers=[...],
    genie=GenieConfig(
        enabled=True,
        qdrant_host="localhost",
        qdrant_port=6333,
    )
)
```

### Step 3: Run Comparison Test
```bash
# Test legacy vs Genie extraction
python scripts/genie_comparison_test.py

# Legacy-only mode (no PromptGenie required)
python scripts/genie_comparison_test.py --legacy-only
```

---

## 5. DSPy Enablement

This integration is the prerequisite for the **DSPy** strategy.

*   Once Genie extracts a template (e.g., `Refine this {{text}} for {{audience}}`), we obtain a stable **Signature**.
*   We can then pass this Signature to `dspy.BootstrapFewShot` to optimize the prompt for Challenger Models (e.g., Llama-3), ensuring fair comparisons in Shadow Mode.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
| :--- | :--- |
| **Latency** | Genie uses embeddings (fast) + LLM extraction (slow). Extraction is only triggered once per cluster (async). Normal routing is vector-only. |
| **Cost** | Embedding cost is negligible (local). LLM extraction cost is managed via Portkey Caching and only running on cluster centroids. |
| **Complexity** | `prompt_genie` adds async complexity. Wrapped in activity for clean workflow integration. |
| **PromptGenie Not Installed** | `GenieTemplateExtractor.initialize()` raises `RuntimeError` with clear install instructions. Workflow falls back to legacy if activity fails. |
