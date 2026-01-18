# Unified Agentic Architecture: DSPy & Metadata Enhancements

**Document Type:** Technical Specification  
**Status:** Planned Enhancement  
**Target:** Phase 2 Implementation  

---

## 1. Executive Summary

This document details the unified strategy for **Self-Optimizing Prompts** (via DSPy) and **Autonomous Governance** (via Agentic Metadata).

By combining `shadow-optic`'s evaluation engine with `prompt_genie`'s structural understanding, we enable a closed-loop system:
1.  **Detect:** Agentic Metadata identifies "Drifting" or "Expensive" prompt clusters.
2.  **Diagnose:** Shadow-Optic evaluates challenger models.
3.  **Optimize:** DSPy rewrites the prompt signature to make the challenger model succeed.
4.  **Deploy:** The new "Agentic Policy" is updated in Portkey.

---

## 2. Agentic Metadata Framework

Metadata is no longer passive logging; it is the **control plane**. We utilize a "Watchdog" pattern where Qdrant clusters act as state machines.

### 2.1 The Agentic Policy Schema (Qdrant Payload)
Every template cluster carries a JSON payload defining its autonomy:

```json
{
  "cluster_id": "finance_earnings_v1",
  "state": "STABLE",  // Options: EMBRYONIC, STABLE, DRIFTING, UNSTABLE
  
  "agentic_policy": {
    "intent": "financial_analysis",
    "sensitivity": "high",
    
    "optimization_target": {
      "metric": "faithfulness",
      "min_score": 0.95,
      "max_cost_per_1k": 0.005
    },
    
    "routing_rules": [
      {"if": "faithfulness < 0.9", "then": "fallback_to_gpt4"},
      {"if": "latency > 2000ms", "then": "switch_provider"}
    ]
  },
  
  "evolution_history": [
    {
      "date": "2026-01-18",
      "action": "DSPy_Optimized",
      "change": "Added Chain-of-Thought for DeepSeek-V3",
      "improvement": "+12% accuracy"
    }
  ]
}
```

### 2.2 The Watchdog Loop
A background worker (`AgenticWatchdog`) scans Qdrant for triggers:
*   **Drift Trigger:** If `variance > 0.15` → Set state to `DRIFTING`.
*   **Cost Trigger:** If `avg_cost > target` → Trigger `ShadowOptic`.
*   **Quality Trigger:** If `eval_score < min_score` → Trigger `DSPyOptimizer`.

**FOSS Resource:**
*   **Evidently AI:** Can be used to calculate the drift metrics (Data Drift) that feed into this schema.
*   **Arize Phoenix:** Already integrated; use its "Project Drift" signals to update the `state` field.

---

## 3. DSPy Integration Strategy

We move from "Prompt Engineering" (manual string editing) to "Prompt Programming" using **MIPROv2** (Multi-prompt Instruction Proposal Optimizer).

### 3.1 The "Fixer" Workflow
When a cheaper model (e.g., Llama-3) fails a Shadow test that GPT-4 passed:

1.  **Extract Signature:** Use `prompt_genie` to define inputs/outputs.
    ```python
    class EarningsSummary(dspy.Signature):
        """Summarize quarterly earnings call transcripts."""
        transcript = dspy.InputField()
        summary = dspy.OutputField(desc="Bullet points with key financial figures")
    ```

2.  **Create Training Set:**
    *   **Inputs:** Historical prompts from the cluster.
    *   **Labels:** The *successful* GPT-4 responses (Golden Data).

3.  **Compile with MIPROv2:**
    MIPROv2 optimizes the *instruction* and *few-shot examples* specifically for the target model (Llama-3).
    ```python
    # Define the metric (from Shadow-Optic evaluator)
    def validate_faithfulness(example, pred, trace=None):
        return shadow_optic.evaluate(pred, example.label) >= 0.95

    # Optimize
    teleprompter = dspy.teleprompt.MIPROv2(metric=validate_faithfulness)
    optimized_program = teleprompter.compile(
        EarningsSummary(), 
        trainset=training_data,
        target_model="llama-3-70b"
    )
    ```

4.  **Deploy:** The optimized instruction is saved as a new version in the Portkey Prompt Library, tagged for Llama-3.

### 3.2 Why MIPROv2?
Research shows MIPROv2 outperforms simple few-shot prompting by optimizing the *instructions* alongside the examples, crucial for helping smaller models understand complex tasks.

**FOSS Resource:**
*   **`stanfordnlp/dspy`:** Core framework.
*   **`haasonsaas/dspy-advanced-prompting`:** Reference for "Manager" style prompts.

---

## 4. Implementation Roadmap

### Phase 1: The "Smart" Registry (Weeks 1-2)
*   Update `prompt_genie` to store `AgenticPolicy` in Qdrant.
*   Implement `IncrementalClusterManager` to update `state` based on variance.

### Phase 2: The DSPy Bridge (Weeks 3-4)
*   Create `DSPyOptimizer` class in `shadow-optic`.
*   Implement the `MIPROv2` loop using Portkey as the LLM backend.

### Phase 3: The Autonomous Loop (Weeks 5-6)
*   Connect the "Watchdog" to the "Optimizer".
*   Enable `auto_deploy: true` for low-risk clusters.

---

## 5. Recommended Open Source Resources

| Tool | Purpose | Integration Point |
| :--- | :--- | :--- |
| **DSPy** | Prompt Optimization | Core Logic (`src/shadow_optic/dspy_optimizer.py`) |
| **Arize Phoenix** | Drift/Trace Visualization | Observability (`src/shadow_optic/observability.py`) |
| **DeepEval** | LLM-as-a-Judge Metrics | Evaluation (`src/shadow_optic/evaluator.py`) |
| **Qdrant** | State/Policy Store | Storage (`src/prompt_genie/storage/vector_store.py`) |
| **Evidently AI** | Statistical Drift Checks | Watchdog (`src/shadow_optic/watchdog.py`) |

---

## 6. Success Metrics

1.  **Self-Correction Rate:** % of failing clusters successfully "fixed" by DSPy without human intervention.
2.  **Drift Detection Latency:** Time from "concept drift" to "alert generation".
3.  **Policy Compliance:** % of prompts accurately routed according to their metadata (e.g., PII staying local).
