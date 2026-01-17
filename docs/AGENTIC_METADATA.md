# Agentic Metadata: The Cognitive Nervous System of Shadow-Optic

**Document Type:** Strategic Architecture Specification  
**Date:** January 17, 2026  
**Status:** Approved for Implementation

---

## 1. Executive Summary

Traditional metadata is **passive** (e.g., tags, timestamps, labels). It describes *what happened*.  
**Agentic Metadata** is **active**. It describes *what should happen* and *why*.

By integrating an Agentic Metadata Infrastructure Layer into Shadow-Optic, we transform the system from a simple "test runner" into a **self-governing cognitive platform**. This layer acts as the "nervous system," connecting **PromptGenie** (Perception) with **Shadow-Optic** (Action) to enable autonomous self-optimization.

### Core Value Proposition

1.  **Context-Aware Execution:** Instead of treating all prompts the same, the system applies specific policies (Safety, Cost, Quality) derived from the prompt's semantic DNA.
2.  **Self-Healing Feedback Loops:** Evaluation results don't just generate reports; they update the metadata, automatically reprogramming the routing logic.
3.  **Reasoning-Level Observability:** Every automated decision is logged with a "Reasoning Trace" (e.g., *"Switched to DeepSeek because Faithfulness > 0.95 and Cost < $0.01"*).

---

## 2. The "Active" Registry Architecture

The central component is the **Policy Store** (implemented within the Qdrant Registry). Instead of storing static templates, we store **Agentic Policy Objects**.

### 2.1 Schema Definition

```json
{
  "template_id": "tpl_finance_earnings_01",
  "pattern": "Summarize Q{{quarter}} earnings for {{company}}",
  "cluster_id": 42,
  
  // --- AGENTIC METADATA STARTS HERE ---
  "agentic_policy": {
    "state": "active",                // Status: discovery | shadow_testing | active | deprecated
    
    "routing_strategy": {             // DIRECTIVE: How to handle live traffic
      "primary_model": "gpt-4o",
      "fallback_model": "deepseek-v3",
      "timeout_ms": 5000
    },
    
    "optimization_strategy": {        // DIRECTIVE: How Shadow-Optic should test this
      "shadow_sample_rate": 0.05,     // Test 5% of traffic
      "challenger_pool": ["deepseek-v3", "llama-4-scout"],
      "success_criteria": {
         "faithfulness": 0.98,        // Strict factual accuracy for finance
         "conciseness": 0.0           // Length doesn't matter
      }
    },
    
    "governance_log": [               // MEMORY: Why is the system this way?
      {
        "timestamp": "2026-01-17T10:00:00Z",
        "action": "policy_update",
        "actor": "Shadow-Optic-Decision-Engine",
        "reasoning": "DeepSeek-V3 failed faithfulness check (0.82 < 0.98). Reverting to GPT-4o."
      }
    ]
  }
}
```

---

## 3. Operational Workflow (The "Cognitive Loop")

The system operates in a continuous cycle of **Perception → Reasoning → Action → Learning**.

### Phase 1: Perception (PromptGenie)
*   **Input:** A new prompt arrives.
*   **Action:** PromptGenie matches it to `tpl_finance_earnings_01`.
*   **Agentic Trigger:** It reads `agentic_policy.routing_strategy`.
    *   *If "active":* Routes to `primary_model`.
    *   *If "shadow_testing":* Routes to `primary` AND asynchronously triggers Shadow-Optic based on `shadow_sample_rate`.

### Phase 2: Reasoning (Shadow-Optic Evaluator)
*   **Input:** Shadow replay results.
*   **Action:** The Evaluator retrieves `agentic_policy.optimization_strategy`.
*   **Contextual Judging:** It sees `faithfulness: 0.98`. It knows this is a "Finance" prompt, so it penalizes hallucinations strictly (unlike a "Creative Writing" prompt where `faithfulness` might be 0.5).

### Phase 3: Learning (Decision Engine)
*   **Input:** Evaluation scores.
*   **Action:** The Decision Engine determines `deepseek-v3` achieved 0.99 faithfulness.
*   **Agentic Update:** It **writes back** to the Policy Store:
    *   Updates `routing_strategy.primary_model` to `deepseek-v3` (Auto-Switch).
    *   Appends to `governance_log`: *"Switching to DeepSeek. Savings: 90%. Quality: Stable."*

---

## 4. Implementation Strategy

### Step 1: Augment the Registry (PromptGenie)
*   **Task:** When creating a new cluster, use an LLM Agent ("The Librarian") to generate the initial `agentic_policy`.
*   **Prompt:** *"Analyze these 5 prompts. Are they factual (Finance/Medical) or creative? Set `faithfulness` threshold accordingly. Suggest a `sample_rate`."*

### Step 2: Policy-Driven Evaluation (Shadow-Optic)
*   **Task:** Modify `ShadowEvaluator` to accept a `policy` object instead of hardcoded config.
*   **Code Change:**
    ```python
    # Before
    threshold = config.faithfulness_threshold
    
    # After (Agentic)
    threshold = template_policy.get("success_criteria", {}).get("faithfulness", 0.8)
    ```

### Step 3: Self-Writing Infrastructure (Decision Engine)
*   **Task:** Allow the Decision Engine to execute a `update_policy(template_id, new_policy)` function.
*   **Outcome:** The system evolves its own configuration database without human deployment.

---

## 5. Strategic Advantage

This architecture explicitly satisfies the **"System Design"** and **"Thoughtful Use of AI"** judging criteria by demonstrating:

1.  **Metadata as Infrastructure:** Metadata isn't just a log; it's the code that runs the system.
2.  **Autonomous Governance:** The system enforces safety and quality rules granularly (per-prompt) rather than globally.
3.  **Evolutionary Architecture:** The system gets smarter and more efficient with every request processed.
