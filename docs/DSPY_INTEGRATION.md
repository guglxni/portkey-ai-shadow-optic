# DSPy Integration Strategy: Self-Optimizing Prompt Pipelines

**Document Type:** Technical Enhancement Proposal  
**Date:** January 17, 2026  
**Status:** Strategic Advisory

---

## 1. Executive Summary

**DSPy (Declarative Self-Improving Language Programs)** is a framework that systematically optimizes LM prompts and weights. By integrating DSPy with **Portkey**, we move Shadow-Optic from "Prompt Engineering" (manual trial & error) to "Prompt Programming" (automated optimization).

**Key Benefit:** instead of just *detecting* that a cheaper model fails, we can use DSPy to *automatically rewrite the prompt* so that it succeeds on the cheaper model.

---

## 2. Portkey + DSPy Architecture

Portkey provides native support for DSPy, acting as the observability and reliability layer for DSPy's compiled programs.

### Configuration
```python
import dspy

# Configure DSPy to route through Portkey
lm = dspy.LM(
    model="openai/gpt-4o",
    api_key="PORTKEY_API_KEY",
    api_base="https://api.portkey.ai/v1",  # Portkey Gateway
    # Portkey specific headers for tracing/caching
    default_headers={
        "x-portkey-virtual-key": "...",
        "x-portkey-trace-id": "..."
    }
)
dspy.configure(lm=lm)
```

---

## 3. High-Value Use Cases in Shadow-Optic

We have identified three critical injection points for DSPy where it provides superior results to standard LLM calls.

### Use Case A: The "Template Architect" (Ingestion)
*Current State:* Regex-based extraction (brittle) or simple LLM prompting (inconsistent).  
*DSPy Solution:* A compiled `Signature` that learns to extract templates perfectly.

```python
class ExtractTemplate(dspy.Signature):
    """Extract a Jinja2 template and variable list from a set of example prompts."""
    examples = dspy.InputField(desc="List of 5-10 raw prompt strings")
    template = dspy.OutputField(desc="Canonical Jinja2 template string")
    variables = dspy.OutputField(desc="List of variable names found")

# We compile this with 20 golden examples to ensure 100% reliable extraction
extractor = dspy.ChainOfThought(ExtractTemplate)
```

### Use Case B: The "Model Porter" (Optimization)
*Current State:* Shadow-Optic reports "Fail" if DeepSeek-V3 cannot handle a GPT-4 prompt.  
*DSPy Solution:* **Automatic Prompt Porting.** If a prompt fails, use a DSPy optimizer (MIPROv2) to rewrite the prompt specifically for the target model's strengths.

**The "Fixer" Module:**
1.  **Input:** Original Prompt + Failure Reason (from DeepEval).
2.  **Logic:** DSPy optimizes instructions (e.g., "Add 'Think step by step'" or "Use XML tags").
3.  **Output:** An optimized prompt that yields the *same* result as GPT-4 but using DeepSeek-V3.

### Use Case C: The "Agentic Policy" Generator
*Current State:* Static metadata rules.  
*DSPy Solution:* A module that looks at a prompt's performance history and *predicts* the optimal routing policy.

---

## 4. Implementation Roadmap

### Phase 1: DSPy-Powered Template Extraction (Track 2)
Replace the `PromptTemplateExtractor` in `templates.py` with a DSPy module. This makes the system robust against complex, non-regex-friendly prompts (like natural language requests).

### Phase 2: The "Auto-Fix" Workflow (Track 4)
Enhance `AdaptiveOptimizationWorkflow`.
*   **If** `Refusal` or `Low Quality`:
*   **Then** Trigger `dspy_optimize_prompt(prompt, target_model)`.
*   **Then** Re-run Shadow Replay with the *new* prompt.
*   **Result:** You don't just report savings; you *create* savings by making the cheap model work.

---

## 5. Why This Wins
1.  **Technical Depth:** Shows mastery of the absolute latest AI stack (DSPy + Portkey).
2.  **Real Utility:** Solves the #1 problem in model switching ("The prompt works on GPT-4 but breaks on Llama").
3.  **Integration:** Portkey provides the *traceability* that DSPy usually lacks, making the "Black Box" of optimization visible.
