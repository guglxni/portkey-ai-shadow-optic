# Latest Models - January 17, 2026

This document confirms the **actual latest models** as of Saturday, January 17, 2026, verified from official vendor pricing pages.

---

##  Code Review Improvements Implemented

Based on strategic analysis, the following enhancements were made:

### 1. Enhanced Refusal Detection (`src/shadow_optic/evaluator.py`)
Expanded from 12 to **45+ refusal patterns** including:
- Direct refusals ("I cannot", "I can't")
- Polite refusals ("I'm afraid I cannot", "I'd prefer not to")
- AI identity refusals ("As an AI, I cannot")
- Policy-based refusals ("This goes against my guidelines")
- Safety refusals ("For safety reasons, I cannot")

### 2. Robust Error Handling (`src/shadow_optic/activities.py`)
- 400 errors (context window, bad params) now treated as failures, not crashes
- Categorized error types: `bad_request`, `auth_error`, `rate_limited`, `server_error`, `timeout`
- Detailed logging for each error category

### 3. Agentic Seed Traffic (`scripts/seed_traffic.py`)
Upgraded from static templates to **LLM-powered generation**:
- Uses `gpt-5-mini` to generate prompts for 7 different personas
- Injects **adversarial prompts** (10% by default) for safety testing
- Includes jailbreak attempts, PII injection, harmful content requests
- Tracks **safety detection rate** in output metrics

---

##  Production Configuration

**File**: `configs/production-config.json`

- **Primary**: `@openai/gpt-5.2` - The best model for coding and agentic tasks (Released Jan 2026)
  - Input: $1.75 / 1M tokens
  - Output: $14.00 / 1M tokens
  
- **Fallback**: `@anthropic/claude-sonnet-4.5` - Optimal balance of intelligence, cost, and speed
  - Input: $3.00 / 1M tokens (≤ 200K tokens)
  - Output: $15.00 / 1M tokens (≤ 200K tokens)

##  Shadow Testing Configuration

**File**: `configs/shadow-config.json`

Load-balanced across three cost-efficient challengers:

1. **`@openai/gpt-5-mini`** (40% weight) - A faster, cheaper version of GPT-5
   - Input: $0.25 / 1M tokens
   - Output: $2.00 / 1M tokens

2. **`@anthropic/claude-haiku-4.5`** (30% weight) - Fastest, most cost-effective Claude model
   - Input: $1.00 / 1M tokens
   - Output: $5.00 / 1M tokens

3. **`@google/gemini-2.5-flash`** (30% weight) - Best price-performance Gemini model
   - Input: $0.30 / 1M tokens
   - Output: $2.50 / 1M tokens

##  Complete Model Pricing Database

**File**: `src/shadow_optic/decision_engine.py`

All pricing verified from official sources as of January 17, 2026:

### OpenAI GPT-5.2 Series (Latest)
- `gpt-5.2`: $1.75 / $14.00 (input/output per 1M tokens)
- `gpt-5.2-pro`: $21.00 / $168.00
- `gpt-5-mini`: $0.25 / $2.00

### OpenAI GPT-4.1 Series
- `gpt-4.1`: $3.00 / $12.00
- `gpt-4.1-mini`: $0.80 / $3.20
- `gpt-4.1-nano`: $0.20 / $0.80

### Anthropic Claude 4.5 Series (Latest)
- `claude-opus-4.5`: $5.00 / $25.00
- `claude-sonnet-4.5`: $3.00 / $15.00
- `claude-haiku-4.5`: $1.00 / $5.00

### Google Gemini 3 Series (Latest)
- `gemini-3-pro`: $2.00 / $12.00
- `gemini-3-flash`: $0.50 / $3.00

### Google Gemini 2.5 Series
- `gemini-2.5-pro`: $1.25 / $10.00
- `gemini-2.5-flash`: $0.30 / $2.50
- `gemini-2.5-flash-lite`: $0.10 / $0.40

##  API Defaults

**File**: `src/shadow_optic/api.py`

Default challenger models:
```python
["gpt-5-mini", "claude-haiku-4.5", "gemini-2.5-flash"]
```

##  Test Suite

**File**: `tests/test_decision_engine.py`

- Tests updated to verify pricing for: `gpt-5.2`, `gpt-5-mini`, `claude-haiku-4.5`, `gemini-2.5-flash`
- Production baseline model: `gpt-5.2`
- All 89 tests passing ✓

## Sources Verified

1. **OpenAI**: https://openai.com/api/pricing/ (Accessed Jan 17, 2026)
   - Confirmed GPT-5.2 as flagship model at $1.75/$14.00
   - Confirmed GPT-5-mini at $0.25/$2.00
   - Confirmed GPT-4.1 series pricing

2. **Anthropic**: https://claude.com/platform/api (Accessed Jan 17, 2026)
   - Confirmed Claude Opus 4.5 at $5.00/$25.00
   - Confirmed Claude Sonnet 4.5 at $3.00/$15.00
   - Confirmed Claude Haiku 4.5 at $1.00/$5.00

3. **Google**: https://ai.google.dev/gemini-api/docs/pricing (Accessed Jan 17, 2026)
   - Confirmed Gemini 3 Pro at $2.00/$12.00 (batch)
   - Confirmed Gemini 3 Flash at $0.50/$3.00 (batch)
   - Confirmed Gemini 2.5 Flash at $0.30/$2.50

## Key Changes from Previous Version

### Before (Incorrect):
- Used `gpt-5-turbo`, `gpt-5`, `gpt-5-nano` (not real model names)
- Used `claude-4-opus`, `claude-4-sonnet`, `claude-4-haiku` (wrong version numbers)
- Mixed pricing sources and outdated data

### After (Correct - Jan 17, 2026):
- **GPT-5.2** (flagship), GPT-5.2-pro, GPT-5-mini (verified real models)
- **Claude Opus 4.5**, Sonnet 4.5, Haiku 4.5 (correct version numbers)
- **Gemini 3 Pro/Flash** (latest generation)
- All pricing verified from official vendor pages

## Test Results

```bash
$ .venv/bin/python -m pytest tests/ -v
======================== 89 passed, 2 warnings in 4.53s ========================
```

All tests passing with latest January 17, 2026 models 

---

**Last Updated**: January 17, 2026  
**Status**:  Production Ready  
**Model Catalog Version**: 2.0.0
