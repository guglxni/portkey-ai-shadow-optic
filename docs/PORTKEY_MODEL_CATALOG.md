# Portkey Model Catalog Reference (January 18, 2026)

**CRITICAL: This is the authoritative reference for Portkey integration in Shadow-Optic.**  
**Last Updated: January 18, 2026**  
**Source: app.portkey.ai → Model Catalog → Models tab**

---

## Overview

Portkey uses the **Model Catalog** system with **slug-based model identifiers** (NOT virtual keys).

- **Documentation**: https://portkey.ai/docs/product/model-catalog
- **Your Providers**: vertex-global, vertex, bedrock, openai, grok, anthropic

---

## AI Providers in Your Workspace

| Provider | Slug | Created | Status |
|----------|------|---------|--------|
| **vertex-global** | `vertex-global` | 17th Jan, 2026 | Active |
| **vertex** | `vertex` | 17th Jan, 2026 | Active |
| **bedrock** | `bedrock` | 17th Jan, 2026 | Active |
| **openai** | `openai` | 17th Jan, 2026 | Active |
| **grok** | `grok` | 17th Jan, 2026 | Active |
| **anthropic** | `anthropic` | 17th Jan, 2026 | Active |

---

## Exact Model Slugs (Verified from Your Portkey Dashboard)

### OpenAI Models (`@openai/...`)

| Model Slug | Description | Token Param |
|------------|-------------|-------------|
| `@openai/gpt-5.2` | GPT-5.2 (Latest flagship) | `max_completion_tokens` |
| `@openai/gpt-5.2-pro` | GPT-5.2 Pro | `max_completion_tokens` |
| `@openai/gpt-5.2-2025-12-11` | GPT-5.2 dated version | `max_completion_tokens` |
| `@openai/gpt-5.2-chat-latest` | GPT-5.2 Chat Latest | `max_completion_tokens` |
| `@openai/gpt-5.1` | GPT-5.1 | `max_completion_tokens` |
| `@openai/gpt-5.1-codex` | GPT-5.1 Codex | `max_completion_tokens` |
| `@openai/gpt-5.1-2025-11-13` | GPT-5.1 dated version | `max_completion_tokens` |
| `@openai/gpt-5-pro` | GPT-5 Pro | `max_completion_tokens` |
| `@openai/gpt-5-pro-2025-10-06` | GPT-5 Pro dated | `max_completion_tokens` |
| `@openai/gpt-5-mini` | GPT-5 Mini ⭐ | `max_completion_tokens` |
| `@openai/gpt-5-mini-2025-08-07` | GPT-5 Mini dated | `max_completion_tokens` |
| `@openai/gpt-5-nano` | GPT-5 Nano (cheapest) | `max_completion_tokens` |
| `@openai/gpt-5-nano-2025-08-07` | GPT-5 Nano dated | `max_completion_tokens` |
| `@openai/gpt-5` | GPT-5 Base | `max_completion_tokens` |
| `@openai/gpt-5-2025-08-14` | GPT-5 dated | `max_completion_tokens` |
| `@openai/gpt-5-codex` | GPT-5 Codex | `max_completion_tokens` |
| `@openai/gpt-5-search-api` | GPT-5 Search API | `max_completion_tokens` |
| `@openai/gpt-4o` | GPT-4o ⭐ | `max_tokens` |
| `@openai/gpt-4o-mini` | GPT-4o Mini ⭐ | `max_tokens` |
| `@openai/gpt-4o-latest` | GPT-4o Latest | `max_tokens` |
| `@openai/gpt-4.1-nano` | GPT-4.1 Nano | `max_completion_tokens` |
| `@openai/gpt-realtime` | GPT Realtime | `max_tokens` |
| `@openai/gpt-realtime-mini` | GPT Realtime Mini | `max_tokens` |

### Anthropic Models (`@anthropic/...`)

| Model Slug | Description | Token Param |
|------------|-------------|-------------|
| `@anthropic/claude-opus-4-5` | Claude Opus 4.5 ⭐ (Judge) | `max_tokens` |
| `@anthropic/claude-opus-4-5-20251101` | Claude Opus 4.5 dated | `max_tokens` |
| `@anthropic/claude-sonnet-4-5` | Claude Sonnet 4.5 ⭐ | `max_tokens` |
| `@anthropic/claude-sonnet-4-5-20250929` | Claude Sonnet 4.5 dated | `max_tokens` |
| `@anthropic/claude-sonnet-4` | Claude Sonnet 4 | `max_tokens` |
| `@anthropic/claude-sonnet-4-0` | Claude Sonnet 4.0 | `max_tokens` |
| `@anthropic/claude-sonnet-4-20250514` | Claude Sonnet 4 dated | `max_tokens` |
| `@anthropic/claude-haiku-4-5-20251001` | Claude Haiku 4.5 ⭐ | `max_tokens` |
| `@anthropic/claude-opus-4-20250514` | Claude Opus 4 | `max_tokens` |
| `@anthropic/claude-opus-4-1-20250805` | Claude Opus 4.1 | `max_tokens` |
| `@anthropic/claude-3-opus-20240229` | Claude 3 Opus | `max_tokens` |
| `@anthropic/claude-3-sonnet-20240229` | Claude 3 Sonnet | `max_tokens` |
| `@anthropic/claude-3-haiku-20240307` | Claude 3 Haiku | `max_tokens` |
| `@anthropic/claude-3-5-sonnet-20240620` | Claude 3.5 Sonnet | `max_tokens` |
| `@anthropic/claude-3-7-sonnet-20250219` | Claude 3.7 Sonnet | `max_tokens` |
| `@anthropic/claude-instant-1.2` | Claude Instant | `max_tokens` |

### Anthropic via Bedrock (`@bedrock/...`)

| Model Slug | Description | Token Param |
|------------|-------------|-------------|
| `@bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | Claude Sonnet 4.5 via Bedrock ⭐ | `max_tokens` |
| `@bedrock/us.anthropic.claude-sonnet-4-5-20250514-v1:0` | Claude Sonnet 4.5 dated | `max_tokens` |
| `@bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0` | Claude Opus 4.5 via Bedrock ⭐ | `max_tokens` |
| `@bedrock/us.anthropic.claude-opus-4-5-20250805-v1:0` | Claude Opus 4.5 dated | `max_tokens` |
| `@bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0` | Claude Haiku 4.5 via Bedrock ⭐ | `max_tokens` |
| `@bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0` | Claude 3 Sonnet via Bedrock | `max_tokens` |
| `@bedrock/us.anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku via Bedrock | `max_tokens` |
| `@bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0` | Claude 3.7 Sonnet via Bedrock | `max_tokens` |
| `@bedrock/global.anthropic.claude-sonnet-4-5-...` | Global Claude Sonnet 4.5 | `max_tokens` |
| `@bedrock/eu.anthropic.claude-opus-4-5-...` | EU Claude Opus 4.5 | `max_tokens` |

### Google Gemini via Vertex (`@vertex-global/...`)

| Model Slug | Description | Token Param |
|------------|-------------|-------------|
| `@vertex-global/gemini-3-flash-preview` | Gemini 3 Flash Preview ⭐ (LATEST) | `max_tokens` |
| `@vertex-global/gemini-3-pro-preview` | Gemini 3 Pro Preview | `max_tokens` |
| `@vertex-global/gemini-3-pro-preview-02-05` | Gemini 3 Pro dated | `max_tokens` |
| `@vertex-global/gemini-3-pro-image-preview` | Gemini 3 Pro Image | `max_tokens` |
| `@vertex-global/gemini-2.5-pro` | Gemini 2.5 Pro | `max_tokens` |
| `@vertex-global/gemini-2.5-pro-preview-05-06` | Gemini 2.5 Pro dated | `max_tokens` |
| `@vertex-global/gemini-2.5-pro-exp-03-25` | Gemini 2.5 Pro Exp | `max_tokens` |
| `@vertex-global/gemini-2.5-flash` | Gemini 2.5 Flash ⭐ | `max_tokens` |
| `@vertex-global/gemini-2.5-flash-preview-04-17` | Gemini 2.5 Flash dated | `max_tokens` |
| `@vertex-global/gemini-2.5-flash-lite-preview-06-17` | Gemini 2.5 Flash Lite | `max_tokens` |
| `@vertex-global/gemini-2.5-flash-image` | Gemini 2.5 Flash Image | `max_tokens` |
| `@vertex-global/gemini-2.0-flash` | Gemini 2.0 Flash | `max_tokens` |
| `@vertex-global/gemini-2.0-flash-lite` | Gemini 2.0 Flash Lite | `max_tokens` |
| `@vertex-global/gemini-2.0-flash-thinking-exp-01-21` | Gemini 2.0 Thinking | `max_tokens` |
| `@vertex-global/gemini-1.5-pro-001` | Gemini 1.5 Pro | `max_tokens` |
| `@vertex-global/gemini-1.5-flash-001` | Gemini 1.5 Flash | `max_tokens` |

### Grok x-ai Models (`@grok/...`)

| Model Slug | Description | Token Param |
|------------|-------------|-------------|
| `@grok/grok-4-latest` | Grok 4 Latest ⭐ | `max_tokens` |
| `@grok/grok-4` | Grok 4 | `max_tokens` |
| `@grok/grok-4-fast` | Grok 4 Fast | `max_tokens` |
| `@grok/grok-4-fast-reasoning` | Grok 4 Fast Reasoning | `max_tokens` |
| `@grok/grok-4-fast-reasoning-latest` | Grok 4 Fast Reasoning Latest | `max_tokens` |
| `@grok/grok-4-fast-non-reasoning` | Grok 4 Fast Non-Reasoning | `max_tokens` |
| `@grok/grok-4-fast-non-reasoning-latest` | Grok 4 Fast Non-Reasoning Latest | `max_tokens` |
| `@grok/grok-4-1-fast-reasoning` | Grok 4.1 Fast Reasoning ⭐ | `max_tokens` |
| `@grok/grok-4-1-fast-reasoning-latest` | Grok 4.1 Fast Reasoning Latest | `max_tokens` |
| `@grok/grok-4-1-fast-non-reasoning` | Grok 4.1 Fast Non-Reasoning | `max_tokens` |
| `@grok/grok-4-1-fast-non-reasoning-latest` | Grok 4.1 Fast Non-Reasoning Latest | `max_tokens` |
| `@grok/grok-4-0709` | Grok 4 dated | `max_tokens` |
| `@grok/grok-3-mini-fast` | Grok 3 Mini Fast ⭐ (Budget) | `max_tokens` |
| `@grok/grok-3-mini-fast-beta` | Grok 3 Mini Fast Beta | `max_tokens` |
| `@grok/grok-3-mini-fast-latest` | Grok 3 Mini Fast Latest | `max_tokens` |
| `@grok/grok-3-mini-latest` | Grok 3 Mini Latest | `max_tokens` |
| `@grok/grok-3-mini` | Grok 3 Mini | `max_tokens` |
| `@grok/grok-3-latest` | Grok 3 Latest | `max_tokens` |
| `@grok/grok-3-fast` | Grok 3 Fast | `max_tokens` |
| `@grok/grok-3-fast-latest` | Grok 3 Fast Latest | `max_tokens` |
| `@grok/grok-3-fast-beta` | Grok 3 Fast Beta | `max_tokens` |
| `@grok/grok-3-beta` | Grok 3 Beta | `max_tokens` |
| `@grok/grok-3` | Grok 3 | `max_tokens` |
| `@grok/grok-vision-beta` | Grok Vision Beta | `max_tokens` |
| `@grok/grok-code-fast` | Grok Code Fast | `max_tokens` |
| `@grok/grok-code-fast-1` | Grok Code Fast 1 | `max_tokens` |
| `@grok/grok-code-fast-1-0825` | Grok Code Fast dated | `max_tokens` |
| `@grok/grok-2-latest` | Grok 2 Latest | `max_tokens` |
| `@grok/grok-2-vision` | Grok 2 Vision | `max_tokens` |
| `@grok/grok-2` | Grok 2 | `max_tokens` |

### DeepSeek via Bedrock (`@bedrock/...`)

| Model Slug | Description | Token Param |
|------------|-------------|-------------|
| `@bedrock/us.deepseek.r1-v1:0` | DeepSeek R1 (Reasoning) ⭐ | `max_tokens` |
| `@bedrock/deepseek.v3-v1:0` | DeepSeek V3 | `max_tokens` |
| `@bedrock/deepseek.r1-v1:0` | DeepSeek R1 | `max_tokens` |

---

## Token Parameter Rules

### Models Using `max_completion_tokens`

**ALL GPT-5.x, GPT-4.1, O1, O3 models:**

```python
# GPT-5, GPT-4.1, O1, O3 models
response = client.chat.completions.create(
    model="@openai/gpt-5.2",
    messages=[...],
    max_completion_tokens=1000  # ← NOT max_tokens!
)
```

### Models Using `max_tokens`

**Claude, Gemini, Grok, DeepSeek, GPT-4o:**

```python
# All other models
response = client.chat.completions.create(
    model="@anthropic/claude-opus-4-5",
    messages=[...],
    max_tokens=1000  # ← Standard parameter
)
```

---

## Recommended Model Selection (Shadow-Optic)

### Production Model
```python
PRODUCTION_MODEL = "@openai/gpt-5.2"  # Latest flagship
```

### Challenger Models (Multi-Provider)
```python
CHALLENGER_MODELS = [
    # OpenAI
    "@openai/gpt-5-mini",                     # GPT-5 Mini
    "@openai/gpt-5-nano",                     # GPT-5 Nano (cheapest)
    # Anthropic
    "@anthropic/claude-sonnet-4-5",           # Claude Sonnet 4.5
    "@anthropic/claude-haiku-4-5-20251001",   # Claude Haiku 4.5
    # Vertex/Gemini
    "@vertex-global/gemini-3-flash-preview",  # Gemini 3 Flash (LATEST)
    "@vertex-global/gemini-2.5-flash",        # Gemini 2.5 Flash
    # Grok
    "@grok/grok-4-latest",                    # Grok 4 Latest
    "@grok/grok-3-mini-fast",                 # Grok 3 Mini (Budget)
    # DeepSeek
    "@bedrock/us.deepseek.r1-v1:0",           # DeepSeek R1 (Reasoning)
]
```

### Judge Model
```python
JUDGE_MODEL = "@anthropic/claude-opus-4-5"    # Best quality for judging
```

---

## DSPy Integration

```python
import dspy
import os

# Portkey Native DSPy Format: openai/@provider/model-name
lm = dspy.LM(
    model="openai/@openai/gpt-5-mini",
    api_base="https://api.portkey.ai/v1",
    api_key=os.environ["PORTKEY_API_KEY"]
)
dspy.configure(lm=lm)
```

---

## Quick Reference

```python
from portkey_ai import Portkey
import os

client = Portkey(api_key=os.environ["PORTKEY_API_KEY"])

# GPT-5.2 (latest - use max_completion_tokens)
client.chat.completions.create(
    model="@openai/gpt-5.2",
    messages=[{"role": "user", "content": "Hello"}],
    max_completion_tokens=500
)

# Claude Opus 4.5 (use max_tokens)
client.chat.completions.create(
    model="@anthropic/claude-opus-4-5",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=500
)

# Gemini 3 Flash (use max_tokens)
client.chat.completions.create(
    model="@vertex-global/gemini-3-flash-preview",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=500
)

# Grok 4 Latest (use max_tokens)
client.chat.completions.create(
    model="@grok/grok-4-latest",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=500
)
```
