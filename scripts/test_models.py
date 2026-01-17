#!/usr/bin/env python3
"""Test which models are available in Portkey Model Catalog."""

import os
import httpx
from portkey_ai import Portkey

api_key = os.getenv('PORTKEY_API_KEY')
portkey = Portkey(api_key=api_key, timeout=httpx.Timeout(20.0))

# Models to test based on project design (LATEST_MODELS_JAN_2026.md)
# Using correct provider slugs from Portkey AI Providers:
# - openai, anthropic, grok, vertex, vertex-global, bedrock
MODELS_TO_TEST = [
    # Production models
    ("@openai/gpt-5.2", "max_completion_tokens", "Production primary"),
    ("@openai/gpt-4o", "max_tokens", "Production fallback option"),
    
    # Challenger models (as per shadow-config.json)
    ("@openai/gpt-5-mini", "max_completion_tokens", "Challenger 1 - 40% weight"),
    ("@openai/gpt-4o-mini", "max_tokens", "Challenger 1 fallback"),
    ("@anthropic/claude-haiku-4-5-20251001", "max_tokens", "Challenger 2 - Haiku 4.5"),
    ("@anthropic/claude-sonnet-4-5-20250929", "max_tokens", "Challenger - Sonnet 4.5"),
    
    # Grok models (provider slug is 'grok' not 'x-ai')
    ("@grok/grok-4-fast", "max_tokens", "Grok 4 Fast"),
    ("@grok/grok-4-latest", "max_tokens", "Grok 4 Latest"),
    
    # Gemini models (provider slug is 'vertex' or 'vertex-global')
    ("@vertex/gemini-3-flash-preview", "max_tokens", "Gemini 3 Flash"),
    ("@vertex-global/gemini-2.5-flash", "max_tokens", "Gemini 2.5 Flash"),
]

print("=" * 60)
print("Shadow-Optic Model Availability Test")
print("=" * 60)

working_models = []
failed_models = []

for model, token_param, description in MODELS_TO_TEST:
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "Say OK"}],
        }
        kwargs[token_param] = 10
        
        response = portkey.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        print(f"✓ {model}")
        print(f"  → {description}")
        working_models.append((model, token_param, description))
    except Exception as e:
        error = str(e)[:100]
        print(f"✗ {model}")
        print(f"  → Error: {error}")
        failed_models.append((model, error))

print("\n" + "=" * 60)
print(f"SUMMARY: {len(working_models)} working, {len(failed_models)} failed")
print("=" * 60)

print("\n✅ Working Models:")
for model, token_param, desc in working_models:
    print(f"  - {model} ({token_param})")

if failed_models:
    print("\n❌ Failed Models:")
    for model, error in failed_models:
        print(f"  - {model}")
