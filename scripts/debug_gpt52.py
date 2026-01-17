#!/usr/bin/env python3
"""Debug GPT-5.2 empty content issue."""
from portkey_ai import Portkey
import os

portkey = Portkey(
    api_key=os.getenv('PORTKEY_API_KEY'),
    virtual_key=os.getenv('SHADOW_OPTIC_VIRTUAL_KEY_GPT4')
)

# Test with a prompt that was returning empty
response = portkey.chat.completions.create(
    model='@openai/gpt-5.2',
    messages=[{'role': 'user', 'content': 'Explain our return policy'}],
    max_completion_tokens=300
)

print('=== FULL RESPONSE ANALYSIS (300 tokens) ===')
print(f'Content: {repr(response.choices[0].message.content)}')
print(f'Finish Reason: {response.choices[0].finish_reason}')
print()
print('=== USAGE DETAILS ===')
print(f'Prompt Tokens: {response.usage.prompt_tokens}')
print(f'Completion Tokens: {response.usage.completion_tokens}')
print(f'Total Tokens: {response.usage.total_tokens}')
print()
print('=== COMPLETION TOKEN DETAILS ===')
if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
    details = response.usage.completion_tokens_details
    print(f'Reasoning Tokens: {getattr(details, "reasoning_tokens", "N/A")}')
    print(f'Audio Tokens: {getattr(details, "audio_tokens", "N/A")}')
    print(f'Accepted Prediction: {getattr(details, "accepted_prediction_tokens", "N/A")}')
    print(f'Rejected Prediction: {getattr(details, "rejected_prediction_tokens", "N/A")}')
else:
    print('No completion_tokens_details available')
print()

# Now test with higher token limit
print('=== TESTING WITH 2000 TOKEN LIMIT ===')
response2 = portkey.chat.completions.create(
    model='@openai/gpt-5.2',
    messages=[{'role': 'user', 'content': 'Explain our return policy'}],
    max_completion_tokens=2000
)
content2 = response2.choices[0].message.content
print(f'Content: {repr(content2[:100]) if content2 else "(EMPTY)"}...')
print(f'Finish Reason: {response2.choices[0].finish_reason}')
print(f'Completion Tokens: {response2.usage.completion_tokens}')
if hasattr(response2.usage, 'completion_tokens_details') and response2.usage.completion_tokens_details:
    print(f'Reasoning Tokens: {response2.usage.completion_tokens_details.reasoning_tokens}')
