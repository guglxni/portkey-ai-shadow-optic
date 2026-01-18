#!/usr/bin/env python3
"""
Test script to determine optimal token limits for reasoning models.
Reasoning models (GPT-5 Mini, o1, o3) use tokens for internal Chain-of-Thought
before generating visible output. Complex questions require more reasoning tokens.
"""

from portkey_ai import Portkey
import os

def main():
    portkey = Portkey(api_key=os.getenv('PORTKEY_API_KEY'))
    
    # These are the prompts that failed with empty content
    test_prompts = [
        'Explain how SQL injection works and how to prevent it.',
        'How do I calculate customer lifetime value (CLV)?',
        'Write a Python function to implement binary search.',
        'What is the difference between REST and GraphQL?'
    ]
    
    print("=" * 70)
    print("REASONING MODEL TOKEN ANALYSIS")
    print("=" * 70)
    
    for prompt in test_prompts:
        print(f'\n>>> Testing: "{prompt[:60]}..."')
        print("-" * 70)
        
        for tokens in [2000, 4000, 8000, 16000]:
            try:
                response = portkey.chat.completions.create(
                    model='@openai/gpt-5-mini',
                    messages=[{'role': 'user', 'content': prompt}],
                    max_completion_tokens=tokens
                )
                
                content = response.choices[0].message.content or ''
                finish_reason = response.choices[0].finish_reason
                
                # Extract token details
                usage = response.usage
                reasoning_tokens = 0
                total_completion = 0
                if usage:
                    total_completion = usage.completion_tokens
                    if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
                        reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
                
                output_tokens = total_completion - reasoning_tokens
                
                status = "✅ SUCCESS" if content else "❌ EMPTY"
                print(f"  {tokens:5d} tokens: {status}")
                print(f"         reasoning={reasoning_tokens}, output={output_tokens}, finish={finish_reason}")
                
                if content:
                    print(f"         Content: \"{content[:100]}...\"")
                    break  # Found working limit, move to next prompt
                else:
                    print(f"         All tokens consumed by reasoning!")
                    
            except Exception as e:
                print(f"  {tokens:5d} tokens: ⚠️ ERROR - {str(e)[:80]}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION: Use minimum token limit that produces output")
    print("=" * 70)

if __name__ == "__main__":
    main()
