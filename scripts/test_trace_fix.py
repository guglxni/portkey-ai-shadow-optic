#!/usr/bin/env python3
"""Test that tracing attributes are correctly set for Phoenix display."""

import os
import time

# Clear any existing endpoint
os.environ.pop('PHOENIX_COLLECTOR_ENDPOINT', None)

from phoenix.otel import register

tracer_provider = register(
    project_name='shadow-optic-trace-test',
    endpoint='http://localhost:6006/v1/traces',
)

from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

tracer = trace.get_tracer('shadow-optic-test')

print("Creating test traces...")

# Test 1: LLM span with proper output
with tracer.start_as_current_span(
    'test.llm.call',
    attributes={
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.LLM_MODEL_NAME: 'gpt-5-test',
        SpanAttributes.LLM_PROVIDER: 'openai',
        SpanAttributes.INPUT_VALUE: 'What is the capital of France?',
        SpanAttributes.INPUT_MIME_TYPE: 'text/plain',
    }
) as span:
    # Simulate some work
    time.sleep(0.05)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, 'The capital of France is Paris. It is known as the City of Light.')
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, 'text/plain')
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, 10)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, 15)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, 25)
    print("  ✓ LLM span created")

# Test 2: Chain span (benchmark)
with tracer.start_as_current_span(
    'test.benchmark.gpt5',
    attributes={
        SpanAttributes.OPENINFERENCE_SPAN_KIND: 'CHAIN',
        SpanAttributes.INPUT_VALUE: 'Benchmark run for GPT-5',
        SpanAttributes.INPUT_MIME_TYPE: 'text/plain',
    }
) as span:
    time.sleep(0.02)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, 'Cost: $0.0012, Tokens: 250, Refusals: 0')
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, 'text/plain')
    print("  ✓ Chain span created")

# Test 3: Evaluator span (A/B test)
with tracer.start_as_current_span(
    'test.ab_test.gpt5_vs_claude',
    attributes={
        SpanAttributes.OPENINFERENCE_SPAN_KIND: 'EVALUATOR',
        SpanAttributes.INPUT_VALUE: 'A/B Test: GPT-5 vs Claude Haiku',
        SpanAttributes.INPUT_MIME_TYPE: 'text/plain',
    }
) as span:
    time.sleep(0.02)
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, 'Challenger wins! 75% cost savings with equivalent quality.')
    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, 'text/plain')
    print("  ✓ Evaluator span created")

# Give time for traces to be sent
time.sleep(2)

print("\n✅ Test traces sent to Phoenix!")
print("🔮 Check http://localhost:6006 - project: shadow-optic-trace-test")
print("\nExpected results:")
print("  - LLM span: kind=llm, output shows 'The capital of France is Paris...'")
print("  - Chain span: kind=chain, output shows 'Cost: $0.0012...'")
print("  - Evaluator span: kind=evaluator, output shows 'Challenger wins!...'")
