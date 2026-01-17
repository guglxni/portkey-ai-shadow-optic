#!/usr/bin/env python3
"""
Import traces into Phoenix from the hackathon demo results.
This script reads the JSON results and sends them to Phoenix for visualization.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime

# Set up Phoenix before importing
os.environ.pop('PHOENIX_COLLECTOR_ENDPOINT', None)

import phoenix as px
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from phoenix.otel import register
from openinference.semconv.trace import SpanAttributes

def main():
    print("🔮 Importing traces to Phoenix...")
    
    # Read the hackathon results
    results_file = Path(__file__).parent.parent / "hackathon_demo_results.json"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file) as f:
        data = json.load(f)
    
    print(f"📊 Found data from {data.get('timestamp', 'unknown')}")
    
    # Connect to Phoenix
    try:
        tracer_provider = register(
            project_name="shadow-optic-hackathon",
            endpoint="http://localhost:6006/v1/traces",
        )
        print("✅ Connected to Phoenix")
    except Exception as e:
        print(f"⚠️ Phoenix connection: {e}")
        return
    
    tracer = trace.get_tracer("shadow-optic-import")
    
    # Import traces from benchmarks
    benchmarks = data.get("benchmarks", {})
    trace_count = 0
    
    for model_id, benchmark in benchmarks.items():
        model_name = benchmark.get("model_name", model_id)
        provider = benchmark.get("provider", "unknown")
        
        # Simulate a realistic latency for the span duration
        avg_latency = benchmark.get("avg_latency_ms", 500)
        
        # Create a parent span for this model's benchmark with CHAIN span kind
        with tracer.start_as_current_span(
            f"benchmark.{model_name}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN",  # Benchmark is a chain of operations
                SpanAttributes.INPUT_VALUE: f"Benchmark run for {model_name}",
                SpanAttributes.OUTPUT_VALUE: f"Cost: ${benchmark.get('total_cost', 0):.4f}, Tokens: {benchmark.get('total_tokens', 0)}, Refusals: {benchmark.get('refusal_count', 0)}",
                "model.id": model_id,
                "model.name": model_name,
                "model.provider": provider,
                "model.tier": benchmark.get("tier", "unknown"),
                "benchmark.total_cost": benchmark.get("total_cost", 0),
                "benchmark.cost_per_request": benchmark.get("cost_per_request", 0),
                "benchmark.total_tokens": benchmark.get("total_tokens", 0),
                "benchmark.avg_latency_ms": avg_latency,
                "benchmark.refusal_count": benchmark.get("refusal_count", 0),
                "benchmark.refusal_rate": benchmark.get("refusal_rate", 0),
                "benchmark.success_rate": benchmark.get("success_rate", 1.0),
            }
        ) as model_span:
            # Simulate some work to give the span non-zero duration
            time.sleep(0.01)
            trace_count += 1
            print(f"  📝 Logged {model_name}: ${benchmark.get('total_cost', 0):.4f}, "
                  f"{benchmark.get('refusal_count', 0)} refusals")
    
    # Import A/B test results
    ab_tests = data.get("ab_tests", [])
    for ab_test in ab_tests:
        control = ab_test.get("control", "unknown")
        challenger = ab_test.get("challenger", "unknown")
        result = ab_test.get("result", "")
        savings = ab_test.get("cost_savings_pct", 0)
        
        with tracer.start_as_current_span(
            f"ab_test.{control}_vs_{challenger}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "EVALUATOR",  # A/B test is an evaluation
                SpanAttributes.INPUT_VALUE: f"A/B Test: {control} vs {challenger}",
                SpanAttributes.OUTPUT_VALUE: f"{result} - Cost savings: {savings:.1f}%, Recommendation: {ab_test.get('recommendation', 'N/A')}",
                "ab_test.control": control,
                "ab_test.challenger": challenger,
                "ab_test.result": result,
                "ab_test.cost_savings_pct": savings,
                "ab_test.safety_delta_pct": ab_test.get("safety_delta_pct", 0),
                "ab_test.confidence": ab_test.get("confidence", 0),
                "ab_test.recommendation": ab_test.get("recommendation", ""),
            }
        ) as ab_span:
            # Simulate some work to give the span non-zero duration
            time.sleep(0.01)
            trace_count += 1
    
    # Give time for traces to be sent
    time.sleep(2)
    
    print(f"\n✅ Imported {trace_count} traces to Phoenix")
    print(f"🔮 View at: http://localhost:6006/projects")
    print(f"   Project: shadow-optic-hackathon")

if __name__ == "__main__":
    main()
