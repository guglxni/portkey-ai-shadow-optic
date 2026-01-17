"""
Observability layer for Shadow-Optic using Arize Phoenix.

Integrates OpenTelemetry and Phoenix to provide deep-dive tracing 
for LLM replays and evaluations.
"""

import logging
import os
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from shadow_optic.models import EvaluationResult, ShadowResult

logger = logging.getLogger(__name__)

class PhoenixObserver:
    """
    Observer for Arize Phoenix integration.
    
    Configures OpenTelemetry to export traces to the Phoenix collector
    and provides helper methods for logging shadow evaluations.
    """
    
    def __init__(self, service_name: str = "shadow-optic"):
        self.service_name = service_name
        self._setup_tracing()
        self.tracer = trace.get_tracer(__name__)

    def _setup_tracing(self):
        """Configure OpenTelemetry OTLP exporter for Phoenix."""
        endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
        
        resource = Resource(attributes={
            "service.name": self.service_name,
            "environment": os.environ.get("ENVIRONMENT", "development")
        })
        
        provider = TracerProvider(resource=resource)
        
        # Phoenix supports OTLP/HTTP
        exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)

    def log_evaluation(
        self, 
        golden_prompt: str, 
        shadow_result: ShadowResult, 
        evaluation: EvaluationResult
    ):
        """
        Logs a shadow evaluation as a structured trace span.
        """
        with self.tracer.start_as_current_span("shadow_evaluation") as span:
            # Set span attributes for filtering in Phoenix
            span.set_attribute("model.challenger", shadow_result.challenger_model)
            span.set_attribute("evaluation.passed", evaluation.passed_thresholds)
            span.set_attribute("evaluation.composite_score", evaluation.composite_score)
            span.set_attribute("cost.shadow", shadow_result.cost)
            span.set_attribute("latency.ms", shadow_result.latency_ms)
            
            # Use OpenInference-like metadata for better UI rendering in Phoenix
            span.set_attribute("input.value", golden_prompt)
            span.set_attribute("output.value", shadow_result.shadow_response)
            
            # Log individual metrics
            span.set_attribute("metrics.faithfulness", evaluation.faithfulness)
            span.set_attribute("metrics.quality", evaluation.quality)
            span.set_attribute("metrics.conciseness", evaluation.conciseness)
            
            if evaluation.reasoning:
                span.set_attribute("evaluation.reasoning", evaluation.reasoning)
            
            if not shadow_result.success:
                span.set_attribute("error", shadow_result.error or "Unknown error")
                span.set_status(trace.Status(trace.StatusCode.ERROR))
            else:
                span.set_status(trace.Status(trace.StatusCode.OK))
