"""
Temporal Activities for Shadow-Optic.

Production-grade activities using REAL SDK integrations:
- AsyncPortkey for Portkey AI Gateway
- AsyncQdrantClient for vector operations  
- DeepEval for LLM-as-Judge evaluation

Each activity handles one specific task with proper error handling,
heartbeats for long operations, and async/await patterns.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from portkey_ai import AsyncPortkey
from qdrant_client import AsyncQdrantClient
from temporalio import activity

from shadow_optic.models import (
    ChallengerConfig,
    ChallengerResult,
    EvaluationResult,
    EvaluatorConfig,
    GoldenPrompt,
    OptimizationReport,
    ProductionTrace,
    SamplerConfig,
    ShadowResult,
    TraceStatus,
    WorkflowConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Client Factory Functions
# =============================================================================

def get_portkey_client() -> AsyncPortkey:
    """Get configured async Portkey client."""
    api_key = os.environ.get("PORTKEY_API_KEY")
    if not api_key:
        raise ValueError("PORTKEY_API_KEY environment variable is required")
    
    return AsyncPortkey(
        api_key=api_key,
        base_url=os.environ.get("PORTKEY_BASE_URL", "https://api.portkey.ai/v1")
    )


def get_qdrant_client() -> AsyncQdrantClient:
    """Get configured async Qdrant client."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY")
    
    return AsyncQdrantClient(url=url, api_key=api_key, timeout=60.0)


# =============================================================================
# Helper Functions
# =============================================================================

def _parse_time_window(window: str) -> int:
    """Parse time window string to hours."""
    window = window.lower().strip()
    if window.endswith("h"):
        return int(window[:-1])
    elif window.endswith("d"):
        return int(window[:-1]) * 24
    elif window.endswith("w"):
        return int(window[:-1]) * 24 * 7
    return int(window)


def _extract_prompt(messages: List[Dict[str, Any]]) -> str:
    """Extract user prompt from chat messages."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
            return str(content)
    return ""


def _extract_response(choices: List[Dict[str, Any]]) -> str:
    """Extract assistant response from completion choices."""
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content", ""))


# =============================================================================
# Activity: Export Portkey Logs (using Logs Export API)
# =============================================================================

@activity.defn
async def export_portkey_logs(
    time_window: str,
    workspace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Export production logs from Portkey using the Logs Export API.
    
    Real implementation using:
    - portkey.logs.exports.create() - Create export job
    - portkey.logs.exports.start() - Start the job
    - portkey.logs.exports.download() - Download results
    """
    activity.logger.info(f"Exporting Portkey logs for window: {time_window}")
    
    portkey = get_portkey_client()
    window_hours = _parse_time_window(time_window)
    start_time = datetime.utcnow() - timedelta(hours=window_hours)
    end_time = datetime.utcnow()
    
    try:
        activity.heartbeat("Creating export job")
        
        filters = {
            "timestamp": {
                "gte": start_time.isoformat() + "Z",
                "lte": end_time.isoformat() + "Z"
            },
            "status": 200,
            "type": "chat.completions"
        }
        
        export_response = await portkey.logs.exports.create(
            filters=filters,
            workspace_id=workspace_id or os.environ.get("PORTKEY_WORKSPACE_ID"),
            requested_data=["trace_id", "request", "response", "model", "latency", "tokens", "cost", "metadata", "timestamp"]
        )
        
        export_id = export_response.id
        activity.logger.info(f"Created export job: {export_id}")
        
        activity.heartbeat("Starting export job")
        await portkey.logs.exports.start(export_id=export_id)
        
        # Poll for completion
        max_wait = 300
        elapsed = 0
        while elapsed < max_wait:
            activity.heartbeat(f"Waiting for export... ({elapsed}s)")
            status = await portkey.logs.exports.retrieve(export_id=export_id)
            
            if status.status == "completed":
                break
            elif status.status == "failed":
                raise RuntimeError(f"Export failed: {status.error}")
            
            await asyncio.sleep(5)
            elapsed += 5
        
        if elapsed >= max_wait:
            raise TimeoutError("Export timed out")
        
        activity.heartbeat("Downloading export data")
        download = await portkey.logs.exports.download(export_id=export_id)
        
        logs = download.data if hasattr(download, 'data') else []
        activity.logger.info(f"Exported {len(logs)} logs")
        return logs
        
    finally:
        await portkey.close()


@activity.defn
async def parse_portkey_logs(raw_logs: List[Dict[str, Any]]) -> List[ProductionTrace]:
    """Parse raw Portkey logs into ProductionTrace objects."""
    activity.logger.info(f"Parsing {len(raw_logs)} logs")
    
    traces = []
    for i, log in enumerate(raw_logs):
        if i % 100 == 0:
            activity.heartbeat(f"Parsing {i}/{len(raw_logs)}")
        
        try:
            request = log.get("request", {})
            response = log.get("response", {})
            usage = response.get("usage", {})
            
            prompt = _extract_prompt(request.get("messages", []))
            resp_text = _extract_response(response.get("choices", []))
            
            if not prompt or not resp_text:
                continue
            
            timestamp = log.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except:
                ts = datetime.utcnow()
            
            traces.append(ProductionTrace(
                trace_id=log.get("trace_id", log.get("id", "")),
                request_timestamp=ts,
                prompt=prompt,
                response=resp_text,
                model=log.get("model", "unknown"),
                latency_ms=float(log.get("latency", 0)),
                tokens_prompt=usage.get("prompt_tokens", 0),
                tokens_completion=usage.get("completion_tokens", 0),
                cost=float(log.get("cost", 0.0)),
                status=TraceStatus.SUCCESS,
                metadata=log.get("metadata", {})
            ))
        except Exception as e:
            logger.warning(f"Parse error: {e}")
    
    activity.logger.info(f"Parsed {len(traces)} valid traces")
    return traces


# =============================================================================
# Activity: Sample Golden Prompts
# =============================================================================

@activity.defn
async def sample_golden_prompts(
    traces: List[ProductionTrace],
    config: SamplerConfig
) -> List[GoldenPrompt]:
    """Select representative golden prompts using semantic sampling."""
    from shadow_optic.sampler import SemanticSampler
    
    activity.logger.info(f"Sampling from {len(traces)} traces")
    
    if not traces:
        return []
    
    portkey = get_portkey_client()
    qdrant = get_qdrant_client()
    
    try:
        sampler = SemanticSampler(qdrant_client=qdrant, portkey_client=portkey, config=config)
        golden_prompts = await sampler.sample(traces, heartbeat_fn=activity.heartbeat)
        
        rate = len(golden_prompts) / len(traces) * 100 if traces else 0
        activity.logger.info(f"Selected {len(golden_prompts)} prompts ({rate:.1f}%)")
        return golden_prompts
    finally:
        await portkey.close()
        await qdrant.close()


# =============================================================================
# Activity: Shadow Replay (via Portkey AI Gateway)
# =============================================================================

@activity.defn
async def replay_shadow_requests(
    golden_prompts: List[GoldenPrompt],
    challenger_config: ChallengerConfig
) -> List[ShadowResult]:
    """Replay golden prompts against challenger model via Portkey."""
    activity.logger.info(f"Replaying {len(golden_prompts)} prompts against {challenger_config.model_id}")
    
    portkey = get_portkey_client()
    results = []
    
    # Concurrency control
    semaphore = asyncio.Semaphore(5)

    async def replay_single(golden: GoldenPrompt) -> ShadowResult:
        async with semaphore:
            try:
                start = datetime.utcnow()
                
                # Model Catalog format: @provider_slug/model_id
                model_catalog_id = challenger_config.model_catalog_id
                
                response = await portkey.chat.completions.create(
                    model=model_catalog_id,
                    messages=[{"role": "user", "content": golden.prompt}],
                    config=challenger_config.portkey_config_id,
                    max_tokens=challenger_config.max_tokens or 4096,
                    temperature=challenger_config.temperature or 0.0
                )
                
                latency = (datetime.utcnow() - start).total_seconds() * 1000
                
                content = ""
                if response.choices:
                    content = response.choices[0].message.content or ""
                
                usage = response.usage
                return ShadowResult(
                    golden_prompt_id=golden.original_trace_id,
                    challenger_model=challenger_config.model_id,
                    shadow_response=content,
                    latency_ms=latency,
                    tokens_prompt=usage.prompt_tokens if usage else 0,
                    tokens_completion=usage.completion_tokens if usage else 0,
                    cost=getattr(response, '_cost', 0.0),
                    success=True,
                    timestamp=datetime.utcnow()
                )
            
            # Handle specific error types gracefully
            except Exception as e:
                error_msg = str(e)
                error_type = "unknown"
                
                # Categorize error for better reporting
                if "400" in error_msg or "BadRequest" in error_msg:
                    error_type = "bad_request"  # Context window, invalid params
                    logger.warning(f"Replay bad request (400): {error_msg[:100]}")
                elif "401" in error_msg or "Unauthorized" in error_msg:
                    error_type = "auth_error"
                    logger.error(f"Replay auth error: {error_msg[:100]}")
                elif "429" in error_msg or "RateLimited" in error_msg:
                    error_type = "rate_limited"
                    logger.warning(f"Replay rate limited: {error_msg[:100]}")
                elif "500" in error_msg or "502" in error_msg or "503" in error_msg:
                    error_type = "server_error"
                    logger.warning(f"Replay server error: {error_msg[:100]}")
                elif "timeout" in error_msg.lower():
                    error_type = "timeout"
                    logger.warning(f"Replay timeout: {error_msg[:100]}")
                else:
                    logger.warning(f"Replay failed: {error_msg[:100]}")
                
                return ShadowResult(
                    golden_prompt_id=golden.original_trace_id,
                    challenger_model=challenger_config.model_id,
                    shadow_response="",
                    latency_ms=0,
                    tokens_prompt=0,
                    tokens_completion=0,
                    cost=0.0,
                    success=False,
                    error=f"{error_type}: {error_msg[:200]}",
                    timestamp=datetime.utcnow()
                )

    try:
        tasks = [replay_single(golden) for golden in golden_prompts]
        
        # Monitor progress via intermediate completion
        pending = set(tasks)
        completed_count = 0
        
        # While tasks are pending, wait for them to complete in chunks or one by one
        # to send heartbeats.
        # Simplification: gather all, but use a periodic heartbeat task if needed.
        # For simplicity and robustness with Temporal, we can just await gather
        # and rely on the fact that individual tasks are short. 
        # However, for 50+ tasks, we should heartbeat.
        
        # Let's use as_completed-like behavior manually or just heartbeat before gather
        activity.heartbeat("Starting parallel replay")
        
        results = await asyncio.gather(*tasks)
        
        success = sum(1 for r in results if r.success)
        activity.logger.info(f"Completed {success}/{len(results)} replays")
        return list(results)
    finally:
        await portkey.close()


# =============================================================================
# Activity: Evaluate Shadow Results (via DeepEval)
# =============================================================================

@activity.defn
async def evaluate_shadow_results(
    golden_prompts: List[GoldenPrompt],
    shadow_results: List[ShadowResult],
    config: EvaluatorConfig
) -> List[EvaluationResult]:
    """Evaluate shadow results using LLM-as-Judge with DeepEval."""
    from shadow_optic.evaluator import ShadowEvaluator
    from shadow_optic.observability import PhoenixObserver
    
    activity.logger.info(f"Evaluating {len(shadow_results)} results")
    
    golden_by_id = {g.original_trace_id: g for g in golden_prompts}
    successful = [r for r in shadow_results if r.success]
    
    if not successful:
        return []
    
    portkey = get_portkey_client()
    observer = PhoenixObserver()
    
    try:
        evaluator = ShadowEvaluator(portkey_client=portkey, config=config)
        evaluations = []
        
        for i, shadow in enumerate(successful):
            if i % 5 == 0:
                activity.heartbeat(f"Evaluating {i}/{len(successful)}")
            
            golden = golden_by_id.get(shadow.golden_prompt_id)
            if not golden:
                continue
            
            try:
                result = await evaluator.evaluate(golden_prompt=golden, shadow_result=shadow)
                
                # Log to Phoenix for observability
                try:
                    observer.log_evaluation(golden.prompt, shadow, result)
                except Exception as obs_err:
                    logger.warning(f"Phoenix logging failed: {obs_err}")
                
                evaluations.append(result)
            except Exception as e:
                evaluations.append(EvaluationResult(
                    shadow_result_id=f"{shadow.golden_prompt_id}_{shadow.challenger_model}",
                    faithfulness=0.0, quality=0.0, conciseness=0.0,
                    is_refusal=False, composite_score=0.0, passed_thresholds=False,
                    judge_model=config.judge_model,
                    reasoning=f"Error: {e}"
                ))
            
            await asyncio.sleep(0.1)
        
        passed = sum(1 for e in evaluations if e.passed_thresholds)
        activity.logger.info(f"Evaluated {len(evaluations)}, {passed} passed")
        return evaluations
    finally:
        await portkey.close()


# =============================================================================
# Activity: Generate Report
# =============================================================================

@activity.defn
async def generate_optimization_report(
    traces: List[ProductionTrace],
    golden_prompts: List[GoldenPrompt],
    evaluations: Dict[str, List[EvaluationResult]],
    config: WorkflowConfig
) -> OptimizationReport:
    """Generate optimization report with recommendations."""
    from shadow_optic.decision_engine import DecisionEngine
    
    activity.logger.info("Generating optimization report")
    
    engine = DecisionEngine(
        config=config.evaluator,
        monthly_request_volume=getattr(config, 'monthly_request_volume', 100_000)
    )
    
    # Aggregate evaluations into challenger results
    challenger_results = {}
    for model_id, model_evals in evaluations.items():
        challenger_results[model_id] = engine.aggregate_results(model_evals, model_id)
    
    report = engine.generate_report(
        challenger_results=challenger_results,
        production_model=traces[0].model if traces else "unknown",
        traces_analyzed=len(traces),
        samples_evaluated=len(golden_prompts)
    )
    
    activity.logger.info(f"Recommendation: {report.recommendation.action}")
    return report


# =============================================================================
# Activity: Submit Feedback to Portkey
# =============================================================================

@activity.defn
async def submit_portkey_feedback(report: OptimizationReport) -> int:
    """Submit evaluation results as feedback to Portkey."""
    activity.logger.info(f"Submitting feedback for report: {report.report_id}")
    
    portkey = get_portkey_client()
    
    try:
        feedbacks = []
        
        # Extract evaluations from challenger results
        for model_id, challenger_result in report.challenger_results.items():
            for eval_result in challenger_result.individual_results:
                feedbacks.append({
                    "trace_id": eval_result.shadow_result_id.split("_")[0],  # Extract original trace ID
                    "value": eval_result.composite_score,
                    "weight": 1.0,
                    "metadata": {
                        "challenger_model": model_id,
                        "faithfulness": eval_result.faithfulness,
                        "quality": eval_result.quality,
                        "conciseness": eval_result.conciseness,
                        "is_refusal": eval_result.is_refusal,
                        "passed_thresholds": eval_result.passed_thresholds,
                        "report_id": report.report_id,
                        "source": "shadow-optic"
                    }
                })
        
        if not feedbacks:
            activity.logger.info("No feedback to submit")
            return 0
        
        activity.logger.info(f"Submitting {len(feedbacks)} feedback entries")
        
        # Batch submit
        submitted = 0
        for i in range(0, len(feedbacks), 100):
            batch = feedbacks[i:i+100]
            activity.heartbeat(f"Submitting batch {i//100 + 1}")
            try:
                await portkey.feedback.bulk_create(feedbacks=batch)
                submitted += len(batch)
            except Exception as e:
                logger.error(f"Feedback batch failed: {e}")
        
        activity.logger.info(f"Submitted {submitted} feedback entries")
        return submitted
    finally:
        await portkey.close()
