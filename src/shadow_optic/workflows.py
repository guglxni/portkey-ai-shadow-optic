"""
Temporal Workflow Definitions for Shadow-Optic.

Implements durable, fault-tolerant workflows for the optimization pipeline:

1. OptimizeModelSelectionWorkflow - Main orchestration workflow
2. SamplingWorkflow - Semantic sampling sub-workflow
3. ReplayWorkflow - Shadow replay sub-workflow
4. EvaluationWorkflow - Quality evaluation sub-workflow

Temporal provides:
- Automatic retries with exponential backoff
- State persistence across failures
- Long-running operation support
- Fan-out/fan-in parallelism
- Workflow versioning for updates
"""

import asyncio
from datetime import timedelta
from typing import Any, Dict, List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

# Import activities (these are defined in activities.py)
with workflow.unsafe.imports_passed_through():
    from shadow_optic.models import (
        ChallengerConfig,
        EvaluationResult,
        GoldenPrompt,
        OptimizationReport,
        ProductionTrace,
        ShadowResult,
        WorkflowConfig,
    )

# Default retry policy for activities
DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=5,
    non_retryable_error_types=["ValueError", "InvalidArgumentError"]
)


@workflow.defn
class OptimizeModelSelectionWorkflow:
    """
    Main orchestration workflow for Shadow-Optic.
    
    Orchestrates the complete optimization pipeline:
        1. Export production logs from Portkey
        2. Semantic sampling to select golden prompts
        3. Parallel shadow replay against challengers
        4. LLM-as-Judge evaluation
        5. Decision engine recommendation
        6. Feedback submission to Portkey
    
    This workflow is scheduled to run daily (configurable) and can
    also be triggered manually via API.
    
    State Management:
        - Workflow state is persisted by Temporal
        - Supports resumption after failures
        - Tracks progress for observability
    
    Example:
        ```python
        client = await Client.connect("localhost:7233")
        
        result = await client.execute_workflow(
            OptimizeModelSelectionWorkflow.run,
            WorkflowConfig(...),
            id="optimize-2024-01-15",
            task_queue="shadow-optic"
        )
        ```
    """
    
    def __init__(self):
        self._progress: float = 0.0
        self._stage: str = "initializing"
        self._message: str = ""
    
    @workflow.run
    async def run(self, config: WorkflowConfig) -> OptimizationReport:
        """
        Execute the complete optimization workflow.
        
        Args:
            config: Workflow configuration with challengers, thresholds, etc.
        
        Returns:
            OptimizationReport with recommendation and all metrics
        """
        workflow.logger.info(f"Starting optimization workflow with config: {config}")
        
        try:
            # Stage 1: Export production logs
            self._update_progress(0.1, "exporting_logs", "Fetching production logs from Portkey")
            traces = await self._export_production_logs(config)
            
            if not traces:
                workflow.logger.warning("No traces found in export window")
                return await self._create_empty_report(config)
            
            # Stage 2: Semantic sampling
            self._update_progress(0.2, "sampling", f"Sampling {len(traces)} traces")
            golden_prompts = await self._sample_golden_prompts(traces, config)
            
            workflow.logger.info(f"Selected {len(golden_prompts)} golden prompts")
            
            # Stage 3: Shadow replay (parallel across challengers)
            self._update_progress(0.4, "shadow_replay", "Replaying against challengers")
            shadow_results = await self._replay_all_challengers(golden_prompts, config)
            
            # Stage 4: Evaluation
            self._update_progress(0.7, "evaluation", "Evaluating with LLM-as-Judge")
            evaluations = await self._evaluate_all_results(
                golden_prompts, shadow_results, config
            )
            
            # Stage 5: Generate report
            self._update_progress(0.9, "generating_report", "Analyzing results")
            report = await self._generate_report(
                traces, golden_prompts, evaluations, config
            )
            
            # Stage 6: Submit feedback to Portkey
            self._update_progress(0.95, "submitting_feedback", "Pushing results to Portkey")
            await self._submit_feedback(report)
            
            self._update_progress(1.0, "complete", "Optimization complete")
            
            return report
            
        except Exception as e:
            workflow.logger.error(f"Workflow failed: {e}")
            self._update_progress(0.0, "failed", str(e))
            raise
    
    async def _export_production_logs(
        self,
        config: WorkflowConfig
    ) -> List[ProductionTrace]:
        """Export logs from Portkey via activity."""
        return await workflow.execute_activity(
            "export_portkey_logs",
            args=[config.log_export_window],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=DEFAULT_RETRY_POLICY
        )
    
    async def _sample_golden_prompts(
        self,
        traces: List[ProductionTrace],
        config: WorkflowConfig
    ) -> List[GoldenPrompt]:
        """Run semantic sampling via activity.
        
        Uses PromptGenie's semantic clustering when config.genie.enabled is True,
        otherwise falls back to the legacy regex-based sampler.
        """
        if config.genie.enabled:
            workflow.logger.info("Using PromptGenie for semantic sampling")
            return await workflow.execute_activity(
                "sample_golden_prompts_genie",
                args=[traces, config.genie, config.sampler.samples_per_cluster],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=DEFAULT_RETRY_POLICY
            )
        else:
            return await workflow.execute_activity(
                "sample_golden_prompts",
                args=[traces, config.sampler],
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=DEFAULT_RETRY_POLICY
            )
    
    async def _replay_all_challengers(
        self,
        golden_prompts: List[GoldenPrompt],
        config: WorkflowConfig
    ) -> Dict[str, List[ShadowResult]]:
        """
        Replay golden prompts against all challenger models in parallel.
        
        Uses fan-out pattern: one child workflow per challenger.
        """
        # Create tasks for parallel execution
        tasks = []
        for challenger in config.challengers:
            task = workflow.execute_activity(
                "replay_shadow_requests",
                args=[golden_prompts, challenger],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=DEFAULT_RETRY_POLICY
            )
            tasks.append((challenger.model_id, task))
        
        # Wait for all to complete
        results = {}
        for model_id, task in tasks:
            try:
                results[model_id] = await task
            except Exception as e:
                workflow.logger.error(f"Replay failed for {model_id}: {e}")
                results[model_id] = []
        
        return results
    
    async def _evaluate_all_results(
        self,
        golden_prompts: List[GoldenPrompt],
        shadow_results: Dict[str, List[ShadowResult]],
        config: WorkflowConfig
    ) -> Dict[str, List[EvaluationResult]]:
        """
        Evaluate all shadow results with LLM-as-Judge.
        
        Batches evaluations to avoid overwhelming the judge model.
        """
        evaluations = {}
        
        for model_id, results in shadow_results.items():
            if not results:
                evaluations[model_id] = []
                continue
            
            model_evals = await workflow.execute_activity(
                "evaluate_shadow_results",
                args=[golden_prompts, results, config.evaluator],
                start_to_close_timeout=timedelta(minutes=45),
                retry_policy=DEFAULT_RETRY_POLICY
            )
            evaluations[model_id] = model_evals
        
        return evaluations
    
    async def _generate_report(
        self,
        traces: List[ProductionTrace],
        golden_prompts: List[GoldenPrompt],
        evaluations: Dict[str, List[EvaluationResult]],
        config: WorkflowConfig
    ) -> OptimizationReport:
        """Generate optimization report via activity."""
        return await workflow.execute_activity(
            "generate_optimization_report",
            args=[traces, golden_prompts, evaluations, config],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=DEFAULT_RETRY_POLICY
        )
    
    async def _submit_feedback(self, report: OptimizationReport) -> None:
        """Submit evaluation feedback to Portkey."""
        await workflow.execute_activity(
            "submit_portkey_feedback",
            args=[report],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=DEFAULT_RETRY_POLICY
        )
    
    async def _create_empty_report(
        self,
        config: WorkflowConfig
    ) -> OptimizationReport:
        """Create empty report when no traces found."""
        from shadow_optic.models import (
            OptimizationReport,
            Recommendation,
            RecommendationAction,
        )
        
        return OptimizationReport(
            time_window=config.log_export_window,
            traces_analyzed=0,
            samples_evaluated=0,
            challenger_results={},
            recommendation=Recommendation(
                action=RecommendationAction.INSUFFICIENT_DATA,
                confidence=0.0,
                reasoning="No production traces found in the export window."
            )
        )
    
    def _update_progress(
        self,
        progress: float,
        stage: str,
        message: str
    ) -> None:
        """Update workflow progress for observability."""
        self._progress = progress
        self._stage = stage
        self._message = message
        workflow.logger.info(f"[{progress:.0%}] {stage}: {message}")
    
    @workflow.query
    def get_progress(self) -> Dict[str, Any]:
        """Query current workflow progress."""
        return {
            "progress": self._progress,
            "stage": self._stage,
            "message": self._message
        }
    
    @workflow.signal
    async def cancel_workflow(self) -> None:
        """Signal to gracefully cancel the workflow."""
        workflow.logger.info("Received cancel signal")
        raise workflow.CancelledError("Cancelled by user signal")


@workflow.defn
class ScheduledOptimizationWorkflow:
    """
    Scheduled wrapper workflow for daily optimization runs.
    
    This workflow:
    1. Runs on a cron schedule (default: 2 AM daily)
    2. Implements circuit breaker pattern
    3. Aggregates results over time for trend detection
    
    Example schedule setup:
        ```python
        await client.start_workflow(
            ScheduledOptimizationWorkflow.run,
            config,
            id="scheduled-optimization",
            task_queue="shadow-optic",
            cron_schedule="0 2 * * *"  # 2 AM daily
        )
        ```
    """
    
    def __init__(self):
        self._consecutive_failures = 0
        self._max_failures = 3
        self._backoff_minutes = 60
    
    @workflow.run
    async def run(self, config: WorkflowConfig) -> OptimizationReport:
        """Execute scheduled optimization with circuit breaker."""
        
        # Circuit breaker check
        if self._consecutive_failures >= self._max_failures:
            workflow.logger.warning(
                f"Circuit breaker open: {self._consecutive_failures} consecutive failures"
            )
            # Wait before retrying
            await asyncio.sleep(self._backoff_minutes * 60)
            self._consecutive_failures = 0
        
        try:
            # Execute main workflow as child
            result = await workflow.execute_child_workflow(
                OptimizeModelSelectionWorkflow.run,
                config,
                id=f"optimize-{workflow.now().strftime('%Y%m%d-%H%M%S')}",
                task_queue="shadow-optic"
            )
            
            # Reset failure counter on success
            self._consecutive_failures = 0
            
            return result
            
        except Exception as e:
            self._consecutive_failures += 1
            workflow.logger.error(
                f"Scheduled run failed ({self._consecutive_failures}/{self._max_failures}): {e}"
            )
            raise


@workflow.defn
class AdaptiveOptimizationWorkflow:
    """
    Enhanced workflow with Thompson Sampling model selection.
    
    Instead of testing all challengers equally, this workflow:
    1. Uses Thompson Sampling to select promising models
    2. Updates beliefs based on evaluation results
    3. Converges to optimal model selection over time
    
    This reduces shadow testing costs by ~80% while maintaining
    exploration of new models.
    """
    
    def __init__(self):
        # We'll initialize the selector in the run method or manage state manually
        # since Temporal workflows must be deterministic and pickling complex objects
        # can be tricky. However, ThompsonSamplingModelSelector is simple enough.
        # Ideally, we persist the alpha/beta params in workflow state.
        self._alpha: Dict[str, float] = {}
        self._beta: Dict[str, float] = {}

    @workflow.run
    async def run(
        self,
        config: WorkflowConfig,
        n_iterations: int = 10
    ) -> OptimizationReport:
        """
        Run adaptive optimization with bandit-based model selection.
        """
        from shadow_optic.decision_engine import ThompsonSamplingModelSelector

        # Initialize state if empty
        challenger_ids = [c.model_id for c in config.challengers]
        for mid in challenger_ids:
            if mid not in self._alpha:
                self._alpha[mid] = 1.0
                self._beta[mid] = 1.0
        
        # Instantiate selector with current workflow state
        selector = ThompsonSamplingModelSelector(challenger_ids)
        selector.alpha = self._alpha
        selector.beta = self._beta

        # Export logs once
        traces = await workflow.execute_activity(
            "export_portkey_logs",
            args=[config.log_export_window],
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=DEFAULT_RETRY_POLICY
        )
        
        # Sample once
        golden_prompts = await workflow.execute_activity(
            "sample_golden_prompts",
            args=[traces, config.sampler],
            start_to_close_timeout=timedelta(minutes=15),
            retry_policy=DEFAULT_RETRY_POLICY
        )
        
        all_evaluations: Dict[str, List[EvaluationResult]] = {
            c.model_id: [] for c in config.challengers
        }
        
        # Iterative optimization
        for i in range(n_iterations):
            # Select model
            selected_model = selector.select_model()
            workflow.logger.info(f"Iteration {i+1}: Selected {selected_model}")
            
            # Find challenger config
            challenger_config = next(
                c for c in config.challengers if c.model_id == selected_model
            )
            
            # Replay
            shadow_results = await workflow.execute_activity(
                "replay_shadow_requests",
                args=[golden_prompts, challenger_config],
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=DEFAULT_RETRY_POLICY
            )
            
            # Evaluate
            evaluations = await workflow.execute_activity(
                "evaluate_shadow_results",
                args=[golden_prompts, shadow_results, config.evaluator],
                start_to_close_timeout=timedelta(minutes=45),
                retry_policy=DEFAULT_RETRY_POLICY
            )
            
            # Update beliefs
            avg_score = sum(e.composite_score for e in evaluations) / len(evaluations) if evaluations else 0
            # Use 0.8 as pass threshold for binary feedback simulation
            passed = avg_score >= 0.8
            selector.update(selected_model, passed, avg_score)
            
            # Sync state back to workflow persistence
            self._alpha = selector.alpha
            self._beta = selector.beta
            
            # Accumulate evaluations
            all_evaluations[selected_model].extend(evaluations)
        
        # Generate final report
        return await workflow.execute_activity(
            "generate_optimization_report",
            args=[traces, golden_prompts, all_evaluations, config],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=DEFAULT_RETRY_POLICY
        )
    
    @workflow.query
    def get_bandit_state(self) -> Dict[str, Dict[str, float]]:
        """Query current bandit state."""
        return {
            model: {
                "alpha": self._alpha[model],
                "beta": self._beta[model],
                "expected_value": self._alpha[model] / (self._alpha[model] + self._beta[model])
            }
            for model in self._alpha
        }


@workflow.defn
class AgenticOptimizationWorkflow:
    """
    Self-optimizing workflow with Agentic Metadata and DSPy integration.
    
    This workflow implements the "Watchdog Loop" pattern:
    1. Scans Qdrant clusters for drift/quality triggers
    2. Invokes DSPy optimization for failing clusters
    3. Updates cluster state machine (EMBRYONIC → STABLE → DRIFTING → UNSTABLE)
    4. Records evolution history for audit
    
    Key Features:
    - Autonomous prompt governance via AgenticPolicy
    - DSPy MIPROv2 prompt optimization
    - Tiered response strategy (cheap → DSPy-enhanced → fallback)
    - Phoenix integration for drift detection
    
    Example:
        ```python
        await client.start_workflow(
            AgenticOptimizationWorkflow.run,
            config,
            id="agentic-optimization",
            task_queue="shadow-optic",
            cron_schedule="*/30 * * * *"  # Every 30 minutes
        )
        ```
    """
    
    def __init__(self):
        self._clusters_scanned: int = 0
        self._optimizations_triggered: int = 0
        self._fallbacks_applied: int = 0
        self._current_stage: str = "idle"
    
    @workflow.run
    async def run(self, config: WorkflowConfig) -> Dict[str, Any]:
        """
        Execute the agentic optimization workflow.
        
        This workflow:
        1. Scans all clusters for trigger conditions
        2. Processes drifting/unstable clusters
        3. Triggers DSPy optimization or fallback as needed
        4. Updates cluster state and records evolution
        
        Args:
            config: Workflow configuration
            
        Returns:
            Summary of actions taken
        """
        workflow.logger.info("Starting agentic optimization workflow")
        self._current_stage = "scanning"
        
        try:
            # Stage 1: Scan clusters for triggers
            scan_results = await workflow.execute_activity(
                "scan_clusters_for_triggers",
                args=[config],
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=DEFAULT_RETRY_POLICY
            )
            
            self._clusters_scanned = scan_results.get("clusters_scanned", 0)
            drifting_clusters = scan_results.get("drifting_clusters", [])
            unstable_clusters = scan_results.get("unstable_clusters", [])
            
            workflow.logger.info(
                f"Scan complete: {len(drifting_clusters)} drifting, "
                f"{len(unstable_clusters)} unstable"
            )
            
            # Stage 2: Process unstable clusters (highest priority)
            self._current_stage = "processing_unstable"
            for cluster_info in unstable_clusters:
                await self._process_unstable_cluster(cluster_info, config)
            
            # Stage 3: Process drifting clusters
            self._current_stage = "processing_drifting"
            for cluster_info in drifting_clusters:
                await self._process_drifting_cluster(cluster_info, config)
            
            # Stage 4: Generate summary
            self._current_stage = "complete"
            
            return {
                "clusters_scanned": self._clusters_scanned,
                "optimizations_triggered": self._optimizations_triggered,
                "fallbacks_applied": self._fallbacks_applied,
                "drifting_processed": len(drifting_clusters),
                "unstable_processed": len(unstable_clusters),
            }
            
        except Exception as e:
            workflow.logger.error(f"Agentic workflow failed: {e}")
            self._current_stage = "failed"
            raise
    
    async def _process_unstable_cluster(
        self,
        cluster_info: Dict[str, Any],
        config: WorkflowConfig
    ) -> None:
        """Process an unstable cluster with immediate action."""
        cluster_id = cluster_info.get("cluster_id")
        policy = cluster_info.get("policy", {})
        
        workflow.logger.info(f"Processing unstable cluster: {cluster_id}")
        
        # Check if DSPy optimization is enabled for this cluster
        if policy.get("dspy_enabled", False):
            # Attempt DSPy optimization
            try:
                result = await workflow.execute_activity(
                    "optimize_cluster_with_dspy",
                    args=[cluster_id, cluster_info],
                    start_to_close_timeout=timedelta(minutes=20),
                    retry_policy=DEFAULT_RETRY_POLICY
                )
                
                if result.get("success"):
                    self._optimizations_triggered += 1
                    workflow.logger.info(
                        f"DSPy optimization succeeded for {cluster_id}: "
                        f"+{result.get('improvement', 0):.1%} improvement"
                    )
                    return
            except Exception as e:
                workflow.logger.warning(f"DSPy optimization failed for {cluster_id}: {e}")
        
        # Apply fallback if DSPy didn't work or isn't enabled
        fallback_model = policy.get("fallback_model")
        if fallback_model:
            await workflow.execute_activity(
                "apply_cluster_fallback",
                args=[cluster_id, fallback_model],
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=DEFAULT_RETRY_POLICY
            )
            self._fallbacks_applied += 1
            workflow.logger.info(f"Applied fallback {fallback_model} for {cluster_id}")
    
    async def _process_drifting_cluster(
        self,
        cluster_info: Dict[str, Any],
        config: WorkflowConfig
    ) -> None:
        """Process a drifting cluster with proactive optimization."""
        cluster_id = cluster_info.get("cluster_id")
        policy = cluster_info.get("policy", {})
        variance = policy.get("variance", 0)
        
        workflow.logger.info(f"Processing drifting cluster: {cluster_id} (variance: {variance:.3f})")
        
        # For drifting clusters, try DSPy optimization if auto_deploy is enabled
        if policy.get("dspy_enabled") and policy.get("auto_deploy"):
            try:
                result = await workflow.execute_activity(
                    "optimize_cluster_with_dspy",
                    args=[cluster_id, cluster_info],
                    start_to_close_timeout=timedelta(minutes=20),
                    retry_policy=DEFAULT_RETRY_POLICY
                )
                
                if result.get("success"):
                    self._optimizations_triggered += 1
                    
                    # Update cluster state to STABLE if optimization succeeded
                    await workflow.execute_activity(
                        "update_cluster_state",
                        args=[cluster_id, "STABLE", {"optimized": True}],
                        start_to_close_timeout=timedelta(minutes=2),
                        retry_policy=DEFAULT_RETRY_POLICY
                    )
            except Exception as e:
                workflow.logger.warning(f"Proactive optimization failed for {cluster_id}: {e}")
        else:
            # Just record that we observed drift but didn't act
            workflow.logger.info(f"Drift observed for {cluster_id} - auto_deploy disabled")
    
    @workflow.query
    def get_status(self) -> Dict[str, Any]:
        """Query current workflow status."""
        return {
            "stage": self._current_stage,
            "clusters_scanned": self._clusters_scanned,
            "optimizations_triggered": self._optimizations_triggered,
            "fallbacks_applied": self._fallbacks_applied,
        }
