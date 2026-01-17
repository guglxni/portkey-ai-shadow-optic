"""
Data models for Shadow-Optic.

Defines all Pydantic models for type-safe data handling throughout
the system. These models are used for:
- API request/response validation
- Temporal workflow payloads
- Database storage
- Cross-component communication
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Enums
# =============================================================================

class TraceStatus(str, Enum):
    """Status of a production trace."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class RecommendationAction(str, Enum):
    """Possible actions from the decision engine."""
    SWITCH_RECOMMENDED = "switch_recommended"
    CONTINUE_MONITORING = "continue_monitoring"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_VIABLE_CHALLENGER = "no_viable_challenger"


class SampleType(str, Enum):
    """Type of sampled prompt."""
    CENTROID = "centroid"       # Cluster center representative
    OUTLIER = "outlier"         # Cluster boundary case
    RANDOM = "random"           # Random sample for diversity


class RolloutStage(str, Enum):
    """Stages of automated rollout."""
    SHADOW = "shadow"           # 0% live traffic
    CANARY = "canary"           # 1% live traffic
    GRADUATED = "graduated"     # 10% live traffic
    RAMPING = "ramping"         # 10-50% live traffic
    MAJORITY = "majority"       # 50-90% live traffic
    FULL = "full"               # 100% live traffic


# =============================================================================
# Core Data Models
# =============================================================================

class ProductionTrace(BaseModel):
    """
    A single production LLM request/response trace exported from Portkey.
    
    Attributes:
        trace_id: Unique identifier from Portkey
        request_timestamp: When the original request was made
        prompt: The user's prompt/input
        response: The production model's response
        model: Production model used (e.g., "gpt-5.2-turbo")
        latency_ms: Response latency in milliseconds
        tokens_prompt: Input token count
        tokens_completion: Output token count
        cost: Actual cost of the request
        metadata: Additional metadata from Portkey
    """
    trace_id: str = Field(..., description="Unique trace identifier from Portkey")
    request_timestamp: datetime = Field(..., description="When the request was made")
    prompt: str = Field(..., description="User prompt/input")
    response: str = Field(..., description="Production model response")
    model: str = Field(..., description="Production model identifier")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    tokens_prompt: int = Field(..., description="Input token count")
    tokens_completion: int = Field(..., description="Output token count")
    cost: float = Field(..., description="Request cost in USD")
    status: TraceStatus = Field(default=TraceStatus.SUCCESS)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )


class GoldenPrompt(BaseModel):
    """
    A selected prompt for shadow testing.
    
    Golden prompts are selected via semantic stratified sampling
    to represent the diversity of production traffic efficiently.
    
    Attributes:
        original_trace_id: Links back to the production trace
        prompt: The prompt text
        production_response: What production model returned (ground truth)
        cluster_id: Which semantic cluster this belongs to
        sample_type: How this was selected (centroid, outlier, random)
        embedding: Vector embedding of the prompt
        template: Extracted prompt template (if applicable)
        variable_values: Extracted variable values from template
    """
    original_trace_id: str
    prompt: str
    production_response: str
    cluster_id: Optional[int] = None
    sample_type: SampleType = SampleType.CENTROID
    embedding: Optional[List[float]] = None
    template: Optional[str] = None
    variable_values: Optional[List[str]] = None
    
    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimensions(cls, v):
        if v is not None and len(v) != 1536:
            raise ValueError(f"Expected 1536-dim embedding, got {len(v)}")
        return v


class ShadowResult(BaseModel):
    """
    Result of replaying a golden prompt against a challenger model.
    
    Attributes:
        golden_prompt_id: Reference to the golden prompt
        challenger_model: Which challenger model was used
        shadow_response: The challenger's response
        latency_ms: Challenger response latency
        tokens_prompt: Challenger input tokens
        tokens_completion: Challenger output tokens
        cost: Challenger request cost
        success: Whether the request completed successfully
        error: Error message if failed
        timestamp: When the shadow request was made
    """
    golden_prompt_id: str
    challenger_model: str
    shadow_response: str
    latency_ms: float
    tokens_prompt: int
    tokens_completion: int
    cost: float
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvaluationResult(BaseModel):
    """
    Evaluation scores from DeepEval for a single shadow replay.
    
    Uses LLM-as-Judge (GPT-5.2) to evaluate challenger response quality
    against the production response (ground truth).
    
    Attributes:
        shadow_result_id: Reference to the shadow result
        faithfulness: How factually consistent with production (0-1)
        quality: Overall response quality via G-Eval (0-1)
        conciseness: Efficiency of response (0-1)
        is_refusal: Whether the challenger refused to answer
        composite_score: Weighted combination of all metrics
        passed_thresholds: Whether all thresholds are met
        judge_model: Which model performed the evaluation
        evaluation_cost: Cost of running the evaluation
        raw_scores: Unprocessed scores from DeepEval
        reasoning: Judge's reasoning (for explainability)
    """
    shadow_result_id: str
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    quality: float = Field(..., ge=0.0, le=1.0)
    conciseness: float = Field(..., ge=0.0, le=1.0)
    is_refusal: bool = False
    composite_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    passed_thresholds: bool = True
    judge_model: str = "gpt-5.2-turbo"
    evaluation_cost: float = 0.0
    raw_scores: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None
    
    @model_validator(mode='before')
    @classmethod
    def calculate_composite(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("composite_score") is None:
            # Weighted average: Faithfulness > Quality > Conciseness
            weights = {"faithfulness": 0.5, "quality": 0.35, "conciseness": 0.15}
            data["composite_score"] = (
                data.get("faithfulness", 0) * weights["faithfulness"] +
                data.get("quality", 0) * weights["quality"] +
                data.get("conciseness", 0) * weights["conciseness"]
            )
        return data


class ChallengerResult(BaseModel):
    """
    Aggregated results for a single challenger model.
    
    Attributes:
        model: Challenger model identifier
        samples_evaluated: Number of golden prompts tested
        avg_faithfulness: Mean faithfulness score
        avg_quality: Mean quality score
        avg_conciseness: Mean conciseness score
        refusal_rate: Percentage of refusals
        avg_latency_ms: Mean response latency
        avg_cost_per_request: Mean cost per request
        success_rate: Percentage of successful requests
        passed_thresholds: Whether aggregates meet all thresholds
        failure_reasons: Why thresholds weren't met
        individual_results: Per-prompt evaluation details
    """
    model: str
    samples_evaluated: int
    avg_faithfulness: float
    avg_quality: float
    avg_conciseness: float
    refusal_rate: float
    avg_latency_ms: float
    avg_cost_per_request: float
    success_rate: float
    passed_thresholds: bool
    failure_reasons: List[str] = Field(default_factory=list)
    individual_results: List[EvaluationResult] = Field(default_factory=list)
    
    @property
    def composite_score(self) -> float:
        """Calculate weighted composite score."""
        return (
            self.avg_faithfulness * 0.5 +
            self.avg_quality * 0.35 +
            self.avg_conciseness * 0.15
        )


class CostAnalysis(BaseModel):
    """
    Cost comparison between production and challenger.
    
    Attributes:
        production_cost_per_1k: Production cost per 1000 requests
        challenger_cost_per_1k: Challenger cost per 1000 requests
        savings_percent: Percentage cost reduction
        estimated_monthly_savings: Projected monthly savings in USD
        breakeven_quality_threshold: Quality score where costs equal
    """
    production_cost_per_1k: float
    challenger_cost_per_1k: float
    savings_percent: Optional[float] = None
    estimated_monthly_savings: float
    breakeven_quality_threshold: float
    
    @model_validator(mode='before')
    @classmethod
    def calculate_savings(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("savings_percent") is None:
            prod = data.get("production_cost_per_1k", 0)
            chal = data.get("challenger_cost_per_1k", 0)
            if prod == 0:
                data["savings_percent"] = 0.0
            else:
                data["savings_percent"] = (prod - chal) / prod
        return data


class Recommendation(BaseModel):
    """
    Final recommendation from the decision engine.
    
    Attributes:
        action: What to do (switch, monitor, etc.)
        challenger: Recommended challenger model (if switching)
        confidence: Confidence in the recommendation (0-1)
        reasoning: Human-readable explanation
        cost_analysis: Cost comparison details
        metrics_summary: Key metrics at a glance
        suggested_actions: Next steps to take
        rollout_config: Recommended rollout configuration
    """
    action: RecommendationAction
    challenger: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    cost_analysis: Optional[CostAnalysis] = None
    metrics_summary: Dict[str, float] = Field(default_factory=dict)
    suggested_actions: List[str] = Field(default_factory=list)
    rollout_config: Optional[Dict[str, Any]] = None


class OptimizationReport(BaseModel):
    """
    Complete optimization report from a workflow run.
    
    Attributes:
        report_id: Unique report identifier
        generated_at: When the report was generated
        time_window: Time period covered by this analysis
        traces_analyzed: Total production traces considered
        samples_evaluated: Golden prompts tested
        challenger_results: Results per challenger model
        best_challenger: Top-performing challenger (if any)
        recommendation: Final recommendation
        execution_stats: Workflow execution statistics
    """
    report_id: str = Field(default_factory=lambda: f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    time_window: str = Field(..., description="e.g., 'last_24h'")
    traces_analyzed: int
    samples_evaluated: int
    challenger_results: Dict[str, ChallengerResult]
    best_challenger: Optional[str] = None
    recommendation: Recommendation
    execution_stats: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )


# =============================================================================
# Configuration Models
# =============================================================================

class SamplerConfig(BaseModel):
    """Configuration for semantic sampling."""
    n_clusters: int = Field(default=50, ge=5, le=500)
    samples_per_cluster: int = Field(default=3, ge=1, le=10)
    min_cluster_size: int = Field(default=3, ge=1)
    embedding_model: str = "text-embedding-3-small"
    qdrant_collection: str = "shadow_optic_prompts"


class EvaluatorConfig(BaseModel):
    """
    Configuration for quality evaluation using Portkey as the LLM backend.
    
    DeepEval routes through Portkey by setting OPENAI_BASE_URL to Portkey's
    gateway. The model specified here should use Model Catalog format:
    @provider_slug/model_name (e.g., @openai/gpt-4o-mini)
    """
    # Model configuration - uses Model Catalog format
    # Claude Opus 4.5 is the recommended judge model for highest quality evaluation
    # Alternative: @open-ai/o3-pro for advanced reasoning tasks
    judge_model: str = Field(
        default="@bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0", 
        description="Model name for LLM-as-Judge using Model Catalog format (@provider/model)"
    )
    # Portkey configuration
    portkey_api_key: Optional[str] = Field(
        default=None, 
        description="Portkey API key (falls back to PORTKEY_API_KEY env var)"
    )
    portkey_base_url: str = Field(
        default="https://api.portkey.ai/v1",
        description="Portkey gateway URL (OpenAI-compatible endpoint)"
    )
    portkey_provider_slug: Optional[str] = Field(
        default=None,
        description="Deprecated - use Model Catalog format in judge_model instead"
    )
    # Thresholds
    faithfulness_threshold: float = Field(default=0.8, ge=0, le=1)
    quality_threshold: float = Field(default=0.7, ge=0, le=1)
    conciseness_threshold: float = Field(default=0.5, ge=0, le=1)
    refusal_rate_threshold: float = Field(default=0.01, ge=0, le=1)
    # Weights for composite score calculation
    faithfulness_weight: float = Field(default=0.5, ge=0, le=1)
    quality_weight: float = Field(default=0.35, ge=0, le=1)
    conciseness_weight: float = Field(default=0.15, ge=0, le=1)


class ChallengerConfig(BaseModel):
    """Configuration for a challenger model (Model Catalog + Workspace architecture).
    
    Uses Portkey's Model Catalog for granular access control and cost tracking.
    Virtual Model Names are defined in the Portkey UI and route to actual models.
    
    Example:
        - virtual_model_name: 'shadow-challenger-1' (defined in Portkey UI)
        - Routes to: 'groq/llama-3-70b' or 'deepseek/deepseek-chat'
        - Tracked under: Shadow Workspace budget
    """
    model_id: str = Field(..., description="Model name (e.g., 'deepseek-chat', 'gpt-5.2')")
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'anthropic', 'deepseek')")
    provider_slug: str = Field(..., description="Portkey AI Provider slug from Model Catalog (e.g., 'openai', 'deepseek')")
    portkey_config_id: Optional[str] = Field(None, description="Portkey Config ID for routing")
    
    # Model Catalog fields (new architecture)
    virtual_model_name: Optional[str] = Field(
        None, 
        description="Virtual Model name defined in Portkey UI (e.g., 'shadow-challenger-1'). Takes precedence over model_id."
    )
    workspace_id: Optional[str] = Field(
        None, 
        description="Portkey Workspace ID for cost segregation (e.g., 'shadow-ops-workspace')"
    )
    
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout_seconds: int = 30
    
    @property
    def model_catalog_id(self) -> str:
        """Get the Model Catalog format: @provider_slug/model_id"""
        return f"@{self.provider_slug}/{self.model_id}"
    
    @property
    def effective_model(self) -> str:
        """Get the effective model name for API calls.
        
        Uses virtual_model_name if set (Enterprise/Model Catalog),
        otherwise falls back to model_catalog_id.
        """
        return self.virtual_model_name or self.model_catalog_id
    
    
class WorkflowConfig(BaseModel):
    """Configuration for the optimization workflow.
    
    Supports Model Catalog with Workspace-based cost segregation.
    """
    schedule_cron: str = "0 2 * * *"  # 2 AM daily
    log_export_window: str = "24h"
    max_retries: int = 3
    retry_initial_interval_seconds: int = 10
    max_budget_per_run: float = 50.0
    monthly_request_volume: int = 100_000  # For cost projections
    
    # Workspace IDs for cost segregation
    production_workspace_id: Optional[str] = Field(
        None, description="Portkey Workspace ID for production traffic"
    )
    shadow_workspace_id: Optional[str] = Field(
        None, description="Portkey Workspace ID for shadow testing"
    )
    
    challengers: List[ChallengerConfig]
    sampler: SamplerConfig = Field(default_factory=SamplerConfig)
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)


# =============================================================================
# API Models
# =============================================================================

class TriggerOptimizationRequest(BaseModel):
    """Request to manually trigger an optimization run."""
    time_window: str = "24h"
    challenger_models: Optional[List[str]] = None
    max_samples: Optional[int] = None


class TriggerOptimizationResponse(BaseModel):
    """Response from triggering an optimization run."""
    workflow_id: str
    status: str
    estimated_completion: datetime


class GetReportRequest(BaseModel):
    """Request to fetch an optimization report."""
    report_id: str


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    temporal_connected: bool
    qdrant_connected: bool
    portkey_connected: bool


# =============================================================================
# Event Models (for Temporal signals/queries)
# =============================================================================

class WorkflowProgressEvent(BaseModel):
    """Progress update from a running workflow."""
    workflow_id: str
    stage: str
    progress_percent: float
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QualityAlertEvent(BaseModel):
    """Alert when quality degradation is detected."""
    alert_id: str
    model: str
    metric: str
    current_value: float
    expected_value: float
    deviation: float
    severity: str  # "info", "warning", "critical"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
