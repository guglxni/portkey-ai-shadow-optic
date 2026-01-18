"""
Shadow-Optic: The Autonomous Cost-Quality Optimization Engine

A shadow-mode optimization system that replays production LLM traffic
against cheaper challenger models, evaluates quality with LLM-as-judge,
and provides data-driven recommendations for model switching.

Built for the Portkey AI Builders Challenge - Track 4: Cost-Quality Optimization
via Historical Replay.

Architecture Overview:
    
    Production Traffic → Portkey Gateway → Logs Export
                                              ↓
                            Temporal Workflow Scheduler (24h)
                                              ↓
                            Qdrant Semantic Sampler (5% golden set)
                                              ↓
                            Shadow Replay Engine (DeepSeek-V3, Llama-4)
                                              ↓
                            DeepEval LLM-as-Judge
                                              ↓
                            Portkey Feedback API → Dashboard

Components:
    - sampler.py: Semantic stratified sampling using Qdrant
    - evaluator.py: DeepEval-based quality evaluation
    - workflows.py: Temporal workflow definitions
    - activities.py: Temporal activity implementations
    - decision_engine.py: Recommendation logic
    - model_registry.py: 250+ model catalog with intelligent selection
    - exporters.py: Portkey feedback and metrics export
"""

__version__ = "0.1.0"
__author__ = "Shadow-Optic Team"
__license__ = "MIT"

from shadow_optic.models import (
    ProductionTrace,
    GoldenPrompt,
    ShadowResult,
    EvaluationResult,
    ChallengerResult,
    CostAnalysis,
    Recommendation,
    OptimizationReport,
    WorkflowConfig,
    SamplerConfig,
    EvaluatorConfig,
    ChallengerConfig,
    RecommendationAction,
)

# Import model registry for intelligent challenger selection
from shadow_optic.model_registry import (
    MODEL_REGISTRY,
    ModelSpec,
    QualityTier,
    Capability,
    Provider,
    get_recommended_challengers,
    suggest_challengers_for_production,
    find_pareto_optimal_models,
    IntelligentChallengerSelector,
)

# Import lightweight core components eagerly
from shadow_optic.observability import PhoenixObserver
from shadow_optic.analytics import QualityTrendDetector, QualityAlert
from shadow_optic.templates import PromptTemplateExtractor
from shadow_optic.decision_engine import DecisionEngine, ThompsonSamplingModelSelector

# Lazy imports for heavy components (DeepEval, spotify-confidence, etc.)
# These are loaded on first access to avoid slowing down CLI tools and scripts
_lazy_imports = {}

def __getattr__(name: str):
    """Lazy import handler for heavy modules."""
    # Evaluator components (triggers DeepEval import)
    evaluator_names = {"ShadowEvaluator", "RefusalDetector", "SemanticDiff"}
    if name in evaluator_names:
        if "evaluator" not in _lazy_imports:
            from shadow_optic import evaluator as _evaluator
            _lazy_imports["evaluator"] = _evaluator
        return getattr(_lazy_imports["evaluator"], name)
    
    # Sampler component (triggers qdrant-client, numpy)
    if name == "SemanticSampler":
        if "sampler" not in _lazy_imports:
            from shadow_optic import sampler as _sampler
            _lazy_imports["sampler"] = _sampler
        return _lazy_imports["sampler"].SemanticSampler
    
    # P2 Enhancement components (triggers spotify-confidence)
    p2_names = {
        "CostPredictionModel", "BudgetOptimizedSampler", "CostFeatures", "CostPrediction",
        "PortkeyGuardrailTuner", "PortkeyRefusalDetector", "RefusalType", "RefusalPattern",
        "AutomatedRolloutController", "ExperimentManager", "ExperimentStage", 
        "ExperimentMetrics", "PromotionDecision",
    }
    if name in p2_names:
        if "p2" not in _lazy_imports:
            from shadow_optic import p2_enhancements as _p2
            _lazy_imports["p2"] = _p2
        return getattr(_lazy_imports["p2"], name)
    
    # Agentic Policy components (lightweight Pydantic models)
    agentic_policy_names = {
        "AgenticPolicy", "ClusterState", "TriggerType", "ActionType",
        "OptimizationTarget", "RoutingRule", "EvolutionRecord",
        "create_default_policy", "policy_to_qdrant_payload", "policy_from_qdrant_payload",
    }
    if name in agentic_policy_names:
        if "agentic_policy" not in _lazy_imports:
            from shadow_optic import agentic_policy as _agentic
            _lazy_imports["agentic_policy"] = _agentic
        return getattr(_lazy_imports["agentic_policy"], name)
    
    # Watchdog components (requires agentic_policy)
    watchdog_names = {
        "AgenticWatchdog", "WatchdogConfig", "WatchdogEvent",
        "WatchdogEventHandler", "AlertingEventHandler", "create_watchdog",
    }
    if name in watchdog_names:
        if "watchdog" not in _lazy_imports:
            from shadow_optic import watchdog as _watchdog
            _lazy_imports["watchdog"] = _watchdog
        return getattr(_lazy_imports["watchdog"], name)
    
    # DSPy Optimizer components (requires dspy-ai)
    dspy_names = {
        "DSPyOptimizer", "DSPyOptimizerConfig", "OptimizationStrategy",
        "SignatureBuilder", "DSPySignature", "DSPyField",
        "OptimizationResult", "PromptFixer", "create_optimizer",
    }
    if name in dspy_names:
        if "dspy_optimizer" not in _lazy_imports:
            from shadow_optic import dspy_optimizer as _dspy
            _lazy_imports["dspy_optimizer"] = _dspy
        return getattr(_lazy_imports["dspy_optimizer"], name)
    
    raise AttributeError(f"module 'shadow_optic' has no attribute '{name}'")

__all__ = [
    # Models
    "ProductionTrace",
    "GoldenPrompt",
    "ShadowResult",
    "EvaluationResult",
    "ChallengerResult",
    "CostAnalysis",
    "Recommendation",
    "OptimizationReport",
    "WorkflowConfig",
    "SamplerConfig",
    "EvaluatorConfig",
    "ChallengerConfig",
    "RecommendationAction",
    # Model Registry
    "MODEL_REGISTRY",
    "ModelSpec",
    "QualityTier",
    "Capability", 
    "Provider",
    "get_recommended_challengers",
    "suggest_challengers_for_production",
    "find_pareto_optimal_models",
    "IntelligentChallengerSelector",
    # Core Components (Microarchitecture)
    "PhoenixObserver",
    "QualityTrendDetector",
    "QualityAlert",
    "PromptTemplateExtractor",
    "SemanticSampler",
    "ShadowEvaluator",
    "RefusalDetector",
    "SemanticDiff",
    "DecisionEngine",
    "ThompsonSamplingModelSelector",
    # P2 Enhancements - Cost Prediction
    "CostPredictionModel",
    "BudgetOptimizedSampler",
    "CostFeatures",
    "CostPrediction",
    # P2 Enhancements - Guardrail Tuning
    "PortkeyGuardrailTuner",
    "PortkeyRefusalDetector",
    "RefusalType",
    "RefusalPattern",
    # P2 Enhancements - A/B Graduation
    "AutomatedRolloutController",
    "ExperimentManager",
    "ExperimentStage",
    "ExperimentMetrics",
    "PromotionDecision",
    # Agentic Policy (DSPy & Agentic Metadata)
    "AgenticPolicy",
    "ClusterState",
    "TriggerType",
    "ActionType",
    "OptimizationTarget",
    "RoutingRule",
    "EvolutionRecord",
    "create_default_policy",
    "policy_to_qdrant_payload",
    "policy_from_qdrant_payload",
    # Watchdog (Autonomous Monitoring)
    "AgenticWatchdog",
    "WatchdogConfig",
    "WatchdogEvent",
    "WatchdogEventHandler",
    "AlertingEventHandler",
    "create_watchdog",
    # DSPy Optimizer (MIPROv2)
    "DSPyOptimizer",
    "DSPyOptimizerConfig",
    "OptimizationStrategy",
    "SignatureBuilder",
    "DSPySignature",
    "DSPyField",
    "OptimizationResult",
    "PromptFixer",
    "create_optimizer",
]
