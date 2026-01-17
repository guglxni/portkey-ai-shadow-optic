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

# Import core components for microarchitecture integration
from shadow_optic.observability import PhoenixObserver
from shadow_optic.analytics import QualityTrendDetector, QualityAlert
from shadow_optic.templates import PromptTemplateExtractor
from shadow_optic.sampler import SemanticSampler
from shadow_optic.evaluator import ShadowEvaluator, RefusalDetector, SemanticDiff
from shadow_optic.decision_engine import DecisionEngine, ThompsonSamplingModelSelector

# P2 Enhancements - Portkey-First Architecture
from shadow_optic.p2_enhancements import (
    # Cost Prediction
    CostPredictionModel,
    BudgetOptimizedSampler,
    CostFeatures,
    CostPrediction,
    # Guardrail Tuning
    PortkeyGuardrailTuner,
    PortkeyRefusalDetector,
    RefusalType,
    RefusalPattern,
    # A/B Graduation
    AutomatedRolloutController,
    ExperimentManager,
    ExperimentStage,
    ExperimentMetrics,
    PromotionDecision,
)

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
]
