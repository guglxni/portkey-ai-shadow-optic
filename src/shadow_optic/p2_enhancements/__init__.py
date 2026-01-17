"""
Shadow-Optic P2 Enhancements - Portkey-First Architecture.

This module provides the Phase 2 enhancements for Shadow-Optic:

1. **Cost Prediction Model** (Portkey Analytics + tiktoken + sklearn)
   - ML-based cost prediction before replay
   - Budget-optimized sampling
   - Training from Portkey historical data

2. **Automatic Guardrail Tuning** (Portkey Guardrails + DSPy)
   - Refusal pattern detection
   - Prompt optimization for false positives
   - Safety validation

3. **Auto A/B Graduation** (Portkey Canary Testing + spotify-confidence)
   - Automated staged rollout
   - Statistical significance testing
   - Quality gates and guardrails

All components follow the Portkey-First design philosophy:
- Maximize Portkey's native features
- Only use FOSS where Portkey doesn't provide coverage
- Deep integration with Portkey ecosystem

Example Usage:
    
    from shadow_optic.p2_enhancements import (
        # Cost Prediction
        CostPredictionModel,
        BudgetOptimizedSampler,
        
        # Guardrail Tuning
        PortkeyGuardrailTuner,
        PortkeyRefusalDetector,
        
        # A/B Graduation
        AutomatedRolloutController,
        ExperimentManager,
        ExperimentStage,
    )
    
    # Create cost prediction model
    cost_model = CostPredictionModel(portkey_api_key="pk-...")
    await cost_model.train_from_portkey(days=30)
    prediction = cost_model.predict(prompt, model="deepseek-v3")
    
    # Create budget-optimized sampler
    sampler = BudgetOptimizedSampler(budget_limit=10.0, cost_model=cost_model)
    selected = await sampler.select_replays_within_budget(candidates, "deepseek-v3")
    
    # Detect refusals
    detector = PortkeyRefusalDetector()
    is_refusal, pattern = await detector.detect_refusal(prompt, response)
    
    # Tune guardrails
    tuner = PortkeyGuardrailTuner(portkey_api_key="pk-...")
    analysis = await tuner.analyze_refusal_patterns(patterns)
    tuning = await tuner.generate_improved_prompt(system_prompt, overcautious)
    
    # Manage A/B experiments
    controller = AutomatedRolloutController(
        experiment_id="exp-001",
        challenger_model="deepseek-v3",
        production_model="gpt-4o-mini"
    )
    result = await controller.check_promotion()
    if result.decision == PromotionDecision.PROMOTE:
        await controller.promote()
"""

__version__ = "0.1.0"

# Cost Prediction Model
from shadow_optic.p2_enhancements.cost_predictor import (
    CostPredictionModel,
    BudgetOptimizedSampler,
    PortkeyCostDataCollector,
    CostFeatureExtractor,
    CostFeatures,
    CostPrediction,
    ModelStatistics,
)

# Guardrail Tuning
from shadow_optic.p2_enhancements.guardrail_tuner import (
    PortkeyGuardrailTuner,
    PortkeyRefusalDetector,
    PortkeyGuardrailManager,
    RefusalType,
    RefusalPattern,
    GuardrailAction,
    GuardrailResult,
    TuningResult,
)

# A/B Graduation
from shadow_optic.p2_enhancements.ab_graduation import (
    AutomatedRolloutController,
    PortkeyConfigManager,
    ExperimentManager,
    ExperimentStage,
    ExperimentMetrics,
    StageConfig,
    PromotionDecision,
    PromotionCheckResult,
    ExperimentHistory,
)

__all__ = [
    # Cost Prediction
    "CostPredictionModel",
    "BudgetOptimizedSampler",
    "PortkeyCostDataCollector",
    "CostFeatureExtractor",
    "CostFeatures",
    "CostPrediction",
    "ModelStatistics",
    
    # Guardrail Tuning
    "PortkeyGuardrailTuner",
    "PortkeyRefusalDetector",
    "PortkeyGuardrailManager",
    "RefusalType",
    "RefusalPattern",
    "GuardrailAction",
    "GuardrailResult",
    "TuningResult",
    
    # A/B Graduation
    "AutomatedRolloutController",
    "PortkeyConfigManager",
    "ExperimentManager",
    "ExperimentStage",
    "ExperimentMetrics",
    "StageConfig",
    "PromotionDecision",
    "PromotionCheckResult",
    "ExperimentHistory",
]
