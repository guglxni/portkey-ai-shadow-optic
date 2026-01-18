#!/usr/bin/env python3
"""Test that all modules can be imported correctly."""

import sys
sys.path.insert(0, 'src')

def test_imports():
    print('Testing imports...\n')
    
    # Test models
    try:
        from shadow_optic.models import (
            ProductionTrace, GoldenPrompt, ShadowResult, EvaluationResult,
            ChallengerResult, CostAnalysis, Recommendation, OptimizationReport,
            WorkflowConfig, SamplerConfig, EvaluatorConfig, ChallengerConfig,
            RecommendationAction, SampleType, TraceStatus
        )
        print('✓ models.py imports OK')
    except Exception as e:
        print(f'✗ models.py FAILED: {e}')
        return False
    
    # Test sampler
    try:
        from shadow_optic.sampler import SemanticSampler, ClusterInfo
        print('✓ sampler.py imports OK')
    except Exception as e:
        print(f'✗ sampler.py FAILED: {e}')
        return False
    
    # Test evaluator
    try:
        from shadow_optic.evaluator import ShadowEvaluator, RefusalDetector, SemanticDiff
        print('✓ evaluator.py imports OK')
    except Exception as e:
        print(f'✗ evaluator.py FAILED: {e}')
        return False
    
    # Test decision engine
    try:
        from shadow_optic.decision_engine import DecisionEngine, ThompsonSamplingModelSelector, MODEL_PRICING
        print('✓ decision_engine.py imports OK')
    except Exception as e:
        print(f'✗ decision_engine.py FAILED: {e}')
        return False
    
    # Test activities
    try:
        from shadow_optic.activities import (
            export_portkey_logs, parse_portkey_logs, sample_golden_prompts,
            replay_shadow_requests, evaluate_shadow_results, 
            generate_optimization_report, submit_portkey_feedback
        )
        print('✓ activities.py imports OK')
    except Exception as e:
        print(f'✗ activities.py FAILED: {e}')
        return False
    
    # Test workflows
    try:
        from shadow_optic.workflows import (
            OptimizeModelSelectionWorkflow, ScheduledOptimizationWorkflow,
            AdaptiveOptimizationWorkflow, DEFAULT_RETRY_POLICY
        )
        print('✓ workflows.py imports OK')
    except Exception as e:
        print(f'✗ workflows.py FAILED: {e}')
        return False
    
    # Test worker
    try:
        from shadow_optic.worker import main as worker_main
        print('✓ worker.py imports OK')
    except Exception as e:
        print(f'✗ worker.py FAILED: {e}')
        return False
    
    # Test API
    try:
        from shadow_optic.api import app
        print('✓ api.py imports OK')
    except Exception as e:
        print(f'✗ api.py FAILED: {e}')
        return False
    
    # Test CLI
    try:
        from shadow_optic.cli import cli
        print('✓ cli.py imports OK')
    except Exception as e:
        print(f'✗ cli.py FAILED: {e}')
        return False
    
    print('\n✓ All imports successful!')
    return True


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)
