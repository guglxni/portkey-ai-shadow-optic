"""
LLM-as-Judge Evaluator using DeepEval with Portkey Integration.

Production-grade implementation using:
- DeepEval OpenAI client routed through Portkey gateway
- DeepEval FaithfulnessMetric for factual consistency
- DeepEval GEval for custom quality/conciseness criteria
- LLMTestCase for structured evaluation

This ensures ALL evaluation LLM calls go through Portkey for:
- Full observability and logging
- Cost tracking
- Virtual key billing segregation

Portkey Integration Pattern:
    DeepEval uses OpenAI-compatible interface. We configure it to route
    through Portkey's gateway URL by setting OPENAI_BASE_URL, which provides
    observability and virtual key routing to any underlying provider.

Metrics Evaluated:
    1. Faithfulness: Factual consistency with production response
    2. Quality: Overall helpfulness and completeness via G-Eval
    3. Conciseness: Response efficiency via G-Eval
    4. Refusal Detection: Pattern matching for model refusals
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from portkey_ai import AsyncPortkey, PORTKEY_GATEWAY_URL

from shadow_optic.models import (
    EvaluationResult,
    EvaluatorConfig,
    GoldenPrompt,
    ShadowResult,
)

logger = logging.getLogger(__name__)


# Comprehensive refusal patterns including polite refusals and edge cases
REFUSAL_PATTERNS = [
    # Direct refusals
    "I cannot",
    "I can't",
    "I'm unable to",
    "I am unable to",
    "I'm not able to",
    "I won't be able to",
    "I must decline",
    # Apology-based refusals
    "I apologize, but I",
    "I'm sorry, but I can't",
    "I'm afraid I cannot",
    "I'm afraid I can't",
    "I regret that I cannot",
    "Unfortunately, I cannot",
    "Unfortunately, I can't",
    # AI identity refusals
    "As an AI, I cannot",
    "As an AI assistant, I'm not able",
    "As a language model, I cannot",
    # Capability-based refusals
    "I don't have the ability",
    "I lack the capability to",
    "I'm not designed to",
    "I'm not equipped to",
    "That's outside my capabilities",
    "That's beyond my abilities",
    # Policy-based refusals
    "This request violates",
    "This goes against my guidelines",
    "I can't assist with that",
    "I'm not allowed to",
    "I'm programmed not to",
    "My guidelines prevent me from",
    # Safety refusals
    "That would be harmful",
    "I can't provide information on",
    "For safety reasons, I cannot",
    "For ethical reasons, I'm unable",
    "I can't help with anything illegal",
    "I can't assist with harmful",
    # Polite deflections
    "I'd prefer not to",
    "I don't feel comfortable",
    "I'd rather not",
    "I need to respectfully decline",
    "I must respectfully refuse",
]


class RefusalDetector:
    """Detect when a model refuses to answer."""
    
    def __init__(self, patterns: Optional[List[str]] = None):
        self.patterns = patterns or REFUSAL_PATTERNS
    
    def is_refusal(self, response: str) -> bool:
        """Check if response contains refusal patterns."""
        response_lower = response.lower()
        for pattern in self.patterns:
            if pattern.lower() in response_lower:
                return True
        return False


class ShadowEvaluator:
    """
    Evaluates shadow responses using LLM-as-Judge with DeepEval.
    
    Uses real DeepEval metrics:
    - FaithfulnessMetric: Factual consistency
    - GEval: Custom criteria evaluation
    
    Example:
        evaluator = ShadowEvaluator(portkey_client=client, config=config)
        result = await evaluator.evaluate(golden_prompt, shadow_result)
    """
    
    def __init__(
        self,
        portkey_client: AsyncPortkey,
        config: Optional[EvaluatorConfig] = None
    ):
        self.portkey = portkey_client
        self.config = config or EvaluatorConfig()
        self.refusal_detector = RefusalDetector()
        
        # Configure DeepEval to route through Portkey gateway
        self._configure_portkey_environment()
        self._init_metrics()
    
    def _configure_portkey_environment(self) -> None:
        """
        Configure environment for DeepEval to route through Portkey with Model Catalog.
        
        DeepEval uses OpenAI-compatible interface. By setting OPENAI_API_KEY
        to our Portkey API key and OPENAI_BASE_URL to Portkey's gateway,
        all DeepEval LLM calls are automatically routed through Portkey.
        
        With Model Catalog, the model name uses format: @provider_slug/model_name
        (e.g., @openai/gpt-4o-mini). This is passed directly to DeepEval metrics.
        
        This enables:
        - Full observability and logging in Portkey dashboard
        - Cost tracking per evaluation
        - Model Catalog routing to any provider (OpenAI, Anthropic, etc.)
        """
        # Get Portkey API key from config or environment
        api_key = self.config.portkey_api_key or os.getenv("PORTKEY_API_KEY")
        
        if not api_key:
            raise ValueError(
                "Portkey API key required. Set PORTKEY_API_KEY env var or "
                "provide portkey_api_key in EvaluatorConfig."
            )
        
        # Configure DeepEval's OpenAI client to use Portkey gateway
        # DeepEval reads these environment variables for its LLM calls
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = self.config.portkey_base_url
        
        # With Model Catalog, the model name (@provider/model) handles routing
        # No need for separate provider header
        
        logger.info(
            f"Configured DeepEval to route through Portkey gateway: "
            f"base_url={self.config.portkey_base_url}, model={self.config.judge_model}"
        )
    
    def _init_metrics(self) -> None:
        """Initialize DeepEval metrics (will use Portkey-routed OpenAI client)."""
        
        # FaithfulnessMetric for factual consistency
        # Model string is passed; DeepEval uses OPENAI_BASE_URL (Portkey gateway)
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=self.config.faithfulness_threshold,
            model=self.config.judge_model,
            include_reason=True,
            async_mode=True
        )
        
        # GEval for quality assessment
        self.quality_metric = GEval(
            name="Quality",
            criteria="""
            Evaluate the quality of the actual output compared to the expected output.
            Consider:
            1. Completeness: Does it cover all key points?
            2. Accuracy: Is the information correct?
            3. Helpfulness: Would this response satisfy the user's needs?
            4. Appropriateness: Is the tone and format suitable?
            
            The actual output should be semantically equivalent to the expected output.
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            threshold=self.config.quality_threshold,
            model=self.config.judge_model,
            async_mode=True
        )
        
        # GEval for conciseness assessment
        self.conciseness_metric = GEval(
            name="Conciseness",
            criteria="""
            Evaluate how concise the actual output is compared to the expected output.
            
            - Score 1.0: Appropriately concise without unnecessary verbosity
            - Penalize: excessive preambles, redundant explanations, filler content
            - Note: Being shorter is not necessarily better - must still be complete
            """,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ],
            threshold=self.config.conciseness_threshold,
            model=self.config.judge_model,
            async_mode=True
        )
    
    async def evaluate(
        self,
        golden_prompt: GoldenPrompt,
        shadow_result: ShadowResult
    ) -> EvaluationResult:
        """
        Evaluate a single shadow result against its golden prompt.
        
        Args:
            golden_prompt: Original prompt with production response
            shadow_result: Challenger model's response
        
        Returns:
            EvaluationResult with all metric scores
        """
        # Check for refusal first
        is_refusal = self.refusal_detector.is_refusal(shadow_result.shadow_response)
        
        if is_refusal:
            return EvaluationResult(
                shadow_result_id=f"{shadow_result.golden_prompt_id}_{shadow_result.challenger_model}",
                faithfulness=0.0,
                quality=0.0,
                conciseness=0.0,
                is_refusal=True,
                composite_score=0.0,
                passed_thresholds=False,
                judge_model=self.config.judge_model,
                reasoning="Model refused to answer the request."
            )
        
        # Create DeepEval test case
        test_case = LLMTestCase(
            input=golden_prompt.prompt,
            actual_output=shadow_result.shadow_response,
            expected_output=golden_prompt.production_response,
            retrieval_context=[golden_prompt.production_response]
        )
        
        # Run evaluations
        scores = await self._run_evaluations(test_case)
        
        # Calculate composite score
        composite = (
            scores["faithfulness"] * self.config.faithfulness_weight +
            scores["quality"] * self.config.quality_weight +
            scores["conciseness"] * self.config.conciseness_weight
        )
        
        # Check thresholds
        passed = (
            scores["faithfulness"] >= self.config.faithfulness_threshold and
            scores["quality"] >= self.config.quality_threshold and
            scores["conciseness"] >= self.config.conciseness_threshold
        )
        
        return EvaluationResult(
            shadow_result_id=f"{shadow_result.golden_prompt_id}_{shadow_result.challenger_model}",
            faithfulness=scores["faithfulness"],
            quality=scores["quality"],
            conciseness=scores["conciseness"],
            is_refusal=False,
            composite_score=composite,
            passed_thresholds=passed,
            judge_model=self.config.judge_model,
            reasoning=scores.get("reasoning", ""),
            raw_scores=scores
        )
    
    async def _run_evaluations(self, test_case: LLMTestCase) -> Dict[str, Any]:
        """Run all evaluation metrics on test case using real LLM-as-Judge.
        
        Uses DeepEval metrics routed through Portkey with Model Catalog.
        No fallbacks - if evaluation fails, the error propagates up.
        """
        scores = {
            "faithfulness": 0.0,
            "quality": 0.0,
            "conciseness": 0.0,
            "reasoning": ""
        }
        
        reasons = []
        
        # Run faithfulness metric (async) - must succeed
        await self.faithfulness_metric.a_measure(test_case)
        scores["faithfulness"] = self.faithfulness_metric.score or 0.0
        if self.faithfulness_metric.reason:
            reasons.append(f"Faithfulness: {self.faithfulness_metric.reason}")
        
        # Run quality metric (async) - must succeed
        await self.quality_metric.a_measure(test_case)
        scores["quality"] = self.quality_metric.score or 0.0
        if self.quality_metric.reason:
            reasons.append(f"Quality: {self.quality_metric.reason}")
        
        # Run conciseness metric (async) - must succeed
        await self.conciseness_metric.a_measure(test_case)
        scores["conciseness"] = self.conciseness_metric.score or 0.0
        if self.conciseness_metric.reason:
            reasons.append(f"Conciseness: {self.conciseness_metric.reason}")
        
        scores["reasoning"] = " | ".join(reasons)
        
        return scores
    
    async def evaluate_batch(
        self,
        golden_prompts: List[GoldenPrompt],
        shadow_results: List[ShadowResult],
        concurrency: int = 5
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple shadow results with controlled concurrency.
        
        Args:
            golden_prompts: List of golden prompts
            shadow_results: List of shadow results
            concurrency: Max concurrent evaluations
        
        Returns:
            List of EvaluationResult
        """
        golden_by_id = {g.original_trace_id: g for g in golden_prompts}
        results = []
        
        semaphore = asyncio.Semaphore(concurrency)
        
        async def eval_with_limit(shadow: ShadowResult):
            async with semaphore:
                golden = golden_by_id.get(shadow.golden_prompt_id)
                if not golden:
                    return None
                return await self.evaluate(golden, shadow)
        
        tasks = [eval_with_limit(s) for s in shadow_results if s.success]
        evaluations = await asyncio.gather(*tasks, return_exceptions=True)
        
        for eval_result in evaluations:
            if isinstance(eval_result, Exception):
                logger.error(f"Batch evaluation error: {eval_result}")
            elif eval_result:
                results.append(eval_result)
        
        return results


class SemanticDiff:
    """Compare responses semantically using embeddings."""
    
    def __init__(self, portkey_client: AsyncPortkey, model: str = "text-embedding-3-small"):
        self.portkey = portkey_client
        self.model = model
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        import numpy as np
        
        try:
            response = await self.portkey.embeddings.create(
                input=[text1[:8000], text2[:8000]],
                model=self.model
            )
            
            emb1 = np.array(response.data[0].embedding)
            emb2 = np.array(response.data[1].embedding)
            
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0
    
    async def compute_diff(
        self,
        production_response: str,
        challenger_response: str
    ) -> Dict[str, Any]:
        """Compute semantic diff between responses."""
        similarity = await self.compute_similarity(production_response, challenger_response)
        
        # Length analysis
        prod_len = len(production_response)
        chal_len = len(challenger_response)
        length_ratio = chal_len / prod_len if prod_len > 0 else 1.0
        
        return {
            "semantic_similarity": similarity,
            "length_ratio": length_ratio,
            "production_length": prod_len,
            "challenger_length": chal_len,
            "is_similar": similarity > 0.85,
            "is_concise": 0.5 <= length_ratio <= 1.5
        }
