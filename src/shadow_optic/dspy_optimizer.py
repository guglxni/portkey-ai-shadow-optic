"""
DSPy Optimizer - Self-optimizing prompts using MIPROv2.

Implements prompt optimization using Stanford DSPy with the MIPROv2 algorithm.
When a cluster's quality degrades or a cheaper model fails, the optimizer:

1. Extracts DSPy Signatures from PromptGenie templates
2. Compiles training data from successful high-quality responses
3. Runs MIPROv2 optimization loop
4. Validates improved prompts against quality thresholds
5. Optionally auto-deploys to production

Integration Points:
- PromptGenie Templates → DSPy Signatures
- Shadow-Optic Evaluator → Validation metric
- Portkey Gateway → LLM backend for optimization
- Qdrant → Training data and deployment targets

References:
- DSPy: https://github.com/stanfordnlp/dspy
- MIPROv2: https://arxiv.org/abs/2406.11695
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Optimization Configuration
# =============================================================================

class OptimizationStrategy(str, Enum):
    """Strategy for prompt optimization."""
    MIPRO_V2 = "mipro_v2"  # Full MIPROv2 with instruction/example optimization
    BOOTSTRAP = "bootstrap"  # BootstrapFewShot (faster, simpler)
    COPRO = "copro"  # Coordinate prompt optimization
    SIGNATURE_OPT = "signature_opt"  # Optimize signature only


class DSPyOptimizerConfig(BaseModel):
    """Configuration for the DSPy optimizer."""
    
    # Strategy selection
    strategy: OptimizationStrategy = Field(
        default=OptimizationStrategy.MIPRO_V2,
        description="Optimization algorithm to use"
    )
    
    # Model configuration
    # NOTE: GPT-5 models use max_completion_tokens (not max_tokens)
    teacher_model: str = Field(
        default="gpt-5",
        description="High-quality teacher model for generating training data"
    )
    student_model: str = Field(
        default="gpt-5-mini",
        description="Cheaper student model to optimize prompts for"
    )
    
    # Optimization parameters
    max_bootstrapped_demos: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of few-shot examples"
    )
    max_labeled_demos: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum labeled examples for training"
    )
    num_candidates: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate prompts to generate"
    )
    num_trials: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of optimization trials"
    )
    
    # Quality thresholds
    min_quality_improvement: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum improvement required to accept optimization"
    )
    target_quality_score: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Target quality score for optimized prompts"
    )
    
    # Training data
    min_training_samples: int = Field(
        default=20,
        ge=5,
        description="Minimum training samples required"
    )
    max_training_samples: int = Field(
        default=200,
        ge=20,
        description="Maximum training samples to use"
    )
    
    # Deployment
    auto_deploy: bool = Field(
        default=False,
        description="Automatically deploy successful optimizations"
    )
    require_validation: bool = Field(
        default=True,
        description="Require validation pass before deployment"
    )
    
    # Portkey Model Catalog integration
    # See docs/PORTKEY_MODEL_CATALOG.md for reference
    use_portkey: bool = Field(
        default=True,
        description="Use Portkey Model Catalog for LLM calls"
    )
    # NOTE: Virtual keys are deprecated. Model Catalog uses @provider/model slugs
    # with x-portkey-provider header instead.


# =============================================================================
# DSPy Signature Builder
# =============================================================================

@dataclass
class DSPyField:
    """Represents a field in a DSPy Signature."""
    name: str
    description: str
    prefix: str = ""
    is_input: bool = True
    
    def to_dspy_field(self) -> str:
        """Generate DSPy field definition."""
        field_type = "dspy.InputField" if self.is_input else "dspy.OutputField"
        return (
            f"{self.name}: str = {field_type}("
            f"desc=\"{self.description}\""
            f"{f', prefix=\"{self.prefix}\"' if self.prefix else ''}"
            f")"
        )


@dataclass
class DSPySignature:
    """Represents a complete DSPy Signature."""
    name: str
    doc: str
    input_fields: List[DSPyField] = field(default_factory=list)
    output_fields: List[DSPyField] = field(default_factory=list)
    
    def to_class_definition(self) -> str:
        """Generate Python class definition for the signature."""
        lines = [
            f"class {self.name}(dspy.Signature):",
            f'    """{self.doc}"""',
        ]
        
        for f in self.input_fields:
            lines.append(f"    {f.to_dspy_field()}")
        
        for f in self.output_fields:
            lines.append(f"    {f.to_dspy_field()}")
        
        return "\n".join(lines)
    
    def create_runtime_class(self) -> type:
        """Create an actual DSPy Signature class at runtime."""
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required for optimization. "
                "Install with: pip install dspy-ai"
            )
        
        # Build class attributes
        attrs = {"__doc__": self.doc}
        
        for f in self.input_fields:
            attrs[f.name] = dspy.InputField(desc=f.description, prefix=f.prefix or None)
        
        for f in self.output_fields:
            attrs[f.name] = dspy.OutputField(desc=f.description, prefix=f.prefix or None)
        
        # Create class dynamically
        return type(self.name, (dspy.Signature,), attrs)


class SignatureBuilder:
    """Builds DSPy Signatures from PromptGenie templates."""
    
    # Pattern for extracting variables from templates
    VARIABLE_PATTERN = re.compile(r'\{\{?\s*(\w+)\s*\}?\}')
    
    @classmethod
    def from_template(
        cls,
        template: str,
        template_name: str = "ExtractedSignature",
        variable_descriptions: Optional[Dict[str, str]] = None,
        output_name: str = "response",
        output_description: str = "The model's response",
    ) -> DSPySignature:
        """Build a DSPy Signature from a prompt template.
        
        Args:
            template: The prompt template with {variable} placeholders
            template_name: Name for the generated signature class
            variable_descriptions: Optional descriptions for each variable
            output_name: Name of the output field
            output_description: Description of the output
            
        Returns:
            DSPySignature ready for optimization
        """
        variable_descriptions = variable_descriptions or {}
        
        # Extract input variables from template
        variables = cls.VARIABLE_PATTERN.findall(template)
        unique_vars = list(dict.fromkeys(variables))  # Preserve order, remove dupes
        
        input_fields = []
        for var in unique_vars:
            description = variable_descriptions.get(
                var, 
                f"The {var.replace('_', ' ')}"
            )
            input_fields.append(DSPyField(
                name=var,
                description=description,
                is_input=True
            ))
        
        output_fields = [
            DSPyField(
                name=output_name,
                description=output_description,
                is_input=False
            )
        ]
        
        # Generate doc from template preview
        doc_preview = template[:100].replace("\n", " ").strip()
        if len(template) > 100:
            doc_preview += "..."
        
        return DSPySignature(
            name=template_name,
            doc=doc_preview,
            input_fields=input_fields,
            output_fields=output_fields
        )
    
    @classmethod
    def from_genie_template(
        cls,
        genie_result: Dict[str, Any],
    ) -> DSPySignature:
        """Build a DSPy Signature from PromptGenie extraction result.
        
        Args:
            genie_result: Result from PromptGenie's extract_template()
            
        Returns:
            DSPySignature with rich variable descriptions
        """
        template = genie_result.get("template", "")
        metadata = genie_result.get("metadata", {})
        
        # Get variable descriptions from Genie metadata
        variables = genie_result.get("variables", {})
        variable_descriptions = {
            name: info.get("description", f"The {name}")
            for name, info in variables.items()
        }
        
        # Build signature name from intent
        intent = metadata.get("intent_tags", ["General"])
        sig_name = "".join(word.capitalize() for word in intent[0].split("_"))
        sig_name += "Signature"
        
        # Determine output based on task type
        task_type = metadata.get("task_type", "completion")
        output_mapping = {
            "classification": ("label", "The classification label"),
            "extraction": ("extracted_data", "Extracted information"),
            "summarization": ("summary", "The summarized content"),
            "translation": ("translation", "The translated text"),
            "generation": ("generated_text", "The generated content"),
        }
        output_name, output_desc = output_mapping.get(
            task_type, 
            ("response", "The model's response")
        )
        
        return cls.from_template(
            template=template,
            template_name=sig_name,
            variable_descriptions=variable_descriptions,
            output_name=output_name,
            output_description=output_desc
        )


# =============================================================================
# Training Data Builder
# =============================================================================

@dataclass
class TrainingExample:
    """A single training example for DSPy optimization."""
    inputs: Dict[str, str]
    output: str
    quality_score: float = 1.0
    source_model: str = "gpt-4o"


class TrainingDataBuilder:
    """Builds training data from Qdrant cluster history."""
    
    def __init__(self, qdrant_client: Any, collection_name: str = "prompt_clusters"):
        self.qdrant = qdrant_client
        self.collection_name = collection_name
    
    async def get_training_data(
        self,
        cluster_id: str,
        min_quality: float = 0.85,
        max_samples: int = 200,
        teacher_model: Optional[str] = None,
    ) -> List[TrainingExample]:
        """Retrieve high-quality examples for training.
        
        Args:
            cluster_id: The cluster to get training data for
            min_quality: Minimum quality score for examples
            max_samples: Maximum number of examples to return
            teacher_model: Prefer examples from this model
            
        Returns:
            List of TrainingExample objects
        """
        examples = []
        
        try:
            from qdrant_client.models import Filter, FieldCondition, Range
            
            # Build filter for high-quality examples in this cluster
            filter_conditions = [
                FieldCondition(
                    key="cluster_id",
                    match={"value": cluster_id}
                ),
                FieldCondition(
                    key="quality_score",
                    range=Range(gte=min_quality)
                )
            ]
            
            if teacher_model:
                filter_conditions.append(
                    FieldCondition(
                        key="model",
                        match={"value": teacher_model}
                    )
                )
            
            # Query Qdrant
            results = await self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=max_samples,
                with_payload=True
            )
            
            for point in results[0]:
                payload = point.payload
                examples.append(TrainingExample(
                    inputs=payload.get("inputs", {}),
                    output=payload.get("output", ""),
                    quality_score=payload.get("quality_score", 1.0),
                    source_model=payload.get("model", "unknown")
                ))
                
        except Exception as e:
            logger.warning(f"Failed to get training data: {e}")
        
        return examples
    
    def examples_to_dspy_format(
        self,
        examples: List[TrainingExample],
        signature: DSPySignature
    ) -> List[Any]:
        """Convert examples to DSPy Example format.
        
        Args:
            examples: Training examples
            signature: The DSPy signature for field mapping
            
        Returns:
            List of dspy.Example objects
        """
        try:
            import dspy
        except ImportError:
            raise ImportError("DSPy required: pip install dspy-ai")
        
        dspy_examples = []
        
        output_field = signature.output_fields[0].name if signature.output_fields else "response"
        
        for ex in examples:
            # Build example dict
            ex_dict = dict(ex.inputs)
            ex_dict[output_field] = ex.output
            
            dspy_examples.append(dspy.Example(**ex_dict).with_inputs(
                *[f.name for f in signature.input_fields]
            ))
        
        return dspy_examples


# =============================================================================
# DSPy Optimizer
# =============================================================================

@dataclass
class OptimizationResult:
    """Result of a DSPy optimization run."""
    success: bool
    original_score: float
    optimized_score: float
    improvement: float
    optimized_program: Optional[Any] = None
    optimized_prompt: Optional[str] = None
    num_demos: int = 0
    trials_run: int = 0
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DSPyOptimizer:
    """
    Self-optimizing prompt system using DSPy MIPROv2.
    
    The optimizer takes a cluster with degraded quality and:
    1. Extracts the prompt template as a DSPy Signature
    2. Gathers high-quality examples from GPT-5 as training data
    3. Runs MIPROv2 to optimize the prompt for cheaper models
    4. Validates the improvement meets thresholds
    5. Optionally deploys the optimized prompt
    
    Example:
        ```python
        optimizer = DSPyOptimizer(
            config=DSPyOptimizerConfig(
                teacher_model="gpt-5",
                student_model="gpt-5-mini",
                auto_deploy=True
            ),
            qdrant_client=qdrant,
        )
        
        result = await optimizer.optimize_cluster(
            cluster_id="customer-support-billing",
            template=extracted_template,
            variable_descriptions={"query": "Customer question"}
        )
        
        if result.success:
            print(f"Improved by {result.improvement:.1%}")
        ```
    """
    
    def __init__(
        self,
        config: Optional[DSPyOptimizerConfig] = None,
        qdrant_client: Optional[Any] = None,
        evaluator: Optional[Any] = None,
        collection_name: str = "prompt_clusters",
    ):
        self.config = config or DSPyOptimizerConfig()
        self.qdrant = qdrant_client
        self.evaluator = evaluator
        self.collection_name = collection_name
        
        self._dspy_configured = False
        self._training_builder: Optional[TrainingDataBuilder] = None
        
        if qdrant_client:
            self._training_builder = TrainingDataBuilder(
                qdrant_client, collection_name
            )
    
    def _configure_dspy(self) -> None:
        """Configure DSPy with Portkey Model Catalog.
        
        Uses Portkey Model Catalog (https://portkey.ai/docs/product/model-catalog)
        with x-portkey-provider header instead of virtual keys.
        
        See docs/PORTKEY_MODEL_CATALOG.md for reference.
        """
        if self._dspy_configured:
            return
        
        try:
            import dspy
        except ImportError:
            raise ImportError(
                "DSPy is required for optimization. "
                "Install with: pip install dspy-ai"
            )
        
        import os
        
        if self.config.use_portkey:
            # Portkey Native DSPy Integration
            # Format: openai/@provider-slug/model-name
            # See: https://portkey.ai/docs/integrations/libraries/dspy
            portkey_api_key = os.environ.get("PORTKEY_API_KEY")
            if not portkey_api_key:
                raise ValueError("PORTKEY_API_KEY environment variable required")
            
            lm = dspy.LM(
                model=f"openai/@openai/{self.config.student_model}",  # Portkey native format
                api_base="https://api.portkey.ai/v1",
                api_key=portkey_api_key
            )
        else:
            # Direct OpenAI (fallback, not recommended)
            lm = dspy.LM(model=f"openai/{self.config.student_model}")
        
        dspy.configure(lm=lm)
        self._dspy_configured = True
    
    async def optimize_cluster(
        self,
        cluster_id: str,
        template: Optional[str] = None,
        genie_result: Optional[Dict[str, Any]] = None,
        variable_descriptions: Optional[Dict[str, str]] = None,
        policy: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> OptimizationResult:
        """Optimize prompts for a cluster using DSPy.
        
        Args:
            cluster_id: The cluster to optimize
            template: Optional prompt template (extracted if not provided)
            genie_result: Optional PromptGenie extraction result
            variable_descriptions: Optional variable descriptions
            policy: Optional AgenticPolicy with constraints
            metrics: Current cluster metrics
            
        Returns:
            OptimizationResult with optimization outcome
        """
        logger.info(f"Starting DSPy optimization for cluster: {cluster_id}")
        
        try:
            self._configure_dspy()
            
            # 1. Build DSPy Signature
            if genie_result:
                signature = SignatureBuilder.from_genie_template(genie_result)
            elif template:
                signature = SignatureBuilder.from_template(
                    template=template,
                    template_name=f"{cluster_id.title().replace('-', '')}Signature",
                    variable_descriptions=variable_descriptions
                )
            else:
                raise ValueError("Either template or genie_result must be provided")
            
            logger.debug(f"Built signature: {signature.name}")
            
            # 2. Get training data
            training_examples = await self._get_training_examples(
                cluster_id, signature
            )
            
            if len(training_examples) < self.config.min_training_samples:
                return OptimizationResult(
                    success=False,
                    original_score=metrics.get("quality", 0) if metrics else 0,
                    optimized_score=0,
                    improvement=0,
                    error=f"Insufficient training data ({len(training_examples)} samples)"
                )
            
            # 3. Run optimization
            result = await self._run_optimization(
                signature=signature,
                training_data=training_examples,
                metrics=metrics
            )
            
            # 4. Validate and optionally deploy
            if result.success and self.config.auto_deploy:
                await self._deploy_optimization(cluster_id, result, policy)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {cluster_id}: {e}")
            return OptimizationResult(
                success=False,
                original_score=metrics.get("quality", 0) if metrics else 0,
                optimized_score=0,
                improvement=0,
                error=str(e)
            )
    
    async def _get_training_examples(
        self,
        cluster_id: str,
        signature: DSPySignature
    ) -> List[Any]:
        """Retrieve and format training examples."""
        if not self._training_builder:
            return []
        
        # Get raw examples
        raw_examples = await self._training_builder.get_training_data(
            cluster_id=cluster_id,
            min_quality=self.config.target_quality_score,
            max_samples=self.config.max_training_samples,
            teacher_model=self.config.teacher_model
        )
        
        # Convert to DSPy format
        return self._training_builder.examples_to_dspy_format(
            raw_examples, signature
        )
    
    async def _run_optimization(
        self,
        signature: DSPySignature,
        training_data: List[Any],
        metrics: Optional[Dict[str, float]] = None
    ) -> OptimizationResult:
        """Run the actual DSPy optimization loop."""
        import dspy
        
        # Create the signature class
        sig_class = signature.create_runtime_class()
        
        # Create base module
        base_module = dspy.Predict(sig_class)
        
        # Define metric for optimization
        def quality_metric(example, prediction, trace=None) -> float:
            """Evaluate prediction quality."""
            # In production, use actual quality evaluation
            # For now, use simple heuristics
            output_field = signature.output_fields[0].name if signature.output_fields else "response"
            pred_output = getattr(prediction, output_field, "")
            
            if not pred_output:
                return 0.0
            
            # Check for refusals
            refusal_patterns = [
                "I cannot", "I can't", "I'm unable",
                "As an AI", "I don't have access"
            ]
            if any(p.lower() in pred_output.lower() for p in refusal_patterns):
                return 0.3
            
            # Check length (too short = bad)
            if len(pred_output) < 50:
                return 0.5
            
            return 0.9
        
        # Split training data
        from random import shuffle
        shuffle(training_data)
        split = int(len(training_data) * 0.8)
        train_set = training_data[:split]
        val_set = training_data[split:]
        
        # Calculate baseline score
        original_scores = []
        for ex in val_set[:10]:  # Quick baseline on subset
            try:
                pred = base_module(**ex.inputs())
                score = quality_metric(ex, pred)
                original_scores.append(score)
            except Exception:
                original_scores.append(0.0)
        
        original_score = sum(original_scores) / len(original_scores) if original_scores else 0.5
        
        # Select and run optimizer
        if self.config.strategy == OptimizationStrategy.MIPRO_V2:
            optimized_module, trials = await self._run_mipro(
                base_module, train_set, val_set, quality_metric
            )
        elif self.config.strategy == OptimizationStrategy.BOOTSTRAP:
            optimized_module, trials = await self._run_bootstrap(
                base_module, train_set, quality_metric
            )
        else:
            optimized_module, trials = await self._run_bootstrap(
                base_module, train_set, quality_metric
            )
        
        # Calculate optimized score
        optimized_scores = []
        for ex in val_set[:10]:
            try:
                pred = optimized_module(**ex.inputs())
                score = quality_metric(ex, pred)
                optimized_scores.append(score)
            except Exception:
                optimized_scores.append(0.0)
        
        optimized_score = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
        improvement = optimized_score - original_score
        
        # Determine success
        success = (
            improvement >= self.config.min_quality_improvement and
            optimized_score >= self.config.target_quality_score * 0.9  # 90% of target
        )
        
        return OptimizationResult(
            success=success,
            original_score=original_score,
            optimized_score=optimized_score,
            improvement=improvement,
            optimized_program=optimized_module if success else None,
            num_demos=len(getattr(optimized_module, 'demos', [])),
            trials_run=trials
        )
    
    async def _run_mipro(
        self,
        module: Any,
        train_set: List[Any],
        val_set: List[Any],
        metric: Callable
    ) -> Tuple[Any, int]:
        """Run MIPROv2 optimization."""
        import dspy
        from dspy.teleprompt import MIPROv2
        
        optimizer = MIPROv2(
            metric=metric,
            num_candidates=self.config.num_candidates,
            init_temperature=1.0,
            verbose=False
        )
        
        # Run optimization in executor to not block
        loop = asyncio.get_event_loop()
        optimized = await loop.run_in_executor(
            None,
            lambda: optimizer.compile(
                module,
                trainset=train_set,
                valset=val_set,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_labeled_demos,
                num_trials=self.config.num_trials
            )
        )
        
        return optimized, self.config.num_trials
    
    async def _run_bootstrap(
        self,
        module: Any,
        train_set: List[Any],
        metric: Callable
    ) -> Tuple[Any, int]:
        """Run BootstrapFewShot optimization."""
        import dspy
        from dspy.teleprompt import BootstrapFewShot
        
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos
        )
        
        loop = asyncio.get_event_loop()
        optimized = await loop.run_in_executor(
            None,
            lambda: optimizer.compile(module, trainset=train_set)
        )
        
        return optimized, len(train_set)
    
    async def _deploy_optimization(
        self,
        cluster_id: str,
        result: OptimizationResult,
        policy: Optional[Any] = None
    ) -> None:
        """Deploy optimized prompt to production."""
        if not self.qdrant or not result.optimized_program:
            return
        
        try:
            # Extract optimized prompt from the program
            # This varies by DSPy version
            optimized_prompt = self._extract_prompt(result.optimized_program)
            result.optimized_prompt = optimized_prompt
            
            # Update Qdrant with new prompt
            from qdrant_client.models import PointStruct
            
            await self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={
                    "optimized_prompt": optimized_prompt,
                    "optimization_score": result.optimized_score,
                    "optimization_timestamp": result.timestamp.isoformat(),
                    "student_model": self.config.student_model,
                },
                points=[cluster_id]
            )
            
            logger.info(
                f"Deployed optimized prompt for {cluster_id} "
                f"(+{result.improvement:.1%} improvement)"
            )
            
        except Exception as e:
            logger.error(f"Failed to deploy optimization: {e}")
    
    def _extract_prompt(self, program: Any) -> str:
        """Extract the optimized prompt text from a DSPy program."""
        # This is implementation-dependent on DSPy version
        # Try common patterns
        try:
            if hasattr(program, 'extended_signature'):
                return str(program.extended_signature)
            elif hasattr(program, 'signature'):
                return str(program.signature)
            elif hasattr(program, 'demos'):
                # Format demos as few-shot examples
                demos = program.demos
                prompt_parts = []
                for demo in demos:
                    prompt_parts.append(str(demo))
                return "\n\n".join(prompt_parts)
            else:
                return str(program)
        except Exception:
            return "Optimized prompt (extraction failed)"


# =============================================================================
# The "Fixer" Pattern
# =============================================================================

class PromptFixer:
    """
    The "Fixer" pattern from the enhancement doc.
    
    When a cheap model produces a low-quality response, the Fixer:
    1. Detects the failure via evaluation
    2. Invokes DSPy to rewrite the prompt
    3. Re-runs with the optimized prompt
    4. Falls back to expensive model if still failing
    
    This creates a tiered response strategy:
    Tier 1: Cheap model with optimized prompt
    Tier 2: Cheap model with DSPy-enhanced prompt  
    Tier 3: Expensive model fallback
    """
    
    def __init__(
        self,
        optimizer: DSPyOptimizer,
        evaluator: Optional[Any] = None,
        quality_threshold: float = 0.75,
        fallback_model: str = "gpt-4o",
    ):
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.quality_threshold = quality_threshold
        self.fallback_model = fallback_model
    
    async def fix_response(
        self,
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str]:
        """
        Attempt to fix a low-quality response.
        
        Args:
            prompt: The original prompt
            response: The low-quality response
            context: Additional context (variables, etc.)
            
        Returns:
            Tuple of (fixed_response, model_used, strategy)
        """
        # 1. Evaluate original response
        quality = await self._evaluate(prompt, response)
        
        if quality >= self.quality_threshold:
            return response, "original", "passed"
        
        # 2. Try DSPy-enhanced prompt
        logger.info(f"Response quality {quality:.2f} below threshold, attempting fix")
        
        try:
            # Build quick optimization
            signature = SignatureBuilder.from_template(
                template=prompt,
                template_name="FixerSignature"
            )
            
            # Use the response as a negative example
            # and generate improved prompt
            enhanced_prompt = await self._enhance_prompt(prompt, response, context)
            
            # Re-run with enhanced prompt
            fixed_response = await self._rerun_prompt(enhanced_prompt, context)
            
            fixed_quality = await self._evaluate(enhanced_prompt, fixed_response)
            
            if fixed_quality >= self.quality_threshold:
                return fixed_response, self.optimizer.config.student_model, "dspy_enhanced"
                
        except Exception as e:
            logger.warning(f"DSPy enhancement failed: {e}")
        
        # 3. Fall back to expensive model
        logger.info("Falling back to expensive model")
        fallback_response = await self._fallback_call(prompt, context)
        
        return fallback_response, self.fallback_model, "fallback"
    
    async def _evaluate(self, prompt: str, response: str) -> float:
        """Evaluate response quality."""
        if self.evaluator:
            # Use actual evaluator
            result = await self.evaluator.evaluate(prompt, response)
            return result.get("overall_score", 0.5)
        
        # Simple heuristics fallback
        if not response or len(response) < 20:
            return 0.3
        if any(p in response.lower() for p in ["i cannot", "i can't", "as an ai"]):
            return 0.4
        return 0.8
    
    async def _enhance_prompt(
        self,
        prompt: str,
        bad_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance prompt using DSPy patterns."""
        # Add explicit quality guidance
        enhancement = """
Please provide a high-quality, helpful response that:
- Directly addresses the user's question
- Provides specific, actionable information  
- Is comprehensive but concise
- Does not refuse to help or deflect

"""
        return enhancement + prompt
    
    async def _rerun_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Re-run the prompt with the student model."""
        # In production, call the actual model
        # For now, placeholder
        return f"[Enhanced response for: {prompt[:50]}...]"
    
    async def _fallback_call(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call the fallback expensive model."""
        # In production, call GPT-4o via Portkey
        return f"[Fallback response for: {prompt[:50]}...]"


# =============================================================================
# Factory Functions
# =============================================================================

def create_optimizer(
    teacher_model: str = "gpt-5",
    student_model: str = "gpt-5-mini",
    auto_deploy: bool = False,
    qdrant_client: Optional[Any] = None,
) -> DSPyOptimizer:
    """Create a configured DSPyOptimizer.
    
    Args:
        teacher_model: High-quality model for training data (default: gpt-5)
        student_model: Cheaper model to optimize for (default: gpt-5-mini)
        auto_deploy: Whether to auto-deploy optimizations
        qdrant_client: Qdrant client for training data
        
    Returns:
        Configured DSPyOptimizer instance
    """
    config = DSPyOptimizerConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        auto_deploy=auto_deploy,
    )
    
    return DSPyOptimizer(
        config=config,
        qdrant_client=qdrant_client,
    )
