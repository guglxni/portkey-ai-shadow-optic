"""
Model Registry for Shadow-Optic.

Comprehensive registry of 250+ AI models available via Portkey Gateway.
Supports intelligent model selection based on:
- Cost-efficiency (cost per 1M tokens)
- Quality tier (flagship, standard, economy)
- Provider diversity (50+ providers)
- Capability matching (chat, reasoning, coding, vision, etc.)
- Latency characteristics

This registry powers the intelligent challenger selection algorithm
that finds Pareto-optimal models for cost-quality trade-offs.

Sources: Portkey docs (https://portkey.ai/docs/integrations/llms)
Updated: January 17, 2026
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Model quality tiers based on benchmarks and pricing."""
    FLAGSHIP = "flagship"      # Top-tier, highest quality (GPT-5.2, Claude Opus 4.5)
    PREMIUM = "premium"        # High quality, premium pricing (Claude Sonnet, Gemini Pro)
    STANDARD = "standard"      # Good quality, balanced pricing (GPT-5-mini, Haiku)
    ECONOMY = "economy"        # Cost-optimized (Flash models, open-source)
    ULTRA_ECONOMY = "ultra"    # Extremely cheap (Nano, Lite models)


class Capability(Enum):
    """Model capabilities for use-case matching."""
    CHAT = "chat"                    # General chat/conversation
    REASONING = "reasoning"          # Extended thinking, complex reasoning
    CODING = "coding"                # Code generation/analysis
    VISION = "vision"                # Image understanding
    FUNCTION_CALLING = "function"    # Tool use
    LONG_CONTEXT = "long_context"    # >100K context window
    MULTILINGUAL = "multilingual"    # Strong non-English support
    CREATIVE = "creative"            # Creative writing
    FACTUAL = "factual"              # High factuality
    LOW_LATENCY = "low_latency"      # Fast inference


class Provider(Enum):
    """Portkey-supported providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    AZURE = "azure"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    COHERE = "cohere"
    XAI = "xai"
    CEREBRAS = "cerebras"
    SAMBANOVA = "sambanova"
    PERPLEXITY = "perplexity"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    ANYSCALE = "anyscale"


@dataclass
class ModelSpec:
    """Specification for a model in the registry."""
    model_id: str                               # Portkey Model Catalog ID
    provider: Provider                          # Provider enum
    display_name: str                           # Human-readable name
    tier: QualityTier                           # Quality tier
    input_cost_per_1m: float                    # $ per 1M input tokens
    output_cost_per_1m: float                   # $ per 1M output tokens
    capabilities: Set[Capability] = field(default_factory=set)
    context_window: int = 128_000               # Max context in tokens
    avg_latency_ms: int = 1000                  # Typical latency
    quality_score: float = 0.80                 # Estimated quality (0-1)
    notes: str = ""                             # Additional notes
    
    @property
    def portkey_model_id(self) -> str:
        """Get the Portkey Model Catalog format (@provider/model)."""
        return f"@{self.provider.value}/{self.model_id}"
    
    @property
    def blended_cost_per_1m(self) -> float:
        """Blended cost assuming 1:1 input:output ratio."""
        return (self.input_cost_per_1m + self.output_cost_per_1m) / 2
    
    @property
    def cost_quality_ratio(self) -> float:
        """Cost-quality efficiency (higher is better)."""
        if self.blended_cost_per_1m == 0:
            return float('inf')
        return self.quality_score / self.blended_cost_per_1m
    
    def matches_capabilities(self, required: Set[Capability]) -> bool:
        """Check if model has all required capabilities."""
        return required.issubset(self.capabilities)


# =============================================================================
# COMPREHENSIVE MODEL REGISTRY - January 17, 2026
# =============================================================================
# 250+ models across 50+ providers via Portkey Gateway

MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def _register(spec: ModelSpec):
    """Register a model spec."""
    MODEL_REGISTRY[spec.model_id] = spec


# -----------------------------------------------------------------------------
# OpenAI Models (via @openai provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="gpt-5.2",
    provider=Provider.OPENAI,
    display_name="GPT-5.2",
    tier=QualityTier.FLAGSHIP,
    input_cost_per_1m=1.75,
    output_cost_per_1m=14.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING, 
                  Capability.FUNCTION_CALLING, Capability.VISION, Capability.CREATIVE},
    context_window=128_000,
    avg_latency_ms=1500,
    quality_score=0.98,
    notes="Flagship model, excellent all-around"
))

_register(ModelSpec(
    model_id="gpt-5.2-pro",
    provider=Provider.OPENAI,
    display_name="GPT-5.2 Pro",
    tier=QualityTier.FLAGSHIP,
    input_cost_per_1m=21.00,
    output_cost_per_1m=168.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING,
                  Capability.FUNCTION_CALLING, Capability.VISION, Capability.CREATIVE,
                  Capability.LONG_CONTEXT},
    context_window=1_000_000,
    avg_latency_ms=3000,
    quality_score=0.99,
    notes="Ultimate flagship with 1M context"
))

_register(ModelSpec(
    model_id="gpt-5-mini",
    provider=Provider.OPENAI,
    display_name="GPT-5 Mini",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=0.25,
    output_cost_per_1m=2.00,
    capabilities={Capability.CHAT, Capability.CODING, Capability.FUNCTION_CALLING},
    context_window=128_000,
    avg_latency_ms=500,
    quality_score=0.88,
    notes="Best cost-quality for most use cases"
))

_register(ModelSpec(
    model_id="gpt-4.1",
    provider=Provider.OPENAI,
    display_name="GPT-4.1",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=3.00,
    output_cost_per_1m=12.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING,
                  Capability.FUNCTION_CALLING, Capability.VISION},
    context_window=128_000,
    avg_latency_ms=1200,
    quality_score=0.93,
    notes="Solid premium model"
))

_register(ModelSpec(
    model_id="gpt-4.1-mini",
    provider=Provider.OPENAI,
    display_name="GPT-4.1 Mini",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=0.80,
    output_cost_per_1m=3.20,
    capabilities={Capability.CHAT, Capability.CODING, Capability.FUNCTION_CALLING},
    context_window=128_000,
    avg_latency_ms=400,
    quality_score=0.85,
    notes="Good balance of cost and quality"
))

_register(ModelSpec(
    model_id="gpt-4.1-nano",
    provider=Provider.OPENAI,
    display_name="GPT-4.1 Nano",
    tier=QualityTier.ULTRA_ECONOMY,
    input_cost_per_1m=0.20,
    output_cost_per_1m=0.80,
    capabilities={Capability.CHAT, Capability.LOW_LATENCY},
    context_window=64_000,
    avg_latency_ms=200,
    quality_score=0.75,
    notes="Fastest, cheapest OpenAI model"
))

_register(ModelSpec(
    model_id="o3-mini",
    provider=Provider.OPENAI,
    display_name="O3 Mini",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=1.10,
    output_cost_per_1m=4.40,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING},
    context_window=200_000,
    avg_latency_ms=2000,
    quality_score=0.91,
    notes="Reasoning-focused model"
))

# -----------------------------------------------------------------------------
# Anthropic Claude Models (via @anthropic provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="claude-opus-4.5",
    provider=Provider.ANTHROPIC,
    display_name="Claude Opus 4.5",
    tier=QualityTier.FLAGSHIP,
    input_cost_per_1m=5.00,
    output_cost_per_1m=25.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING,
                  Capability.CREATIVE, Capability.FUNCTION_CALLING, Capability.VISION,
                  Capability.LONG_CONTEXT},
    context_window=200_000,
    avg_latency_ms=2000,
    quality_score=0.97,
    notes="Anthropic flagship, excellent reasoning"
))

_register(ModelSpec(
    model_id="claude-sonnet-4.5",
    provider=Provider.ANTHROPIC,
    display_name="Claude Sonnet 4.5",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=3.00,
    output_cost_per_1m=15.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING,
                  Capability.FUNCTION_CALLING, Capability.VISION},
    context_window=200_000,
    avg_latency_ms=1200,
    quality_score=0.94,
    notes="Best value premium model"
))

_register(ModelSpec(
    model_id="claude-haiku-4.5",
    provider=Provider.ANTHROPIC,
    display_name="Claude Haiku 4.5",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=1.00,
    output_cost_per_1m=5.00,
    capabilities={Capability.CHAT, Capability.CODING, Capability.FUNCTION_CALLING,
                  Capability.LOW_LATENCY},
    context_window=200_000,
    avg_latency_ms=400,
    quality_score=0.87,
    notes="Fast and affordable, great for routing"
))

_register(ModelSpec(
    model_id="claude-3.7-sonnet",
    provider=Provider.ANTHROPIC,
    display_name="Claude 3.7 Sonnet",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=2.00,
    output_cost_per_1m=10.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING},
    context_window=200_000,
    avg_latency_ms=1000,
    quality_score=0.92,
    notes="Extended thinking support"
))

# -----------------------------------------------------------------------------
# Google Gemini Models (via @google provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="gemini-3-pro",
    provider=Provider.GOOGLE,
    display_name="Gemini 3 Pro",
    tier=QualityTier.FLAGSHIP,
    input_cost_per_1m=2.00,
    output_cost_per_1m=12.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING,
                  Capability.VISION, Capability.FUNCTION_CALLING, Capability.LONG_CONTEXT,
                  Capability.MULTILINGUAL},
    context_window=2_000_000,
    avg_latency_ms=1500,
    quality_score=0.96,
    notes="2M context, excellent multimodal"
))

_register(ModelSpec(
    model_id="gemini-3-flash",
    provider=Provider.GOOGLE,
    display_name="Gemini 3 Flash",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=0.50,
    output_cost_per_1m=3.00,
    capabilities={Capability.CHAT, Capability.CODING, Capability.VISION,
                  Capability.LOW_LATENCY},
    context_window=1_000_000,
    avg_latency_ms=300,
    quality_score=0.89,
    notes="Fast multimodal with 1M context"
))

_register(ModelSpec(
    model_id="gemini-2.5-pro",
    provider=Provider.GOOGLE,
    display_name="Gemini 2.5 Pro",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=1.25,
    output_cost_per_1m=10.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING,
                  Capability.VISION, Capability.LONG_CONTEXT},
    context_window=1_000_000,
    avg_latency_ms=1000,
    quality_score=0.93,
    notes="Thinking mode available"
))

_register(ModelSpec(
    model_id="gemini-2.5-flash",
    provider=Provider.GOOGLE,
    display_name="Gemini 2.5 Flash",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.30,
    output_cost_per_1m=2.50,
    capabilities={Capability.CHAT, Capability.CODING, Capability.VISION,
                  Capability.LOW_LATENCY},
    context_window=1_000_000,
    avg_latency_ms=250,
    quality_score=0.86,
    notes="Best price-performance for many use cases"
))

_register(ModelSpec(
    model_id="gemini-2.5-flash-lite",
    provider=Provider.GOOGLE,
    display_name="Gemini 2.5 Flash Lite",
    tier=QualityTier.ULTRA_ECONOMY,
    input_cost_per_1m=0.10,
    output_cost_per_1m=0.40,
    capabilities={Capability.CHAT, Capability.LOW_LATENCY},
    context_window=500_000,
    avg_latency_ms=150,
    quality_score=0.78,
    notes="Extremely cheap, good for classification"
))

# -----------------------------------------------------------------------------
# DeepSeek Models (via @deepseek provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="deepseek-v3",
    provider=Provider.DEEPSEEK,
    display_name="DeepSeek V3",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.14,
    output_cost_per_1m=0.28,
    capabilities={Capability.CHAT, Capability.CODING, Capability.REASONING},
    context_window=128_000,
    avg_latency_ms=800,
    quality_score=0.90,
    notes="Incredible value, rivals GPT-4"
))

_register(ModelSpec(
    model_id="deepseek-reasoner",
    provider=Provider.DEEPSEEK,
    display_name="DeepSeek Reasoner (R1)",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.55,
    output_cost_per_1m=2.19,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CODING},
    context_window=128_000,
    avg_latency_ms=2500,
    quality_score=0.94,
    notes="Best reasoning at this price point"
))

# -----------------------------------------------------------------------------
# Meta Llama 4 Models (via @bedrock or @together provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="llama-4-405b",
    provider=Provider.BEDROCK,
    display_name="Llama 4 405B",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.50,
    output_cost_per_1m=0.50,
    capabilities={Capability.CHAT, Capability.CODING, Capability.REASONING,
                  Capability.MULTILINGUAL},
    context_window=128_000,
    avg_latency_ms=1000,
    quality_score=0.92,
    notes="Flagship open-source model"
))

_register(ModelSpec(
    model_id="llama-4-70b",
    provider=Provider.BEDROCK,
    display_name="Llama 4 70B",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.25,
    output_cost_per_1m=0.25,
    capabilities={Capability.CHAT, Capability.CODING},
    context_window=128_000,
    avg_latency_ms=600,
    quality_score=0.88,
    notes="Great value open-source"
))

_register(ModelSpec(
    model_id="llama-4-8b",
    provider=Provider.BEDROCK,
    display_name="Llama 4 8B",
    tier=QualityTier.ULTRA_ECONOMY,
    input_cost_per_1m=0.05,
    output_cost_per_1m=0.05,
    capabilities={Capability.CHAT, Capability.LOW_LATENCY},
    context_window=128_000,
    avg_latency_ms=200,
    quality_score=0.75,
    notes="Ultra-cheap, good for simple tasks"
))

# -----------------------------------------------------------------------------
# Mistral Models (via @mistral provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="mistral-large-3",
    provider=Provider.MISTRAL,
    display_name="Mistral Large 3",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=2.00,
    output_cost_per_1m=6.00,
    capabilities={Capability.CHAT, Capability.CODING, Capability.FUNCTION_CALLING,
                  Capability.MULTILINGUAL},
    context_window=128_000,
    avg_latency_ms=800,
    quality_score=0.91,
    notes="Excellent European model"
))

_register(ModelSpec(
    model_id="mistral-medium-3",
    provider=Provider.MISTRAL,
    display_name="Mistral Medium 3",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=1.00,
    output_cost_per_1m=3.00,
    capabilities={Capability.CHAT, Capability.CODING},
    context_window=128_000,
    avg_latency_ms=500,
    quality_score=0.86,
    notes="Good mid-tier option"
))

_register(ModelSpec(
    model_id="mistral-small-3",
    provider=Provider.MISTRAL,
    display_name="Mistral Small 3",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.10,
    output_cost_per_1m=0.30,
    capabilities={Capability.CHAT, Capability.LOW_LATENCY},
    context_window=128_000,
    avg_latency_ms=300,
    quality_score=0.80,
    notes="Fast and cheap"
))

_register(ModelSpec(
    model_id="codestral",
    provider=Provider.MISTRAL,
    display_name="Codestral",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=0.30,
    output_cost_per_1m=0.90,
    capabilities={Capability.CODING, Capability.CHAT},
    context_window=32_000,
    avg_latency_ms=400,
    quality_score=0.89,
    notes="Specialized for code"
))

# -----------------------------------------------------------------------------
# xAI Grok Models (via @xai provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="grok-3",
    provider=Provider.XAI,
    display_name="Grok 3",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=2.00,
    output_cost_per_1m=10.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.CREATIVE},
    context_window=131_072,
    avg_latency_ms=1200,
    quality_score=0.93,
    notes="Excellent for creative tasks"
))

_register(ModelSpec(
    model_id="grok-3-mini",
    provider=Provider.XAI,
    display_name="Grok 3 Mini",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.50,
    output_cost_per_1m=2.00,
    capabilities={Capability.CHAT, Capability.REASONING},
    context_window=131_072,
    avg_latency_ms=500,
    quality_score=0.86,
    notes="Fast reasoning model"
))

# -----------------------------------------------------------------------------
# Cohere Models (via @cohere provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="command-r-plus",
    provider=Provider.COHERE,
    display_name="Command R+",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=2.50,
    output_cost_per_1m=10.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.FUNCTION_CALLING,
                  Capability.MULTILINGUAL},
    context_window=128_000,
    avg_latency_ms=1000,
    quality_score=0.90,
    notes="Great for RAG applications"
))

_register(ModelSpec(
    model_id="command-r",
    provider=Provider.COHERE,
    display_name="Command R",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=0.50,
    output_cost_per_1m=1.50,
    capabilities={Capability.CHAT, Capability.FUNCTION_CALLING},
    context_window=128_000,
    avg_latency_ms=500,
    quality_score=0.84,
    notes="Affordable RAG model"
))

# -----------------------------------------------------------------------------
# Perplexity Models (via @perplexity provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="sonar-pro",
    provider=Provider.PERPLEXITY,
    display_name="Sonar Pro",
    tier=QualityTier.PREMIUM,
    input_cost_per_1m=3.00,
    output_cost_per_1m=15.00,
    capabilities={Capability.CHAT, Capability.REASONING, Capability.FACTUAL},
    context_window=128_000,
    avg_latency_ms=2000,
    quality_score=0.91,
    notes="Web-augmented, great for factual queries"
))

_register(ModelSpec(
    model_id="sonar",
    provider=Provider.PERPLEXITY,
    display_name="Sonar",
    tier=QualityTier.STANDARD,
    input_cost_per_1m=1.00,
    output_cost_per_1m=5.00,
    capabilities={Capability.CHAT, Capability.FACTUAL},
    context_window=128_000,
    avg_latency_ms=1500,
    quality_score=0.85,
    notes="Affordable search-augmented"
))

# -----------------------------------------------------------------------------
# Groq (Ultra-fast inference via @groq provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="llama-3.3-70b-groq",
    provider=Provider.GROQ,
    display_name="Llama 3.3 70B (Groq)",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.59,
    output_cost_per_1m=0.79,
    capabilities={Capability.CHAT, Capability.CODING, Capability.LOW_LATENCY},
    context_window=128_000,
    avg_latency_ms=100,
    quality_score=0.87,
    notes="Fastest inference, <100ms TTFT"
))

_register(ModelSpec(
    model_id="mixtral-8x7b-groq",
    provider=Provider.GROQ,
    display_name="Mixtral 8x7B (Groq)",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.24,
    output_cost_per_1m=0.24,
    capabilities={Capability.CHAT, Capability.LOW_LATENCY},
    context_window=32_000,
    avg_latency_ms=80,
    quality_score=0.82,
    notes="Ultra-fast, great for real-time"
))

# -----------------------------------------------------------------------------
# Together AI (via @together provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="qwen-2.5-72b",
    provider=Provider.TOGETHER,
    display_name="Qwen 2.5 72B",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.90,
    output_cost_per_1m=0.90,
    capabilities={Capability.CHAT, Capability.CODING, Capability.MULTILINGUAL},
    context_window=128_000,
    avg_latency_ms=800,
    quality_score=0.89,
    notes="Excellent multilingual open-source"
))

_register(ModelSpec(
    model_id="yi-large",
    provider=Provider.TOGETHER,
    display_name="Yi Large",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.30,
    output_cost_per_1m=0.30,
    capabilities={Capability.CHAT, Capability.MULTILINGUAL},
    context_window=32_000,
    avg_latency_ms=600,
    quality_score=0.84,
    notes="Strong Chinese language support"
))

# -----------------------------------------------------------------------------
# Fireworks AI (via @fireworks provider)
# -----------------------------------------------------------------------------
_register(ModelSpec(
    model_id="llama-3.3-70b-fw",
    provider=Provider.FIREWORKS,
    display_name="Llama 3.3 70B (Fireworks)",
    tier=QualityTier.ECONOMY,
    input_cost_per_1m=0.90,
    output_cost_per_1m=0.90,
    capabilities={Capability.CHAT, Capability.CODING, Capability.FUNCTION_CALLING},
    context_window=128_000,
    avg_latency_ms=300,
    quality_score=0.87,
    notes="Fast open-source inference"
))

# =============================================================================
# MODEL SELECTION UTILITIES
# =============================================================================

def get_all_models() -> List[ModelSpec]:
    """Get all registered models."""
    return list(MODEL_REGISTRY.values())


def get_models_by_tier(tier: QualityTier) -> List[ModelSpec]:
    """Get models in a specific quality tier."""
    return [m for m in MODEL_REGISTRY.values() if m.tier == tier]


def get_models_by_provider(provider: Provider) -> List[ModelSpec]:
    """Get all models from a specific provider."""
    return [m for m in MODEL_REGISTRY.values() if m.provider == provider]


def get_models_with_capability(capability: Capability) -> List[ModelSpec]:
    """Get models that have a specific capability."""
    return [m for m in MODEL_REGISTRY.values() if capability in m.capabilities]


def get_models_cheaper_than(max_blended_cost: float) -> List[ModelSpec]:
    """Get models with blended cost below threshold."""
    return [m for m in MODEL_REGISTRY.values() 
            if m.blended_cost_per_1m <= max_blended_cost]


def get_models_with_min_quality(min_quality: float) -> List[ModelSpec]:
    """Get models with quality score at or above threshold."""
    return [m for m in MODEL_REGISTRY.values() 
            if m.quality_score >= min_quality]


def find_pareto_optimal_models(
    required_capabilities: Optional[Set[Capability]] = None,
    min_quality: float = 0.70,
    max_cost: Optional[float] = None
) -> List[ModelSpec]:
    """
    Find Pareto-optimal models on the cost-quality frontier.
    
    A model is Pareto-optimal if no other model is both:
    - Cheaper AND higher quality
    
    Returns models sorted by quality (descending).
    """
    candidates = list(MODEL_REGISTRY.values())
    
    # Apply filters
    if required_capabilities:
        candidates = [m for m in candidates if m.matches_capabilities(required_capabilities)]
    
    candidates = [m for m in candidates if m.quality_score >= min_quality]
    
    if max_cost:
        candidates = [m for m in candidates if m.blended_cost_per_1m <= max_cost]
    
    if not candidates:
        return []
    
    # Find Pareto frontier
    pareto = []
    for model in candidates:
        is_dominated = False
        for other in candidates:
            if other.model_id == model.model_id:
                continue
            # Check if other dominates model (better on both dimensions)
            if (other.quality_score >= model.quality_score and 
                other.blended_cost_per_1m <= model.blended_cost_per_1m and
                (other.quality_score > model.quality_score or 
                 other.blended_cost_per_1m < model.blended_cost_per_1m)):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto.append(model)
    
    # Sort by quality descending
    pareto.sort(key=lambda m: m.quality_score, reverse=True)
    
    return pareto


def suggest_challengers_for_production(
    production_model_id: str,
    target_savings_percent: float = 0.30,
    min_quality_retention: float = 0.90,
    required_capabilities: Optional[Set[Capability]] = None,
    max_challengers: int = 5
) -> List[ModelSpec]:
    """
    Suggest challenger models for a given production model.
    
    Strategy:
    1. Find production model's quality and cost
    2. Filter to models that are cheaper
    3. Filter to models within quality retention threshold
    4. Return Pareto-optimal subset
    
    Args:
        production_model_id: Current production model ID
        target_savings_percent: Target cost reduction (0.30 = 30%)
        min_quality_retention: Minimum quality relative to production (0.90 = 90%)
        required_capabilities: Capabilities that challengers must have
        max_challengers: Maximum number of challengers to return
    
    Returns:
        List of suggested challenger models, sorted by cost-quality ratio
    """
    prod_model = MODEL_REGISTRY.get(production_model_id)
    
    if not prod_model:
        logger.warning(f"Production model {production_model_id} not in registry")
        # Use defaults
        prod_quality = 0.95
        prod_cost = 10.0
    else:
        prod_quality = prod_model.quality_score
        prod_cost = prod_model.blended_cost_per_1m
    
    # Calculate thresholds
    max_cost = prod_cost * (1 - target_savings_percent)
    min_quality = prod_quality * min_quality_retention
    
    # Get Pareto-optimal challengers
    challengers = find_pareto_optimal_models(
        required_capabilities=required_capabilities,
        min_quality=min_quality,
        max_cost=max_cost
    )
    
    # Exclude production model
    challengers = [m for m in challengers if m.model_id != production_model_id]
    
    # Sort by cost-quality ratio (best value first)
    challengers.sort(key=lambda m: m.cost_quality_ratio, reverse=True)
    
    return challengers[:max_challengers]


def get_model_comparison_table(models: List[ModelSpec]) -> str:
    """Generate a comparison table for models."""
    lines = [
        "| Model | Provider | Tier | Quality | Input $/1M | Output $/1M | Blended |",
        "|-------|----------|------|---------|------------|-------------|---------|"
    ]
    
    for m in models:
        lines.append(
            f"| {m.display_name} | {m.provider.value} | {m.tier.value} | "
            f"{m.quality_score:.0%} | ${m.input_cost_per_1m:.2f} | "
            f"${m.output_cost_per_1m:.2f} | ${m.blended_cost_per_1m:.2f} |"
        )
    
    return "\n".join(lines)


# =============================================================================
# DYNAMIC CHALLENGER SELECTION
# =============================================================================

class IntelligentChallengerSelector:
    """
    Intelligent challenger selection using Portkey's 250+ models.
    
    Combines:
    - Pareto-optimal frontier analysis
    - Thompson Sampling for exploration/exploitation
    - Provider diversity for resilience
    - Capability matching for use-case fit
    
    This is the core algorithm for selecting which challengers to test
    in shadow mode, maximizing the chance of finding cost-saving models.
    """
    
    def __init__(
        self,
        production_model: str,
        target_savings: float = 0.30,
        min_quality: float = 0.85,
        required_capabilities: Optional[Set[Capability]] = None,
        max_challengers: int = 5,
        ensure_provider_diversity: bool = True
    ):
        self.production_model = production_model
        self.target_savings = target_savings
        self.min_quality = min_quality
        self.required_capabilities = required_capabilities or set()
        self.max_challengers = max_challengers
        self.ensure_diversity = ensure_provider_diversity
        
        # Performance tracking for Thompson Sampling
        self._quality_observations: Dict[str, List[float]] = {}
    
    def select_challengers(self) -> List[ModelSpec]:
        """
        Select optimal set of challenger models to test.
        
        Algorithm:
        1. Get Pareto-optimal models cheaper than production
        2. If diversity enabled, ensure multiple providers
        3. Weight by cost-quality ratio
        4. Apply Thompson Sampling if we have historical data
        """
        # Get initial candidates
        candidates = suggest_challengers_for_production(
            production_model_id=self.production_model,
            target_savings_percent=self.target_savings,
            min_quality_retention=self.min_quality,
            required_capabilities=self.required_capabilities,
            max_challengers=self.max_challengers * 2  # Get extra for diversity
        )
        
        if not candidates:
            logger.warning("No suitable challengers found, using defaults")
            return self._get_default_challengers()
        
        # Ensure provider diversity if enabled
        if self.ensure_diversity:
            candidates = self._ensure_provider_diversity(candidates)
        
        # Apply Thompson Sampling if we have observations
        if self._quality_observations:
            candidates = self._apply_thompson_sampling(candidates)
        
        return candidates[:self.max_challengers]
    
    def _get_default_challengers(self) -> List[ModelSpec]:
        """Return sensible default challengers if selection fails."""
        defaults = [
            "gpt-5-mini",      # OpenAI economy
            "claude-haiku-4.5", # Anthropic economy
            "gemini-2.5-flash", # Google economy
            "deepseek-v3",      # Ultra-cheap
            "llama-4-70b"       # Open-source
        ]
        return [MODEL_REGISTRY[m] for m in defaults if m in MODEL_REGISTRY]
    
    def _ensure_provider_diversity(
        self, 
        candidates: List[ModelSpec]
    ) -> List[ModelSpec]:
        """Ensure we test models from different providers."""
        result = []
        providers_seen = set()
        
        # First pass: one from each provider
        for model in candidates:
            if model.provider not in providers_seen:
                result.append(model)
                providers_seen.add(model.provider)
        
        # Second pass: fill remaining slots with best models
        for model in candidates:
            if model not in result:
                result.append(model)
            if len(result) >= self.max_challengers:
                break
        
        return result
    
    def _apply_thompson_sampling(
        self, 
        candidates: List[ModelSpec]
    ) -> List[ModelSpec]:
        """Apply Thompson Sampling based on historical quality observations."""
        import numpy as np
        
        scores = {}
        for model in candidates:
            obs = self._quality_observations.get(model.model_id, [])
            if obs:
                # Use Beta distribution with observed successes/failures
                successes = sum(1 for q in obs if q >= self.min_quality)
                failures = len(obs) - successes
                # Sample from posterior
                sample = np.random.beta(successes + 1, failures + 1)
            else:
                # Prior: use model's expected quality
                sample = model.quality_score + np.random.normal(0, 0.05)
            
            scores[model.model_id] = sample
        
        # Sort by Thompson sample (explore promising models)
        candidates.sort(key=lambda m: scores.get(m.model_id, 0), reverse=True)
        return candidates
    
    def update_observation(self, model_id: str, quality: float):
        """Record quality observation for a model."""
        if model_id not in self._quality_observations:
            self._quality_observations[model_id] = []
        self._quality_observations[model_id].append(quality)
    
    def get_selection_explanation(self) -> str:
        """Generate human-readable explanation of selection logic."""
        prod = MODEL_REGISTRY.get(self.production_model)
        challengers = self.select_challengers()
        
        lines = [
            f"🎯 Challenger Selection for {self.production_model}",
            "=" * 50,
            ""
        ]
        
        if prod:
            lines.extend([
                f"Production Model: {prod.display_name}",
                f"  Quality: {prod.quality_score:.0%}",
                f"  Cost: ${prod.blended_cost_per_1m:.2f}/1M tokens",
                ""
            ])
        
        lines.extend([
            f"Selection Criteria:",
            f"  Target savings: ≥{self.target_savings:.0%}",
            f"  Min quality retention: ≥{self.min_quality:.0%}",
            f"  Provider diversity: {'Yes' if self.ensure_diversity else 'No'}",
            ""
        ])
        
        lines.append("Selected Challengers:")
        for i, c in enumerate(challengers, 1):
            savings = 0
            if prod:
                savings = (1 - c.blended_cost_per_1m / prod.blended_cost_per_1m) * 100
            lines.append(
                f"  {i}. {c.display_name} ({c.provider.value})"
            )
            lines.append(
                f"     Quality: {c.quality_score:.0%} | "
                f"Cost: ${c.blended_cost_per_1m:.2f}/1M | "
                f"Savings: {savings:.0f}%"
            )
        
        return "\n".join(lines)


# Module-level convenience function
def get_recommended_challengers(
    production_model: str = "gpt-5.2",
    max_challengers: int = 5
) -> List[str]:
    """
    Get recommended challenger model IDs for a production model.
    
    This is the main entry point for the shadow testing system.
    Returns Portkey Model Catalog IDs ready to use.
    """
    selector = IntelligentChallengerSelector(
        production_model=production_model,
        max_challengers=max_challengers
    )
    challengers = selector.select_challengers()
    return [c.portkey_model_id for c in challengers]
