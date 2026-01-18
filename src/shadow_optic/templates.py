"""
Prompt Template Extraction and Clustering.

Provides two implementations:
1. PromptTemplateExtractor: Legacy regex-based extraction (fast, simple)
2. GenieTemplateExtractor: PromptGenie-powered semantic extraction (intelligent)

The Genie extractor integrates with the PromptGenie pipeline for:
- Semantic clustering using SentenceTransformers embeddings
- LLM-powered template extraction via Portkey
- Agentic metadata for intelligent routing
"""

import asyncio
import logging
import os
import re
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from shadow_optic.models import GoldenPrompt, ProductionTrace, SampleType, GenieConfig

logger = logging.getLogger(__name__)


@dataclass
class PromptInstance:
    """A specific instance of a prompt template."""
    prompt: str
    variables: List[str]
    trace: ProductionTrace


class PromptTemplateExtractor:
    """
    Extracts prompt templates from production logs.
    
    Example:
    Input prompts:
    - "Write a poem about cats"
    - "Write a poem about dogs"
    
    Extracted template:
    - "Write a poem about {var_0}"
    - Variables: ["cats", "dogs"]
    """
    
    VARIABLE_PATTERNS = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # Dates
        r'\b\d+\b',  # Numbers
        r'"[^"]*"',  # Quoted strings
        r"'[^']*'",  # Single-quoted strings
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns (simple heuristic)
    ]
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def extract_template(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Extract template and variables from a prompt.
        
        Returns: (template_string, list_of_variable_values)
        """
        variables = []
        template = prompt
        
        # Replace variable patterns with placeholders
        for i, pattern in enumerate(self.VARIABLE_PATTERNS):
            matches = re.findall(pattern, template)
            for match in matches:
                # Avoid re-replacing already replaced placeholders
                if match in variables: 
                    continue
                    
                variables.append(match)
                # Escape the match for regex replacement to handle special chars
                escaped_match = re.escape(match)
                template = re.sub(escaped_match, f"{{var_{len(variables)-1}}}", template, count=1)
        
        return template, variables
    
    def cluster_into_template_families(
        self, 
        traces: List[ProductionTrace]
    ) -> Dict[str, List[PromptInstance]]:
        """
        Cluster traces into template families.
        
        Algorithm:
        1. Extract template for each prompt
        2. Group templates by similarity
        3. Merge similar templates into families
        """
        families = defaultdict(list)
        
        for trace in traces:
            template, variables = self.extract_template(trace.prompt)
            
            # Find matching family
            matched = False
            for family_template in list(families.keys()):
                # Quick length check optimization
                if abs(len(template) - len(family_template)) / max(len(template), len(family_template)) > (1 - self.similarity_threshold):
                    continue
                    
                similarity = SequenceMatcher(
                    None, template, family_template
                ).ratio()
                
                if similarity >= self.similarity_threshold:
                    families[family_template].append(
                        PromptInstance(prompt=trace.prompt, variables=variables, trace=trace)
                    )
                    matched = True
                    break
            
            if not matched:
                families[template].append(
                    PromptInstance(prompt=trace.prompt, variables=variables, trace=trace)
                )
        
        return families
    
    def select_test_cases(
        self, 
        family: List[PromptInstance],
        n_cases: int = 3
    ) -> List[PromptInstance]:
        """
        Select diverse test cases from a template family.
        
        Strategy:
        1. Include shortest and longest variable combinations
        2. Random sample for diversity
        """
        if len(family) <= n_cases:
            return family
        
        selected = []
        
        # Shortest prompt (often simplest case)
        selected.append(min(family, key=lambda x: len(x.prompt)))
        
        # Longest prompt (often most complex case)
        longest = max(family, key=lambda x: len(x.prompt))
        if longest not in selected:
            selected.append(longest)
        
        # Fill remaining with random samples
        remaining = [p for p in family if p not in selected]
        if remaining:
            needed = n_cases - len(selected)
            if needed > 0:
                selected.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return selected

    def sample_with_templates(
        self,
        traces: List[ProductionTrace],
        samples_per_template: int = 3
    ) -> List[GoldenPrompt]:
        """
        Main entry point: Sample golden prompts using template extraction.
        """
        families = self.cluster_into_template_families(traces)
        golden_prompts = []
        
        for template, instances in families.items():
            test_cases = self.select_test_cases(instances, n_cases=samples_per_template)
            
            for case in test_cases:
                golden_prompts.append(GoldenPrompt(
                    original_trace_id=case.trace.trace_id,
                    prompt=case.trace.prompt,
                    production_response=case.trace.response,
                    sample_type=SampleType.CENTROID, # Using CENTROID as generic type here
                    template=template,
                    variable_values=case.variables
                ))
                
        return golden_prompts


# =============================================================================
# PromptGenie Integration: GenieTemplateExtractor
# =============================================================================

class GenieTemplateExtractor:
    """
    PromptGenie-powered semantic template extraction adapter.
    
    Replaces the legacy regex-based PromptTemplateExtractor with intelligent
    semantic clustering and LLM-powered template extraction.
    
    Features:
    - Semantic clustering using SentenceTransformers embeddings (local FOSS)
    - LLM-powered Jinja2 template extraction via Portkey
    - Agentic metadata for intelligent routing decisions
    - Drift detection for evolving prompts
    
    Integration Pattern:
    1. Convert ProductionTrace objects to PromptGenie's AgenticPrompt format
    2. Process through PromptGenie pipeline (embedding, clustering, extraction)
    3. Map results back to Shadow-Optic's GoldenPrompt format
    
    Example:
        extractor = GenieTemplateExtractor(genie_config)
        await extractor.initialize()
        golden_prompts = await extractor.sample_with_templates(traces)
    """
    
    def __init__(self, config: GenieConfig):
        """Initialize the Genie adapter with configuration.
        
        Args:
            config: GenieConfig with Qdrant, LLM, and processing settings
        """
        self.config = config
        self._pipeline = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the PromptGenie pipeline.
        
        Lazy loads PromptGenie components to avoid import errors
        when PromptGenie is not installed.
        """
        if self._initialized:
            return
            
        try:
            # Import PromptGenie components
            from prompt_genie.config import Config as GenieFullConfig
            from prompt_genie.config import (
                PortkeyConfig, 
                QdrantConfig, 
                ProcessingConfig, 
                StorageConfig
            )
            from prompt_genie.core.pipeline import PromptGeniePipeline
            
            # Build PromptGenie config from Shadow-Optic's GenieConfig
            genie_full_config = GenieFullConfig(
                portkey=PortkeyConfig(
                    api_key=os.environ.get("PORTKEY_API_KEY", ""),
                    openai_provider_slug=os.environ.get("PORTKEY_OPENAI_PROVIDER_SLUG", "openai"),
                    anthropic_provider_slug=os.environ.get("PORTKEY_ANTHROPIC_PROVIDER_SLUG"),
                ),
                qdrant=QdrantConfig(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    collection=self.config.qdrant_collection,
                    api_key=self.config.qdrant_api_key,
                ),
                processing=ProcessingConfig(
                    similarity_threshold=self.config.similarity_threshold,
                    min_cluster_size=self.config.min_cluster_size,
                    extraction_model=self.config.extraction_model,
                    fallback_model=self.config.fallback_model,
                    enforce_privacy=self.config.enforce_privacy,
                ),
                storage=StorageConfig(
                    sqlite_path=self.config.sqlite_path,
                ),
            )
            
            self._pipeline = PromptGeniePipeline(genie_full_config)
            await self._pipeline.initialize()
            self._initialized = True
            
            logger.info(
                f"GenieTemplateExtractor initialized "
                f"(Qdrant: {self.config.qdrant_host}:{self.config.qdrant_port}, "
                f"Collection: {self.config.qdrant_collection})"
            )
            
        except ImportError as e:
            logger.error(
                f"Failed to import PromptGenie. Ensure it's installed: {e}. "
                "Falling back to legacy extractor."
            )
            raise RuntimeError(
                "PromptGenie is not installed. Install with: "
                "pip install -e ../prompt_genie"
            ) from e
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the pipeline."""
        if self._pipeline:
            await self._pipeline.shutdown()
            self._initialized = False
    
    def _trace_to_agentic_prompt(self, trace: ProductionTrace) -> "AgenticPrompt":
        """Convert Shadow-Optic ProductionTrace to PromptGenie AgenticPrompt.
        
        Maps trace metadata to agentic metadata for intelligent routing.
        """
        from prompt_genie.core.models import AgenticPrompt, AgenticMetadata
        
        # Determine sensitivity from trace metadata
        sensitivity = trace.metadata.get("sensitivity", "internal")
        if sensitivity not in ["public", "internal", "restricted"]:
            sensitivity = "internal"
        
        # Extract intent tags from metadata - ensure it's always a list
        raw_intent_tags = trace.metadata.get("intent_tags", [])
        if isinstance(raw_intent_tags, list):
            intent_tags = [str(tag) for tag in raw_intent_tags]
        elif isinstance(raw_intent_tags, str):
            intent_tags = [raw_intent_tags]
        else:
            intent_tags = []
        
        # Determine priority (higher cost traces get higher priority)
        priority = min(100, int(trace.cost * 1000) + 50)
        
        return AgenticPrompt(
            id=trace.trace_id,
            text=trace.prompt,
            timestamp=trace.request_timestamp.isoformat(),
            source="shadow-optic",
            metadata=AgenticMetadata(
                sensitivity=sensitivity,
                priority=priority,
                intent_tags=intent_tags,
                processing_flags={"cache": True},
                state="NEW"
            )
        )
    
    async def _process_trace(
        self, 
        trace: ProductionTrace
    ) -> Tuple[ProductionTrace, Optional[dict]]:
        """Process a single trace through PromptGenie pipeline.
        
        Returns:
            Tuple of (original_trace, processing_result_dict)
        """
        try:
            agentic_prompt = self._trace_to_agentic_prompt(trace)
            result = await self._pipeline.process_single(agentic_prompt)
            
            return (trace, {
                "cluster_id": result.cluster_id,
                "action": result.action,
                "similarity_score": result.similarity_score,
                "template_triggered": result.template_triggered,
                "routing_decision": result.routing_decision,
            })
            
        except Exception as e:
            logger.warning(f"Failed to process trace {trace.trace_id}: {e}")
            return (trace, None)
    
    async def sample_with_templates(
        self,
        traces: List[ProductionTrace],
        samples_per_cluster: int = 3
    ) -> List[GoldenPrompt]:
        """
        Main entry point: Sample golden prompts using PromptGenie's semantic clustering.
        
        Algorithm:
        1. Convert all traces to AgenticPrompts
        2. Process through PromptGenie pipeline (parallel)
        3. Group results by cluster_id
        4. Select diverse samples from each cluster
        5. Enrich with Genie metadata
        
        Args:
            traces: Production traces from Portkey
            samples_per_cluster: Number of samples to select per semantic cluster
            
        Returns:
            List of GoldenPrompt objects enriched with Genie fields
        """
        await self.initialize()
        
        logger.info(f"Processing {len(traces)} traces through PromptGenie pipeline")
        
        # Process all traces in parallel (with concurrency limit)
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        
        async def process_with_limit(trace):
            async with semaphore:
                return await self._process_trace(trace)
        
        results = await asyncio.gather(
            *[process_with_limit(t) for t in traces],
            return_exceptions=True
        )
        
        # Group by cluster
        clusters: Dict[str, List[Tuple[ProductionTrace, dict]]] = defaultdict(list)
        skipped = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Processing exception: {result}")
                skipped += 1
                continue
                
            trace, genie_result = result
            if genie_result is None:
                skipped += 1
                continue
                
            cluster_id = genie_result.get("cluster_id", "unknown")
            clusters[cluster_id].append((trace, genie_result))
        
        logger.info(
            f"Clustered into {len(clusters)} semantic clusters "
            f"(processed: {len(traces) - skipped}, skipped: {skipped})"
        )
        
        # Get templates for each cluster (if available)
        cluster_templates = await self._get_cluster_templates(list(clusters.keys()))
        
        # Select samples from each cluster
        golden_prompts = []
        
        for cluster_id, items in clusters.items():
            # Sort by similarity score (centroids first)
            items.sort(key=lambda x: x[1].get("similarity_score", 0), reverse=True)
            
            # Select diverse samples
            selected = self._select_diverse_samples(items, samples_per_cluster)
            
            # Get template for this cluster
            template_info = cluster_templates.get(cluster_id, {})
            
            for trace, genie_result in selected:
                # Determine sample type based on position
                sample_type = SampleType.CENTROID
                if genie_result.get("similarity_score", 1.0) < 0.7:
                    sample_type = SampleType.OUTLIER
                elif items.index((trace, genie_result)) > 0:
                    sample_type = SampleType.RANDOM
                
                golden_prompts.append(GoldenPrompt(
                    original_trace_id=trace.trace_id,
                    prompt=trace.prompt,
                    production_response=trace.response,
                    cluster_id=hash(cluster_id) % 1000000,  # Convert string to int
                    sample_type=sample_type,
                    template=template_info.get("template"),
                    variable_values=template_info.get("variables", []),
                    # Genie-specific fields
                    genie_cluster_id=cluster_id,
                    genie_confidence=template_info.get("confidence"),
                    genie_template=template_info.get("template"),
                    genie_routing_decision=genie_result.get("routing_decision"),
                    genie_intent_tags=trace.metadata.get("intent_tags"),
                ))
        
        logger.info(f"Selected {len(golden_prompts)} golden prompts via PromptGenie")
        
        return golden_prompts
    
    async def _get_cluster_templates(
        self, 
        cluster_ids: List[str]
    ) -> Dict[str, dict]:
        """Fetch templates for clusters from PromptGenie's state store.
        
        Returns:
            Dict mapping cluster_id to template info (template, variables, confidence)
        """
        templates = {}
        
        if not self._pipeline:
            return templates
        
        try:
            # Query state store for cluster templates
            for cluster_id in cluster_ids:
                try:
                    state = await self._pipeline.state_store.get_cluster_state(cluster_id)
                    if state and hasattr(state, 'template'):
                        templates[cluster_id] = {
                            "template": getattr(state, 'template', None),
                            "variables": getattr(state, 'variables', []),
                            "confidence": getattr(state, 'confidence', None),
                        }
                except Exception:
                    # Cluster may not have template yet
                    pass
                    
        except Exception as e:
            logger.warning(f"Failed to fetch cluster templates: {e}")
        
        return templates
    
    def _select_diverse_samples(
        self,
        items: List[Tuple[ProductionTrace, dict]],
        n_samples: int
    ) -> List[Tuple[ProductionTrace, dict]]:
        """Select diverse samples from a cluster.
        
        Strategy:
        1. Include highest similarity (centroid representative)
        2. Include lowest similarity (boundary case)
        3. Random samples for diversity
        """
        if len(items) <= n_samples:
            return items
        
        selected = []
        
        # Centroid (highest similarity)
        selected.append(items[0])
        
        # Boundary (lowest similarity)
        if len(items) > 1:
            selected.append(items[-1])
        
        # Random samples
        remaining = [x for x in items if x not in selected]
        if remaining and len(selected) < n_samples:
            needed = n_samples - len(selected)
            selected.extend(random.sample(remaining, min(needed, len(remaining))))
        
        return selected


# =============================================================================
# Factory Function
# =============================================================================

def create_template_extractor(
    genie_config: Optional[GenieConfig] = None
) -> PromptTemplateExtractor:
    """
    Factory function to create the appropriate template extractor.
    
    If GenieConfig is provided and enabled, returns GenieTemplateExtractor.
    Otherwise, returns the legacy PromptTemplateExtractor.
    
    Note: GenieTemplateExtractor is async and requires explicit initialization.
    
    Args:
        genie_config: Optional GenieConfig for PromptGenie integration
        
    Returns:
        PromptTemplateExtractor (sync) or GenieTemplateExtractor (async)
    """
    if genie_config and genie_config.enabled:
        logger.info("Using GenieTemplateExtractor (PromptGenie integration)")
        return GenieTemplateExtractor(genie_config)
    else:
        logger.info("Using legacy PromptTemplateExtractor (regex-based)")
        return PromptTemplateExtractor()
