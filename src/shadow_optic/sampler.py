"""
Semantic Stratified Sampler using AsyncQdrantClient.

Production-grade implementation using:
- AsyncQdrantClient for vector operations
- AsyncPortkey for embeddings via OpenAI-compatible API
- K-Means clustering for semantic stratification

Achieves ~95% reduction in evaluation volume while maintaining
semantic coverage of production traffic.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from portkey_ai import AsyncPortkey
from qdrant_client import AsyncQdrantClient, models as qdrant_models
from sklearn.cluster import KMeans

from shadow_optic.models import GoldenPrompt, ProductionTrace, SampleType, SamplerConfig

logger = logging.getLogger(__name__)


class PromptTemplateExtractor:
    """
    Extract template patterns from prompts and cluster similar prompts.
    
    Identifies variable components (dates, numbers, etc.) and creates
    normalized templates for prompt deduplication.
    """
    
    # Patterns to identify variable components
    DATE_PATTERN = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
    NUMBER_PATTERN = re.compile(r'\b\d+\b')
    UUID_PATTERN = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self._variable_counter = 0
    
    def extract_template(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Extract a template from a prompt by replacing variable components.
        
        Returns:
            Tuple of (template_string, list_of_extracted_variables)
        """
        variables = []
        template = prompt
        
        # Extract UUIDs
        for match in self.UUID_PATTERN.finditer(template):
            variables.append(match.group())
        template = self.UUID_PATTERN.sub('{uuid}', template)
        
        # Extract emails
        for match in self.EMAIL_PATTERN.finditer(template):
            variables.append(match.group())
        template = self.EMAIL_PATTERN.sub('{email}', template)
        
        # Extract dates
        date_count = 0
        for match in self.DATE_PATTERN.finditer(template):
            variables.append(match.group())
            date_count += 1
        template = self.DATE_PATTERN.sub(lambda m: f'{{date_{self._get_counter()}}}', template)
        
        # Extract numbers (be more selective - only standalone numbers)
        for match in self.NUMBER_PATTERN.finditer(template):
            variables.append(match.group())
        template = self.NUMBER_PATTERN.sub('{number}', template)
        
        return template, variables
    
    def _get_counter(self) -> int:
        """Get and increment variable counter."""
        self._variable_counter += 1
        return self._variable_counter
    
    def similarity(self, template1: str, template2: str) -> float:
        """Calculate similarity between two templates."""
        return SequenceMatcher(None, template1, template2).ratio()
    
    def cluster_into_template_families(self, prompts: List[str]) -> Dict[str, List[str]]:
        """
        Cluster prompts into template families based on structural similarity.
        
        Returns:
            Dict mapping template pattern to list of matching prompts
        """
        families: Dict[str, List[str]] = {}
        
        for prompt in prompts:
            template, _ = self.extract_template(prompt)
            
            # Find matching family
            matched = False
            for existing_template in list(families.keys()):
                if self.similarity(template, existing_template) >= self.similarity_threshold:
                    families[existing_template].append(prompt)
                    matched = True
                    break
            
            if not matched:
                families[template] = [prompt]
        
        return families


@dataclass
class ClusterInfo:
    """Information about a semantic cluster."""
    cluster_id: int
    centroid_idx: int
    member_indices: List[int]
    size: int


class SemanticSampler:
    """
    Semantic stratified sampler using AsyncQdrantClient.
    
    Algorithm:
        1. Compute embeddings for all prompts via Portkey
        2. Store embeddings in Qdrant for persistence
        3. Cluster embeddings using K-Means
        4. For each cluster: select 1 centroid + 2 outliers
        5. Return golden prompts
    """
    
    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        portkey_client: AsyncPortkey,
        config: Optional[SamplerConfig] = None
    ):
        self.qdrant = qdrant_client
        self.portkey = portkey_client
        self.config = config or SamplerConfig()
        self._collection_initialized = False
    
    async def _ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        if self._collection_initialized:
            return
        
        collection_name = self.config.qdrant_collection
        
        try:
            exists = await self.qdrant.collection_exists(collection_name)
            
            if not exists:
                await self.qdrant.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=1536,
                        distance=qdrant_models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            
            self._collection_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise
    
    async def sample(
        self,
        traces: List[ProductionTrace],
        max_samples: Optional[int] = None,
        heartbeat_fn: Optional[Callable[[str], None]] = None
    ) -> List[GoldenPrompt]:
        """Select representative golden prompts from production traces."""
        if not traces:
            logger.warning("No traces provided for sampling")
            return []
        
        def heartbeat(msg: str):
            if heartbeat_fn:
                heartbeat_fn(msg)
        
        await self._ensure_collection()
        
        heartbeat("Computing embeddings...")
        logger.info(f"Computing embeddings for {len(traces)} traces")
        embeddings = await self._compute_embeddings([t.prompt for t in traces], heartbeat_fn)
        
        heartbeat("Storing embeddings in Qdrant...")
        await self._store_embeddings(traces, embeddings)
        
        heartbeat("Clustering embeddings...")
        n_clusters = min(self.config.n_clusters, len(traces) // 3)
        if n_clusters < 2:
            n_clusters = 2
        
        logger.info(f"Clustering into {n_clusters} clusters")
        clusters = self._cluster_embeddings(embeddings, n_clusters)
        
        heartbeat("Selecting golden prompts...")
        golden_prompts = self._sample_from_clusters(traces, embeddings, clusters)
        
        if max_samples and len(golden_prompts) > max_samples:
            golden_prompts = golden_prompts[:max_samples]
        
        sample_rate = len(golden_prompts) / len(traces) * 100
        logger.info(f"Selected {len(golden_prompts)} prompts ({sample_rate:.1f}%)")
        
        return golden_prompts
    
    async def _compute_embeddings(
        self,
        texts: List[str],
        heartbeat_fn: Optional[Callable[[str], None]] = None,
        batch_size: int = 100
    ) -> np.ndarray:
        """Compute embeddings using Portkey."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:8000] if len(t) > 8000 else t for t in batch]
            
            if heartbeat_fn:
                heartbeat_fn(f"Embeddings batch {i//batch_size + 1}")
            
            try:
                response = await self.portkey.embeddings.create(
                    input=batch,
                    model=self.config.embedding_model
                )
                batch_embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Embedding batch failed: {e}")
                all_embeddings.extend([[0.0] * 1536 for _ in batch])
            
            if i + batch_size < len(texts):
                await asyncio.sleep(0.1)
        
        return np.array(all_embeddings)
    
    async def _store_embeddings(
        self,
        traces: List[ProductionTrace],
        embeddings: np.ndarray
    ) -> None:
        """Store embeddings in Qdrant."""
        collection_name = self.config.qdrant_collection
        
        points = []
        for trace, embedding in zip(traces, embeddings):
            point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "trace_id": trace.trace_id,
                    "prompt": trace.prompt[:1000],
                    "model": trace.model,
                    "timestamp": trace.request_timestamp.isoformat(),
                    "cost": trace.cost
                }
            )
            points.append(point)
        
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                await self.qdrant.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            except Exception as e:
                logger.error(f"Qdrant upsert failed: {e}")
        
        logger.info(f"Stored {len(points)} embeddings in Qdrant")
    
    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
        n_clusters: Optional[int] = None
    ) -> List[ClusterInfo]:
        """Cluster embeddings using K-Means."""
        if n_clusters is None:
            # Auto-determine based on config and data size
            n_clusters = min(self.config.n_clusters, len(embeddings) // self.config.min_cluster_size)
            n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        # Ensure we don't have more clusters than samples
        n_clusters = min(n_clusters, len(embeddings))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        clusters = []
        for cluster_id in range(n_clusters):
            member_indices = np.where(labels == cluster_id)[0].tolist()
            
            if len(member_indices) < self.config.min_cluster_size:
                continue
            
            cluster_embeddings = embeddings[member_indices]
            centroid = centroids[cluster_id]
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            centroid_local_idx = int(np.argmin(distances))
            centroid_idx = member_indices[centroid_local_idx]
            
            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                centroid_idx=centroid_idx,
                member_indices=member_indices,
                size=len(member_indices)
            ))
        
        return clusters
    
    def _find_outliers(
        self,
        embeddings: np.ndarray,
        cluster: ClusterInfo,
        n_outliers: int = 2
    ) -> List[int]:
        """
        Find outliers within a cluster based on distance from centroid.
        
        Args:
            embeddings: Full embeddings array
            cluster: ClusterInfo object
            n_outliers: Number of outliers to find
            
        Returns:
            List of global indices for outlier points
        """
        if cluster.size <= 1:
            return []
        
        cluster_embeddings = embeddings[cluster.member_indices]
        centroid_embedding = embeddings[cluster.centroid_idx]
        
        # Calculate distances from centroid
        distances = np.linalg.norm(cluster_embeddings - centroid_embedding, axis=1)
        
        # Sort by distance (descending) to find furthest points
        sorted_local_indices = np.argsort(distances)[::-1]
        
        outliers = []
        for local_idx in sorted_local_indices:
            global_idx = cluster.member_indices[local_idx]
            # Don't include centroid as outlier
            if global_idx != cluster.centroid_idx:
                outliers.append(global_idx)
                if len(outliers) >= n_outliers:
                    break
        
        return outliers
    
    def _sample_from_clusters(
        self,
        traces: List[ProductionTrace],
        embeddings: np.ndarray,
        clusters: List[ClusterInfo]
    ) -> List[GoldenPrompt]:
        """Sample centroids and outliers from each cluster."""
        golden_prompts = []
        
        for cluster in clusters:
            if cluster.size == 0:
                continue
            
            cluster_embeddings = embeddings[cluster.member_indices]
            centroid_embedding = embeddings[cluster.centroid_idx]
            
            centroid_trace = traces[cluster.centroid_idx]
            golden_prompts.append(GoldenPrompt(
                original_trace_id=centroid_trace.trace_id,
                prompt=centroid_trace.prompt,
                production_response=centroid_trace.response,
                cluster_id=cluster.cluster_id,
                sample_type=SampleType.CENTROID,
                embedding=centroid_embedding.tolist()
            ))
            
            if cluster.size > 1:
                distances = np.linalg.norm(cluster_embeddings - centroid_embedding, axis=1)
                sorted_indices = np.argsort(distances)[::-1]
                # Subtract 1 for centroid
                n_outliers = min(self.config.samples_per_cluster - 1, cluster.size - 1)
                
                for j in range(n_outliers):
                    outlier_local_idx = sorted_indices[j]
                    outlier_idx = cluster.member_indices[outlier_local_idx]
                    
                    if outlier_idx == cluster.centroid_idx:
                        continue
                    
                    outlier_trace = traces[outlier_idx]
                    golden_prompts.append(GoldenPrompt(
                        original_trace_id=outlier_trace.trace_id,
                        prompt=outlier_trace.prompt,
                        production_response=outlier_trace.response,
                        cluster_id=cluster.cluster_id,
                        sample_type=SampleType.OUTLIER,
                        embedding=embeddings[outlier_idx].tolist()
                    ))
        
        return golden_prompts
    
    async def find_similar(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar prompts in the vector database."""
        await self._ensure_collection()
        
        response = await self.portkey.embeddings.create(
            input=[query_text],
            model=self.config.embedding_model
        )
        query_embedding = response.data[0].embedding
        
        results = await self.qdrant.query_points(
            collection_name=self.config.qdrant_collection,
            query=query_embedding,
            limit=limit,
            with_payload=True
        )
        
        return [
            {"score": p.score, "trace_id": p.payload.get("trace_id"), "prompt": p.payload.get("prompt")}
            for p in results.points
        ]
