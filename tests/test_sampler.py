"""
Tests for the Semantic Sampler.
"""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from shadow_optic.sampler import SemanticSampler, PromptTemplateExtractor
from shadow_optic.models import ProductionTrace, GoldenPrompt, SamplerConfig, SampleType


class TestSemanticSampler:
    """Tests for SemanticSampler class."""
    
    @pytest.fixture
    def mock_qdrant(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.get_collections.return_value = MagicMock(collections=[])
        return client
    
    @pytest.fixture
    def mock_portkey(self):
        """Create mock Portkey client."""
        client = MagicMock()
        return client
    
    @pytest.fixture
    def sample_traces(self):
        """Create sample production traces."""
        return [
            ProductionTrace(
                trace_id=f"trace_{i}",
                request_timestamp=datetime.now(timezone.utc),
                prompt=f"Test prompt {i}",
                response=f"Test response {i}",
                model="gpt-4",
                latency_ms=100.0,
                tokens_prompt=50,
                tokens_completion=100,
                cost=0.01
            )
            for i in range(100)
        ]
    
    def test_sampler_initialization(self, mock_qdrant, mock_portkey):
        """Test sampler initializes correctly."""
        sampler = SemanticSampler(
            qdrant_client=mock_qdrant,
            portkey_client=mock_portkey,
            config=SamplerConfig(n_clusters=10)
        )
        
        assert sampler.config.n_clusters == 10
        # Collection is created lazily, not on init
        assert sampler._collection_initialized is False
    
    def test_cluster_embeddings(self, mock_qdrant, mock_portkey):
        """Test embedding clustering."""
        sampler = SemanticSampler(
            qdrant_client=mock_qdrant,
            portkey_client=mock_portkey,
            config=SamplerConfig(n_clusters=5, min_cluster_size=2)
        )
        
        # Create test embeddings
        embeddings = np.random.randn(50, 1536)
        
        clusters = sampler._cluster_embeddings(embeddings)
        
        # Should have clusters
        assert len(clusters) > 0
        
        # Each cluster should have required attributes
        for cluster in clusters:
            assert cluster.cluster_id >= 0
            assert cluster.centroid_idx >= 0
            assert cluster.size >= 2
    
    def test_find_outliers(self, mock_qdrant, mock_portkey):
        """Test outlier detection in clusters."""
        sampler = SemanticSampler(
            qdrant_client=mock_qdrant,
            portkey_client=mock_portkey
        )
        
        # Create embeddings with clear outliers
        embeddings = np.array([
            [0, 0],  # Center
            [0.1, 0.1],  # Close to center
            [0.2, 0.2],  # Close to center
            [5, 5],  # Outlier
            [-5, -5],  # Outlier
        ])
        
        from shadow_optic.sampler import ClusterInfo
        cluster = ClusterInfo(
            cluster_id=0,
            centroid_idx=0,
            member_indices=[0, 1, 2, 3, 4],
            size=5
        )
        
        outliers = sampler._find_outliers(embeddings, cluster, n_outliers=2)
        
        # Should find the actual outliers (indices 3 and 4)
        assert 3 in outliers or 4 in outliers
        assert len(outliers) == 2
    
    @pytest.mark.asyncio
    async def test_sample_empty_traces(self, mock_qdrant, mock_portkey):
        """Test sampling with empty trace list."""
        sampler = SemanticSampler(
            qdrant_client=mock_qdrant,
            portkey_client=mock_portkey
        )
        
        result = await sampler.sample([])
        
        assert result == []


class TestPromptTemplateExtractor:
    """Tests for PromptTemplateExtractor class."""
    
    def test_extract_date_variables(self):
        """Test extraction of date patterns."""
        extractor = PromptTemplateExtractor()
        
        template, variables = extractor.extract_template(
            "Show me data from 2024-01-15 to 2024-01-20"
        )
        
        assert "2024-01-15" in variables
        assert "2024-01-20" in variables
        assert "{date_" in template
    
    def test_extract_number_variables(self):
        """Test extraction of number patterns."""
        extractor = PromptTemplateExtractor()
        
        template, variables = extractor.extract_template(
            "Give me the top 10 results for category 5"
        )
        
        assert "10" in variables
        assert "5" in variables
    
    def test_cluster_similar_templates(self):
        """Test clustering prompts into template families."""
        extractor = PromptTemplateExtractor(similarity_threshold=0.7)
        
        prompts = [
            "Write a poem about cats",
            "Write a poem about dogs",
            "Write a poem about birds",
            "Explain quantum physics",
            "Explain machine learning",
        ]
        
        families = extractor.cluster_into_template_families(prompts)
        
        # Should group poem prompts together
        assert len(families) >= 2
        
        # Find the poem family
        poem_family = None
        for template, instances in families.items():
            if "poem" in template.lower():
                poem_family = instances
                break
        
        assert poem_family is not None
        assert len(poem_family) >= 2


class TestSamplerIntegration:
    """Integration tests for the sampler."""
    
    @pytest.fixture
    def realistic_embeddings(self):
        """Create realistic embeddings with clusters."""
        np.random.seed(42)
        
        # Create 3 distinct clusters
        cluster1 = np.random.randn(30, 1536) + np.array([5] * 1536)
        cluster2 = np.random.randn(30, 1536) + np.array([-5] * 1536)
        cluster3 = np.random.randn(30, 1536)
        
        return np.vstack([cluster1, cluster2, cluster3])
    
    def test_cluster_count_adjustment(self, realistic_embeddings):
        """Test that cluster count adjusts for small datasets."""
        mock_qdrant = MagicMock()
        mock_qdrant.get_collections.return_value = MagicMock(collections=[])
        mock_portkey = MagicMock()
        
        sampler = SemanticSampler(
            qdrant_client=mock_qdrant,
            portkey_client=mock_portkey,
            config=SamplerConfig(n_clusters=100)  # More than data points
        )
        
        # With 90 points and min 3 per cluster, max is 30 clusters
        clusters = sampler._cluster_embeddings(realistic_embeddings)
        
        assert len(clusters) <= 30
