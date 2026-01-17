"""
Temporal Worker for Shadow-Optic.

This module runs the Temporal worker that executes workflows and activities.
The worker connects to the Temporal server and polls for tasks.

Usage:
    python -m shadow_optic.worker
    
Environment Variables:
    TEMPORAL_ADDRESS: Temporal server address (default: localhost:7233)
    TEMPORAL_NAMESPACE: Temporal namespace (default: default)
    PORTKEY_API_KEY: Portkey API key (required)
    QDRANT_HOST: Qdrant server host (default: localhost)
    QDRANT_PORT: Qdrant server port (default: 6333)
    OPENAI_API_KEY: OpenAI API key for DeepEval (required)
"""

import asyncio
import logging
import os
import sys
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker

from shadow_optic.activities import (
    evaluate_shadow_results,
    export_portkey_logs,
    generate_optimization_report,
    replay_shadow_requests,
    sample_golden_prompts,
    submit_portkey_feedback,
)
from shadow_optic.workflows import (
    AdaptiveOptimizationWorkflow,
    OptimizeModelSelectionWorkflow,
    ScheduledOptimizationWorkflow,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# Task queue name
TASK_QUEUE = "shadow-optic"


async def main():
    """Start the Temporal worker."""
    
    # Validate required environment variables
    required_vars = ["PORTKEY_API_KEY", "OPENAI_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        sys.exit(1)
    
    # Connect to Temporal
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    temporal_namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    
    logger.info(f"Connecting to Temporal at {temporal_address}")
    
    try:
        client = await Client.connect(
            temporal_address,
            namespace=temporal_namespace
        )
    except Exception as e:
        logger.error(f"Failed to connect to Temporal: {e}")
        sys.exit(1)
    
    logger.info(f"Connected to Temporal namespace: {temporal_namespace}")
    
    # Create worker
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[
            OptimizeModelSelectionWorkflow,
            ScheduledOptimizationWorkflow,
            AdaptiveOptimizationWorkflow,
        ],
        activities=[
            export_portkey_logs,
            sample_golden_prompts,
            replay_shadow_requests,
            evaluate_shadow_results,
            generate_optimization_report,
            submit_portkey_feedback,
        ],
        # Worker configuration
        max_concurrent_activities=10,
        max_concurrent_workflow_tasks=10,
    )
    
    logger.info(f"Starting worker on task queue: {TASK_QUEUE}")
    
    try:
        await worker.run()
    except Exception as e:
        logger.error(f"Worker error: {e}")
        raise
    finally:
        logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
