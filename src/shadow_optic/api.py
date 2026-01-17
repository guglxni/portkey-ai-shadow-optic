"""
FastAPI Server for Shadow-Optic.

Provides REST API endpoints for:
- Triggering optimization workflows
- Querying workflow status
- Fetching reports
- Health checks

This server is optional - workflows can also be triggered
directly via Temporal client.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
from temporalio.client import Client

from shadow_optic.models import (
    ChallengerConfig,
    GetReportRequest,
    HealthCheckResponse,
    OptimizationReport,
    TriggerOptimizationRequest,
    TriggerOptimizationResponse,
    WorkflowConfig,
    SamplerConfig,
    EvaluatorConfig,
)
from shadow_optic.workflows import (
    OptimizeModelSelectionWorkflow,
    ScheduledOptimizationWorkflow,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Shadow-Optic API",
    description="Autonomous Cost-Quality Optimization Engine for LLMs",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Prometheus metrics instrumentation
# Exposes metrics at /metrics endpoint for Grafana scraping
Instrumentator().instrument(app).expose(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Temporal client (initialized on startup)
temporal_client: Optional[Client] = None


# =============================================================================
# Startup & Shutdown
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize Temporal client on startup."""
    global temporal_client
    
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    temporal_namespace = os.environ.get("TEMPORAL_NAMESPACE", "shadow-optic")
    
    try:
        temporal_client = await Client.connect(
            temporal_address,
            namespace=temporal_namespace
        )
        logger.info(f"Connected to Temporal at {temporal_address}")
    except Exception as e:
        logger.error(f"Failed to connect to Temporal: {e}")
        # Don't fail startup - allow health checks to report status


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global temporal_client
    if temporal_client:
        # Temporal client doesn't have explicit close
        temporal_client = None


# =============================================================================
# Health Endpoints
# =============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns status of all connected services.
    """
    temporal_ok = temporal_client is not None
    
    # Check Qdrant
    qdrant_ok = False
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", "6333")),
            timeout=5
        )
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass
    
    # Check Portkey
    portkey_ok = bool(os.environ.get("PORTKEY_API_KEY"))
    
    return HealthCheckResponse(
        status="healthy" if all([temporal_ok, qdrant_ok, portkey_ok]) else "degraded",
        version="0.1.0",
        temporal_connected=temporal_ok,
        qdrant_connected=qdrant_ok,
        portkey_connected=portkey_ok
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Shadow-Optic API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# =============================================================================
# Workflow Endpoints
# =============================================================================

class TriggerRequest(BaseModel):
    """Request to trigger an optimization run."""
    time_window: str = "24h"
    challenger_models: Optional[List[str]] = None
    max_samples: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "time_window": "24h",
                "challenger_models": ["deepseek-chat", "gemini-3-flash", "haiku-4.5"],
                "max_samples": 100
            }
        }


class TriggerResponse(BaseModel):
    """Response from triggering an optimization run."""
    workflow_id: str
    status: str
    message: str
    estimated_completion: str


@app.post("/api/v1/optimize", response_model=TriggerResponse)
async def trigger_optimization(request: TriggerRequest):
    """
    Trigger an optimization workflow.
    
    Starts a new optimization run that will:
    1. Export production logs from Portkey
    2. Sample golden prompts
    3. Replay against challenger models
    4. Evaluate with LLM-as-Judge
    5. Generate recommendations
    
    Returns immediately with workflow ID for status polling.
    """
    if not temporal_client:
        raise HTTPException(
            status_code=503,
            detail="Temporal client not connected"
        )
    
    # Build workflow config
    challengers = request.challenger_models or [
        "gpt-5-mini",
        "claude-haiku-4.5",
        "gemini-2.5-flash"
    ]
    
    config = WorkflowConfig(
        log_export_window=request.time_window,
        challengers=[
            ChallengerConfig(
                model_id=model,
                provider=_get_provider(model),
                provider_slug=_get_provider(model)  # Model Catalog uses provider slug
            )
            for model in challengers
        ],
        sampler=SamplerConfig(),
        evaluator=EvaluatorConfig()
    )
    
    # Generate workflow ID
    workflow_id = f"optimize-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    try:
        # Start workflow (non-blocking)
        handle = await temporal_client.start_workflow(
            OptimizeModelSelectionWorkflow.run,
            config,
            id=workflow_id,
            task_queue="shadow-optic"
        )
        
        logger.info(f"Started workflow: {workflow_id}")
        
        return TriggerResponse(
            workflow_id=workflow_id,
            status="started",
            message="Optimization workflow started successfully",
            estimated_completion=(
                datetime.utcnow() + timedelta(minutes=30)
            ).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to start workflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start workflow: {str(e)}"
        )


class WorkflowStatusResponse(BaseModel):
    """Workflow status response."""
    workflow_id: str
    status: str
    progress: Optional[float] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


@app.get("/api/v1/optimize/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get status of an optimization workflow.
    
    Poll this endpoint to track workflow progress.
    """
    if not temporal_client:
        raise HTTPException(
            status_code=503,
            detail="Temporal client not connected"
        )
    
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        
        # Try to get progress via query
        try:
            progress = await handle.query(
                OptimizeModelSelectionWorkflow.get_progress
            )
        except Exception:
            progress = None
        
        # Check if workflow is completed
        try:
            result = await asyncio.wait_for(
                handle.result(),
                timeout=0.1  # Quick check
            )
            return WorkflowStatusResponse(
                workflow_id=workflow_id,
                status="completed",
                progress=1.0,
                result=result.dict() if result else None
            )
        except asyncio.TimeoutError:
            # Still running
            return WorkflowStatusResponse(
                workflow_id=workflow_id,
                status="running",
                progress=progress.get("progress") if progress else None,
                stage=progress.get("stage") if progress else None,
                message=progress.get("message") if progress else None
            )
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(status_code=404, detail="Workflow not found")
            return WorkflowStatusResponse(
                workflow_id=workflow_id,
                status="failed",
                message=str(e)
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workflow status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting workflow status: {str(e)}"
        )


@app.delete("/api/v1/optimize/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """
    Cancel a running optimization workflow.
    """
    if not temporal_client:
        raise HTTPException(
            status_code=503,
            detail="Temporal client not connected"
        )
    
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        await handle.cancel()
        
        return {"message": f"Workflow {workflow_id} cancelled"}
        
    except Exception as e:
        logger.error(f"Error cancelling workflow: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error cancelling workflow: {str(e)}"
        )


# =============================================================================
# Report Endpoints
# =============================================================================

@app.get("/api/v1/reports")
async def list_reports(limit: int = 10):
    """
    List recent optimization reports.
    
    Returns summary of recent workflow executions.
    """
    if not temporal_client:
        raise HTTPException(
            status_code=503,
            detail="Temporal client not connected"
        )
    
    try:
        # List workflows
        workflows = []
        async for workflow in temporal_client.list_workflows(
            query="WorkflowType = 'OptimizeModelSelectionWorkflow'",
            page_size=limit
        ):
            workflows.append({
                "workflow_id": workflow.id,
                "status": workflow.status.name,
                "start_time": workflow.start_time.isoformat() if workflow.start_time else None,
                "close_time": workflow.close_time.isoformat() if workflow.close_time else None
            })
        
        return {"reports": workflows}
        
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing reports: {str(e)}"
        )


@app.get("/api/v1/reports/{workflow_id}")
async def get_report(workflow_id: str):
    """
    Get detailed report from a completed workflow.
    """
    if not temporal_client:
        raise HTTPException(
            status_code=503,
            detail="Temporal client not connected"
        )
    
    try:
        handle = temporal_client.get_workflow_handle(workflow_id)
        result = await handle.result()
        
        if result:
            return result.dict()
        else:
            raise HTTPException(status_code=404, detail="Report not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting report: {str(e)}"
        )


# =============================================================================
# Helper Functions
# =============================================================================

def _get_provider(model: str) -> str:
    """Get provider slug for a model ID (Model Catalog format)."""
    providers = {
        # OpenAI GPT-5.2 models (Latest Jan 17, 2026)
        "gpt-5.2": "openai",
        "gpt-5.2-pro": "openai",
        "gpt-5-mini": "openai",
        "gpt-4.1": "openai",
        "gpt-4.1-mini": "openai",
        "gpt-4.1-nano": "openai",
        # Anthropic Claude 4.5 models (Latest Jan 17, 2026)
        "claude-opus-4.5": "anthropic",
        "claude-sonnet-4.5": "anthropic",
        "claude-haiku-4.5": "anthropic",
        # Google Gemini 3/2.5 models (Latest Jan 17, 2026)
        "gemini-3-pro": "google",
        "gemini-3-flash": "google",
        "gemini-2.5-pro": "google",
        "gemini-2.5-flash": "google",
        # Meta Llama 4 models (Latest Jan 2026)
        "llama-4-405b": "bedrock",
        "llama-4-70b": "bedrock",
        "llama-4-8b": "bedrock",
        # Mistral 3 models (Latest Jan 2026)
        "mistral-large-3": "bedrock",
        "mistral-medium-3": "bedrock",
        # DeepSeek V3 models (Latest Jan 2026)
        "deepseek-v3": "bedrock",
        "deepseek-reasoner": "bedrock",
        # Grok 3 models (Latest Jan 2026)
        "grok-3": "grok",
        "grok-3-mini": "grok",
    }
    return providers.get(model, "openai")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "shadow_optic.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
