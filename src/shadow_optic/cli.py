"""
Command Line Interface for Shadow-Optic.

Provides CLI commands for:
- Triggering optimization runs
- Checking workflow status
- Viewing reports
- Managing configuration

Usage:
    shadow-optic optimize --time-window 24h
    shadow-optic status <workflow_id>
    shadow-optic report <workflow_id>
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional

import click
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Shadow-Optic: Autonomous Cost-Quality Optimization Engine"""
    pass


@cli.command()
@click.option("--time-window", "-t", default="24h", help="Time window for log export (e.g., 24h, 12h, 7d)")
@click.option("--challengers", "-c", multiple=True, default=["deepseek-chat", "gemini-3-flash", "haiku-4.5"], help="Challenger models to test")
@click.option("--max-samples", "-m", type=int, default=None, help="Maximum samples to evaluate")
@click.option("--wait", "-w", is_flag=True, help="Wait for workflow completion")
def optimize(time_window: str, challengers: tuple, max_samples: Optional[int], wait: bool):
    """Trigger an optimization workflow."""
    asyncio.run(_optimize(time_window, list(challengers), max_samples, wait))


async def _optimize(time_window: str, challengers: list, max_samples: Optional[int], wait: bool):
    """Execute optimization workflow."""
    from temporalio.client import Client
    from shadow_optic.models import WorkflowConfig, ChallengerConfig, SamplerConfig, EvaluatorConfig
    from shadow_optic.workflows import OptimizeModelSelectionWorkflow
    
    click.echo("🔮 Shadow-Optic Optimization")
    click.echo("=" * 50)
    
    # Connect to Temporal
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    click.echo(f"Connecting to Temporal at {temporal_address}...")
    
    try:
        client = await Client.connect(temporal_address)
    except Exception as e:
        click.echo(f"❌ Failed to connect to Temporal: {e}", err=True)
        sys.exit(1)
    
    click.echo("✅ Connected to Temporal")
    
    # Build config
    config = WorkflowConfig(
        log_export_window=time_window,
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
    
    click.echo(f"\n📋 Configuration:")
    click.echo(f"   Time window: {time_window}")
    click.echo(f"   Challengers: {', '.join(challengers)}")
    click.echo(f"   Workflow ID: {workflow_id}")
    
    # Start workflow
    click.echo(f"\n🚀 Starting optimization workflow...")
    
    try:
        handle = await client.start_workflow(
            OptimizeModelSelectionWorkflow.run,
            config,
            id=workflow_id,
            task_queue="shadow-optic"
        )
        
        click.echo(f"✅ Workflow started: {workflow_id}")
        
        if wait:
            click.echo("\n⏳ Waiting for completion...")
            
            with click.progressbar(length=100, label="Progress") as bar:
                last_progress = 0
                while True:
                    try:
                        progress = await handle.query(
                            OptimizeModelSelectionWorkflow.get_progress
                        )
                        current = int(progress.get("progress", 0) * 100)
                        bar.update(current - last_progress)
                        last_progress = current
                        
                        if current >= 100:
                            break
                    except Exception:
                        pass
                    
                    await asyncio.sleep(2)
            
            result = await handle.result()
            _print_report(result)
        else:
            click.echo(f"\n💡 Track progress: shadow-optic status {workflow_id}")
            
    except Exception as e:
        click.echo(f"❌ Failed to start workflow: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("workflow_id")
def status(workflow_id: str):
    """Check status of an optimization workflow."""
    asyncio.run(_status(workflow_id))


async def _status(workflow_id: str):
    """Get workflow status."""
    from temporalio.client import Client
    from shadow_optic.workflows import OptimizeModelSelectionWorkflow
    
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    
    try:
        client = await Client.connect(temporal_address)
    except Exception as e:
        click.echo(f"❌ Failed to connect to Temporal: {e}", err=True)
        sys.exit(1)
    
    try:
        handle = client.get_workflow_handle(workflow_id)
        
        # Try to get progress
        try:
            progress = await handle.query(OptimizeModelSelectionWorkflow.get_progress)
            click.echo(f"\n📊 Workflow Status: {workflow_id}")
            click.echo("=" * 50)
            click.echo(f"   Progress: {progress.get('progress', 0) * 100:.0f}%")
            click.echo(f"   Stage: {progress.get('stage', 'unknown')}")
            click.echo(f"   Message: {progress.get('message', '')}")
        except Exception:
            click.echo(f"\n📊 Workflow Status: {workflow_id}")
            click.echo("=" * 50)
            click.echo("   Progress: Unknown (workflow may be completed or failed)")
        
        # Check if completed
        try:
            result = await asyncio.wait_for(handle.result(), timeout=0.5)
            click.echo(f"   Status: ✅ COMPLETED")
            click.echo(f"\n💡 View report: shadow-optic report {workflow_id}")
        except asyncio.TimeoutError:
            click.echo(f"   Status: 🔄 RUNNING")
        except Exception as e:
            click.echo(f"   Status: ❌ FAILED - {e}")
            
    except Exception as e:
        click.echo(f"❌ Workflow not found: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("workflow_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def report(workflow_id: str, as_json: bool):
    """View report from a completed workflow."""
    asyncio.run(_report(workflow_id, as_json))


async def _report(workflow_id: str, as_json: bool):
    """Get workflow report."""
    from temporalio.client import Client
    
    temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
    
    try:
        client = await Client.connect(temporal_address)
    except Exception as e:
        click.echo(f"❌ Failed to connect to Temporal: {e}", err=True)
        sys.exit(1)
    
    try:
        handle = client.get_workflow_handle(workflow_id)
        result = await handle.result()
        
        if as_json:
            click.echo(json.dumps(result.dict(), indent=2, default=str))
        else:
            _print_report(result)
            
    except Exception as e:
        click.echo(f"❌ Failed to get report: {e}", err=True)
        sys.exit(1)


def _print_report(report):
    """Print formatted report."""
    click.echo("\n" + "=" * 60)
    click.echo("🔮 SHADOW-OPTIC OPTIMIZATION REPORT")
    click.echo("=" * 60)
    
    click.echo(f"\n📅 Generated: {report.generated_at}")
    click.echo(f"📊 Time Window: {report.time_window}")
    click.echo(f"📈 Traces Analyzed: {report.traces_analyzed}")
    click.echo(f"🎯 Samples Evaluated: {report.samples_evaluated}")
    
    click.echo("\n" + "-" * 60)
    click.echo("CHALLENGER RESULTS")
    click.echo("-" * 60)
    
    for model, result in report.challenger_results.items():
        status = "✅ PASSED" if result.passed_thresholds else "❌ FAILED"
        click.echo(f"\n{model} {status}")
        click.echo(f"   Faithfulness: {result.avg_faithfulness:.2f}")
        click.echo(f"   Quality: {result.avg_quality:.2f}")
        click.echo(f"   Conciseness: {result.avg_conciseness:.2f}")
        click.echo(f"   Refusal Rate: {result.refusal_rate:.2%}")
        
        if result.failure_reasons:
            for reason in result.failure_reasons:
                click.echo(f"   ⚠️  {reason}")
    
    click.echo("\n" + "-" * 60)
    click.echo("RECOMMENDATION")
    click.echo("-" * 60)
    
    rec = report.recommendation
    
    action_emoji = {
        "switch_recommended": "✅",
        "continue_monitoring": "🔄",
        "no_viable_challenger": "❌",
        "insufficient_data": "⚠️"
    }
    
    click.echo(f"\n{action_emoji.get(rec.action.value, '❓')} {rec.action.value.upper()}")
    
    if rec.challenger:
        click.echo(f"Best Challenger: {rec.challenger}")
    
    click.echo(f"Confidence: {rec.confidence:.0%}")
    click.echo(f"\nReasoning:\n{rec.reasoning}")
    
    if rec.suggested_actions:
        click.echo("\nSuggested Actions:")
        for action in rec.suggested_actions:
            click.echo(f"   • {action}")
    
    if rec.cost_analysis:
        click.echo("\n" + "-" * 60)
        click.echo("COST ANALYSIS")
        click.echo("-" * 60)
        ca = rec.cost_analysis
        click.echo(f"   Production cost: ${ca.production_cost_per_1k:.2f}/1K requests")
        click.echo(f"   Challenger cost: ${ca.challenger_cost_per_1k:.2f}/1K requests")
        click.echo(f"   Savings: {ca.savings_percent:.0%}")
        click.echo(f"   Monthly savings: ${ca.estimated_monthly_savings:,.0f}")
    
    click.echo("\n" + "=" * 60)


@cli.command()
def worker():
    """Start the Temporal worker."""
    click.echo("🚀 Starting Shadow-Optic Worker...")
    
    # Import and run worker
    from shadow_optic.worker import main
    asyncio.run(main())


@cli.command()
@click.option("--host", default="0.0.0.0", help="API server host")
@click.option("--port", default=8000, type=int, help="API server port")
def serve(host: str, port: int):
    """Start the API server."""
    click.echo(f"🚀 Starting Shadow-Optic API at http://{host}:{port}")
    
    import uvicorn
    uvicorn.run(
        "shadow_optic.api:app",
        host=host,
        port=port,
        reload=False
    )


@cli.command()
def health():
    """Check health of all services."""
    asyncio.run(_health())


async def _health():
    """Check service health."""
    click.echo("\n🏥 Shadow-Optic Health Check")
    click.echo("=" * 50)
    
    # Check Temporal
    try:
        from temporalio.client import Client
        temporal_address = os.environ.get("TEMPORAL_ADDRESS", "localhost:7233")
        client = await Client.connect(temporal_address)
        click.echo(f"✅ Temporal: Connected ({temporal_address})")
    except Exception as e:
        click.echo(f"❌ Temporal: {e}")
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
        qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
        client.get_collections()
        click.echo(f"✅ Qdrant: Connected ({qdrant_host}:{qdrant_port})")
    except Exception as e:
        click.echo(f"❌ Qdrant: {e}")
    
    # Check Portkey
    if os.environ.get("PORTKEY_API_KEY"):
        click.echo("✅ Portkey: API key configured")
    else:
        click.echo("❌ Portkey: API key not set")
    
    # Check OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        click.echo("✅ OpenAI: API key configured")
    else:
        click.echo("❌ OpenAI: API key not set (required for DeepEval)")
    
    click.echo()


def _get_provider(model: str) -> str:
    """Get provider for a model ID."""
    providers = {
        # DeepSeek models
        "deepseek-chat": "deepseek",
        "deepseek-reasoner": "deepseek",
        # Google models
        "gemini-3-pro": "google",
        "gemini-3-flash": "google",
        "gemini-2.5-flash": "google",
        "gemini-2.5-pro": "google",
        # Anthropic models
        "opus-4.5": "anthropic",
        "sonnet-4.5": "anthropic",
        "haiku-4.5": "anthropic",
        # OpenAI models
        "gpt-5.2": "openai",
        "gpt-5": "openai",
        "gpt-5-mini": "openai",
        "gpt-5-nano": "openai",
    }
    return providers.get(model, "openai")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
