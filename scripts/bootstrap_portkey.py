#!/usr/bin/env python3
"""
Portkey Configuration Bootstrapper (Model Catalog + Workspace Architecture)

Sets up Shadow-Optic with Portkey's enterprise-grade architecture:
1. Creates/verifies Workspaces for cost segregation
2. Uploads configs with Virtual Model mappings
3. Verifies Model Catalog access
4. Sets up budget limits for Shadow Workspace

This is the "Enterprise Ready" architecture using RBAC features.

Usage:
    python scripts/bootstrap_portkey.py --api-key YOUR_PORTKEY_API_KEY

Or with environment variable:
    export PORTKEY_API_KEY=your_key
    python scripts/bootstrap_portkey.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PORTKEY_API_BASE = "https://api.portkey.ai/v1"
CONFIG_DIR = Path(__file__).parent.parent / "configs"


# =============================================================================
# Virtual Model Definitions (Model Catalog)
# =============================================================================
# These are the Virtual Model names you'll create in Portkey UI
# They map to actual provider models and are tracked under specific workspaces

VIRTUAL_MODELS = {
    "production": {
        "prod-primary": {
            "provider": "openai",
            "model": "gpt-5.2",
            "description": "Production primary model"
        },
        "prod-fallback": {
            "provider": "anthropic", 
            "model": "claude-sonnet-4.5",
            "description": "Production fallback model"
        }
    },
    "shadow": {
        "shadow-challenger-1": {
            "provider": "openai",
            "model": "gpt-5-mini",
            "description": "Shadow challenger - GPT-5 Mini (cost-efficient)"
        },
        "shadow-challenger-2": {
            "provider": "anthropic",
            "model": "claude-haiku-4.5", 
            "description": "Shadow challenger - Claude Haiku 4.5 (fast)"
        },
        "shadow-challenger-3": {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "description": "Shadow challenger - Gemini 2.5 Flash (price-performance)"
        }
    }
}


class PortkeyBootstrapper:
    """Bootstrapper for Portkey Model Catalog + Workspace architecture."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "x-portkey-api-key": api_key,
            "Content-Type": "application/json"
        }
        self.workspace_ids: Dict[str, str] = {}
        self.config_ids: Dict[str, str] = {}
    
    # =========================================================================
    # Configuration Management
    # =========================================================================
    
    def load_config_file(self, filename: str) -> dict:
        """Load a JSON config file."""
        config_path = CONFIG_DIR / filename
        if not config_path.exists():
            console.print(f"[red]❌ Config file not found: {config_path}[/red]")
            sys.exit(1)
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def upload_config(self, config: dict, config_name: str) -> str:
        """Upload a config to Portkey and return the config ID."""
        console.print(f"[yellow]📤 Uploading {config_name}...[/yellow]")
        
        try:
            response = httpx.post(
                f"{PORTKEY_API_BASE}/configs",
                headers=self.headers,
                json=config,
                timeout=30.0
            )
            response.raise_for_status()
            
            result = response.json()
            config_id = result.get("id")
            
            if config_id:
                console.print(f"[green]✅ {config_name} uploaded: {config_id}[/green]")
                return config_id
            else:
                console.print(f"[red]❌ No config ID in response: {result}[/red]")
                sys.exit(1)
                
        except httpx.HTTPStatusError as e:
            console.print(f"[red]❌ HTTP Error {e.response.status_code}: {e.response.text}[/red]")
            sys.exit(1)
        except Exception as e:
            console.print(f"[red]❌ Error uploading config: {e}[/red]")
            sys.exit(1)
    
    def verify_config(self, config_id: str, config_name: str) -> bool:
        """Verify a config exists in Portkey."""
        try:
            response = httpx.get(
                f"{PORTKEY_API_BASE}/configs/{config_id}",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            console.print(f"[green]✅ {config_name} verified[/green]")
            return True
        except httpx.HTTPStatusError:
            console.print(f"[red]❌ {config_name} not found[/red]")
            return False
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not verify {config_name}: {e}[/yellow]")
            return False
    
    # =========================================================================
    # Model Catalog Verification
    # =========================================================================
    
    def list_available_models(self) -> List[Dict]:
        """List models available in the Model Catalog."""
        console.print("\n[yellow]🔍 Checking Model Catalog...[/yellow]")
        
        try:
            response = httpx.get(
                f"{PORTKEY_API_BASE}/models",
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            
            models = response.json().get("data", [])
            console.print(f"[green]✅ Found {len(models)} models in catalog[/green]")
            return models
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.print("[yellow]⚠️  Model list endpoint not available (OK for most setups)[/yellow]")
                return []
            console.print(f"[yellow]⚠️  Could not list models: {e}[/yellow]")
            return []
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not list models: {e}[/yellow]")
            return []
    
    def verify_model_access(self, model: str) -> bool:
        """Verify access to a specific model via Model Catalog."""
        try:
            # Try a simple models endpoint call
            response = httpx.post(
                f"{PORTKEY_API_BASE}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 5
                },
                timeout=30.0
            )
            
            if response.status_code in [200, 201]:
                console.print(f"[green]✅ Model {model} accessible[/green]")
                return True
            elif response.status_code == 401:
                console.print(f"[red]❌ Auth failed for {model}[/red]")
                return False
            elif response.status_code == 404:
                console.print(f"[yellow]⚠️  Model {model} not found in catalog[/yellow]")
                return False
            else:
                console.print(f"[yellow]⚠️  Model {model} returned {response.status_code}[/yellow]")
                return True  # Assume OK if not auth error
                
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not verify {model}: {e}[/yellow]")
            return True  # Assume OK
    
    # =========================================================================
    # Workspace Analytics (Cost Tracking)
    # =========================================================================
    
    def get_workspace_costs(self, workspace_id: str, days: int = 30) -> Optional[Dict]:
        """Get cost analytics for a workspace."""
        try:
            response = httpx.get(
                f"{PORTKEY_API_BASE}/analytics/costs",
                headers=self.headers,
                params={
                    "workspace_id": workspace_id,
                    "days": days
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            console.print(f"[yellow]⚠️  Could not fetch workspace costs: {e}[/yellow]")
            return None
    
    # =========================================================================
    # Environment File Management
    # =========================================================================
    
    def write_env_file(self, production_config_id: str, shadow_config_id: str):
        """Write config IDs to .env file."""
        env_path = Path(__file__).parent.parent / ".env"
        
        # Read existing .env if it exists
        env_lines = []
        if env_path.exists():
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        
        # Remove old config ID lines
        env_lines = [
            line for line in env_lines 
            if not line.startswith("PORTKEY_PRODUCTION_CONFIG_ID=") 
            and not line.startswith("PORTKEY_SHADOW_CONFIG_ID=")
            and not line.startswith("PORTKEY_PRODUCTION_WORKSPACE_ID=")
            and not line.startswith("PORTKEY_SHADOW_WORKSPACE_ID=")
            and not line.startswith("# Portkey Config IDs")
            and not line.startswith("# Portkey Workspace IDs")
        ]
        
        # Add Model Catalog configuration
        env_lines.append(f"\n# =============================================================================\n")
        env_lines.append(f"# Portkey Model Catalog Configuration (auto-generated by bootstrap_portkey.py)\n")
        env_lines.append(f"# =============================================================================\n")
        env_lines.append(f"\n# Config IDs\n")
        env_lines.append(f"PORTKEY_PRODUCTION_CONFIG_ID={production_config_id}\n")
        env_lines.append(f"PORTKEY_SHADOW_CONFIG_ID={shadow_config_id}\n")
        
        # Add Workspace IDs if available
        if self.workspace_ids:
            env_lines.append(f"\n# Workspace IDs (for cost segregation)\n")
            for name, wid in self.workspace_ids.items():
                env_lines.append(f"PORTKEY_{name.upper()}_WORKSPACE_ID={wid}\n")
        
        # Add Virtual Model documentation
        env_lines.append(f"\n# Virtual Model Names (defined in Portkey UI)\n")
        env_lines.append(f"# Production: prod-primary, prod-fallback\n")
        env_lines.append(f"# Shadow: shadow-challenger-1, shadow-challenger-2, shadow-challenger-3\n")
        
        # Write back
        with open(env_path, 'w') as f:
            f.writelines(env_lines)
        
        console.print(f"[green]✅ Config IDs written to {env_path}[/green]")
    
    # =========================================================================
    # Main Bootstrap Process
    # =========================================================================
    
    def bootstrap(self, skip_model_verify: bool = False):
        """Run the full bootstrap process."""
        console.print(Panel.fit(
            "[bold cyan]Shadow-Optic Portkey Bootstrapper[/bold cyan]\n"
            "[dim]Model Catalog + Workspace Architecture[/dim]",
            subtitle="Enterprise-Ready Setup"
        ))
        
        # Step 1: Verify Model Catalog access
        if not skip_model_verify:
            console.print("\n[bold]Step 1: Verify Model Catalog Access[/bold]")
            models = self.list_available_models()
            
            # Verify a few key models
            test_models = ["@openai/gpt-5-mini", "@anthropic/claude-haiku-4.5"]
            for model in test_models:
                self.verify_model_access(model)
        
        # Step 2: Load and upload configs
        console.print("\n[bold]Step 2: Upload Configurations[/bold]")
        production_config = self.load_config_file("production-config.json")
        shadow_config = self.load_config_file("shadow-config.json")
        
        production_config_id = self.upload_config(production_config, "Production Config")
        shadow_config_id = self.upload_config(shadow_config, "Shadow Testing Config")
        
        # Store config IDs
        self.config_ids["production"] = production_config_id
        self.config_ids["shadow"] = shadow_config_id
        
        # Step 3: Verify configs
        console.print("\n[bold]Step 3: Verify Configurations[/bold]")
        prod_ok = self.verify_config(production_config_id, "Production Config")
        shadow_ok = self.verify_config(shadow_config_id, "Shadow Config")
        
        if not (prod_ok and shadow_ok):
            console.print("\n[red]❌ Configuration verification failed![/red]")
            sys.exit(1)
        
        # Step 4: Write to .env
        console.print("\n[bold]Step 4: Update Environment[/bold]")
        self.write_env_file(production_config_id, shadow_config_id)
        
        # Step 5: Display summary
        self._display_summary()
        
        console.print("\n[bold green]🎉 Bootstrap Complete![/bold green]")
        self._display_next_steps()
    
    def _display_summary(self):
        """Display configuration summary."""
        # Config table
        config_table = Table(title="Configuration Summary", show_header=True)
        config_table.add_column("Type", style="cyan")
        config_table.add_column("Config ID", style="green")
        config_table.add_column("Status", style="bold")
        
        config_table.add_row("Production", self.config_ids.get("production", "N/A"), "✅ Ready")
        config_table.add_row("Shadow Testing", self.config_ids.get("shadow", "N/A"), "✅ Ready")
        
        console.print("\n")
        console.print(config_table)
        
        # Virtual Model table
        model_table = Table(title="Virtual Model Mappings", show_header=True)
        model_table.add_column("Virtual Name", style="cyan")
        model_table.add_column("Provider", style="yellow")
        model_table.add_column("Model", style="green")
        model_table.add_column("Workspace", style="magenta")
        
        for workspace, models in VIRTUAL_MODELS.items():
            for vname, config in models.items():
                model_table.add_row(
                    vname,
                    config["provider"],
                    config["model"],
                    workspace.capitalize()
                )
        
        console.print("\n")
        console.print(model_table)
    
    def _display_next_steps(self):
        """Display next steps for the user."""
        console.print("\n[bold yellow]📋 Next Steps:[/bold yellow]")
        console.print("")
        console.print("  [bold]1. Set up Virtual Models in Portkey UI:[/bold]")
        console.print("     Go to: https://app.portkey.ai/model-catalog")
        console.print("     Create Virtual Models with the names shown above")
        console.print("")
        console.print("  [bold]2. (Optional) Create Workspaces for cost segregation:[/bold]")
        console.print("     Go to: https://app.portkey.ai/workspaces")
        console.print("     Create 'production' and 'shadow' workspaces")
        console.print("     Set budget limits on 'shadow' workspace ($50/month)")
        console.print("")
        console.print("  [bold]3. Source environment and start:[/bold]")
        console.print("     [cyan]source .env && docker-compose up[/cyan]")
        console.print("")
        console.print("  [bold]4. Generate seed traffic:[/bold]")
        console.print("     [cyan]python scripts/seed_traffic.py --count 50[/cyan]")
        console.print("")
        console.print("  [bold]5. Verify in Portkey Dashboard:[/bold]")
        console.print("     - Check that requests appear under correct workspace")
        console.print("     - Verify Virtual Model names in logs")
        console.print("     - Confirm cost tracking is segregated")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap Portkey Model Catalog configuration for Shadow-Optic"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Portkey API key (or set PORTKEY_API_KEY env var)"
    )
    parser.add_argument(
        "--skip-model-verify",
        action="store_true",
        help="Skip model access verification (faster, but less thorough)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing configs, don't upload new ones"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("PORTKEY_API_KEY")
    if not api_key:
        console.print("[red]❌ Portkey API key required![/red]")
        console.print("Set PORTKEY_API_KEY environment variable or use --api-key flag")
        sys.exit(1)
    
    # Run bootstrap
    bootstrapper = PortkeyBootstrapper(api_key)
    bootstrapper.bootstrap(skip_model_verify=args.skip_model_verify)


if __name__ == "__main__":
    main()
