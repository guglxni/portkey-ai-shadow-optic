#!/usr/bin/env python3
"""
Shadow-Optic Pre-Flight Check

Comprehensive verification that all components are ready for the demo.

This script checks:
1. ✅ Environment variables are set
2. ✅ Portkey connectivity
3. ✅ Qdrant connectivity
4. ✅ Temporal connectivity
5. ✅ Portkey configs exist
6. ✅ Required Python packages

Usage:
    python scripts/preflight_check.py
"""

import asyncio
import os
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


class PreFlightChecker:
    """Comprehensive pre-flight checks."""
    
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0
    
    def check_env_var(self, var_name: str, required: bool = True) -> bool:
        """Check if environment variable is set."""
        value = os.getenv(var_name)
        if value:
            console.print(f"  [green]✅[/green] {var_name}: [dim]{value[:20]}...[/dim]")
            return True
        else:
            if required:
                console.print(f"  [red]❌[/red] {var_name}: [red]Not set[/red]")
                return False
            else:
                console.print(f"  [yellow]⚠️[/yellow]  {var_name}: [dim]Not set (optional)[/dim]")
                return True
    
    async def check_portkey_connectivity(self) -> bool:
        """Check Portkey API connectivity."""
        api_key = os.getenv("PORTKEY_API_KEY")
        if not api_key:
            console.print("  [red]❌ PORTKEY_API_KEY not set, skipping connectivity check[/red]")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.portkey.ai/v1/configs",
                    headers={"x-portkey-api-key": api_key},
                    timeout=10.0
                )
                response.raise_for_status()
            console.print("  [green]✅ Portkey API is reachable[/green]")
            return True
        except Exception as e:
            console.print(f"  [red]❌ Portkey API error: {e}[/red]")
            return False
    
    async def check_portkey_configs(self) -> bool:
        """Check if Portkey configs exist."""
        api_key = os.getenv("PORTKEY_API_KEY")
        prod_config_id = os.getenv("PORTKEY_PRODUCTION_CONFIG_ID")
        shadow_config_id = os.getenv("PORTKEY_SHADOW_CONFIG_ID")
        
        if not api_key:
            console.print("  [red]❌ PORTKEY_API_KEY not set[/red]")
            return False
        
        if not prod_config_id or not shadow_config_id:
            console.print("  [yellow]⚠️  Config IDs not set. Run bootstrap_portkey.py first[/yellow]")
            return False
        
        try:
            async with httpx.AsyncClient() as client:
                # Check production config
                response = await client.get(
                    f"https://api.portkey.ai/v1/configs/{prod_config_id}",
                    headers={"x-portkey-api-key": api_key},
                    timeout=10.0
                )
                response.raise_for_status()
                console.print(f"  [green]✅ Production config exists: {prod_config_id}[/green]")
                
                # Check shadow config
                response = await client.get(
                    f"https://api.portkey.ai/v1/configs/{shadow_config_id}",
                    headers={"x-portkey-api-key": api_key},
                    timeout=10.0
                )
                response.raise_for_status()
                console.print(f"  [green]✅ Shadow config exists: {shadow_config_id}[/green]")
                
                return True
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.print("  [red]❌ Config not found. Run bootstrap_portkey.py first[/red]")
            else:
                console.print(f"  [red]❌ HTTP error: {e.response.status_code}[/red]")
            return False
        except Exception as e:
            console.print(f"  [red]❌ Error: {e}[/red]")
            return False
    
    async def check_qdrant_connectivity(self) -> bool:
        """Check Qdrant connectivity."""
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{qdrant_url}/collections",
                    timeout=10.0
                )
                response.raise_for_status()
            console.print(f"  [green]✅ Qdrant is reachable at {qdrant_url}[/green]")
            return True
        except Exception as e:
            console.print(f"  [red]❌ Qdrant error: {e}[/red]")
            console.print(f"  [dim]Start Qdrant: docker run -p 6333:6333 qdrant/qdrant[/dim]")
            return False
    
    async def check_temporal_connectivity(self) -> bool:
        """Check Temporal connectivity."""
        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        
        try:
            from temporalio.client import Client
            
            client = await Client.connect(temporal_host, namespace="shadow-optic")
            await client.close()
            
            console.print(f"  [green]✅ Temporal is reachable at {temporal_host}[/green]")
            return True
        except Exception as e:
            console.print(f"  [red]❌ Temporal error: {e}[/red]")
            console.print(f"  [dim]Start Temporal: temporal server start-dev[/dim]")
            return False
    
    def check_python_packages(self) -> bool:
        """Check if required packages are installed."""
        required_packages = [
            "portkey_ai",
            "temporalio",
            "deepeval",
            "qdrant_client",
            "fastapi",
            "prometheus_fastapi_instrumentator",
        ]
        
        all_installed = True
        for package in required_packages:
            try:
                __import__(package)
                console.print(f"  [green]✅[/green] {package}")
            except ImportError:
                console.print(f"  [red]❌[/red] {package} [red]not installed[/red]")
                all_installed = False
        
        if not all_installed:
            console.print("\n  [yellow]Run: pip install -r requirements.txt[/yellow]")
        
        return all_installed
    
    async def run_checks(self):
        """Run all pre-flight checks."""
        console.print(Panel.fit(
            "[bold cyan]Shadow-Optic Pre-Flight Check[/bold cyan]",
            subtitle="Verifying all components"
        ))
        
        # Environment variables
        console.print("\n[bold cyan]🔐 Environment Variables[/bold cyan]")
        env_checks = [
            ("PORTKEY_API_KEY", True),
            ("PORTKEY_PRODUCTION_CONFIG_ID", False),
            ("PORTKEY_SHADOW_CONFIG_ID", False),
            ("QDRANT_URL", False),
            ("TEMPORAL_HOST", False),
        ]
        
        env_ok = all(self.check_env_var(var, required) for var, required in env_checks)
        
        # Python packages
        console.print("\n[bold cyan]📦 Python Packages[/bold cyan]")
        packages_ok = self.check_python_packages()
        
        # Service connectivity
        console.print("\n[bold cyan]🌐 Service Connectivity[/bold cyan]")
        portkey_ok = await self.check_portkey_connectivity()
        qdrant_ok = await self.check_qdrant_connectivity()
        temporal_ok = await self.check_temporal_connectivity()
        
        # Portkey configs
        console.print("\n[bold cyan]⚙️  Portkey Configurations[/bold cyan]")
        configs_ok = await self.check_portkey_configs()
        
        # Summary
        console.print("\n" + "="*60)
        
        all_checks = [
            ("Environment Variables", env_ok),
            ("Python Packages", packages_ok),
            ("Portkey Connectivity", portkey_ok),
            ("Qdrant Connectivity", qdrant_ok),
            ("Temporal Connectivity", temporal_ok),
            ("Portkey Configs", configs_ok),
        ]
        
        table = Table(title="Pre-Flight Summary", box=box.ROUNDED, show_header=True)
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        
        passed = 0
        for check_name, result in all_checks:
            status = "[green]✅ PASS[/green]" if result else "[red]❌ FAIL[/red]"
            table.add_row(check_name, status)
            if result:
                passed += 1
        
        console.print("\n")
        console.print(table)
        
        # Final verdict
        console.print()
        if passed == len(all_checks):
            console.print(Panel.fit(
                "[bold green]🎉 ALL CHECKS PASSED! Ready for launch! 🚀[/bold green]",
                border_style="green"
            ))
            console.print("\n[bold]Next steps:[/bold]")
            console.print("  1. Generate traffic: [cyan]python scripts/seed_traffic.py[/cyan]")
            console.print("  2. Trigger optimization: [cyan]curl -X POST http://localhost:8000/api/v1/optimize[/cyan]")
            console.print("  3. Monitor workflow: [cyan]http://localhost:8233[/cyan]")
            return True
        else:
            console.print(Panel.fit(
                f"[bold yellow]⚠️  {passed}/{len(all_checks)} checks passed[/bold yellow]",
                border_style="yellow"
            ))
            console.print("\n[bold]Action required:[/bold]")
            if not configs_ok:
                console.print("  • Run: [cyan]python scripts/bootstrap_portkey.py[/cyan]")
            if not packages_ok:
                console.print("  • Run: [cyan]pip install -r requirements.txt[/cyan]")
            if not qdrant_ok:
                console.print("  • Start Qdrant: [cyan]docker run -p 6333:6333 qdrant/qdrant[/cyan]")
            if not temporal_ok:
                console.print("  • Start Temporal: [cyan]temporal server start-dev[/cyan]")
            return False


async def main():
    """Main entry point."""
    checker = PreFlightChecker()
    success = await checker.run_checks()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
