#!/usr/bin/env python3
"""
Autonomous Quantitative Research Agency - Main Entry Point

This script starts the perpetual motion engine for quantitative trading strategy innovation.
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path

# Optional imports with fallbacks
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Create mock classes
    class MockConsole:
        def print(self, *args, **kwargs): print(*args)
        def rule(self, *args, **kwargs): print("="*50)

    class MockTable:
        def __init__(self, **kwargs): pass
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): pass

    class MockPanel:
        @staticmethod
        def fit(text, **kwargs): return text

    class MockText:
        def __init__(self, text): self.text = text

    Console = MockConsole
    Table = MockTable
    Panel = MockPanel
    Text = MockText

# Add autonomous_agency to path
sys.path.insert(0, str(Path(__file__).parent))

from autonomous_agency.config import AgencyConfig
from autonomous_agency.orchestrator import PerpetualOrchestrator
from autonomous_agency.deployer import LiveDeployer
from autonomous_agency.monitoring import SystemMonitor


console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_agency.log'),
        logging.StreamHandler()
    ]
)


def print_banner():
    """Print the agency banner"""
    banner = """
    █████╗ ██╗   ██╗████████╗ ██████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ ██████╗ ██╗   ██╗███████╗
   ██╔══██╗██║   ██║╚══██╔══╝██╔═══██╗████╗  ██║██╔═══██╗████╗ ████║██╔═══██╗██║   ██║██╔════╝
   ███████║██║   ██║   ██║   ██║   ██║██╔██╗ ██║██║   ██║██╔████╔██║██║   ██║██║   ██║███████╗
   ██╔══██║██║   ██║   ██║   ██║   ██║██║╚██╗██║██║   ██║██║╚██╔╝██║██║   ██║██║   ██║╚════██║
   ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚████║╚██████╔╝██║ ╚═╝ ██║╚██████╔╝╚██████╔╝███████║
   ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝

                        AUTONOMOUS QUANTITATIVE RESEARCH AGENCY
                        Perpetual Motion for Strategy Innovation

    """
    console.print(Panel.fit(banner, border_style="blue"))


def print_system_status(orchestrator: PerpetualOrchestrator, deployer: LiveDeployer):
    """Print comprehensive system status"""
    status = orchestrator.get_system_status()
    deployment_status = deployer.get_deployment_status()

    # Main status table
    table = Table(title="🔄 System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    table.add_row("Orchestrator", "🟢 Running" if status['is_running'] else "🔴 Stopped",
                  f"Loop #{status['loop_count']}")
    table.add_row("Population", f"{status['population_stats']['population_size']} strategies",
                  f"Gen {status['population_stats']['generation']}, Avg Fitness: {status['population_stats']['avg_fitness']:.3f}")
    table.add_row("Archive", f"{status['archive_stats']['total_strategies']} total",
                  f"Active: {status['archive_stats']['status_distribution'].get('active', 0)}")
    table.add_row("Deployments", f"{deployment_status['active_deployments']} active",
                  f"Total Exposure: ${deployment_status['total_exposure']:,.2f}")
    table.add_row("Health", f"CPU: {status['health']['cpu_usage']:.1f}%",
                  f"Memory: {status['health']['memory_usage']:.1f}%, Error Rate: {status['health']['error_rate']:.3f}")

    console.print(table)

    # Best performers
    if status['archive_stats']['best_performers']:
        best_table = Table(title="🏆 Best Performers")
        best_table.add_column("Strategy", style="cyan")
        best_table.add_column("Score", style="green")
        best_table.add_column("Name", style="yellow")

        for performer in status['archive_stats']['best_performers'][:3]:
            best_table.add_row(
                performer['id'][:16] + "...",
                f"{performer['score']:.3f}",
                performer['name'][:30]
            )

        console.print(best_table)


async def run_agency(config: AgencyConfig, mode: str = 'full'):
    """Run the autonomous agency"""

    # Initialize components
    orchestrator = PerpetualOrchestrator(config)
    deployer = LiveDeployer(config.deployer)

    if mode == 'status':
        print_system_status(orchestrator, deployer)
        return

    if mode == 'init':
        console.print("[bold blue]🌱 Initializing population...[/bold blue]")
        await orchestrator._initialize_population()
        console.print("[bold green]✅ Initialization complete[/bold green]")
        return

    # Start deployment monitoring
    if config.deployer.max_live_strategies > 0:
        deployment_task = asyncio.create_task(deployer.monitor_deployments())
    else:
        deployment_task = None

    try:
        # Start the perpetual loop
        await orchestrator.start_perpetual_loop()

    except KeyboardInterrupt:
        console.print("\n[bold yellow]⚠ Received interrupt signal[/bold yellow]")

    except Exception as e:
        console.print(f"\n[bold red]💥 Critical error: {e}[/bold red]")
        raise

    finally:
        # Cleanup
        orchestrator.stop()

        if deployment_task:
            deployment_task.cancel()
            try:
                await deployment_task
            except asyncio.CancelledError:
                pass

        # Emergency stop all deployments
        await deployer.emergency_stop_all()

        console.print("[bold green]🛑 Agency shutdown complete[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Quantitative Research Agency")
    parser.add_argument('--mode', choices=['full', 'init', 'status'], default='full',
                       help='Operation mode: full (run perpetually), init (initialize only), status (show status)')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    # Load configuration
    config = AgencyConfig.from_env()
    if args.config:
        # Could load from file here
        pass

    if args.debug:
        config.debug_mode = True
        logging.getLogger().setLevel(logging.DEBUG)

    print_banner()

    console.print(f"[bold blue]🚀 Starting Autonomous Agency (Mode: {args.mode})[/bold blue]")
    console.print(f"[dim]Config: {config.to_dict()}[/dim]\n")

    try:
        asyncio.run(run_agency(config, args.mode))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]👋 Agency interrupted by user[/bold yellow]")
    except Exception as e:
        console.print(f"\n[bold red]💥 Fatal error: {e}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()