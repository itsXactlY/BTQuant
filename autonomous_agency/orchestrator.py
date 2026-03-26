"""
Orchestrator for the Autonomous Quantitative Research Agency

This module manages the perpetual motion engine of the agency, coordinating
all components in an endless loop of strategy innovation, refinement, and
execution. It handles scheduling, resource management, and ensures continuous
operation without human intervention.
"""

import logging
import asyncio
import signal
import threading
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from autonomous_agency.config import config, AgencyConfig
from autonomous_agency.hypothesis_generator import HypothesisGenerator
from autonomous_agency.strategy_factory import StrategyFactory
from autonomous_agency.backtester import AutomatedBacktester as Backtester
from autonomous_agency.evaluator import StrategyEvaluator as Evaluator
from autonomous_agency.evolution_engine import StrategyEvolutionEngine as EvolutionEngine
from autonomous_agency.archiver import StrategyArchiver
from autonomous_agency.live_deployment import LiveDeploymentManager


@dataclass
class AgencyStatus:
    """Current status of the autonomous agency"""
    is_running: bool = False
    current_generation: int = 0
    total_strategies: int = 0
    active_strategies: int = 0
    elite_strategies: int = 0
    last_evolution_cycle: Optional[datetime] = None
    next_evolution_cycle: Optional[datetime] = None
    system_health: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)


@dataclass
class ResourceUsage:
    """System resource usage tracking"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    gpu_usage: Optional[float] = None


class PerpetualOrchestrator:
    """
    Main orchestrator for the autonomous quantitative research agency.

    Manages the perpetual loop of:
    1. Hypothesis generation
    2. Strategy creation and backtesting
    3. Evaluation and validation
    4. Evolution and refinement
    5. Archiving and documentation
    6. Live deployment of elite strategies
    """

    def __init__(self, config: AgencyConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config

        # Core components
        self.hypothesis_generator = HypothesisGenerator()
        self.strategy_factory = StrategyFactory()
        self.backtester = Backtester()
        self.evaluator = Evaluator()
        self.evolution_engine = EvolutionEngine()
        self.archiver = StrategyArchiver()
        self.live_deployment = LiveDeploymentManager()

        # Scheduling
        self.scheduler = AsyncIOScheduler()
        self.loop = None

        # Status and monitoring
        self.status = AgencyStatus()
        self.resource_monitor = ResourceMonitor()

        # Control flags
        self.shutdown_event = threading.Event()
        self.pause_event = threading.Event()

        # Thread pools for parallel processing
        self.backtest_executor = ThreadPoolExecutor(max_workers=config.max_parallel_backtests)
        self.evaluation_executor = ThreadPoolExecutor(max_workers=config.max_parallel_evaluations)

        # Performance tracking
        self.cycle_times: List[float] = []
        self.generation_start_time: Optional[datetime] = None

        # Setup signal handlers
        self._setup_signal_handlers()

    async def start_agency(self):
        """
        Start the autonomous agency in perpetual motion mode
        """
        self.logger.info("Starting Autonomous Quantitative Research Agency...")

        try:
            # Initialize components
            await self._initialize_components()

            # Start resource monitoring
            self.resource_monitor.start()

            # Start the scheduler
            self.scheduler.start()

            # Schedule the main evolution cycle
            await self._schedule_evolution_cycle()

            # Schedule maintenance tasks
            await self._schedule_maintenance_tasks()

            # Update status
            self.status.is_running = True
            self.status.last_evolution_cycle = datetime.now()

            self.logger.info("Agency started successfully. Entering perpetual evolution loop...")

            # Main loop - keep running until shutdown
            while not self.shutdown_event.is_set():
                try:
                    # Check system health
                    await self._check_system_health()

                    # Process any pending alerts
                    await self._process_alerts()

                    # Small delay to prevent busy waiting
                    await asyncio.sleep(1)

                except Exception as e:
                    self.logger.error(f"Error in main agency loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying

            # Graceful shutdown
            await self._shutdown_agency()

        except Exception as e:
            self.logger.error(f"Failed to start agency: {e}")
            await self._shutdown_agency()
            raise

    async def stop_agency(self):
        """
        Stop the autonomous agency gracefully
        """
        self.logger.info("Stopping Autonomous Quantitative Research Agency...")

        self.shutdown_event.set()
        self.status.is_running = False

        # Stop scheduler
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)

        # Stop resource monitoring
        self.resource_monitor.stop()

        # Shutdown thread pools
        self.backtest_executor.shutdown(wait=True)
        self.evaluation_executor.shutdown(wait=True)

        # Final archive
        try:
            await self._final_archive()
        except Exception as e:
            self.logger.error(f"Error during final archive: {e}")

        self.logger.info("Agency stopped successfully.")

    async def pause_agency(self):
        """
        Pause the agency's evolution cycles
        """
        self.logger.info("Pausing agency evolution cycles...")
        self.pause_event.set()

        # Remove evolution jobs from scheduler
        jobs_to_remove = []
        for job in self.scheduler.get_jobs():
            if 'evolution' in job.id:
                jobs_to_remove.append(job)

        for job in jobs_to_remove:
            job.remove()

    async def resume_agency(self):
        """
        Resume the agency's evolution cycles
        """
        self.logger.info("Resuming agency evolution cycles...")
        self.pause_event.clear()

        # Re-schedule evolution cycle
        await self._schedule_evolution_cycle()

    async def get_status(self) -> AgencyStatus:
        """
        Get current agency status

        Returns:
            Current agency status
        """
        # Update real-time metrics
        self.status.system_health = await self._get_system_health()
        self.status.performance_metrics = self._calculate_performance_metrics()

        return self.status

    async def force_evolution_cycle(self):
        """
        Force an immediate evolution cycle (for testing/debugging)
        """
        self.logger.info("Forcing immediate evolution cycle...")
        await self._run_evolution_cycle()

    async def _initialize_components(self):
        """Initialize all agency components"""

        self.logger.info("Initializing agency components...")

        # Initialize hypothesis generator
        await self.hypothesis_generator.initialize()

        # Initialize strategy factory
        await self.strategy_factory.initialize()

        # Initialize backtester
        await self.backtester.initialize()

        # Initialize evaluator
        await self.evaluator.initialize()

        # Initialize evolution engine
        await self.evolution_engine.initialize()

        # Initialize archiver
        await self.archiver.initialize()

        # Initialize live deployment
        await self.live_deployment.initialize()

        self.logger.info("All components initialized successfully.")

    async def _schedule_evolution_cycle(self):
        """Schedule the main evolution cycle"""

        # Schedule based on configuration
        if config.evolution_cycle_interval_hours > 0:
            # Interval-based scheduling
            trigger = IntervalTrigger(hours=config.evolution_cycle_interval_hours)
            self.scheduler.add_job(
                self._run_evolution_cycle,
                trigger=trigger,
                id='evolution_cycle',
                name='Evolution Cycle',
                max_instances=1,
                replace_existing=True
            )
            self.logger.info(f"Scheduled evolution cycle every {config.evolution_cycle_interval_hours} hours")

        elif config.evolution_cycle_cron:
            # Cron-based scheduling
            trigger = CronTrigger.from_crontab(config.evolution_cycle_cron)
            self.scheduler.add_job(
                self._run_evolution_cycle,
                trigger=trigger,
                id='evolution_cycle',
                name='Evolution Cycle',
                max_instances=1,
                replace_existing=True
            )
            self.logger.info(f"Scheduled evolution cycle with cron: {config.evolution_cycle_cron}")

        # Calculate next run time
        job = self.scheduler.get_job('evolution_cycle')
        if job:
            self.status.next_evolution_cycle = job.next_run_time

    async def _schedule_maintenance_tasks(self):
        """Schedule maintenance and monitoring tasks"""

        # Daily system health check
        self.scheduler.add_job(
            self._daily_health_check,
            trigger=CronTrigger(hour=2, minute=0),  # 2 AM daily
            id='daily_health_check',
            name='Daily Health Check'
        )

        # Weekly archive cleanup
        self.scheduler.add_job(
            self._weekly_archive_cleanup,
            trigger=CronTrigger(day_of_week=0, hour=3, minute=0),  # Sunday 3 AM
            id='weekly_archive_cleanup',
            name='Weekly Archive Cleanup'
        )

        # Performance metrics update (every 15 minutes)
        self.scheduler.add_job(
            self._update_performance_metrics,
            trigger=IntervalTrigger(minutes=15),
            id='performance_metrics_update',
            name='Performance Metrics Update'
        )

    async def _run_evolution_cycle(self):
        """
        Execute one complete evolution cycle
        """
        if self.pause_event.is_set():
            self.logger.info("Evolution cycle paused, skipping...")
            return

        cycle_start = time.time()
        self.generation_start_time = datetime.now()

        try:
            self.logger.info(f"Starting evolution cycle {self.status.current_generation + 1}")

            # Phase 1: Hypothesis Generation
            hypotheses = await self._generate_hypotheses()

            # Phase 2: Strategy Creation
            strategies = await self._create_strategies(hypotheses)

            # Phase 3: Backtesting
            backtest_results = await self._run_backtests(strategies)

            # Phase 4: Evaluation
            evaluation_results = await self._run_evaluations(backtest_results)

            # Phase 5: Evolution
            evolution_result = await self._run_evolution(evaluation_results)

            # Phase 6: Archiving
            await self._run_archiving(evolution_result)

            # Phase 7: Live Deployment
            await self._run_live_deployment(evolution_result)

            # Update status
            self.status.current_generation += 1
            self.status.last_evolution_cycle = datetime.now()
            self.status.total_strategies = len(strategies)
            self.status.active_strategies = len(evolution_result.elite_strategies)
            self.status.elite_strategies = len([s for s in evolution_result.elite_strategies
                                              if s.fitness_score > config.elite_fitness_threshold])

            # Track cycle time
            cycle_time = time.time() - cycle_start
            self.cycle_times.append(cycle_time)

            self.logger.info(f"Evolution cycle {self.status.current_generation} completed in {cycle_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Error in evolution cycle: {e}")
            self.status.alerts.append(f"Evolution cycle failed: {str(e)}")

    async def _generate_hypotheses(self) -> List[Dict[str, Any]]:
        """Generate novel trading hypotheses"""

        self.logger.info("Generating trading hypotheses...")

        try:
            # Generate hypotheses using AI
            hypotheses = await self.hypothesis_generator.generate_hypotheses(
                population_size=config.hypothesis_population_size,
                generation=self.status.current_generation
            )

            self.logger.info(f"Generated {len(hypotheses)} hypotheses")
            return hypotheses

        except Exception as e:
            self.logger.error(f"Failed to generate hypotheses: {e}")
            return []

    async def _create_strategies(self, hypotheses: List[Dict[str, Any]]) -> List[Any]:
        """Create executable strategies from hypotheses"""

        self.logger.info("Creating strategies from hypotheses...")

        try:
            strategies = []

            for hypothesis in hypotheses:
                try:
                    strategy = await self.strategy_factory.create_strategy(hypothesis)
                    if strategy:
                        strategies.append(strategy)
                except Exception as e:
                    self.logger.warning(f"Failed to create strategy from hypothesis: {e}")
                    continue

            self.logger.info(f"Created {len(strategies)} executable strategies")
            return strategies

        except Exception as e:
            self.logger.error(f"Failed to create strategies: {e}")
            return []

    async def _run_backtests(self, strategies: List[Any]) -> List[Dict[str, Any]]:
        """Run backtests for all strategies"""

        self.logger.info("Running parallel backtests...")

        try:
            backtest_results = []

            # Submit backtest jobs
            future_to_strategy = {
                self.backtest_executor.submit(self._backtest_strategy, strategy): strategy
                for strategy in strategies
            }

            # Collect results
            for future in as_completed(future_to_strategy):
                try:
                    result = future.result()
                    if result:
                        backtest_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Backtest failed: {e}")
                    continue

            self.logger.info(f"Completed {len(backtest_results)} backtests")
            return backtest_results

        except Exception as e:
            self.logger.error(f"Failed to run backtests: {e}")
            return []

    async def _run_evaluations(self, backtest_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate backtest results"""

        self.logger.info("Evaluating backtest results...")

        try:
            evaluation_results = []

            # Submit evaluation jobs
            future_to_result = {
                self.evaluation_executor.submit(self._evaluate_result, result): result
                for result in backtest_results
            }

            # Collect results
            for future in as_completed(future_to_result):
                try:
                    result = future.result()
                    if result:
                        evaluation_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Evaluation failed: {e}")
                    continue

            self.logger.info(f"Completed {len(evaluation_results)} evaluations")
            return evaluation_results

        except Exception as e:
            self.logger.error(f"Failed to run evaluations: {e}")
            return []

    async def _run_evolution(self, evaluation_results: List[Dict[str, Any]]) -> Any:
        """Run evolution on evaluation results"""

        self.logger.info("Running evolution engine...")

        try:
            evolution_result = await self.evolution_engine.evolve_population(
                evaluation_results,
                generation=self.status.current_generation
            )

            self.logger.info(f"Evolution completed. Elite strategies: {len(evolution_result.elite_strategies)}")
            return evolution_result

        except Exception as e:
            self.logger.error(f"Failed to run evolution: {e}")
            return None

    async def _run_archiving(self, evolution_result: Any):
        """Archive evolution results"""

        self.logger.info("Archiving evolution results...")

        try:
            if evolution_result:
                await self.archiver.archive_evolution_result(evolution_result)

            self.logger.info("Archiving completed")

        except Exception as e:
            self.logger.error(f"Failed to archive results: {e}")

    async def _run_live_deployment(self, evolution_result: Any):
        """Deploy elite strategies to live trading"""

        self.logger.info("Checking for live deployment opportunities...")

        try:
            if evolution_result and config.enable_live_deployment:
                elite_strategies = [s for s in evolution_result.elite_strategies
                                  if s.fitness_score > config.live_deployment_threshold]

                for strategy in elite_strategies:
                    try:
                        await self.live_deployment.deploy_strategy(strategy)
                        self.logger.info(f"Deployed strategy {strategy.strategy_name} to live trading")
                    except Exception as e:
                        self.logger.warning(f"Failed to deploy strategy {strategy.strategy_name}: {e}")

        except Exception as e:
            self.logger.error(f"Failed to run live deployment: {e}")

    def _backtest_strategy(self, strategy: Any) -> Optional[Dict[str, Any]]:
        """Backtest a single strategy (synchronous wrapper)"""

        try:
            # Run backtest synchronously
            result = asyncio.run(self.backtester.run_backtest(strategy))
            return result
        except Exception as e:
            self.logger.error(f"Backtest failed for strategy: {e}")
            return None

    def _evaluate_result(self, backtest_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate a single backtest result (synchronous wrapper)"""

        try:
            # Run evaluation synchronously
            result = asyncio.run(self.evaluator.evaluate_result(backtest_result))
            return result
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return None

    async def _check_system_health(self):
        """Check overall system health"""

        try:
            health_status = await self._get_system_health()

            # Check resource thresholds
            if health_status['cpu_percent'] > config.cpu_threshold:
                self.status.alerts.append(f"High CPU usage: {health_status['cpu_percent']:.1f}%")

            if health_status['memory_percent'] > config.memory_threshold:
                self.status.alerts.append(f"High memory usage: {health_status['memory_percent']:.1f}%")

            if health_status['disk_usage'] > config.disk_threshold:
                self.status.alerts.append(f"High disk usage: {health_status['disk_usage']:.1f}%")

            # Check for evolution cycle delays
            if self.status.last_evolution_cycle:
                time_since_last_cycle = (datetime.now() - self.status.last_evolution_cycle).total_seconds()
                expected_interval = config.evolution_cycle_interval_hours * 3600

                if time_since_last_cycle > expected_interval * 1.5:  # 50% overdue
                    self.status.alerts.append("Evolution cycle overdue")

        except Exception as e:
            self.logger.error(f"Failed to check system health: {e}")

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get current system health metrics"""

        return self.resource_monitor.get_current_usage()

    async def _process_alerts(self):
        """Process and handle system alerts"""

        if not self.status.alerts:
            return

        # Log alerts
        for alert in self.status.alerts:
            self.logger.warning(f"System alert: {alert}")

        # Clear old alerts (keep last 10)
        if len(self.status.alerts) > 10:
            self.status.alerts = self.status.alerts[-10:]

    async def _daily_health_check(self):
        """Perform daily system health check"""

        self.logger.info("Performing daily health check...")

        try:
            # Comprehensive system check
            health_report = await self._comprehensive_health_check()

            # Archive health report
            health_file = Path(config.archive_dir) / "health" / f"daily_health_{datetime.now().strftime('%Y%m%d')}.json"
            health_file.parent.mkdir(parents=True, exist_ok=True)

            with open(health_file, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)

            # Check for concerning patterns
            if health_report.get('concerning_patterns'):
                self.status.alerts.append("Concerning patterns detected in daily health check")

        except Exception as e:
            self.logger.error(f"Daily health check failed: {e}")

    async def _weekly_archive_cleanup(self):
        """Perform weekly archive cleanup"""

        self.logger.info("Performing weekly archive cleanup...")

        try:
            # Clean old archives based on retention policy
            await self._cleanup_old_archives()

            # Optimize archive storage
            await self._optimize_archive_storage()

        except Exception as e:
            self.logger.error(f"Weekly archive cleanup failed: {e}")

    async def _update_performance_metrics(self):
        """Update performance metrics"""

        try:
            self.status.performance_metrics = self._calculate_performance_metrics()
        except Exception as e:
            self.logger.error(f"Failed to update performance metrics: {e}")

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics"""

        metrics = {
            'avg_cycle_time': np.mean(self.cycle_times) if self.cycle_times else 0,
            'total_cycles': len(self.cycle_times),
            'uptime_hours': (datetime.now() - self.status.last_evolution_cycle).total_seconds() / 3600 if self.status.last_evolution_cycle else 0,
            'strategies_per_cycle': self.status.total_strategies / max(1, self.status.current_generation),
            'elite_strategy_ratio': self.status.elite_strategies / max(1, self.status.total_strategies)
        }

        return metrics

    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""

        # This would include detailed checks of all components
        # For now, return basic structure
        return {
            'timestamp': datetime.now().isoformat(),
            'system_resources': await self._get_system_health(),
            'component_status': {
                'hypothesis_generator': 'operational',
                'strategy_factory': 'operational',
                'backtester': 'operational',
                'evaluator': 'operational',
                'evolution_engine': 'operational',
                'archiver': 'operational',
                'live_deployment': 'operational'
            },
            'concerning_patterns': []
        }

    async def _cleanup_old_archives(self):
        """Clean up old archive files based on retention policy"""

        try:
            archive_path = Path(config.archive_dir)

            # Remove old report files (keep last 30 days)
            reports_path = archive_path / "reports"
            if reports_path.exists():
                cutoff_date = datetime.now() - timedelta(days=config.archive_retention_days)

                for file_path in reports_path.glob("*.md"):
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()

                for file_path in reports_path.glob("*.pdf"):
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        file_path.unlink()

        except Exception as e:
            self.logger.error(f"Failed to cleanup old archives: {e}")

    async def _optimize_archive_storage(self):
        """Optimize archive storage (compression, deduplication, etc.)"""

        # Placeholder for storage optimization logic
        pass

    async def _final_archive(self):
        """Perform final archiving before shutdown"""

        try:
            # Archive final status
            final_status = await self.get_status()
            final_file = Path(config.archive_dir) / "final_status.json"

            with open(final_file, 'w') as f:
                json.dump({
                    'shutdown_time': datetime.now().isoformat(),
                    'final_status': asdict(final_status),
                    'total_runtime_hours': (datetime.now() - self.status.last_evolution_cycle).total_seconds() / 3600 if self.status.last_evolution_cycle else 0
                }, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to perform final archive: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop_agency())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _shutdown_agency(self):
        """Perform graceful shutdown of all components"""

        self.logger.info("Performing graceful shutdown...")

        # Shutdown components in reverse order
        try:
            await self.live_deployment.shutdown()
            await self.archiver.shutdown()
            await self.evolution_engine.shutdown()
            await self.evaluator.shutdown()
            await self.backtester.shutdown()
            await self.strategy_factory.shutdown()
            await self.hypothesis_generator.shutdown()

        except Exception as e:
            self.logger.error(f"Error during component shutdown: {e}")

        self.logger.info("Agency shutdown complete.")


class ResourceMonitor:
    """Monitor system resource usage"""

    def __init__(self):
        self.monitoring = False
        self.thread = None
        self.current_usage = ResourceUsage()

    def start(self):
        """Start resource monitoring"""

        if not self.monitoring:
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop resource monitoring"""

        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5)

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""

        return {
            'cpu_percent': self.current_usage.cpu_percent,
            'memory_percent': self.current_usage.memory_percent,
            'disk_usage': self.current_usage.disk_usage,
            'network_io': self.current_usage.network_io,
            'gpu_usage': self.current_usage.gpu_usage,
            'timestamp': datetime.now().isoformat()
        }

    def _monitor_loop(self):
        """Main monitoring loop"""

        while self.monitoring:
            try:
                # CPU usage
                self.current_usage.cpu_percent = psutil.cpu_percent(interval=1)

                # Memory usage
                memory = psutil.virtual_memory()
                self.current_usage.memory_percent = memory.percent

                # Disk usage
                disk = psutil.disk_usage('/')
                self.current_usage.disk_usage = disk.percent

                # Network I/O (simplified)
                net = psutil.net_io_counters()
                self.current_usage.network_io = {
                    'bytes_sent': net.bytes_sent,
                    'bytes_recv': net.bytes_recv
                }

                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.current_usage.gpu_usage = gpus[0].load * 100
                except ImportError:
                    self.current_usage.gpu_usage = None

            except Exception as e:
                logging.error(f"Resource monitoring error: {e}")

            time.sleep(5)  # Update every 5 seconds