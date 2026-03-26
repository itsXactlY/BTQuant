"""
Configuration for the Autonomous Quantitative Research Agency
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any


def default_hypothesis_complexity_levels():
    return ["simple", "moderate", "complex"]


def default_strategy_types():
    return ["trend_following", "mean_reversion", "momentum", "arbitrage", "volatility"]


@dataclass
class MonitoringConfig:
    """Configuration for system monitoring"""
    cpu_threshold: float = 80.0  # %
    memory_threshold: float = 85.0  # %
    disk_threshold: float = 90.0  # %
    error_rate_threshold: float = 0.1  # 10%
    max_parallel_backtests: int = 4
    max_parallel_evaluations: int = 2
    archive_retention_days: int = 30
    evolution_cycle_interval_hours: int = 24
    evolution_cycle_cron: str = ""


@dataclass
class DeployerConfig:
    """Configuration for live deployment"""
    max_live_strategies: int = 5
    max_total_exposure: float = 10000.0  # USD
    position_size_limit: float = 0.1  # 10% of capital per position
    min_confidence_score: float = 0.8
    max_drawdown_limit: float = 0.2  # 20%
    max_daily_loss: float = 0.05  # 5%
    min_live_sharpe_ratio: float = 1.0
    max_consecutive_losses: int = 5
    live_exchange: str = "binance"
    live_sandbox_mode: bool = True
    live_base_currency: str = "USDT"
    live_monitoring_interval_minutes: int = 5
    max_deployment_days: int = 30


@dataclass
class AIModelConfig:
    """Configuration for AI model integration"""
    model_name: str = "Xiaomi: MiMo-V2-Flash"
    api_endpoint: str = os.getenv("AI_MODEL_ENDPOINT", "http://localhost:8000")
    temperature: float = 0.7
    max_tokens: int = 2000
    kilo_code_cli_path: str = os.getenv("KILO_CODE_PATH", "kilo-code")
    kilo_timeout_seconds: int = 300


@dataclass
class AgencyConfig:
    """Main configuration class for the autonomous agency"""

    # AI Model Configuration
    ai_model_name: str = "Xiaomi: MiMo-V2-Flash"
    ai_model_endpoint: str = os.getenv("AI_MODEL_ENDPOINT", "http://localhost:8000")
    ai_temperature: float = 0.7
    ai_max_tokens: int = 2000

    # Kilo Code CLI Configuration
    kilo_code_path: str = os.getenv("KILO_CODE_PATH", "kilo-code")
    kilo_code_timeout: int = 300  # seconds

    # Strategy Generation Parameters
    max_hypotheses_per_cycle: int = 10
    hypothesis_complexity_levels: List[str] = field(default_factory=default_hypothesis_complexity_levels)
    strategy_types: List[str] = field(default_factory=default_strategy_types)

    # Backtesting Configuration
    backtest_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h", "4h", "1d"])
    backtest_periods: List[str] = field(default_factory=lambda: ["1M", "3M", "6M", "1Y", "2Y"])
    out_of_sample_ratio: float = 0.3
    walk_forward_window: str = "3M"

    # Evaluation Thresholds
    min_sharpe_ratio: float = 1.0
    min_win_rate: float = 0.55
    max_drawdown_limit: float = 0.15
    min_profit_factor: float = 1.2

    # Evolution Parameters
    survival_rate: float = 0.2  # Top 20% survive each cycle
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    max_generations: int = 50

    # Database Configuration
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "1433"))
    db_name: str = os.getenv("DB_NAME", "btquant")
    db_user: str = os.getenv("DB_USER", "sa")
    db_password: str = os.getenv("DB_PASSWORD", "")

    # File Paths
    strategies_dir: str = "autonomous_agency/strategies"
    results_dir: str = "autonomous_agency/results"
    archive_dir: str = "autonomous_agency/archive"
    logs_dir: str = "autonomous_agency/logs"

    # Operational Parameters
    cycle_interval_hours: int = 24
    max_concurrent_backtests: int = 4
    memory_limit_gb: int = 8
    cpu_limit_cores: int = 4

    # Risk Management
    max_position_size_pct: float = 0.02  # 2% of portfolio
    max_portfolio_risk_pct: float = 0.05  # 5% max risk
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05

    # Monitoring and Alerts
    enable_alerts: bool = True
    alert_email: str = os.getenv("ALERT_EMAIL", "")
    log_level: str = "INFO"

    # Live Trading (when enabled)
    enable_live_trading: bool = False
    live_broker: str = "ccxt"
    live_exchange: str = "binance"
    live_api_key: str = os.getenv("LIVE_API_KEY", "")
    live_api_secret: str = os.getenv("LIVE_API_SECRET", "")


# Global configuration instance
config = AgencyConfig()


def load_config_from_env() -> AgencyConfig:
    """Load configuration from environment variables"""
    return AgencyConfig(
        ai_model_endpoint=os.getenv("AI_MODEL_ENDPOINT", config.ai_model_endpoint),
        kilo_code_path=os.getenv("KILO_CODE_PATH", config.kilo_code_path),
        db_host=os.getenv("DB_HOST", config.db_host),
        db_port=int(os.getenv("DB_PORT", str(config.db_port))),
        db_name=os.getenv("DB_NAME", config.db_name),
        db_user=os.getenv("DB_USER", config.db_user),
        db_password=os.getenv("DB_PASSWORD", config.db_password),
        alert_email=os.getenv("ALERT_EMAIL", config.alert_email),
        live_api_key=os.getenv("LIVE_API_KEY", config.live_api_key),
        live_api_secret=os.getenv("LIVE_API_SECRET", config.live_api_secret),
    )


def validate_config(config: AgencyConfig) -> List[str]:
    """Validate configuration and return list of issues"""
    issues = []

    if not config.db_password:
        issues.append("Database password not set")

    if config.enable_live_trading and (not config.live_api_key or not config.live_api_secret):
        issues.append("Live trading enabled but API credentials not provided")

    if not os.path.exists(config.kilo_code_path) and not os.path.exists(f"/usr/local/bin/{config.kilo_code_path}"):
        issues.append(f"Kilo Code CLI not found at {config.kilo_code_path}")

    return issues