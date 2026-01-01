"""
LLM Configuration Management

Centralized configuration management for LLM integration with environment-based
settings, validation, and runtime configuration updates.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


class GenerationMode(Enum):
    """Strategy generation modes"""
    LLM_ONLY = "llm_only"
    TEMPLATE_ONLY = "template_only"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class LLMConfig:
    """LLM configuration dataclass"""
    
    # Provider settings
    provider: LLMProvider = LLMProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b"
    api_key: Optional[str] = None
    
    # Performance settings
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Generation settings
    temperature: float = 0.8
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1
    
    # Routing settings
    generation_mode: GenerationMode = GenerationMode.ADAPTIVE
    fallback_threshold: float = 0.6
    max_consecutive_failures: int = 3
    latency_threshold: float = 5.0
    
    # Quality settings
    min_novelty_score: float = 0.7
    min_validation_score: float = 0.8
    max_generation_time: float = 10.0
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_retention_days: int = 30
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMConfig':
        """Create from dictionary"""
        # Convert string enums to actual enum values
        if 'provider' in data and isinstance(data['provider'], str):
            data['provider'] = LLMProvider(data['provider'])
        if 'generation_mode' in data and isinstance(data['generation_mode'], str):
            data['generation_mode'] = GenerationMode(data['generation_mode'])
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'LLMConfig':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> tuple[bool, List[str]]:
        """Validate configuration"""
        errors = []
        
        # Validate provider-specific settings
        if self.provider == LLMProvider.OLLAMA:
            if not self.base_url:
                errors.append("base_url is required for Ollama provider")
            if not self.model_name:
                errors.append("model_name is required for Ollama provider")
        
        # Validate performance settings
        if self.timeout <= 0:
            errors.append("timeout must be positive")
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        
        # Validate generation settings
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        if self.max_tokens <= 0:
            errors.append("max_tokens must be positive")
        
        # Validate routing settings
        if not 0 <= self.fallback_threshold <= 1:
            errors.append("fallback_threshold must be between 0 and 1")
        if self.max_consecutive_failures <= 0:
            errors.append("max_consecutive_failures must be positive")
        if self.latency_threshold <= 0:
            errors.append("latency_threshold must be positive")
        
        # Validate quality settings
        if not 0 <= self.min_novelty_score <= 1:
            errors.append("min_novelty_score must be between 0 and 1")
        if not 0 <= self.min_validation_score <= 1:
            errors.append("min_validation_score must be between 0 and 1")
        if self.max_generation_time <= 0:
            errors.append("max_generation_time must be positive")
        
        # Validate circuit breaker settings
        if self.circuit_breaker_threshold <= 0:
            errors.append("circuit_breaker_threshold must be positive")
        if self.circuit_breaker_timeout <= 0:
            errors.append("circuit_breaker_timeout must be positive")
        
        return len(errors) == 0, errors


class ConfigManager:
    """Configuration manager for LLM settings"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory for configuration files
        """
        self.logger = logging.getLogger('ConfigManager')
        
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), 'llm_configs')
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.default_config = LLMConfig()
        
        # Runtime configuration overrides
        self.runtime_overrides: Dict[str, Any] = {}
        
        self.logger.info(f"Configuration manager initialized with directory: {self.config_dir}")
    
    def load_config(self, profile: str = "default") -> LLMConfig:
        """
        Load configuration for a specific profile
        
        Args:
            profile: Configuration profile name
            
        Returns:
            LLMConfig object
        """
        config_path = self.config_dir / f"{profile}.json"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                config = LLMConfig.from_dict(config_data)
                self.logger.info(f"Loaded configuration from {config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
                config = self.default_config
        else:
            self.logger.info(f"No config file found for profile '{profile}', using defaults")
            config = self.default_config
        
        # Apply environment overrides
        config = self._apply_environment_overrides(config)
        
        # Apply runtime overrides
        config = self._apply_runtime_overrides(config)
        
        # Validate configuration
        is_valid, errors = config.validate()
        if not is_valid:
            self.logger.warning(f"Configuration validation failed: {errors}")
        
        return config
    
    def save_config(self, config: LLMConfig, profile: str = "default") -> bool:
        """
        Save configuration for a specific profile
        
        Args:
            config: LLMConfig object to save
            profile: Configuration profile name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = self.config_dir / f"{profile}.json"
            
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self.logger.info(f"Saved configuration to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save config: {str(e)}")
            return False
    
    def create_profile(self, profile: str, base_profile: str = "default", 
                      overrides: Dict[str, Any] = None) -> bool:
        """
        Create a new configuration profile
        
        Args:
            profile: New profile name
            base_profile: Base profile to copy from
            overrides: Dictionary of configuration overrides
            
        Returns:
            True if successful, False otherwise
        """
        base_config = self.load_config(base_profile)
        new_config_dict = base_config.to_dict()
        
        if overrides:
            new_config_dict.update(overrides)
        
        new_config = LLMConfig.from_dict(new_config_dict)
        return self.save_config(new_config, profile)
    
    def list_profiles(self) -> List[str]:
        """List all available configuration profiles"""
        profiles = []
        for file_path in self.config_dir.glob("*.json"):
            profiles.append(file_path.stem)
        return sorted(profiles)
    
    def get_active_profile(self) -> str:
        """Get the active profile from environment"""
        return os.getenv("LLM_PROFILE", "default")
    
    def set_runtime_override(self, key: str, value: Any) -> None:
        """
        Set a runtime configuration override
        
        Args:
            key: Configuration key
            value: Override value
        """
        self.runtime_overrides[key] = value
        self.logger.info(f"Runtime override set: {key} = {value}")
    
    def clear_runtime_overrides(self) -> None:
        """Clear all runtime overrides"""
        self.runtime_overrides.clear()
        self.logger.info("Cleared all runtime overrides")
    
    def _apply_environment_overrides(self, config: LLMConfig) -> LLMConfig:
        """Apply environment variable overrides"""
        env_overrides = {}
        
        # LLM provider
        if "LLM_PROVIDER" in os.environ:
            try:
                env_overrides['provider'] = LLMProvider(os.environ["LLM_PROVIDER"])
            except ValueError:
                self.logger.warning(f"Invalid LLM_PROVIDER value: {os.environ['LLM_PROVIDER']}")
        
        # Base URL
        if "LLM_BASE_URL" in os.environ:
            env_overrides['base_url'] = os.environ["LLM_BASE_URL"]
        
        # Model name
        if "LLM_MODEL" in os.environ:
            env_overrides['model_name'] = os.environ["LLM_MODEL"]
        
        # API key
        if "LLM_API_KEY" in os.environ:
            env_overrides['api_key'] = os.environ["LLM_API_KEY"]
        
        # Generation mode
        if "LLM_GENERATION_MODE" in os.environ:
            try:
                env_overrides['generation_mode'] = GenerationMode(os.environ["LLM_GENERATION_MODE"])
            except ValueError:
                self.logger.warning(f"Invalid LLM_GENERATION_MODE value: {os.environ['LLM_GENERATION_MODE']}")
        
        # Performance settings
        if "LLM_TIMEOUT" in os.environ:
            try:
                env_overrides['timeout'] = int(os.environ["LLM_TIMEOUT"])
            except ValueError:
                self.logger.warning(f"Invalid LLM_TIMEOUT value: {os.environ['LLM_TIMEOUT']}")
        
        if "LLM_MAX_RETRIES" in os.environ:
            try:
                env_overrides['max_retries'] = int(os.environ["LLM_MAX_RETRIES"])
            except ValueError:
                self.logger.warning(f"Invalid LLM_MAX_RETRIES value: {os.environ['LLM_MAX_RETRIES']}")
        
        # Quality thresholds
        if "LLM_FALLBACK_THRESHOLD" in os.environ:
            try:
                env_overrides['fallback_threshold'] = float(os.environ["LLM_FALLBACK_THRESHOLD"])
            except ValueError:
                self.logger.warning(f"Invalid LLM_FALLBACK_THRESHOLD value: {os.environ['LLM_FALLBACK_THRESHOLD']}")
        
        if "LLM_MIN_NOVELTY_SCORE" in os.environ:
            try:
                env_overrides['min_novelty_score'] = float(os.environ["LLM_MIN_NOVELTY_SCORE"])
            except ValueError:
                self.logger.warning(f"Invalid LLM_MIN_NOVELTY_SCORE value: {os.environ['LLM_MIN_NOVELTY_SCORE']}")
        
        # Apply overrides
        if env_overrides:
            config_dict = config.to_dict()
            config_dict.update(env_overrides)
            config = LLMConfig.from_dict(config_dict)
            self.logger.info(f"Applied {len(env_overrides)} environment overrides")
        
        return config
    
    def _apply_runtime_overrides(self, config: LLMConfig) -> LLMConfig:
        """Apply runtime configuration overrides"""
        if not self.runtime_overrides:
            return config
        
        config_dict = config.to_dict()
        config_dict.update(self.runtime_overrides)
        
        try:
            config = LLMConfig.from_dict(config_dict)
            self.logger.info(f"Applied {len(self.runtime_overrides)} runtime overrides")
        except Exception as e:
            self.logger.error(f"Failed to apply runtime overrides: {str(e)}")
        
        return config
    
    def export_config(self, profile: str = "default", export_path: Optional[str] = None) -> bool:
        """
        Export configuration to a file
        
        Args:
            profile: Profile to export
            export_path: Path to export to (defaults to current directory)
            
        Returns:
            True if successful, False otherwise
        """
        config = self.load_config(profile)
        
        if export_path is None:
            export_path = f"llm_config_{profile}.json"
        
        try:
            with open(export_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            self.logger.info(f"Exported configuration to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {str(e)}")
            return False
    
    def import_config(self, import_path: str, profile: str) -> bool:
        """
        Import configuration from a file
        
        Args:
            import_path: Path to import from
            profile: Profile name to save as
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_path, 'r') as f:
                config_data = json.load(f)
            
            config = LLMConfig.from_dict(config_data)
            return self.save_config(config, profile)
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {str(e)}")
            return False


# Global configuration instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_llm_config(profile: str = None) -> LLMConfig:
    """
    Get LLM configuration for a profile
    
    Args:
        profile: Configuration profile name (defaults to active profile)
        
    Returns:
        LLMConfig object
    """
    if profile is None:
        profile = os.getenv("LLM_PROFILE", "default")
    
    manager = get_config_manager()
    return manager.load_config(profile)


def create_llm_config_profile(profile: str, **overrides) -> bool:
    """
    Create a new LLM configuration profile
    
    Args:
        profile: Profile name
        **overrides: Configuration overrides
        
    Returns:
        True if successful, False otherwise
    """
    manager = get_config_manager()
    return manager.create_profile(profile, overrides=overrides)


def set_llm_runtime_override(key: str, value: Any) -> None:
    """
    Set a runtime LLM configuration override
    
    Args:
        key: Configuration key
        value: Override value
    """
    manager = get_config_manager()
    manager.set_runtime_override(key, value)


def reset_llm_config() -> None:
    """Reset LLM configuration to defaults"""
    manager = get_config_manager()
    manager.clear_runtime_overrides()