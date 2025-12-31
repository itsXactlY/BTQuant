"""
Error Handling Module

Provides custom exceptions and error handling utilities for the system.
"""

import logging
from typing import Any, Optional

class SystemError(Exception):
    """Base class for system exceptions"""
    pass

class ConfigurationError(SystemError):
    """Exception for configuration-related errors"""
    pass

class StrategyGenerationError(SystemError):
    """Exception for strategy generation errors"""
    pass

class BacktestingError(SystemError):
    """Exception for backtesting-related errors"""
    pass

class EvolutionarySelectionError(SystemError):
    """Exception for evolutionary selection errors"""
    pass

class DocumentationError(SystemError):
    """Exception for documentation-related errors"""
    pass

class DeploymentError(SystemError):
    """Exception for deployment-related errors"""
    pass

class ErrorHandler:
    """Class for handling and logging errors consistently"""
    
    def __init__(self):
        self.logger = logging.getLogger('ErrorHandler')
        
    def handle_error(self, error: Exception, context: str = "", severity: str = "error") -> None:
        """
        Handle an error with appropriate logging and context
        
        Args:
            error: The exception to handle
            context: Additional context about where the error occurred
            severity: Severity level (debug, info, warning, error, critical)
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Log the error with context
        if severity.lower() == "debug":
            self.logger.debug(f"{context} - {error_type}: {error_message}", exc_info=True)
        elif severity.lower() == "info":
            self.logger.info(f"{context} - {error_type}: {error_message}", exc_info=True)
        elif severity.lower() == "warning":
            self.logger.warning(f"{context} - {error_type}: {error_message}", exc_info=True)
        elif severity.lower() == "critical":
            self.logger.critical(f"{context} - {error_type}: {error_message}", exc_info=True)
        else:  # default to error
            self.logger.error(f"{context} - {error_type}: {error_message}", exc_info=True)
    
    def handle_system_error(self, error: SystemError, context: str = "") -> None:
        """
        Handle a system-specific error
        
        Args:
            error: The SystemError to handle
            context: Additional context about where the error occurred
        """
        self.handle_error(error, context, "error")
    
    def log_exception(self, exception: Exception, message: str = "") -> None:
        """
        Log an exception with a custom message
        
        Args:
            exception: The exception to log
            message: Custom message to include in the log
        """
        if message:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(f"Exception occurred: {str(exception)}", exc_info=True)