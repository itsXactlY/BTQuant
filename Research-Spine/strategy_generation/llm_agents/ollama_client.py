"""
Ollama Client Module

Provides interface to local Ollama instance for LLM-powered strategy generation.
"""

import logging
import requests
import json
from typing import Dict, List, Any, Optional
import time
from requests.exceptions import RequestException


class OllamaClient:
    """
    Client for interacting with local Ollama instance
    
    Handles connection management, request/response patterns, and error handling
    for Mistral-3:8B model integration.
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client
         
        Args:
            model_name: Name of the model to use (default: qwen2.5:7b)
            base_url: Base URL of Ollama server (default: http://localhost:11434)
        """
        self.logger = logging.getLogger('OllamaClient')
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Enhanced connection state
        self.connected = False
        self.last_connection_check = 0
        self.connection_retry_interval = 30  # shorter interval for faster recovery
        self.max_retries = 3  # maximum retry attempts
        self.retry_delay = 2  # seconds between retries
        self.connection_timeout = 10  # seconds for connection timeout
        self.read_timeout = 60  # seconds for read timeout
        
        self.logger.info(f"OllamaClient initialized for model: {model_name}")
        self.logger.info(f"Ollama server URL: {base_url}")
        self.logger.info(f"Connection settings: retries={self.max_retries}, timeout={self.connection_timeout}s")
        
        # Initial connection validation with retry
        self._validate_connection_with_retry()
    
    def _validate_connection(self) -> bool:
        """
        Validate connection to Ollama server
        
        Returns:
            True if connection is valid, False otherwise
        """
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_connection_check < self.connection_retry_interval:
            return self.connected
        
        self.last_connection_check = current_time
        
        try:
            # Simple health check
            response = self.session.get(f"{self.base_url}", timeout=5)
            
            if response.status_code == 200:
                self.connected = True
                self.logger.info("✅ Successfully connected to Ollama server")
                return True
            else:
                self.connected = False
                self.logger.warning(f"⚠️  Ollama server connection issue: HTTP {response.status_code}")
                return False
                
        except RequestException as e:
            self.connected = False
            self.logger.error(f"❌ Failed to connect to Ollama server: {str(e)}")
            return False
        except Exception as e:
            self.connected = False
            self.logger.error(f"❌ Unexpected error validating Ollama connection: {str(e)}")
            return False

    def _validate_connection_with_retry(self) -> bool:
        """
        Validate connection to Ollama server with retry logic and exponential backoff
        
        Returns:
            True if connection is valid after retries, False otherwise
        """
        for attempt in range(self.max_retries):
            try:
                if self._validate_connection():
                    return True
                
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.info(f"Retrying connection in {wait_time:.1f} seconds... (attempt {attempt + 2}/{self.max_retries})")
                    time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
        
        self.logger.error(f"❌ Failed to connect to Ollama server after {self.max_retries} attempts")
        return False

    def _adjust_timeouts_based_on_system(self):
        """
        Adjust timeout settings based on system resource monitoring
        
        Uses adaptive timeout logic for older hardware
        """
        try:
            # Get system information
            import psutil
            import platform
            
            # Check CPU and memory resources
            cpu_count = psutil.cpu_count(logical=False)
            total_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
            system_info = platform.system()
            
            # Adjust timeouts based on system capabilities
            if cpu_count <= 2 or total_memory <= 4:
                # Older/slower hardware - increase timeouts
                self.connection_timeout = 15
                self.read_timeout = 90
                self.logger.info("🐢 Detected limited hardware resources - adjusted timeouts for better reliability")
            elif cpu_count <= 4 or total_memory <= 8:
                # Moderate hardware - slight timeout increase
                self.connection_timeout = 12
                self.read_timeout = 75
                self.logger.info("🖥️  Detected moderate hardware - optimized timeouts for performance")
            else:
                # Powerful hardware - use default timeouts
                self.connection_timeout = 10
                self.read_timeout = 60
                self.logger.info("🚀 Detected powerful hardware - using optimal timeout settings")
                
        except ImportError:
            self.logger.warning("⚠️  psutil not available - using default timeout settings")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not adjust timeouts based on system: {str(e)}")

    def _analyze_connection_error(self, error: Exception) -> str:
        """
        Analyze connection errors to distinguish between different failure types
        
        Args:
            error: Exception to analyze
            
        Returns:
            Error type classification
        """
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return "network_timeout"
        elif "connection refused" in error_str or "refused" in error_str:
            return "service_unavailable"
        elif "404" in error_str or "not found" in error_str:
            return "model_not_found"
        elif "500" in error_str or "server error" in error_str:
            return "server_error"
        elif "memory" in error_str or "resource" in error_str:
            return "hardware_limitation"
        else:
            return "unknown_error"

    def _enhanced_generate_with_retry(self, prompt: str, system_message: str = "",
                                    temperature: float = 0.7, max_tokens: Optional[int] = None,
                                    top_p: float = 0.9) -> str:
        """
        Enhanced generation with intelligent retry logic and error classification
        
        Args:
            prompt: The input prompt for generation
            system_message: Optional system message for context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            
        Returns:
            Generated text from the LLM
            
        Raises:
            RuntimeError: If generation fails after retries
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Adjust timeouts based on system resources
                self._adjust_timeouts_based_on_system()
                
                # Prepare request payload
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p
                }
                
                if system_message:
                    payload["system"] = system_message
                
                if max_tokens:
                    payload["max_tokens"] = max_tokens
                
                self.logger.debug(f"Sending generation request to Ollama (attempt {attempt + 1}): {json.dumps(payload, indent=2)}")
                
                # Send request with adaptive timeout
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    data=json.dumps(payload),
                    timeout=(self.connection_timeout, self.read_timeout)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get("response", "")
                    
                    # Log token usage if available
                    if "context" in result:
                        self.logger.debug(f"Generation completed. Context length: {len(result['context'])}")
                    
                    self.logger.info(f"✅ Generation successful (attempt {attempt + 1}). Generated {len(generated_text)} characters")
                    return generated_text
                else:
                    error_msg = f"LLM generation failed: HTTP {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text}"
                    last_error = RuntimeError(error_msg)
                    self.logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                    
            except RequestException as e:
                error_type = self._analyze_connection_error(e)
                error_msg = f"{error_type}: {str(e)}"
                last_error = RuntimeError(error_msg)
                self.logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                
                # Adjust retry strategy based on error type
                if error_type == "network_timeout" and attempt < self.max_retries - 1:
                    # Network issues - use exponential backoff
                    wait_time = self.retry_delay * (2 ** attempt)
                    self.logger.info(f"Network timeout detected - retrying in {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                elif error_type == "service_unavailable":
                    # Service unavailable - retry immediately
                    time.sleep(self.retry_delay)
                elif error_type == "hardware_limitation":
                    # Hardware issues - increase timeouts and retry
                    self.connection_timeout = min(30, self.connection_timeout * 1.5)
                    self.read_timeout = min(120, self.read_timeout * 1.5)
                    time.sleep(self.retry_delay * 2)
                else:
                    # Other errors - standard retry
                    time.sleep(self.retry_delay)
                    
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse LLM response: {str(e)}"
                last_error = RuntimeError(error_msg)
                self.logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                time.sleep(self.retry_delay)
                
            except Exception as e:
                error_msg = f"Unexpected error during LLM generation: {str(e)}"
                last_error = RuntimeError(error_msg)
                self.logger.error(f"Attempt {attempt + 1} failed: {error_msg}")
                time.sleep(self.retry_delay)
        
        # If all attempts failed, provide detailed error information
        error_msg = f"❌ All {self.max_retries} generation attempts failed"
        if last_error:
            error_msg += f": {str(last_error)}"
        self.logger.error(error_msg)
        raise RuntimeError(error_msg)

    def generate(self, prompt: str, system_message: str = "", temperature: float = 0.7,
                 max_tokens: Optional[int] = None, top_p: float = 0.9) -> str:
        """
        Generate text using the LLM with enhanced error handling and retry logic
        
        Args:
            prompt: The input prompt for generation
            system_message: Optional system message for context
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            
        Returns:
            Generated text from the LLM
            
        Raises:
            RuntimeError: If generation fails after retries
        """
        # Use enhanced generation with retry logic
        return self._enhanced_generate_with_retry(prompt, system_message, temperature, max_tokens, top_p)
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, 
             max_tokens: Optional[int] = None, top_p: float = 0.9) -> str:
        """
        Engage in chat conversation with the LLM
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            
        Returns:
            Generated response from the LLM
            
        Raises:
            RuntimeError: If chat fails
        """
        if not self._validate_connection():
            raise RuntimeError("Cannot chat: Not connected to Ollama server")
        
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            self.logger.debug(f"Sending chat request to Ollama with {len(messages)} messages")
            
            # Send request with timeout
            response = self.session.post(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("message", {}).get("content", "")
                
                self.logger.info(f"✅ Chat successful. Generated {len(generated_text)} characters")
                return generated_text
            else:
                error_msg = f"LLM chat failed: HTTP {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
                
        except RequestException as e:
            error_msg = f"Network error during LLM chat: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse LLM chat response: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during LLM chat: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        
        Returns:
            Dictionary containing model information
        """
        if not self._validate_connection():
            raise RuntimeError("Cannot get model info: Not connected to Ollama server")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/show",
                data=json.dumps({"name": self.model_name}),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Failed to get model info: HTTP {response.status_code}"
                self.logger.warning(error_msg)
                return {}
                
        except Exception as e:
            self.logger.warning(f"Failed to get model info: {str(e)}")
            return {}
    
    def list_available_models(self) -> List[str]:
        """
        List available models on the Ollama server
        
        Returns:
            List of available model names
        """
        if not self._validate_connection():
            return []
        
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                self.logger.warning(f"Failed to list models: HTTP {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.warning(f"Failed to list models: {str(e)}")
            return []
    
    def set_model(self, model_name: str) -> None:
        """
        Change the active model
        
        Args:
            model_name: Name of the model to switch to
        """
        if model_name == self.model_name:
            return
        
        self.model_name = model_name
        self.logger.info(f"Switched to model: {model_name}")
        
        # Validate the new model is available
        available_models = self.list_available_models()
        if model_name not in available_models:
            self.logger.warning(f"Model {model_name} may not be available on the server")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check
        
        Returns:
            Dictionary containing health check results
        """
        health_info = {
            "connected": self._validate_connection(),
            "model": self.model_name,
            "base_url": self.base_url,
            "available_models": [],
            "model_info": {}
        }
        
        if health_info["connected"]:
            health_info["available_models"] = self.list_available_models()
            health_info["model_info"] = self.get_model_info()
        
        return health_info
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.session.close()
        self.logger.info("OllamaClient session closed")
    
    def close(self) -> None:
        """Close the client session"""
        self.session.close()
        self.connected = False
        self.logger.info("OllamaClient closed")