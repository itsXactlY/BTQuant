"""
JSON Serialization Utilities

Provides robust JSON serialization with comprehensive type handling,
including boolean values, numpy types, and complex nested structures.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class JSONSerializationError(Exception):
    """Custom exception for JSON serialization errors"""
    pass


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles non-standard types including:
    - Boolean values (True/False)
    - Numpy types (np.bool_, np.int_, np.float_)
    - Pandas types (Timestamp, etc.)
    - Datetime objects
    - Custom objects with __dict__
    """
    
    def default(self, obj):
        # Handle boolean types
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        
        # Handle numpy integer types
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        
        # Handle numpy floating types - updated for NumPy 2.0 compatibility
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        
        # Handle numpy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Handle pandas Timestamp
        elif isinstance(obj, (pd.Timestamp, datetime, date)):
            return obj.isoformat()
        
        # Handle pandas DataFrame
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        
        # Handle pandas Series
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        
        # Handle custom objects with __dict__
        elif hasattr(obj, '__dict__'):
            return self.default(obj.__dict__)
        
        # Handle sets and other iterables
        elif isinstance(obj, set):
            return list(obj)
        
        # Let the base class default method raise the TypeError
        return super().default(obj)


def make_json_serializable(data: Any) -> Any:
    """
    Recursively convert data to JSON-serializable format
    
    Args:
        data: Data structure to convert
        
    Returns:
        JSON-serializable version of the data
    """
    if data is None:
        return None
    
    # Handle basic JSON-serializable types
    if isinstance(data, (str, int, float)):
        return data
    
    # Handle boolean values
    if isinstance(data, (bool, np.bool_)):
        return bool(data)
    
    # Handle numpy types
    if isinstance(data, (np.integer, np.int_)):
        return int(data)
    if isinstance(data, (np.floating, np.float64, np.float32)):
        return float(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    
    # Handle datetime objects
    if isinstance(data, (datetime, date, pd.Timestamp)):
        return data.isoformat()
    
    # Handle dictionaries
    if isinstance(data, dict):
        return {key: make_json_serializable(value) for key, value in data.items()}
    
    # Handle lists and tuples
    if isinstance(data, (list, tuple)):
        return [make_json_serializable(item) for item in data]
    
    # Handle pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return make_json_serializable(data.to_dict())
    
    # Handle pandas Series
    if isinstance(data, pd.Series):
        return make_json_serializable(data.to_dict())
    
    # Handle sets
    if isinstance(data, set):
        return [make_json_serializable(item) for item in data]
    
    # Handle custom objects
    if hasattr(data, '__dict__'):
        return make_json_serializable(data.__dict__)
    
    # Fallback: convert to string
    return str(data)


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely serialize data to JSON string
    
    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string representation
        
    Raises:
        JSONSerializationError: If serialization fails
    """
    try:
        # Ensure we have a custom encoder if not provided
        kwargs.setdefault('cls', JSONEncoder)
        
        # Make data serializable first
        serializable_data = make_json_serializable(data)
        
        return json.dumps(serializable_data, **kwargs)
    
    except (TypeError, ValueError) as e:
        logging.error(f"JSON serialization failed: {e}")
        raise JSONSerializationError(f"Failed to serialize data: {e}")


def safe_json_dump(data: Any, file_path: Union[str, Path], **kwargs) -> None:
    """
    Safely serialize data to JSON file
    
    Args:
        data: Data to serialize
        file_path: Path to output file
        **kwargs: Additional arguments for json.dump
        
    Raises:
        JSONSerializationError: If serialization fails
        IOError: If file operations fail
    """
    try:
        # Ensure we have a custom encoder if not provided
        kwargs.setdefault('cls', JSONEncoder)
        
        # Make data serializable first
        serializable_data = make_json_serializable(data)
        
        # Ensure directory exists
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, **kwargs)
            
    except (TypeError, ValueError) as e:
        logging.error(f"JSON serialization failed: {e}")
        raise JSONSerializationError(f"Failed to serialize data: {e}")
    except IOError as e:
        logging.error(f"File operation failed: {e}")
        raise


def safe_json_loads(json_string: str) -> Any:
    """
    Safely deserialize JSON string
    
    Args:
        json_string: JSON string to deserialize
        
    Returns:
        Deserialized data
        
    Raises:
        JSONSerializationError: If deserialization fails
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"JSON deserialization failed: {e}")
        raise JSONSerializationError(f"Failed to deserialize JSON: {e}")


def safe_json_load(file_path: Union[str, Path]) -> Any:
    """
    Safely deserialize JSON from file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Deserialized data
        
    Raises:
        JSONSerializationError: If deserialization fails
        IOError: If file operations fail
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, TypeError) as e:
        logging.error(f"JSON deserialization failed: {e}")
        raise JSONSerializationError(f"Failed to deserialize JSON from file: {e}")
    except IOError as e:
        logging.error(f"File operation failed: {e}")
        raise


def validate_json_serializable(data: Any) -> bool:
    """
    Validate if data can be serialized to JSON
    
    Args:
        data: Data to validate
        
    Returns:
        True if data is JSON-serializable, False otherwise
    """
    try:
        safe_json_dumps(data)
        return True
    except JSONSerializationError:
        return False


def get_serialization_report(data: Any) -> Dict[str, Any]:
    """
    Generate a report about data serialization compatibility
    
    Args:
        data: Data to analyze
        
    Returns:
        Dictionary with serialization analysis
    """
    report = {
        'is_serializable': False,
        'issues': [],
        'data_type': type(data).__name__,
        'complexity': 'simple'
    }
    
    try:
        # Test serialization
        safe_json_dumps(data)
        report['is_serializable'] = True
        return report
    except JSONSerializationError as e:
        report['issues'].append(str(e))
    
    # Analyze nested structure
    if isinstance(data, dict):
        report['complexity'] = 'dictionary'
        for key, value in data.items():
            if not validate_json_serializable(value):
                report['issues'].append(f"Key '{key}' of type {type(value).__name__} is not serializable")
    
    elif isinstance(data, (list, tuple)):
        report['complexity'] = 'list'
        for i, item in enumerate(data):
            if not validate_json_serializable(item):
                report['issues'].append(f"Item at index {i} of type {type(item).__name__} is not serializable")
    
    return report


# Convenience functions
def to_json(data: Any, indent: int = 2) -> str:
    """Convert data to JSON string with standard formatting"""
    return safe_json_dumps(data, indent=indent)


def from_json(json_string: str) -> Any:
    """Convert JSON string to data"""
    return safe_json_loads(json_string)


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file with standard formatting"""
    safe_json_dump(data, file_path, indent=indent)


def load_json(file_path: Union[str, Path]) -> Any:
    """Load data from JSON file"""
    return safe_json_load(file_path)