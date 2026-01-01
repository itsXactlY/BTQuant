#!/usr/bin/env python3
"""
Test cases to verify the JSON serialization fix for boolean values

This test suite validates that the boolean serialization error has been resolved
and that all JSON serialization operations work correctly with various data types.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from utils.json_serialization import (
    JSONEncoder,
    make_json_serializable,
    safe_json_dumps,
    safe_json_dump,
    safe_json_loads,
    safe_json_load,
    JSONSerializationError,
    validate_json_serializable,
    get_serialization_report
)


class TestJSONSerializationFix:
    """Test suite for JSON serialization fix"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def test_boolean_serialization(self):
        """Test that boolean values are properly serialized"""
        print("Testing boolean serialization...")
        
        # Test basic boolean values
        test_cases = [
            True,
            False,
            np.bool_(True),
            np.bool_(False),
            {'enabled': True, 'disabled': False},
            [True, False, True],
            {'nested': {'flag': True, 'data': [False, True]}},
            {'mixed_types': [1, True, 'string', False, 3.14]}
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                result = safe_json_dumps(test_case)
                # Verify it can be loaded back
                loaded = json.loads(result)
                self.test_results.append({
                    'test': f'boolean_serialization_{i}',
                    'status': 'PASS',
                    'data': str(test_case),
                    'result': result[:100] + '...' if len(result) > 100 else result
                })
                self.passed += 1
                print(f"  ✓ Test {i+1} passed")
            except Exception as e:
                self.test_results.append({
                    'test': f'boolean_serialization_{i}',
                    'status': 'FAIL',
                    'data': str(test_case),
                    'error': str(e)
                })
                self.failed += 1
                print(f"  ✗ Test {i+1} failed: {e}")
    
    def test_numpy_types_serialization(self):
        """Test that numpy types are properly converted"""
        print("\nTesting numpy types serialization...")
        
        test_cases = [
            np.int32(42),
            np.int64(123),
            np.float32(3.14),
            np.float64(2.718),
            np.array([1, 2, 3]),
            np.array([True, False, True]),
            {'numpy_int': np.int32(5), 'numpy_float': np.float64(6.7)}
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                result = safe_json_dumps(test_case)
                loaded = json.loads(result)
                self.test_results.append({
                    'test': f'numpy_serialization_{i}',
                    'status': 'PASS',
                    'data': f"{type(test_case).__name__}: {test_case}",
                    'result': result[:100] + '...' if len(result) > 100 else result
                })
                self.passed += 1
                print(f"  ✓ Test {i+1} passed")
            except Exception as e:
                self.test_results.append({
                    'test': f'numpy_serialization_{i}',
                    'status': 'FAIL',
                    'data': f"{type(test_case).__name__}: {test_case}",
                    'error': str(e)
                })
                self.failed += 1
                print(f"  ✗ Test {i+1} failed: {e}")
    
    def test_datetime_serialization(self):
        """Test that datetime objects are properly serialized"""
        print("\nTesting datetime serialization...")
        
        test_cases = [
            datetime.now(),
            date.today(),
            pd.Timestamp.now(),
            {'created_at': datetime.now(), 'updated_at': date.today()},
            [datetime.now(), pd.Timestamp.now()]
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                result = safe_json_dumps(test_case)
                loaded = json.loads(result)
                self.test_results.append({
                    'test': f'datetime_serialization_{i}',
                    'status': 'PASS',
                    'data': str(test_case),
                    'result': result[:100] + '...' if len(result) > 100 else result
                })
                self.passed += 1
                print(f"  ✓ Test {i+1} passed")
            except Exception as e:
                self.test_results.append({
                    'test': f'datetime_serialization_{i}',
                    'status': 'FAIL',
                    'data': str(test_case),
                    'error': str(e)
                })
                self.failed += 1
                print(f"  ✗ Test {i+1} failed: {e}")
    
    def test_complex_nested_structures(self):
        """Test complex nested data structures with mixed types"""
        print("\nTesting complex nested structures...")
        
        # Simulate a strategy configuration with boolean parameters
        strategy_config = {
            'id': 'strategy_001',
            'template': 'moving_average_crossover',
            'enabled': True,
            'parameters': {
                'fast_ma_period': 20,
                'slow_ma_period': 50,
                'use_rsi_filter': True,
                'rsi_threshold': 30,
                'risk_management': {
                    'stop_loss_enabled': True,
                    'stop_loss_percent': 0.02,
                    'take_profit_enabled': False,
                    'take_profit_percent': 0.05,
                    'position_sizing': 'fixed'
                },
                'backtest_settings': {
                    'start_date': datetime.now(),
                    'end_date': date.today(),
                    'initial_capital': 100000,
                    'commission': 0.001,
                    'slippage': 0.0005
                }
            },
            'performance_metrics': {
                'sharpe_ratio': np.float64(1.85),
                'max_drawdown': np.float32(0.15),
                'win_rate': np.float64(0.62),
                'total_return': np.float64(0.25),
                'significance': {
                    'p_value': np.float64(0.021),
                    'significant': True,
                    'confidence_interval': [0.28, 0.36]
                }
            },
            'metadata': {
                'created': datetime.now(),
                'version': 2,
                'active': True,
                'tags': ['trend', 'momentum', 'tested']
            }
        }
        
        try:
            result = safe_json_dumps(strategy_config, indent=2)
            loaded = json.loads(result)
            
            # Verify boolean values are preserved
            assert loaded['enabled'] == True
            assert loaded['parameters']['use_rsi_filter'] == True
            assert loaded['parameters']['risk_management']['stop_loss_enabled'] == True
            assert loaded['parameters']['risk_management']['take_profit_enabled'] == False
            assert loaded['performance_metrics']['significance']['significant'] == True
            assert loaded['metadata']['active'] == True
            
            self.test_results.append({
                'test': 'complex_nested_structure',
                'status': 'PASS',
                'data': 'Complex strategy configuration',
                'result': f"Successfully serialized and verified {len(result)} bytes"
            })
            self.passed += 1
            print("  ✓ Complex nested structure test passed")
            
        except Exception as e:
            self.test_results.append({
                'test': 'complex_nested_structure',
                'status': 'FAIL',
                'data': 'Complex strategy configuration',
                'error': str(e)
            })
            self.failed += 1
            print(f"  ✗ Complex nested structure test failed: {e}")
    
    def test_file_operations(self):
        """Test file-based JSON operations"""
        print("\nTesting file operations...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test_data.json"
            
            # Test data with boolean values
            test_data = {
                'test_id': 'file_test_001',
                'active': True,
                'settings': {
                    'debug': False,
                    'verbose': True,
                    'max_retries': 3
                },
                'timestamp': datetime.now()
            }
            
            # Test save
            try:
                safe_json_dump(test_data, test_file, indent=2)
                self.test_results.append({
                    'test': 'file_save',
                    'status': 'PASS',
                    'data': 'Test data with booleans',
                    'result': f"Saved to {test_file}"
                })
                self.passed += 1
                print("  ✓ File save test passed")
            except Exception as e:
                self.test_results.append({
                    'test': 'file_save',
                    'status': 'FAIL',
                    'data': 'Test data with booleans',
                    'error': str(e)
                })
                self.failed += 1
                print(f"  ✗ File save test failed: {e}")
                return
            
            # Test load
            try:
                loaded_data = safe_json_load(test_file)
                
                # Verify boolean values
                assert loaded_data['active'] == True
                assert loaded_data['settings']['debug'] == False
                assert loaded_data['settings']['verbose'] == True
                
                self.test_results.append({
                    'test': 'file_load',
                    'status': 'PASS',
                    'data': 'Load and verify',
                    'result': 'Boolean values preserved correctly'
                })
                self.passed += 1
                print("  ✓ File load test passed")
                
            except Exception as e:
                self.test_results.append({
                    'test': 'file_load',
                    'status': 'FAIL',
                    'data': 'Load and verify',
                    'error': str(e)
                })
                self.failed += 1
                print(f"  ✗ File load test failed: {e}")
    
    def test_validation_functions(self):
        """Test the validation and reporting functions"""
        print("\nTesting validation functions...")
        
        # Test valid data
        valid_data = {'flag': True, 'count': 5, 'name': 'test'}
        is_valid = validate_json_serializable(valid_data)
        
        if is_valid:
            self.test_results.append({
                'test': 'validation_valid',
                'status': 'PASS',
                'data': 'Valid data',
                'result': 'Validation passed'
            })
            self.passed += 1
            print("  ✓ Validation test passed")
        else:
            self.test_results.append({
                'test': 'validation_valid',
                'status': 'FAIL',
                'data': 'Valid data',
                'error': 'Validation failed unexpectedly'
            })
            self.failed += 1
            print("  ✗ Validation test failed")
        
        # Test serialization report
        test_data = {
            'bool_value': True,
            'numpy_int': np.int32(42),
            'datetime': datetime.now(),
            'array': np.array([1, 2, 3])
        }
        
        try:
            report = get_serialization_report(test_data)
            if report['is_serializable']:
                self.test_results.append({
                    'test': 'serialization_report',
                    'status': 'PASS',
                    'data': 'Mixed types',
                    'result': 'Report generated successfully'
                })
                self.passed += 1
                print("  ✓ Serialization report test passed")
            else:
                self.test_results.append({
                    'test': 'serialization_report',
                    'status': 'FAIL',
                    'data': 'Mixed types',
                    'error': f"Report indicated issues: {report['issues']}"
                })
                self.failed += 1
                print("  ✗ Serialization report test failed")
        except Exception as e:
            self.test_results.append({
                'test': 'serialization_report',
                'status': 'FAIL',
                'data': 'Mixed types',
                'error': str(e)
            })
            self.failed += 1
            print(f"  ✗ Serialization report test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling for edge cases"""
        print("\nTesting error handling...")
        
        # Test with a custom object that can't be serialized
        class CustomObject:
            def __init__(self):
                self.value = True
                self.data = [1, 2, 3]
        
        custom_obj = CustomObject()
        
        try:
            # This should work because our custom encoder handles objects with __dict__
            result = safe_json_dumps(custom_obj)
            self.test_results.append({
                'test': 'custom_object',
                'status': 'PASS',
                'data': 'Custom object with __dict__',
                'result': 'Successfully serialized'
            })
            self.passed += 1
            print("  ✓ Custom object test passed")
        except Exception as e:
            self.test_results.append({
                'test': 'custom_object',
                'status': 'FAIL',
                'data': 'Custom object with __dict__',
                'error': str(e)
            })
            self.failed += 1
            print(f"  ✗ Custom object test failed: {e}")
    
    def run_all_tests(self):
        """Run all test cases"""
        print("=" * 60)
        print("JSON Serialization Fix Test Suite")
        print("=" * 60)
        
        self.test_boolean_serialization()
        self.test_numpy_types_serialization()
        self.test_datetime_serialization()
        self.test_complex_nested_structures()
        self.test_file_operations()
        self.test_validation_functions()
        self.test_error_handling()
        
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed / (self.passed + self.failed) * 100):.1f}%")
        
        if self.failed == 0:
            print("\n🎉 ALL TESTS PASSED! The boolean serialization fix is working correctly.")
        else:
            print(f"\n⚠️  {self.failed} test(s) failed. Please review the errors above.")
        
        return self.failed == 0


def main():
    """Main test execution"""
    tester = TestJSONSerializationFix()
    success = tester.run_all_tests()
    
    # Save detailed results to file
    results_file = Path("test_results.json")
    try:
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': tester.passed + tester.failed,
                    'passed': tester.passed,
                    'failed': tester.failed,
                    'success_rate': (tester.passed / (tester.passed + tester.failed) * 100) if (tester.passed + tester.failed) > 0 else 0
                },
                'detailed_results': tester.test_results
            }, f, indent=2)
        print(f"\nDetailed results saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save detailed results: {e}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())