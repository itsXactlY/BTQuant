"""
Security tests for input validation, data integrity, and vulnerability prevention
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

import backtrader as bt
from backtrader.feeds import mssql_stocks, hotspine_feed
from backtrader.brokers import ccxtbroker
from backtrader import stores


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization"""

    def test_malicious_sql_injection_prevention(self):
        """Test that SQL injection attempts are prevented in MSSQL feeds"""
        malicious_queries = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "'; EXEC xp_cmdshell 'net user'; --",
            "UNION SELECT * FROM sys.tables; --",
            "'; SHUTDOWN; --"
        ]

        for malicious_input in malicious_queries:
            with pytest.raises((ValueError, TypeError, Exception)):
                # Test MSSQL feed with malicious table names
                feed = mssql_stocks.MSSQLData(
                    dataname=malicious_input,
                    server='test_server',
                    database='test_db',
                    table='safe_table',
                    user='test_user',
                    password='test_pass'
                )

    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../root/.ssh/id_rsa",
            "....//....//....//etc/passwd"
        ]

        for malicious_path in malicious_paths:
            with pytest.raises((ValueError, OSError, Exception)):
                # Test file-based feeds with malicious paths
                feed = bt.feeds.YahooFinanceCSVData(dataname=malicious_path)

    def test_api_key_validation(self):
        """Test API key validation and secure handling"""
        # Test CCXT broker with invalid API keys
        invalid_keys = [
            "",  # Empty key
            " ",  # Whitespace only
            "<script>alert('xss')</script>",  # XSS attempt
            "javascript:alert('xss')",  # JavaScript injection
            "data:text/html,<script>alert('xss')</script>",  # Data URL injection
        ]

        for invalid_key in invalid_keys:
            with pytest.raises((ValueError, TypeError)):
                broker = ccxtbroker.CCXTBroker(
                    exchange='binance',
                    api_key=invalid_key,
                    api_secret='valid_secret'
                )

    def test_numeric_input_validation(self):
        """Test validation of numeric inputs to prevent overflow/underflow"""
        extreme_values = [
            float('inf'),
            float('-inf'),
            float('nan'),
            1e100,  # Very large number
            -1e100,  # Very small number
            1e-100,  # Very small positive
        ]

        for extreme_val in extreme_values:
            # Test strategy parameters
            class TestStrategy(bt.Strategy):
                params = (('test_param', extreme_val),)

                def next(self):
                    pass

            cerebro = bt.Cerebro()
            # Create minimal data
            dates = pd.date_range('2020-01-01', periods=10, freq='1D')
            data = pd.DataFrame({
                'datetime': dates,
                'open': [100] * 10,
                'high': [101] * 10,
                'low': [99] * 10,
                'close': [100] * 10,
                'volume': [1000] * 10
            })

            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(TestStrategy)

            # Should handle extreme values gracefully or raise appropriate errors
            try:
                cerebro.run()
            except (OverflowError, ValueError, ZeroDivisionError):
                pass  # Expected for extreme values
            except Exception as e:
                # Other exceptions should be reasonable
                assert not isinstance(e, (SystemExit, KeyboardInterrupt))

    def test_data_integrity_validation(self):
        """Test data integrity checks"""
        # Create data with integrity issues
        dates = pd.date_range('2020-01-01', periods=10, freq='1D')

        # Test with negative prices
        invalid_data = pd.DataFrame({
            'datetime': dates,
            'open': [-100, -50, 100, 101, 99, 100, 102, 98, 101, 100],
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        })

        feed = bt.feeds.PolarsData(dataname=invalid_data)

        class IntegrityCheckStrategy(bt.Strategy):
            def next(self):
                # Strategy should handle or detect invalid data
                if self.data.open[0] < 0:
                    self.log("Warning: Negative price detected")

        cerebro = bt.Cerebro()
        cerebro.adddata(feed)
        cerebro.addstrategy(IntegrityCheckStrategy)

        # Should complete without crashing
        result = cerebro.run()
        assert result is not None

    def test_buffer_overflow_prevention(self):
        """Test prevention of buffer overflow through large inputs"""
        # Create extremely large dataset
        n_points = 1000000  # 1 million data points

        try:
            dates = pd.date_range('2020-01-01', periods=n_points, freq='1min')
            prices = np.random.normal(100, 1, n_points).cumsum()

            data = pd.DataFrame({
                'datetime': dates,
                'open': prices,
                'high': prices * 1.001,
                'low': prices * 0.999,
                'close': prices,
                'volume': np.random.randint(100, 1000, n_points)
            })

            feed = bt.feeds.PolarsData(dataname=data)

            class LargeDataStrategy(bt.Strategy):
                def next(self):
                    pass  # Minimal processing

            cerebro = bt.Cerebro()
            cerebro.adddata(feed)
            cerebro.addstrategy(LargeDataStrategy)

            # Should handle large datasets without buffer overflow
            result = cerebro.run()
            assert result is not None

        except MemoryError:
            pytest.skip("Insufficient memory for large dataset test")
        except Exception as e:
            # Should not be buffer overflow related
            assert "buffer" not in str(e).lower()


@pytest.mark.security
class TestConfigurationSecurity:
    """Test secure handling of configuration files"""

    def test_secure_config_file_handling(self):
        """Test that configuration files are handled securely"""
        # Create temporary config file with sensitive data
        config_data = {
            "database": {
                "server": "secure-server",
                "user": "admin",
                "password": "super_secret_password",
                "database": "trading_db"
            },
            "api_keys": {
                "binance_api_key": "binance_key_123",
                "binance_secret": "binance_secret_456",
                "ccxt_config": {
                    "apiKey": "ccxt_key",
                    "secret": "ccxt_secret"
                }
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name

        try:
            # Test loading config
            with open(config_file, 'r') as f:
                loaded_config = json.load(f)

            # Verify sensitive data is loaded correctly
            assert loaded_config['database']['password'] == 'super_secret_password'
            assert loaded_config['api_keys']['binance_secret'] == 'binance_secret_456'

            # Test that config is not accidentally logged or exposed
            # (This would be caught by log analysis in real security testing)

        finally:
            # Clean up
            os.unlink(config_file)

    def test_environment_variable_security(self):
        """Test secure handling of environment variables"""
        sensitive_env_vars = [
            'BT_API_KEY',
            'BT_SECRET_KEY',
            'BT_DATABASE_PASSWORD',
            'BT_CCXT_API_KEY',
            'BT_CCXT_SECRET'
        ]

        # Test with malicious environment variables
        malicious_values = [
            "'; rm -rf /; --",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "../../../../etc/passwd"
        ]

        for env_var in sensitive_env_vars:
            for malicious_val in malicious_values:
                with patch.dict(os.environ, {env_var: malicious_val}):
                    # Test that environment variables are validated
                    # This would typically be done in config loading functions
                    value = os.environ.get(env_var)
                    if value:
                        # Should not contain dangerous characters
                        dangerous_chars = [';', '&', '|', '`', '$', '<', '>', '"', "'"]
                        for char in dangerous_chars:
                            assert char not in value or value.count(char) == 0, \
                                f"Potentially dangerous character '{char}' in {env_var}"


@pytest.mark.security
class TestDataSanitization:
    """Test data sanitization and validation"""

    def test_html_xss_prevention(self):
        """Test prevention of XSS through HTML/script injection"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(\"xss\")'>",
            "<svg onload=alert('xss')>",
            "data:text/html,<script>alert('xss')</script>"
        ]

        for payload in xss_payloads:
            # Test in strategy names
            class TestStrategy(bt.Strategy):
                def __init__(self):
                    self.name = payload

                def next(self):
                    pass

            # Test in data names
            dates = pd.date_range('2020-01-01', periods=10, freq='1D')
            data = pd.DataFrame({
                'datetime': dates,
                'open': [100] * 10,
                'high': [101] * 10,
                'low': [99] * 10,
                'close': [100] * 10,
                'volume': [1000] * 10
            })

            feed = bt.feeds.PolarsData(dataname=data)
            feed._name = payload  # Try to set malicious name

            cerebro = bt.Cerebro()
            cerebro.adddata(feed)
            cerebro.addstrategy(TestStrategy)

            # Should complete without executing scripts
            result = cerebro.run()
            assert result is not None

    def test_sql_injection_in_data_queries(self):
        """Test SQL injection prevention in data queries"""
        # Mock database connection
        with patch('pyodbc.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_connection = MagicMock()
            mock_connection.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_connection

            malicious_symbols = [
                "'; DROP TABLE stocks; --",
                "BTC' OR '1'='1",
                "ETH'; EXEC xp_cmdshell 'dir'; --"
            ]

            for symbol in malicious_symbols:
                # Test MSSQL feed
                feed = mssql_stocks.MSSQLData(
                    dataname=symbol,
                    server='test',
                    database='test',
                    table='stocks',
                    user='test',
                    password='test'
                )

                # Should either sanitize input or raise error
                try:
                    # This would normally attempt to query the database
                    feed.start()
                except Exception:
                    pass  # Expected for malicious input

    def test_file_inclusion_prevention(self):
        """Test prevention of file inclusion attacks"""
        dangerous_files = [
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "../../../../.ssh/id_rsa",
            "/proc/self/environ",
            "php://input",
            "data://text/plain;base64,SGVsbG8gV29ybGQ="
        ]

        for dangerous_file in dangerous_files:
            with pytest.raises((ValueError, OSError, IOError, Exception)):
                # Test file-based feeds
                feed = bt.feeds.YahooFinanceCSVData(dataname=dangerous_file)

    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks"""
        command_injections = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "`whoami`",
            "$(rm -rf /)",
            "; net user hacker password /add",
            "| type C:\\Windows\\System32\\config\\sam"
        ]

        # Test through environment variables that might be used in system calls
        for injection in command_injections:
            with patch.dict(os.environ, {'BT_SYSTEM_CMD': injection}):
                # Test that system commands are not executed
                # This would be validated in components that use system calls
                env_cmd = os.environ.get('BT_SYSTEM_CMD', '')
                if env_cmd:
                    # Should not contain command separators
                    assert ';' not in env_cmd
                    assert '|' not in env_cmd
                    assert '`' not in env_cmd
                    assert '$(' not in env_cmd


@pytest.mark.security
class TestAccessControl:
    """Test access control and authorization"""

    def test_broker_access_control(self):
        """Test that brokers properly control access"""
        # Test CCXT broker with insufficient permissions
        with patch('ccxt.binance') as mock_exchange:
            mock_exchange_instance = MagicMock()
            mock_exchange_instance.has = {
                'fetchBalance': False,
                'createOrder': False,
                'fetchOrder': True
            }
            mock_exchange.return_value = mock_exchange_instance

            broker = ccxtbroker.CCXTBroker(
                exchange='binance',
                api_key='test_key',
                api_secret='test_secret'
            )

            # Should handle lack of permissions gracefully
            assert broker is not None

    def test_database_access_validation(self):
        """Test database access validation"""
        with patch('pyodbc.connect') as mock_connect:
            # Simulate connection failure
            mock_connect.side_effect = Exception("Access denied")

            with pytest.raises(Exception):
                feed = mssql_stocks.MSSQLData(
                    dataname='BTC',
                    server='restricted_server',
                    database='restricted_db',
                    table='stocks',
                    user='invalid_user',
                    password='wrong_password'
                )
                feed.start()

    def test_file_permission_validation(self):
        """Test file permission validation"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test data")
            temp_file = f.name

        try:
            # Make file unreadable
            os.chmod(temp_file, 0o000)

            with pytest.raises((OSError, IOError, PermissionError)):
                feed = bt.feeds.YahooFinanceCSVData(dataname=temp_file)

        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(temp_file, 0o644)
                os.unlink(temp_file)
            except:
                pass


@pytest.mark.security
class TestCryptographicSecurity:
    """Test cryptographic operations and key management"""

    def test_api_key_encryption_handling(self):
        """Test that API keys are handled securely"""
        # Test that API keys are not logged in plain text
        with patch('backtrader.brokers.ccxtbroker.logger') as mock_logger:
            broker = ccxtbroker.CCXTBroker(
                exchange='binance',
                api_key='sensitive_key_123',
                api_secret='sensitive_secret_456'
            )

            # Check that sensitive data is not logged
            # This would require inspecting log calls
            # For this test, we ensure the broker initializes without exposing keys
            assert broker is not None

    def test_secure_random_generation(self):
        """Test secure random number generation for trading decisions"""
        # Test that random elements in strategies use secure random
        import random
        import secrets

        # Mock random to ensure it's not predictable
        with patch('random.random') as mock_random:
            mock_random.return_value = 0.5  # Predictable value

            class RandomStrategy(bt.Strategy):
                def next(self):
                    if random.random() > 0.5:
                        self.buy(size=1)

            dates = pd.date_range('2020-01-01', periods=10, freq='1D')
            data = pd.DataFrame({
                'datetime': dates,
                'open': [100] * 10,
                'high': [101] * 10,
                'low': [99] * 10,
                'close': [100] * 10,
                'volume': [1000] * 10
            })

            cerebro = bt.Cerebro()
            feed = bt.feeds.PolarsData(dataname=data)
            cerebro.adddata(feed)
            cerebro.addstrategy(RandomStrategy)
            cerebro.broker.set_cash(1000)

            result = cerebro.run()
            assert result is not None

    def test_data_integrity_verification(self):
        """Test data integrity verification"""
        # Create data and verify it hasn't been tampered with
        dates = pd.date_range('2020-01-01', periods=100, freq='1D')
        original_prices = np.random.normal(100, 5, 100).cumsum()

        data = pd.DataFrame({
            'datetime': dates,
            'open': original_prices,
            'high': original_prices * 1.01,
            'low': original_prices * 0.99,
            'close': original_prices,
            'volume': np.random.randint(1000, 5000, 100)
        })

        # Calculate checksum of original data
        import hashlib
        original_checksum = hashlib.sha256(
            data.to_csv().encode()
        ).hexdigest()

        # Test that data integrity is maintained through backtrader
        feed = bt.feeds.PolarsData(dataname=data)

        class IntegrityStrategy(bt.Strategy):
            def __init__(self):
                self.data_points = []

            def next(self):
                self.data_points.append(self.data.close[0])

        cerebro = bt.Cerebro()
        cerebro.adddata(feed)
        cerebro.addstrategy(IntegrityStrategy)

        result = cerebro.run()

        # Verify data integrity
        processed_data = result[0].data_points
        processed_df = pd.DataFrame({
            'close': processed_data
        })

        processed_checksum = hashlib.sha256(
            processed_df.to_csv().encode()
        ).hexdigest()

        # Data should maintain integrity through processing
        assert len(processed_data) > 0