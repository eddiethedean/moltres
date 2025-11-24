"""Tests for retry and backoff utilities."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from moltres.utils.retry import (
    RetryConfig,
    is_retryable_error,
    retry_with_backoff,
    retry_with_backoff_async,
)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_initialization(self):
        """Test default initialization."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert ConnectionError in config.retryable_errors
        assert TimeoutError in config.retryable_errors
        assert OSError in config.retryable_errors

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        custom_errors = (ValueError, TypeError)
        config = RetryConfig(
            max_attempts=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
            retryable_errors=custom_errors,
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_errors == custom_errors

    def test_calculate_delay_no_jitter(self):
        """Test delay calculation without jitter."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(initial_delay=1.0, exponential_base=2.0, jitter=True)
        # With jitter, delay should be >= base delay and <= base delay * 1.1
        delay = config.calculate_delay(0)
        assert 1.0 <= delay <= 1.1

        delay = config.calculate_delay(1)
        assert 2.0 <= delay <= 2.2

    def test_calculate_delay_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(initial_delay=10.0, exponential_base=2.0, max_delay=50.0, jitter=False)
        # Attempt 0: 10.0
        assert config.calculate_delay(0) == 10.0
        # Attempt 1: 20.0
        assert config.calculate_delay(1) == 20.0
        # Attempt 2: 40.0
        assert config.calculate_delay(2) == 40.0
        # Attempt 3: 80.0, but capped at 50.0
        assert config.calculate_delay(3) == 50.0
        # Attempt 4: 160.0, but capped at 50.0
        assert config.calculate_delay(4) == 50.0

    def test_calculate_delay_custom_exponential_base(self):
        """Test delay calculation with custom exponential base."""
        config = RetryConfig(initial_delay=1.0, exponential_base=3.0, jitter=False)
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 3.0
        assert config.calculate_delay(2) == 9.0
        assert config.calculate_delay(3) == 27.0


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_retryable_error_type(self):
        """Test that retryable error types are detected."""
        config = RetryConfig()
        assert is_retryable_error(ConnectionError("test"), config) is True
        assert is_retryable_error(TimeoutError("test"), config) is True
        assert is_retryable_error(OSError("test"), config) is True

    def test_non_retryable_error_type(self):
        """Test that non-retryable error types are not detected."""
        config = RetryConfig()
        assert is_retryable_error(ValueError("test"), config) is False
        assert is_retryable_error(TypeError("test"), config) is False
        assert is_retryable_error(KeyError("test"), config) is False

    def test_retryable_error_message_indicators(self):
        """Test that error messages with transient indicators are detected."""
        config = RetryConfig(retryable_errors=())
        # Test various transient error message indicators
        assert is_retryable_error(ValueError("connection timeout"), config) is True
        assert is_retryable_error(ValueError("timed out"), config) is True
        assert is_retryable_error(ValueError("network error"), config) is True
        assert is_retryable_error(ValueError("temporary failure"), config) is True
        assert is_retryable_error(ValueError("please retry"), config) is True
        assert is_retryable_error(ValueError("deadlock detected"), config) is True
        assert is_retryable_error(ValueError("database lock"), config) is True
        assert is_retryable_error(ValueError("database busy"), config) is True

    def test_non_retryable_error_message(self):
        """Test that non-transient error messages are not detected."""
        config = RetryConfig(retryable_errors=())
        assert is_retryable_error(ValueError("invalid input"), config) is False
        assert is_retryable_error(ValueError("syntax error"), config) is False

    def test_custom_retryable_errors(self):
        """Test with custom retryable error types."""
        config = RetryConfig(retryable_errors=(ValueError, KeyError))
        assert is_retryable_error(ValueError("test"), config) is True
        assert is_retryable_error(KeyError("test"), config) is True
        assert is_retryable_error(ConnectionError("test"), config) is False


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    def test_successful_execution_first_attempt(self):
        """Test successful execution on first attempt."""
        func = Mock(return_value=42)
        result = retry_with_backoff(func)
        assert result == 42
        assert func.call_count == 1

    def test_successful_execution_after_retries(self):
        """Test successful execution after retries."""
        func = Mock(side_effect=[ConnectionError("test"), ConnectionError("test"), "success"])
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("time.sleep"):
            result = retry_with_backoff(func, config=config)
        assert result == "success"
        assert func.call_count == 3

    def test_exhaustion_of_retries(self):
        """Test that exception is raised when all retries are exhausted."""
        func = Mock(side_effect=ConnectionError("test"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("time.sleep"):
            with pytest.raises(ConnectionError, match="test"):
                retry_with_backoff(func, config=config)
        assert func.call_count == 3

    def test_non_retryable_error_not_retried(self):
        """Test that non-retryable errors are not retried."""
        func = Mock(side_effect=ValueError("invalid input"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01, retryable_errors=())
        with pytest.raises(ValueError, match="invalid input"):
            retry_with_backoff(func, config=config)
        assert func.call_count == 1  # Should not retry

    def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        func = Mock(side_effect=[ConnectionError("test"), "success"])
        callback = Mock()
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("time.sleep"):
            retry_with_backoff(func, config=config, on_retry=callback)
        assert callback.call_count == 1
        assert callback.call_args[0][0].args[0] == "test"
        assert callback.call_args[0][1] == 1  # First retry attempt

    def test_on_retry_callback_failure(self):
        """Test that callback failures don't stop retry."""
        func = Mock(side_effect=[ConnectionError("test"), "success"])
        callback = Mock(side_effect=ValueError("callback error"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("time.sleep"):
            result = retry_with_backoff(func, config=config, on_retry=callback)
        assert result == "success"
        assert callback.call_count == 1

    def test_delay_calculation(self):
        """Test that delays are calculated correctly."""
        func = Mock(side_effect=[ConnectionError("test"), ConnectionError("test"), "success"])
        config = RetryConfig(max_attempts=3, initial_delay=0.1, exponential_base=2.0, jitter=False)
        with patch("time.sleep") as mock_sleep:
            retry_with_backoff(func, config=config)
        # Should sleep twice (after first and second failures)
        assert mock_sleep.call_count == 2
        # First delay: 0.1 * 2^0 = 0.1
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(0.1, abs=0.01)
        # Second delay: 0.1 * 2^1 = 0.2
        assert mock_sleep.call_args_list[1][0][0] == pytest.approx(0.2, abs=0.01)

    def test_default_config(self):
        """Test that default config is used when not provided."""
        func = Mock(return_value=42)
        result = retry_with_backoff(func)
        assert result == 42
        assert func.call_count == 1

    def test_error_message_indicators_retry(self):
        """Test that errors with transient message indicators are retried."""
        func = Mock(side_effect=[ValueError("connection timeout"), "success"])
        config = RetryConfig(max_attempts=3, initial_delay=0.01, retryable_errors=(), jitter=False)
        with patch("time.sleep"):
            result = retry_with_backoff(func, config=config)
        assert result == "success"
        assert func.call_count == 2


class TestRetryWithBackoffAsync:
    """Tests for retry_with_backoff_async function."""

    @pytest.mark.asyncio
    async def test_successful_execution_first_attempt(self):
        """Test successful async execution on first attempt."""

        async def func():
            return 42

        result = await retry_with_backoff_async(func)
        assert result == 42

    @pytest.mark.asyncio
    async def test_successful_execution_after_retries(self):
        """Test successful async execution after retries."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("test")
            return "success"

        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("asyncio.sleep"):
            result = await retry_with_backoff_async(func, config=config)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_exhaustion_of_retries_async(self):
        """Test that exception is raised when all async retries are exhausted."""

        async def func():
            raise ConnectionError("test")

        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("asyncio.sleep"):
            with pytest.raises(ConnectionError, match="test"):
                await retry_with_backoff_async(func, config=config)

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried_async(self):
        """Test that non-retryable errors are not retried in async."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        config = RetryConfig(max_attempts=3, initial_delay=0.01, retryable_errors=())
        with pytest.raises(ValueError, match="invalid input"):
            await retry_with_backoff_async(func, config=config)
        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_on_retry_callback_async(self):
        """Test that on_retry callback is called in async."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("test")
            return "success"

        callback = Mock()
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("asyncio.sleep"):
            result = await retry_with_backoff_async(func, config=config, on_retry=callback)
        assert result == "success"
        assert callback.call_count == 1
        assert callback.call_args[0][0].args[0] == "test"
        assert callback.call_args[0][1] == 1  # First retry attempt

    @pytest.mark.asyncio
    async def test_on_retry_callback_failure_async(self):
        """Test that callback failures don't stop async retry."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("test")
            return "success"

        callback = Mock(side_effect=ValueError("callback error"))
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        with patch("asyncio.sleep"):
            result = await retry_with_backoff_async(func, config=config, on_retry=callback)
        assert result == "success"
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_delay_calculation_async(self):
        """Test that delays are calculated correctly in async."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("test")
            return "success"

        config = RetryConfig(max_attempts=3, initial_delay=0.1, exponential_base=2.0, jitter=False)
        with patch("asyncio.sleep") as mock_sleep:
            await retry_with_backoff_async(func, config=config)
        # Should sleep twice (after first and second failures)
        assert mock_sleep.call_count == 2
        # First delay: 0.1 * 2^0 = 0.1
        assert mock_sleep.call_args_list[0][0][0] == pytest.approx(0.1, abs=0.01)
        # Second delay: 0.1 * 2^1 = 0.2
        assert mock_sleep.call_args_list[1][0][0] == pytest.approx(0.2, abs=0.01)

    @pytest.mark.asyncio
    async def test_default_config_async(self):
        """Test that default config is used when not provided in async."""

        async def func():
            return 42

        result = await retry_with_backoff_async(func)
        assert result == 42

    @pytest.mark.asyncio
    async def test_error_message_indicators_retry_async(self):
        """Test that errors with transient message indicators are retried in async."""
        call_count = 0

        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("connection timeout")
            return "success"

        config = RetryConfig(max_attempts=3, initial_delay=0.01, retryable_errors=(), jitter=False)
        with patch("asyncio.sleep"):
            result = await retry_with_backoff_async(func, config=config)
        assert result == "success"
        assert call_count == 2
