"""Error handling and recovery utilities for robust grid search execution."""

from __future__ import annotations

import functools
import sys
import time
import traceback
from typing import Any, Callable, Optional

from logging_utils.logger import DualLogger


class RecoveryError(Exception):
    """Base exception for recovery-related errors."""

    pass


class RecoveryStrategy:
    """Error recovery strategy with retry logic."""

    def __init__(
        self,
        logger: DualLogger,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
    ) -> None:
        self.logger = logger
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.retry_count = 0

    def reset(self) -> None:
        """Reset retry counter."""
        self.retry_count = 0

    def exponential_backoff_retry(
        self,
        func: Callable,
        *args: Any,
        error_context: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute function with exponential backoff retry logic."""
        self.retry_count = 0

        while self.retry_count <= self.max_retries:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                self.retry_count += 1
                if self.retry_count > self.max_retries:
                    self.logger.error(f"Max retries ({self.max_retries}) exceeded" + (f" for {error_context}" if error_context else ""))
                    self.logger.error(f"Final error: {exc}")
                    raise RecoveryError(f"Failed after {self.max_retries} retries: {exc}") from exc

                delay = self.initial_delay * (self.backoff_factor ** (self.retry_count - 1))
                self.logger.warning(
                    f"Attempt {self.retry_count} failed"
                    + (f" ({error_context})" if error_context else "")
                    + f". Retrying in {delay:.1f} seconds..."
                )
                time.sleep(delay)

    def retry_decorator(self, error_context: Optional[str] = None) -> Callable:
        """Decorator for automatic retry logic."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return self.exponential_backoff_retry(
                    func,
                    *args,
                    error_context=error_context or func.__name__,
                    **kwargs,
                )

            return wrapper

        return decorator


class ErrorHandler:
    """Central error handling and categorization."""

    # Error categories
    RECOVERABLE_ERRORS = (
        OSError,
        IOError,
        TimeoutError,
        RuntimeError,
        MemoryError,
    )
    CRITICAL_ERRORS = (
        KeyboardInterrupt,
        SystemExit,
    )

    def __init__(self, logger: DualLogger) -> None:
        self.logger = logger
        self.error_count = 0
        self.error_log: list[dict[str, Any]] = []

    def is_recoverable(self, exc: Exception) -> bool:
        """Check if error is recoverable."""
        return isinstance(exc, self.RECOVERABLE_ERRORS) or "recoverable" in str(exc).lower()

    def is_critical(self, exc: Exception) -> bool:
        """Check if error is critical."""
        return isinstance(exc, self.CRITICAL_ERRORS)

    def log_error(
        self,
        exc: Exception,
        context: str = "",
        severity: str = "ERROR",
    ) -> None:
        """Log error with context."""
        self.error_count += 1
        error_record = {
            "count": self.error_count,
            "type": exc.__class__.__name__,
            "message": str(exc),
            "context": context,
            "severity": severity,
            "traceback": traceback.format_exc(),
        }
        self.error_log.append(error_record)

        if severity == "CRITICAL":
            self.logger.error(f"CRITICAL ERROR [{self.error_count}] in {context}: {exc}")
        else:
            self.logger.warning(f"ERROR [{self.error_count}] in {context}: {exc}")

    def handle_exception(
        self,
        exc: Exception,
        context: str = "",
        allow_continue: bool = False,
    ) -> bool:
        """
        Handle exception and return whether to continue.

        Parameters
        ----------
        exc : Exception
            The exception to handle
        context : str
            Context description for logging
        allow_continue : bool
            If True, recoverable errors allow continuation

        Returns
        -------
        bool
            True if execution should continue, False if should stop
        """
        if self.is_critical(exc):
            self.log_error(exc, context, severity="CRITICAL")
            return False

        if self.is_recoverable(exc):
            self.log_error(exc, context, severity="WARNING")
            return allow_continue

        self.log_error(exc, context, severity="ERROR")
        return False

    def save_error_log(self, path: str) -> None:
        """Save error log to file."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.error_log, f, indent=2, default=str)

    def get_summary(self) -> dict[str, Any]:
        """Get error summary."""
        return {
            "total_errors": self.error_count,
            "by_type": self._count_by_type(),
            "by_severity": self._count_by_severity(),
        }

    def _count_by_type(self) -> dict[str, int]:
        """Count errors by type."""
        counts = {}
        for error in self.error_log:
            error_type = error["type"]
            counts[error_type] = counts.get(error_type, 0) + 1
        return counts

    def _count_by_severity(self) -> dict[str, int]:
        """Count errors by severity."""
        counts = {}
        for error in self.error_log:
            severity = error["severity"]
            counts[severity] = counts.get(severity, 0) + 1
        return counts


def safe_cleanup(func: Callable) -> Callable:
    """Decorator to ensure cleanup happens even on error."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        cleanup_funcs = kwargs.pop("cleanup_funcs", [])
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            for cleanup_func in cleanup_funcs:
                try:
                    cleanup_func()
                except Exception as cleanup_exc:
                    print(f"Error during cleanup: {cleanup_exc}")
            raise

    return wrapper
