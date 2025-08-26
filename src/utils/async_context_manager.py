"""
Async Context Manager for reliable event loop management across the codebase.

This module provides comprehensive async context management to prevent
"Task got Future attached to different loop" errors and other asyncio issues.
"""

import asyncio
import functools
import logging
import threading
import weakref
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Global registry for tracking active event loops
_active_loops = weakref.WeakSet()
_loop_lock = threading.Lock()


class AsyncContextError(Exception):
    """Exception raised for async context management errors."""

    pass


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """
    Get current event loop or create a new one safely.

    Returns:
        Event loop instance
    """
    try:
        loop = asyncio.get_running_loop()
        return loop
    except RuntimeError:
        # No running loop, create new one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


def is_async_context() -> bool:
    """Check if we're currently in an async context."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


@asynccontextmanager
async def safe_async_context():
    """
    Context manager for safe async operations.

    Ensures proper event loop handling and cleanup.
    """
    loop = None
    try:
        loop = get_or_create_event_loop()
        with _loop_lock:
            _active_loops.add(loop)

        yield loop

    except Exception as e:
        logger.error(f"Error in async context: {e}")
        raise AsyncContextError(f"Async context failed: {e}") from e
    finally:
        if loop:
            try:
                # Don't close the loop if it's still running
                if not loop.is_running() and not loop.is_closed():
                    # Clean up pending tasks
                    pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
                    if pending_tasks:
                        logger.debug(f"Cancelling {len(pending_tasks)} pending tasks")
                        for task in pending_tasks:
                            task.cancel()

                        # Wait for tasks to be cancelled
                        try:
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                        except Exception as e:
                            logger.debug(f"Error cancelling tasks: {e}")
            except Exception as e:
                logger.debug(f"Error in async context cleanup: {e}")


def run_in_thread_pool(coro_func: Callable[..., Coroutine[Any, Any, T]], *args, **kwargs) -> T:
    """
    Run async function in a thread pool with proper event loop isolation.

    Args:
        coro_func: Async function to run
        *args: Arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the async function
    """
    import concurrent.futures

    def run_in_new_loop():
        """Run coroutine in a new event loop."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro_func(*args, **kwargs))
        finally:
            try:
                # Cancel remaining tasks
                pending = asyncio.all_tasks(new_loop)
                for task in pending:
                    task.cancel()

                if pending:
                    new_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.debug(f"Error cleaning up event loop: {e}")
            finally:
                new_loop.close()

    # Check if we're in an async context
    try:
        asyncio.get_running_loop()
        # We're in an async context, run in thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_loop)
            return future.result(timeout=300)  # 5-minute timeout
    except RuntimeError:
        # No running loop, safe to run directly
        return asyncio.run(coro_func(*args, **kwargs))


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """
    Decorator to convert async function to sync with proper loop handling.

    Args:
        func: Async function to convert

    Returns:
        Sync wrapper function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        return run_in_thread_pool(func, *args, **kwargs)

    return wrapper


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator to convert sync function to async.

    Args:
        func: Sync function to convert

    Returns:
        Async wrapper function
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_running_loop()
        # Run sync function in thread pool to avoid blocking the event loop
        return await loop.run_in_executor(None, func, *args, **kwargs)

    return wrapper


def ensure_async_safe(func: Callable | None = None, *, timeout: float = 300):
    """
    Decorator to ensure async operations are safe across different contexts.

    Args:
        func: Function to decorate (optional for decorator with parameters)
        timeout: Timeout in seconds for async operations

    Returns:
        Decorated function
    """

    def decorator(f):
        if asyncio.iscoroutinefunction(f):

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                try:
                    # Check if we're in an async context
                    asyncio.get_running_loop()
                    # We're in an async context, run directly
                    return f(*args, **kwargs)
                except RuntimeError:
                    # No running loop, run with timeout
                    return asyncio.wait_for(asyncio.run(f(*args, **kwargs)), timeout=timeout)

        else:

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


class AsyncExecutor:
    """Helper class for executing async operations safely."""

    def __init__(self, timeout: float = 300):
        """
        Initialize async executor.

        Args:
            timeout: Default timeout for operations
        """
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def run(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """
        Run coroutine safely with proper event loop handling.

        Args:
            coro: Coroutine to run
            timeout: Optional timeout override

        Returns:
            Result of the coroutine
        """
        timeout = timeout or self.timeout

        try:
            # Check if we're in an async context
            asyncio.get_running_loop()
            # Run in thread pool to avoid loop conflicts
            return run_in_thread_pool(lambda: coro)
        except RuntimeError:
            # No running loop, safe to run directly
            async def run_with_timeout():
                return await asyncio.wait_for(coro, timeout=timeout)

            return asyncio.run(run_with_timeout())

    async def run_sync(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Run sync function in async context.

        Args:
            func: Sync function to run
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the function
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)


# Global async executor instance
async_executor = AsyncExecutor()


def handle_async_errors(func: Callable) -> Callable:
    """
    Decorator to handle common async errors gracefully.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with error handling
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout in {func.__name__}: {e}")
            raise AsyncContextError(f"Operation timed out: {func.__name__}") from e
        except asyncio.CancelledError as e:
            logger.warning(f"Operation cancelled in {func.__name__}: {e}")
            raise
        except RuntimeError as e:
            if "event loop" in str(e).lower():
                logger.error(f"Event loop error in {func.__name__}: {e}")
                raise AsyncContextError(f"Event loop error: {func.__name__}") from e
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
