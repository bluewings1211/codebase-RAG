"""Utilities for handling async generator issues in sync contexts."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any


async def safe_async_generator_to_list(async_gen: AsyncGenerator) -> list[Any]:
    """Convert async generator to list safely.

    Args:
        async_gen: An async generator to convert

    Returns:
        A list containing all items from the async generator
    """
    try:
        result = []
        async for item in async_gen:
            result.append(item)
        return result
    except Exception:
        return []


def sync_wrapper_for_async_generator(async_gen_func):
    """Wrapper to handle async generators in sync context.

    This wrapper handles the common case where an async generator needs
    to be consumed from a synchronous context, properly managing the
    event loop in both running and non-running states.

    Args:
        async_gen_func: An async generator function to wrap

    Returns:
        A synchronous wrapper function
    """

    def wrapper(*args, **kwargs):
        try:
            # Get the event loop
            try:
                loop = asyncio.get_running_loop()
                is_running = True
            except RuntimeError:
                loop = asyncio.new_event_loop()
                is_running = False

            if is_running:
                # If in async context, create new thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(lambda: asyncio.run(safe_async_generator_to_list(async_gen_func(*args, **kwargs))))
                    return future.result()
            else:
                # If not in async context, run normally
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(safe_async_generator_to_list(async_gen_func(*args, **kwargs)))
                finally:
                    loop.close()
        except Exception:
            return []

    return wrapper


async def run_async_in_sync(coro):
    """Run a coroutine from a potentially sync context.

    Handles the case where we need to run an async function but might
    already be inside an event loop.

    Args:
        coro: A coroutine to run

    Returns:
        The result of the coroutine
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, create one
        return asyncio.run(coro)

    # Already in a loop - use ThreadPoolExecutor
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()


__all__ = [
    "safe_async_generator_to_list",
    "sync_wrapper_for_async_generator",
    "run_async_in_sync",
]
