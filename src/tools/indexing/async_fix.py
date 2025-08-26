"""Temporary fix for async generator issues in index_tools.py"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any


async def fix_async_generator_issue():
    """
    Fix for the async generator coroutine issue.
    This function properly handles async generators and converts them to sync iterables.
    """

    async def safe_async_generator_to_list(async_gen: AsyncGenerator) -> list[Any]:
        """Convert async generator to list safely."""
        try:
            result = []
            async for item in async_gen:
                result.append(item)
            return result
        except Exception:
            # Error converting async generator: {e}
            return []

    def sync_wrapper_for_async_generator(async_gen_func):
        """Wrapper to handle async generators in sync context."""

        def wrapper(*args, **kwargs):
            try:
                # Get the event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If in async context, create new thread
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(lambda: asyncio.run(safe_async_generator_to_list(async_gen_func(*args, **kwargs))))
                        return future.result()
                else:
                    # If not in async context, run normally
                    return asyncio.run(safe_async_generator_to_list(async_gen_func(*args, **kwargs)))
            except Exception:
                # Error in async generator wrapper: {e}
                return []

        return wrapper

    return safe_async_generator_to_list, sync_wrapper_for_async_generator


# Export the fix
__all__ = ["fix_async_generator_issue"]
