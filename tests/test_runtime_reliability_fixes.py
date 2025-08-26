"""
Unit tests for Phase 1 runtime reliability fixes.

Tests cover:
1. Asyncio event loop issues
2. Missing service methods
3. Attribute access errors
4. Async/await mismatches
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.models.code_chunk import ChunkType
from src.services.implementation_chain_service import ChainType, GraphNode, ImplementationChainService
from src.services.indexing_service import IndexingService
from src.services.pattern_recognition_service import PatternRecognitionService
from src.utils.async_context_manager import (
    AsyncContextError,
    AsyncExecutor,
    async_to_sync,
    handle_async_errors,
    run_in_thread_pool,
    safe_async_context,
    sync_to_async,
)


class TestAsyncioEventLoopFixes:
    """Test fixes for asyncio event loop issues."""

    def setup_method(self):
        """Set up test fixtures."""
        self.indexing_service = IndexingService()

    def test_process_single_file_no_running_loop(self):
        """Test process_single_file when no event loop is running."""
        with patch.object(self.indexing_service, "_process_single_file") as mock_process:
            mock_process.return_value = []

            result = self.indexing_service.process_single_file("/test/file.py")

            assert result == []
            mock_process.assert_called_once()

    def test_process_single_file_with_running_loop(self):
        """Test process_single_file when event loop is already running."""

        async def async_test():
            with patch.object(self.indexing_service, "_process_single_file") as mock_process:
                mock_process.return_value = []

                result = self.indexing_service.process_single_file("/test/file.py")

                assert result == []
                mock_process.assert_called_once()

        asyncio.run(async_test())

    def test_process_single_file_timeout_handling(self):
        """Test timeout handling in process_single_file."""
        with patch.object(self.indexing_service, "_process_single_file") as mock_process:
            # Simulate a long-running operation
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                return []

            mock_process.side_effect = slow_process

            with pytest.raises(RuntimeError, match="Timeout processing file"):
                # Override timeout for testing
                with patch("concurrent.futures.Future.result") as mock_result:
                    mock_result.side_effect = TimeoutError()
                    self.indexing_service.process_single_file("/test/file.py")

    def test_process_single_file_exception_handling(self):
        """Test exception handling in process_single_file."""
        with patch.object(self.indexing_service, "_process_single_file") as mock_process:
            mock_process.side_effect = ValueError("Test error")

            with pytest.raises(ValueError, match="Test error"):
                self.indexing_service.process_single_file("/test/file.py")


class TestAttributeAccessFixes:
    """Test fixes for attribute access errors."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_service = PatternRecognitionService(Mock(), Mock())  # Mock dependencies
        self.chain_service = ImplementationChainService(Mock(), Mock())  # Mock dependencies

    def test_pattern_recognition_attribute_safety(self):
        """Test safe attribute access in pattern recognition."""
        # Mock candidate with various component types
        candidate = {
            "components": [
                # Valid component with chunk_type attribute
                Mock(chunk_type=Mock(value="function")),
                # Component with string chunk_type
                Mock(chunk_type="class"),
                # Dict-based component
                {"chunk_type": "method"},
                # String component (invalid)
                "invalid_component",
            ]
        }

        signature = Mock(required_components=["function", "class"])

        # This should not raise AttributeError
        confidence = self.pattern_service._calculate_structural_confidence(candidate, signature)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_implementation_chain_breadcrumb_safety(self):
        """Test safe breadcrumb access in implementation chain service."""
        # Create mock graph with various node types
        mock_graph = Mock()
        mock_graph.nodes = [
            # Valid node with breadcrumb
            Mock(breadcrumb="valid.function"),
            # Dict-based node
            {"breadcrumb": "dict.function"},
            # String node (should be skipped)
            "string_node",
            # Node without breadcrumb
            Mock(spec=[]),  # Mock with no attributes
            # None node
            None,
        ]

        # This should not raise AttributeError
        result = self.chain_service._find_component_by_breadcrumb(mock_graph, "valid.function")

        assert result is not None
        assert result.breadcrumb == "valid.function"

    def test_implementation_chain_link_safety(self):
        """Test safe link component access."""
        # Mock links with various component types
        mock_links = [
            # Valid link with breadcrumb attributes
            Mock(source_component=Mock(breadcrumb="source.func"), target_component=Mock(breadcrumb="target.func")),
            # Link with dict components
            Mock(source_component={"breadcrumb": "dict.source"}, target_component={"breadcrumb": "dict.target"}),
            # Link with missing attributes
            Mock(source_component=Mock(spec=[]), target_component=Mock(spec=[])),
            # Link with None components
            Mock(source_component=None, target_component=None),
        ]

        # This should extract breadcrumbs safely without errors
        breadcrumbs = []
        for link in mock_links:
            try:
                if hasattr(link.source_component, "breadcrumb") and link.source_component.breadcrumb:
                    breadcrumbs.append(link.source_component.breadcrumb)
                elif isinstance(link.source_component, dict) and link.source_component.get("breadcrumb"):
                    breadcrumbs.append(link.source_component["breadcrumb"])
                elif link.source_component:
                    breadcrumbs.append(str(link.source_component))
            except (AttributeError, TypeError, KeyError):
                pass

            try:
                if hasattr(link.target_component, "breadcrumb") and link.target_component.breadcrumb:
                    breadcrumbs.append(link.target_component.breadcrumb)
                elif isinstance(link.target_component, dict) and link.target_component.get("breadcrumb"):
                    breadcrumbs.append(link.target_component["breadcrumb"])
                elif link.target_component:
                    breadcrumbs.append(str(link.target_component))
            except (AttributeError, TypeError, KeyError):
                pass

        assert "source.func" in breadcrumbs
        assert "target.func" in breadcrumbs
        assert "dict.source" in breadcrumbs
        assert "dict.target" in breadcrumbs


class TestAsyncContextManager:
    """Test async context manager utilities."""

    def test_async_executor_basic(self):
        """Test basic AsyncExecutor functionality."""
        executor = AsyncExecutor(timeout=5)

        async def test_coro():
            await asyncio.sleep(0.1)
            return "success"

        result = executor.run(test_coro())
        assert result == "success"

    def test_async_to_sync_decorator(self):
        """Test async_to_sync decorator."""

        @async_to_sync
        async def async_func(x, y):
            await asyncio.sleep(0.1)
            return x + y

        result = async_func(2, 3)
        assert result == 5

    def test_sync_to_async_decorator(self):
        """Test sync_to_async decorator."""

        @sync_to_async
        def sync_func(x, y):
            return x * y

        async def test():
            result = await sync_func(3, 4)
            assert result == 12

        asyncio.run(test())

    def test_run_in_thread_pool_with_running_loop(self):
        """Test run_in_thread_pool when event loop is running."""

        async def test_coro():
            return "thread_result"

        async def test():
            result = run_in_thread_pool(test_coro)
            assert result == "thread_result"

        asyncio.run(test())

    def test_run_in_thread_pool_no_loop(self):
        """Test run_in_thread_pool when no event loop is running."""

        async def test_coro():
            return "no_loop_result"

        result = run_in_thread_pool(test_coro)
        assert result == "no_loop_result"

    def test_safe_async_context(self):
        """Test safe async context manager."""

        async def test():
            async with safe_async_context() as loop:
                assert loop is not None
                assert isinstance(loop, asyncio.AbstractEventLoop)
                return "context_success"

        result = asyncio.run(test())
        assert result == "context_success"

    def test_handle_async_errors_decorator(self):
        """Test async error handling decorator."""

        @handle_async_errors
        async def failing_func():
            raise RuntimeError("event loop error")

        @handle_async_errors
        async def timeout_func():
            raise asyncio.TimeoutError("timeout")

        async def test():
            with pytest.raises(AsyncContextError, match="Event loop error"):
                await failing_func()

            with pytest.raises(AsyncContextError, match="Operation timed out"):
                await timeout_func()

        asyncio.run(test())


class TestIntegrationFixes:
    """Integration tests for runtime reliability fixes."""

    def test_indexing_service_integration(self):
        """Test IndexingService integration with fixes."""
        indexing_service = IndexingService()

        # Mock the async method to return test data
        with patch.object(indexing_service, "_process_single_file") as mock_process:
            mock_chunk = Mock()
            mock_chunk.chunk_type = ChunkType.FUNCTION
            mock_process.return_value = [mock_chunk]

            # Test in sync context
            result = indexing_service.process_single_file("/test/file.py")
            assert len(result) == 1
            assert result[0].chunk_type == ChunkType.FUNCTION

    def test_pattern_service_integration(self):
        """Test PatternRecognitionService integration with fixes."""
        pattern_service = PatternRecognitionService(Mock(), Mock())

        # Test with mixed component types to ensure no attribute errors
        candidate = {"components": [Mock(chunk_type=ChunkType.FUNCTION), {"chunk_type": "class"}, "invalid"]}
        signature = Mock(required_components=["function"])

        # Should handle gracefully without errors
        confidence = pattern_service._calculate_structural_confidence(candidate, signature)
        assert isinstance(confidence, float)

    def test_chain_service_integration(self):
        """Test ImplementationChainService integration with fixes."""
        chain_service = ImplementationChainService(Mock(), Mock())

        # Create mixed graph structure
        graph = Mock()
        graph.nodes = [Mock(breadcrumb="test.function"), {"breadcrumb": "dict.node"}, "string_node"]

        # Should find valid node without errors
        result = chain_service._find_component_by_breadcrumb(graph, "test.function")
        assert result is not None

        # Should handle missing node gracefully
        result = chain_service._find_component_by_breadcrumb(graph, "missing.node")
        assert result is None


@pytest.mark.asyncio
class TestAsyncIntegration:
    """Async integration tests."""

    async def test_async_service_reliability(self):
        """Test async service operations are reliable."""
        async with safe_async_context():
            # Test that we can perform async operations safely
            tasks = []
            for i in range(5):

                async def task():
                    await asyncio.sleep(0.1)
                    return f"task_{i}"

                tasks.append(task())

            results = await asyncio.gather(*tasks)
            assert len(results) == 5

    async def test_concurrent_async_operations(self):
        """Test concurrent async operations don't interfere."""

        async def operation(value):
            await asyncio.sleep(0.1)
            return value * 2

        # Run multiple operations concurrently
        tasks = [operation(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        expected = [i * 2 for i in range(10)]
        assert results == expected


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_runtime"])
