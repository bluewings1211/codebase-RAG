"""
Tests for ManualIndexingTool.

Tests the refactored manual indexing tool coordinator functionality.
"""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Now we can import the manual indexing tool
sys.path.insert(0, str(Path(__file__).parent.parent))
from manual_indexing import ManualIndexingTool


def async_test(coro):
    """Decorator to run async test methods."""

    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro(*args, **kwargs))

    return wrapper


class TestManualIndexingTool(unittest.TestCase):
    """Test ManualIndexingTool functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default(self):
        """Test tool initialization with default parameters."""
        tool = ManualIndexingTool()

        self.assertFalse(tool.verbose)
        self.assertIsNotNone(tool.file_discovery)
        self.assertIsNotNone(tool.pipeline)
        self.assertIsNotNone(tool.reporter)
        self.assertIsNone(tool.error_report_dir)
        self.assertIsNotNone(tool.logger)

    def test_init_verbose(self):
        """Test tool initialization with verbose mode."""
        tool = ManualIndexingTool(verbose=True)

        self.assertTrue(tool.verbose)

    @patch("manual_indexing.setup_logging")
    def test_setup_logging_default(self, mock_setup_logging):
        """Test logging setup with default (non-verbose) mode."""
        ManualIndexingTool(verbose=False)

        # Verify logging was configured via setup_logging
        mock_setup_logging.assert_called_once()
        call_kwargs = mock_setup_logging.call_args[1]
        self.assertFalse(call_kwargs.get("verbose", True))

    @patch("manual_indexing.setup_logging")
    def test_setup_logging_verbose(self, mock_setup_logging):
        """Test logging setup with verbose mode."""
        ManualIndexingTool(verbose=True)

        # Verify logging was configured via setup_logging
        mock_setup_logging.assert_called_once()
        call_kwargs = mock_setup_logging.call_args[1]
        self.assertTrue(call_kwargs.get("verbose", False))

    def test_validate_arguments_valid_mode_and_directory(self):
        """Test argument validation with valid inputs."""
        tool = ManualIndexingTool()

        with patch.object(tool.file_discovery, "validate_directory") as mock_validate:
            mock_validate.return_value = (True, "")

            is_valid, error_msg = tool.validate_arguments(str(self.temp_path), "clear_existing")

            self.assertTrue(is_valid)
            self.assertEqual(error_msg, "")
            mock_validate.assert_called_once_with(str(self.temp_path))

    def test_validate_arguments_invalid_mode(self):
        """Test argument validation with invalid mode."""
        tool = ManualIndexingTool()

        is_valid, error_msg = tool.validate_arguments(str(self.temp_path), "invalid_mode")

        self.assertFalse(is_valid)
        self.assertIn("Invalid mode", error_msg)
        self.assertIn("invalid_mode", error_msg)

    def test_validate_arguments_invalid_directory(self):
        """Test argument validation with invalid directory."""
        tool = ManualIndexingTool()

        with patch.object(tool.file_discovery, "validate_directory") as mock_validate:
            mock_validate.return_value = (False, "Directory not found")

            is_valid, error_msg = tool.validate_arguments("/nonexistent", "clear_existing")

            self.assertFalse(is_valid)
            self.assertEqual(error_msg, "Directory not found")

    def test_check_dependencies_all_available(self):
        """Test dependency checking when all services are available."""
        tool = ManualIndexingTool()

        # Mock successful Qdrant connection
        mock_collections = MagicMock()
        mock_collections.collections = ["collection1", "collection2"]

        with patch.object(tool.pipeline.qdrant_service.client, "get_collections") as mock_get_collections:
            mock_get_collections.return_value = mock_collections

            deps_ok, missing = tool.check_dependencies()

            self.assertTrue(deps_ok)
            self.assertEqual(missing, [])

    def test_check_dependencies_qdrant_failure(self):
        """Test dependency checking when Qdrant is unavailable."""
        tool = ManualIndexingTool()

        with patch.object(tool.pipeline.qdrant_service.client, "get_collections") as mock_get_collections:
            mock_get_collections.side_effect = Exception("Connection failed")

            deps_ok, missing = tool.check_dependencies()

            self.assertFalse(deps_ok)
            self.assertEqual(len(missing), 1)
            self.assertIn("Qdrant", missing[0])

    def test_check_dependencies_embedding_service_missing(self):
        """Test dependency checking when embedding service is misconfigured."""
        tool = ManualIndexingTool()

        # Mock successful Qdrant but missing embedding service method
        mock_collections = MagicMock()
        mock_collections.collections = []

        with (
            patch.object(tool.pipeline.qdrant_service.client, "get_collections") as mock_get_collections,
            patch.object(tool.pipeline, "embedding_service", spec=[]),
        ):  # No generate_embeddings method
            mock_get_collections.return_value = mock_collections

            deps_ok, missing = tool.check_dependencies()

            self.assertFalse(deps_ok)
            self.assertTrue(any("Embedding service" in m for m in missing))

    @patch("builtins.print")
    def test_show_pre_indexing_summary_with_files(self, mock_print):
        """Test pre-indexing summary display with files to process."""
        tool = ManualIndexingTool()

        mock_estimates = {"file_count": 100, "total_size_mb": 25.5, "estimated_minutes": 2.5}

        with patch.object(tool.file_discovery, "estimate_processing_requirements") as mock_estimate:
            mock_estimate.return_value = mock_estimates

            tool.show_pre_indexing_summary(str(self.temp_path), "clear_existing")

            # Verify print was called multiple times
            self.assertTrue(mock_print.called)
            mock_estimate.assert_called_once_with(str(self.temp_path), "clear_existing")

    @patch("builtins.print")
    def test_show_pre_indexing_summary_incremental_no_changes(self, mock_print):
        """Test pre-indexing summary for incremental with no changes."""
        tool = ManualIndexingTool()

        mock_estimates = {"file_count": 0, "total_size_mb": 0, "estimated_minutes": 0}

        with patch.object(tool.file_discovery, "estimate_processing_requirements") as mock_estimate:
            mock_estimate.return_value = mock_estimates

            tool.show_pre_indexing_summary(str(self.temp_path), "incremental")

            self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_show_pre_indexing_summary_long_operation(self, mock_print):
        """Test pre-indexing summary display for long operations."""
        tool = ManualIndexingTool()

        mock_estimates = {"file_count": 5000, "total_size_mb": 500.0, "estimated_minutes": 30.0}

        with patch.object(tool.file_discovery, "estimate_processing_requirements") as mock_estimate:
            mock_estimate.return_value = mock_estimates

            tool.show_pre_indexing_summary(str(self.temp_path), "clear_existing")

            self.assertTrue(mock_print.called)

    @async_test
    @patch("builtins.print")
    async def test_perform_indexing_clear_existing_success(self, mock_print):
        """Test successful clear_existing indexing operation."""
        tool = ManualIndexingTool()

        # Mock discovery result
        mock_discovery = {"project_context": {"project_name": "test_project"}, "relevant_files": ["/test/file1.py"], "file_count": 1}

        # Mock pipeline result - all attributes must be proper values (not MagicMock) for comparisons
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_files_processed = 1
        mock_result.total_chunks_generated = 5
        mock_result.total_points_stored = 5
        mock_result.collections_used = ["project_test_project_code"]
        mock_result.performance_metrics = {"memory_usage_mb": 256}
        mock_result.embedding_failures = 0  # Integer for comparison
        mock_result.embedding_success_rate = 100.0  # Float for formatting
        mock_result.failed_chunks = []

        with (
            patch.object(tool.file_discovery, "discover_project_files") as mock_discover,
            patch.object(tool.reporter, "start_report") as mock_start_report,
            patch.object(tool.pipeline, "execute_full_indexing") as mock_execute,
            patch.object(tool.reporter, "finalize_report") as mock_finalize,
            patch.object(tool.reporter, "update_file_counts") as mock_update,
            patch.object(tool.reporter, "extract_syntax_errors_from_indexing_service") as mock_extract,
            patch.object(tool.reporter, "print_summary") as mock_summary,
        ):
            mock_discover.return_value = mock_discovery
            mock_execute.return_value = mock_result

            success = await tool.perform_indexing(str(self.temp_path), "clear_existing")

            self.assertTrue(success)
            mock_discover.assert_called_once()
            mock_start_report.assert_called_once_with("clear_existing", str(self.temp_path), "test_project")
            mock_execute.assert_called_once_with(str(self.temp_path), "test_project", clear_existing=True)
            mock_finalize.assert_called_once()
            mock_update.assert_called_once()
            mock_extract.assert_called_once()
            mock_summary.assert_called_once()

    @async_test
    @patch("builtins.print")
    async def test_perform_indexing_incremental_success(self, mock_print):
        """Test successful incremental indexing operation."""
        tool = ManualIndexingTool()

        # Mock discovery result
        mock_discovery = {"project_context": {"project_name": "test_project"}, "relevant_files": ["/test/file1.py"], "file_count": 1}

        # Mock pipeline result - all attributes must be proper values (not MagicMock) for comparisons
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_files_processed = 0  # No changes
        mock_result.total_chunks_generated = 0
        mock_result.total_points_stored = 0
        mock_result.collections_used = []
        mock_result.performance_metrics = {"memory_usage_mb": 128}
        mock_result.embedding_failures = 0  # Integer for comparison
        mock_result.embedding_success_rate = 100.0  # Float for formatting
        mock_result.failed_chunks = []

        with (
            patch.object(tool.file_discovery, "discover_project_files") as mock_discover,
            patch.object(tool.reporter, "start_report"),
            patch.object(tool.pipeline, "execute_incremental_indexing") as mock_execute,
            patch.object(tool.reporter, "finalize_report"),
            patch.object(tool.reporter, "update_file_counts"),
            patch.object(tool.reporter, "extract_syntax_errors_from_indexing_service"),
            patch.object(tool.reporter, "print_summary"),
        ):
            mock_discover.return_value = mock_discovery
            mock_execute.return_value = mock_result

            success = await tool.perform_indexing(str(self.temp_path), "incremental")

            self.assertTrue(success)
            mock_execute.assert_called_once_with(str(self.temp_path), "test_project")

    @async_test
    @patch("builtins.print")
    async def test_perform_indexing_discovery_failure(self, mock_print):
        """Test indexing operation with discovery failure."""
        tool = ManualIndexingTool()

        # Mock discovery failure
        mock_discovery = {"error": "Failed to discover files"}

        with patch.object(tool.file_discovery, "discover_project_files") as mock_discover:
            mock_discover.return_value = mock_discovery

            success = await tool.perform_indexing(str(self.temp_path), "clear_existing")

            self.assertFalse(success)

    @async_test
    @patch("builtins.print")
    async def test_perform_indexing_unknown_mode(self, mock_print):
        """Test indexing operation with unknown mode."""
        tool = ManualIndexingTool()

        # Mock discovery result
        mock_discovery = {"project_context": {"project_name": "test_project"}, "relevant_files": ["/test/file1.py"]}

        with patch.object(tool.file_discovery, "discover_project_files") as mock_discover, patch.object(tool.reporter, "start_report"):
            mock_discover.return_value = mock_discovery

            success = await tool.perform_indexing(str(self.temp_path), "unknown_mode")

            self.assertFalse(success)

    @async_test
    @patch("builtins.print")
    async def test_perform_indexing_exception_handling(self, mock_print):
        """Test indexing operation with exception handling."""
        tool = ManualIndexingTool()

        with patch.object(tool.file_discovery, "discover_project_files") as mock_discover:
            mock_discover.side_effect = Exception("Test exception")

            success = await tool.perform_indexing(str(self.temp_path), "clear_existing")

            self.assertFalse(success)

    @async_test
    @patch("builtins.print")
    async def test_perform_indexing_with_error_report_saving(self, mock_print):
        """Test indexing operation with error report saving."""
        tool = ManualIndexingTool(verbose=True)  # Verbose mode to trigger report saving
        tool.error_report_dir = str(self.temp_path)

        # Mock discovery result
        mock_discovery = {"project_context": {"project_name": "test_project"}, "relevant_files": ["/test/file1.py"]}

        # Mock pipeline result with success
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_files_processed = 1
        mock_result.total_chunks_generated = 5
        mock_result.total_points_stored = 5
        mock_result.collections_used = ["test_collection"]
        mock_result.performance_metrics = {}
        mock_result.embedding_failures = 0  # Set as integer for comparison
        mock_result.embedding_success_rate = 100.0

        # Mock report with some errors to trigger saving
        mock_report = MagicMock()
        mock_report.get_error_summary.return_value = {"syntax": 1}
        mock_report.embedding_failures = 0  # Set as integer for comparison
        mock_report.has_critical_errors.return_value = False

        with (
            patch.object(tool.file_discovery, "discover_project_files") as mock_discover,
            patch.object(tool.reporter, "start_report"),
            patch.object(tool.pipeline, "execute_full_indexing") as mock_execute,
            patch.object(tool.reporter, "finalize_report"),
            patch.object(tool.reporter, "update_file_counts"),
            patch.object(tool.reporter, "extract_syntax_errors_from_indexing_service"),
            patch.object(tool.reporter, "print_summary"),
            patch.object(tool.reporter, "current_report", mock_report),
            patch.object(tool.reporter, "save_report") as mock_save,
        ):
            mock_discover.return_value = mock_discovery
            mock_execute.return_value = mock_result
            mock_save.return_value = "/path/to/report.json"

            success = await tool.perform_indexing(str(self.temp_path), "clear_existing")

            self.assertTrue(success)
            mock_save.assert_called_once_with(str(self.temp_path))

    def test_handle_pipeline_error(self):
        """Test pipeline error handling."""
        tool = ManualIndexingTool()

        with patch.object(tool.reporter, "add_error") as mock_add_error:
            tool._handle_pipeline_error("syntax", "/test/file.py", "Syntax error", "Fix the syntax")

            mock_add_error.assert_called_once_with(
                error_type="syntax", file_path="/test/file.py", error_message="Syntax error", suggestion="Fix the syntax"
            )

    @patch("builtins.print")
    def test_handle_progress_verbose(self, mock_print):
        """Test progress handling in verbose mode."""
        tool = ManualIndexingTool(verbose=True)

        tool._handle_progress("Processing files...")

        mock_print.assert_called_once_with("🔄 Processing files...")

    @patch("builtins.print")
    def test_handle_progress_non_verbose(self, mock_print):
        """Test progress handling in non-verbose mode."""
        tool = ManualIndexingTool(verbose=False)

        tool._handle_progress("Processing files...")

        # Should not print in non-verbose mode
        mock_print.assert_not_called()


class TestManualIndexingToolIntegration(unittest.TestCase):
    """Integration tests for ManualIndexingTool with actual service coordination."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create test files
        test_file = self.temp_path / "test.py"
        test_file.write_text("print('hello world')")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_callback_integration(self):
        """Test that callbacks are properly set up between services."""
        tool = ManualIndexingTool()

        # Verify callbacks are set
        self.assertIsNotNone(tool.pipeline._error_callback)
        self.assertIsNotNone(tool.pipeline._progress_callback)

        # Test error callback integration
        with patch.object(tool.reporter, "add_error") as mock_add_error:
            tool.pipeline._error_callback("test_type", "test_location", "test_message")
            mock_add_error.assert_called_once()

        # Test progress callback integration
        # The progress callback should call the actual handle progress method
        tool.pipeline._progress_callback("test_message")
        # Since _handle_progress may just log in non-verbose mode, we just verify it doesn't error


if __name__ == "__main__":
    unittest.main()
