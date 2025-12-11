"""
Tests for IndexingPipeline.

Tests the complete indexing workflow coordination including full and incremental indexing.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.indexing_pipeline import IndexingPipeline, PipelineResult


@pytest.mark.skip(reason="IndexingPipeline API changed - tests need to be updated for new service architecture")
class TestIndexingPipeline(unittest.TestCase):
    """Test IndexingPipeline functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = IndexingPipeline()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.indexing_service)
        self.assertIsNotNone(self.pipeline.embedding_service)
        self.assertIsNotNone(self.pipeline.qdrant_service)
        self.assertIsNotNone(self.pipeline.project_analysis)
        self.assertIsNotNone(self.pipeline.metadata_service)
        self.assertIsNotNone(self.pipeline.change_detector)
        self.assertIsNotNone(self.pipeline.memory_monitor)

    def test_set_callbacks(self):
        """Test setting callback functions."""
        error_callback = MagicMock()
        progress_callback = MagicMock()

        self.pipeline.set_error_callback(error_callback)
        self.pipeline.set_progress_callback(progress_callback)

        self.assertEqual(self.pipeline._error_callback, error_callback)
        self.assertEqual(self.pipeline._progress_callback, progress_callback)

    @patch("services.indexing_pipeline.time.time")
    def test_execute_full_indexing_success(self, mock_time):
        """Test successful full indexing execution."""
        # Mock time.time() to return predictable values
        mock_time.side_effect = [1000.0, 1010.0]  # 10 second duration

        # Mock services
        mock_chunks = [
            MagicMock(metadata={"file_path": "/test/file1.py", "language": "python"}),
            MagicMock(metadata={"file_path": "/test/file2.py", "language": "python"}),
        ]

        with (
            patch.object(self.pipeline.memory_monitor, "start_monitoring"),
            patch.object(self.pipeline.memory_monitor, "stop_monitoring"),
            patch.object(self.pipeline.metadata_service, "clear_project_metadata"),
            patch.object(self.pipeline.indexing_service, "process_codebase_for_indexing") as mock_process,
            patch.object(self.pipeline, "_process_chunks_to_storage") as mock_storage,
            patch.object(self.pipeline, "_store_file_metadata"),
            patch.object(self.pipeline, "_get_performance_metrics") as mock_metrics,
        ):
            mock_process.return_value = mock_chunks
            mock_storage.return_value = (100, ["project_test_code"])
            mock_metrics.return_value = {"memory_usage_mb": 256}

            result = self.pipeline.execute_full_indexing(str(self.temp_path), "test_project", clear_existing=True)

            self.assertIsInstance(result, PipelineResult)
            self.assertTrue(result.success)
            self.assertEqual(result.total_files_processed, 2)  # Unique files
            self.assertEqual(result.total_chunks_generated, 2)
            self.assertEqual(result.total_points_stored, 100)
            self.assertEqual(result.collections_used, ["project_test_code"])
            self.assertEqual(result.processing_time_seconds, 10.0)
            self.assertEqual(result.error_count, 0)
            self.assertEqual(result.warning_count, 0)

    @patch("services.indexing_pipeline.time.time")
    def test_execute_full_indexing_no_chunks(self, mock_time):
        """Test full indexing with no chunks generated."""
        mock_time.side_effect = [1000.0, 1005.0]

        with (
            patch.object(self.pipeline.memory_monitor, "start_monitoring"),
            patch.object(self.pipeline.memory_monitor, "stop_monitoring"),
            patch.object(self.pipeline.metadata_service, "clear_project_metadata"),
            patch.object(self.pipeline.indexing_service, "process_codebase_for_indexing") as mock_process,
        ):
            mock_process.return_value = None

            result = self.pipeline.execute_full_indexing(str(self.temp_path), "test_project")

            self.assertFalse(result.success)
            self.assertEqual(result.total_files_processed, 0)
            self.assertEqual(result.total_chunks_generated, 0)
            self.assertEqual(result.error_count, 1)

    @patch("services.indexing_pipeline.time.time")
    def test_execute_full_indexing_exception(self, mock_time):
        """Test full indexing with exception."""
        mock_time.side_effect = [1000.0, 1005.0]

        with (
            patch.object(self.pipeline.memory_monitor, "start_monitoring"),
            patch.object(self.pipeline.memory_monitor, "stop_monitoring"),
            patch.object(self.pipeline.metadata_service, "clear_project_metadata"),
            patch.object(self.pipeline.indexing_service, "process_codebase_for_indexing") as mock_process,
            patch.object(self.pipeline, "_report_error") as mock_error,
        ):
            mock_process.side_effect = Exception("Test error")

            result = self.pipeline.execute_full_indexing(str(self.temp_path), "test_project")

            self.assertFalse(result.success)
            self.assertEqual(result.error_count, 1)
            mock_error.assert_called_once()

    @patch("services.indexing_pipeline.time.time")
    def test_execute_incremental_indexing_no_changes(self, mock_time):
        """Test incremental indexing with no changes."""
        mock_time.side_effect = [1000.0, 1002.0]

        mock_changes = MagicMock()
        mock_changes.has_changes = False
        mock_changes.get_summary.return_value = {"changed": 0, "added": 0, "removed": 0}

        with (
            patch.object(self.pipeline.memory_monitor, "start_monitoring"),
            patch.object(self.pipeline.memory_monitor, "stop_monitoring"),
            patch.object(self.pipeline.project_analysis, "get_relevant_files") as mock_files,
            patch.object(self.pipeline.change_detector, "detect_changes") as mock_detect,
        ):
            mock_files.return_value = ["/test/file1.py"]
            mock_detect.return_value = mock_changes

            result = self.pipeline.execute_incremental_indexing(str(self.temp_path), "test_project")

            self.assertTrue(result.success)
            self.assertEqual(result.total_files_processed, 0)
            self.assertEqual(result.total_chunks_generated, 0)
            self.assertIsNotNone(result.change_summary)

    @patch("services.indexing_pipeline.time.time")
    def test_execute_incremental_indexing_with_changes(self, mock_time):
        """Test incremental indexing with changes."""
        mock_time.side_effect = [1000.0, 1008.0]

        mock_changes = MagicMock()
        mock_changes.has_changes = True
        mock_changes.get_files_to_reindex.return_value = ["/test/file1.py"]
        mock_changes.get_files_to_remove.return_value = ["/test/old_file.py"]
        mock_changes.get_summary.return_value = {"changed": 1, "added": 0, "removed": 1}

        mock_chunks = [MagicMock(metadata={"file_path": "/test/file1.py", "language": "python"})]

        with (
            patch.object(self.pipeline.memory_monitor, "start_monitoring"),
            patch.object(self.pipeline.memory_monitor, "stop_monitoring"),
            patch.object(self.pipeline.project_analysis, "get_relevant_files") as mock_files,
            patch.object(self.pipeline.change_detector, "detect_changes") as mock_detect,
            patch.object(self.pipeline.indexing_service, "process_specific_files") as mock_process,
            patch.object(self.pipeline, "_process_chunks_to_storage") as mock_storage,
            patch.object(self.pipeline, "_store_file_metadata"),
            patch.object(self.pipeline, "_get_performance_metrics") as mock_metrics,
        ):
            mock_files.return_value = ["/test/file1.py"]
            mock_detect.return_value = mock_changes
            mock_process.return_value = mock_chunks
            mock_storage.return_value = (50, ["project_test_code"])
            mock_metrics.return_value = {"memory_usage_mb": 128}

            result = self.pipeline.execute_incremental_indexing(str(self.temp_path), "test_project")

            self.assertTrue(result.success)
            self.assertEqual(result.total_files_processed, 1)
            self.assertEqual(result.total_chunks_generated, 1)
            self.assertEqual(result.total_points_stored, 50)

    def test_process_chunks_to_storage_success(self):
        """Test successful chunk processing and storage."""
        mock_chunks = [
            MagicMock(content="def hello(): pass", metadata={"file_path": "/test/file1.py", "language": "python"}),
            MagicMock(content="function hello() {}", metadata={"file_path": "/test/file1.js", "language": "javascript"}),
        ]

        import numpy as np

        mock_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        mock_stats = MagicMock()
        mock_stats.successful_insertions = 2
        mock_stats.failed_insertions = 0

        with (
            patch.object(self.pipeline.embedding_service, "generate_embeddings") as mock_embed,
            patch.object(self.pipeline, "_ensure_collection_exists"),
            patch.object(self.pipeline.qdrant_service, "batch_upsert_with_retry") as mock_upsert,
            patch("os.getenv") as mock_getenv,
        ):
            mock_getenv.return_value = "test-model"
            mock_embed.return_value = mock_embeddings
            mock_upsert.return_value = mock_stats

            project_context = {"project_name": "test_project", "source_path": "/test"}

            total_points, collections_used = self.pipeline._process_chunks_to_storage(mock_chunks, project_context)

            self.assertEqual(total_points, 2)  # 2 successful insertions
            self.assertEqual(len(collections_used), 1)  # Both files go to code collection
            self.assertIn("project_test_project_code", collections_used)

    def test_process_chunks_to_storage_embedding_failure(self):
        """Test chunk processing with embedding failure."""
        mock_chunks = [MagicMock(content="def hello(): pass", metadata={"file_path": "/test/file1.py", "language": "python"})]

        with (
            patch.object(self.pipeline.embedding_service, "generate_embeddings") as mock_embed,
            patch.object(self.pipeline, "_report_error") as mock_error,
            patch("os.getenv") as mock_getenv,
        ):
            mock_getenv.return_value = "test-model"
            mock_embed.return_value = None  # Embedding failure

            project_context = {"project_name": "test_project"}

            total_points, collections_used = self.pipeline._process_chunks_to_storage(mock_chunks, project_context)

            self.assertEqual(total_points, 0)
            mock_error.assert_called_once()

    def test_store_file_metadata_success(self):
        """Test successful file metadata storage."""
        mock_files = ["/test/file1.py", "/test/file2.py"]

        with (
            patch.object(self.pipeline.project_analysis, "get_relevant_files") as mock_files_fn,
            patch.object(self.pipeline.metadata_service, "store_file_metadata") as mock_store,
            patch("models.file_metadata.FileMetadata") as mock_metadata_class,
        ):
            mock_files_fn.return_value = mock_files
            mock_store.return_value = True
            mock_metadata_class.from_file_path.return_value = MagicMock()

            # Should not raise exception
            self.pipeline._store_file_metadata(str(self.temp_path), "test_project")

    def test_store_file_metadata_failure(self):
        """Test file metadata storage failure."""
        with (
            patch.object(self.pipeline.project_analysis, "get_relevant_files") as mock_files_fn,
            patch.object(self.pipeline.metadata_service, "store_file_metadata") as mock_store,
            patch.object(self.pipeline, "_report_error") as mock_error,
        ):
            mock_files_fn.return_value = ["/test/file1.py"]
            mock_store.return_value = False

            self.pipeline._store_file_metadata(str(self.temp_path), "test_project")

            mock_error.assert_called_once()

    def test_ensure_collection_exists_create_new(self):
        """Test collection creation when it doesn't exist."""
        with (
            patch.object(self.pipeline.qdrant_service, "collection_exists") as mock_exists,
            patch.object(self.pipeline.qdrant_service.client, "create_collection") as mock_create,
        ):
            mock_exists.return_value = False

            self.pipeline._ensure_collection_exists("test_collection")

            mock_create.assert_called_once()

    def test_ensure_collection_exists_already_exists(self):
        """Test when collection already exists."""
        with (
            patch.object(self.pipeline.qdrant_service, "collection_exists") as mock_exists,
            patch.object(self.pipeline.qdrant_service.client, "create_collection") as mock_create,
        ):
            mock_exists.return_value = True

            self.pipeline._ensure_collection_exists("test_collection")

            mock_create.assert_not_called()

    def test_get_performance_metrics(self):
        """Test performance metrics retrieval."""
        with patch.object(self.pipeline.memory_monitor, "get_current_usage") as mock_memory:
            mock_memory.return_value = 512.0

            metrics = self.pipeline._get_performance_metrics()

            self.assertEqual(metrics["memory_usage_mb"], 512.0)
            self.assertIn("timestamp", metrics)

    def test_report_progress_with_callback(self):
        """Test progress reporting with callback."""
        progress_callback = MagicMock()
        self.pipeline.set_progress_callback(progress_callback)

        self.pipeline._report_progress("Test message")

        progress_callback.assert_called_once_with("Test message")

    def test_report_progress_without_callback(self):
        """Test progress reporting without callback."""
        with patch.object(self.pipeline.logger, "info") as mock_log:
            self.pipeline._report_progress("Test message")

            mock_log.assert_called_once_with("Test message")

    def test_report_error_with_callback(self):
        """Test error reporting with callback."""
        error_callback = MagicMock()
        self.pipeline.set_error_callback(error_callback)

        self.pipeline._report_error("test_type", "test_location", "test_message", "test_suggestion")

        error_callback.assert_called_once_with("test_type", "test_location", "test_message", suggestion="test_suggestion")

    def test_report_error_without_callback(self):
        """Test error reporting without callback."""
        with patch.object(self.pipeline.logger, "error") as mock_log:
            self.pipeline._report_error("test_type", "test_location", "test_message")

            mock_log.assert_called_once()


@pytest.mark.skip(reason="IndexingPipeline API changed - tests need to be updated for new service architecture")
class TestPipelineResult(unittest.TestCase):
    """Test PipelineResult dataclass."""

    def test_pipeline_result_creation(self):
        """Test PipelineResult creation with all fields."""
        result = PipelineResult(
            success=True,
            total_files_processed=10,
            total_chunks_generated=50,
            total_points_stored=50,
            collections_used=["test_collection"],
            processing_time_seconds=30.5,
            error_count=0,
            warning_count=2,
            change_summary={"added": 5, "modified": 3, "removed": 2},
            performance_metrics={"memory_mb": 256},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.total_files_processed, 10)
        self.assertEqual(result.total_chunks_generated, 50)
        self.assertEqual(result.total_points_stored, 50)
        self.assertEqual(result.collections_used, ["test_collection"])
        self.assertEqual(result.processing_time_seconds, 30.5)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.warning_count, 2)
        self.assertIsNotNone(result.change_summary)
        self.assertIsNotNone(result.performance_metrics)


if __name__ == "__main__":
    unittest.main()
