"""
Indexing Pipeline Service for coordinating the entire indexing workflow.

This service orchestrates the complete indexing process, including file discovery,
change detection, processing, embedding generation, and storage operations.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from qdrant_client.http.models import PointStruct
from services.change_detector_service import ChangeDetectorService
from services.embedding_service import EmbeddingService
from services.file_metadata_service import FileMetadataService
from services.indexing_service import IndexingService
from services.project_analysis_service import ProjectAnalysisService
from services.qdrant_service import QdrantService
from utils.performance_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of an indexing pipeline execution."""

    success: bool
    total_files_processed: int
    total_chunks_generated: int
    total_points_stored: int
    collections_used: list[str]
    processing_time_seconds: float
    error_count: int
    warning_count: int
    change_summary: dict[str, Any] | None = None
    performance_metrics: dict[str, Any] | None = None


class IndexingPipeline:
    """
    Complete indexing pipeline for coordinating all indexing operations.

    This service manages the entire indexing workflow from file discovery
    through final storage, with support for both full and incremental indexing.
    """

    def __init__(self):
        """Initialize the indexing pipeline."""
        self.logger = logger

        # Initialize core services
        self.indexing_service = IndexingService()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        self.project_analysis = ProjectAnalysisService()
        self.metadata_service = FileMetadataService(self.qdrant_service)
        self.change_detector = ChangeDetectorService(self.metadata_service)
        self.memory_monitor = MemoryMonitor()

        # Pipeline state
        self._current_operation = None
        self._error_callback = None
        self._progress_callback = None

    def set_error_callback(self, callback):
        """Set callback function for error reporting."""
        self._error_callback = callback

    def set_progress_callback(self, callback):
        """Set callback function for progress reporting."""
        self._progress_callback = callback

    def execute_full_indexing(self, directory: str, project_name: str, clear_existing: bool = True) -> PipelineResult:
        """
        Execute full indexing pipeline.

        Args:
            directory: Directory to index
            project_name: Name of the project
            clear_existing: Whether to clear existing data

        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()

        try:
            self._current_operation = "full_indexing"
            self._report_progress("Starting full indexing pipeline")

            # Initialize monitoring
            self.memory_monitor.start_monitoring()

            # Clear existing metadata if requested
            if clear_existing:
                self._report_progress("Clearing existing metadata")
                self.metadata_service.clear_project_metadata(project_name)

            # Process codebase
            self._report_progress("Processing codebase for intelligent chunking")
            chunks = self.indexing_service.process_codebase_for_indexing(directory)

            if not chunks:
                return PipelineResult(
                    success=False,
                    total_files_processed=0,
                    total_chunks_generated=0,
                    total_points_stored=0,
                    collections_used=[],
                    processing_time_seconds=time.time() - start_time,
                    error_count=1,
                    warning_count=0,
                )

            # Generate embeddings and store
            self._report_progress("Generating embeddings and storing to vector database")
            points_stored, collections_used = self._process_chunks_to_storage(
                chunks, {"project_name": project_name, "source_path": directory}
            )

            # Store file metadata
            self._report_progress("Storing file metadata for change detection")
            self._store_file_metadata(directory, project_name)

            # Calculate final metrics
            processing_time = time.time() - start_time

            return PipelineResult(
                success=True,
                total_files_processed=len({chunk.metadata.get("file_path") for chunk in chunks}),
                total_chunks_generated=len(chunks),
                total_points_stored=points_stored,
                collections_used=collections_used,
                processing_time_seconds=processing_time,
                error_count=0,
                warning_count=0,
                performance_metrics=self._get_performance_metrics(),
            )

        except Exception as e:
            self.logger.error(f"Full indexing pipeline failed: {e}")
            self._report_error("pipeline", "Full indexing failed", str(e))

            return PipelineResult(
                success=False,
                total_files_processed=0,
                total_chunks_generated=0,
                total_points_stored=0,
                collections_used=[],
                processing_time_seconds=time.time() - start_time,
                error_count=1,
                warning_count=0,
            )
        finally:
            try:
                self.memory_monitor.stop_monitoring()
            except Exception:
                pass

    def execute_incremental_indexing(self, directory: str, project_name: str) -> PipelineResult:
        """
        Execute incremental indexing pipeline.

        Args:
            directory: Directory to index
            project_name: Name of the project

        Returns:
            PipelineResult with execution details
        """
        start_time = time.time()

        try:
            self._current_operation = "incremental_indexing"
            self._report_progress("Starting incremental indexing pipeline")

            # Initialize monitoring
            self.memory_monitor.start_monitoring()

            # Get current files and detect changes
            self._report_progress("Analyzing files for changes")
            relevant_files = self.project_analysis.get_relevant_files(directory)
            changes = self.change_detector.detect_changes(
                project_name=project_name,
                current_files=relevant_files,
                project_root=directory,
            )

            if not changes.has_changes:
                self._report_progress("No changes detected - indexing complete")
                return PipelineResult(
                    success=True,
                    total_files_processed=0,
                    total_chunks_generated=0,
                    total_points_stored=0,
                    collections_used=[],
                    processing_time_seconds=time.time() - start_time,
                    error_count=0,
                    warning_count=0,
                    change_summary=changes.get_summary(),
                )

            # Process changed files
            files_to_reindex = changes.get_files_to_reindex()
            files_to_remove = changes.get_files_to_remove()

            total_points_stored = 0
            collections_used = []

            # Remove obsolete entries
            if files_to_remove:
                self._report_progress(f"Removing {len(files_to_remove)} obsolete entries")

                # Import deletion function
                from tools.project.project_utils import delete_file_chunks

                total_removed_points = 0
                successful_deletions = 0

                for file_path in files_to_remove:
                    try:
                        # Delete chunks for this file from all collections
                        result = delete_file_chunks(file_path)

                        if result.get("error") is None:
                            deleted_points = result.get("deleted_points", 0)
                            total_removed_points += deleted_points
                            successful_deletions += 1

                            if deleted_points > 0:
                                self.logger.info(f"Removed {deleted_points} points for deleted file: {file_path}")
                            else:
                                self.logger.debug(f"No points found for deleted file: {file_path}")
                        else:
                            error_msg = result.get("error", "Unknown error")
                            self.logger.warning(f"Failed to remove file {file_path}: {error_msg}")

                    except Exception as e:
                        self.logger.error(f"Error removing file {file_path}: {e}")
                        continue

                self.logger.info(
                    f"Deletion summary: {successful_deletions}/{len(files_to_remove)} files processed, "
                    f"{total_removed_points} points removed from vector database"
                )

            # Process changed files
            if files_to_reindex:
                self._report_progress(f"Processing {len(files_to_reindex)} changed files")
                chunks = self.indexing_service.process_specific_files(files_to_reindex, project_name, directory)

                if chunks:
                    points_stored, collections = self._process_chunks_to_storage(
                        chunks, {"project_name": project_name, "source_path": directory}
                    )
                    total_points_stored = points_stored
                    collections_used = collections

            # Update file metadata for processed files
            self._report_progress("Updating file metadata")
            files_to_update_metadata = files_to_reindex if files_to_reindex else []
            self._store_file_metadata(directory, project_name, specific_files=files_to_update_metadata)

            # Remove file metadata for deleted files
            if files_to_remove:
                self._report_progress("Removing metadata for deleted files")

                try:
                    # Remove metadata for all deleted files at once
                    success = self.metadata_service.remove_file_metadata(project_name, files_to_remove)

                    if success:
                        self.logger.info(f"Successfully removed metadata for {len(files_to_remove)} deleted files")
                        for file_path in files_to_remove:
                            self.logger.debug(f"Removed metadata for: {file_path}")
                    else:
                        self.logger.warning(f"Failed to remove metadata for {len(files_to_remove)} deleted files")

                except Exception as e:
                    self.logger.error(f"Error during metadata cleanup: {e}")

            processing_time = time.time() - start_time

            return PipelineResult(
                success=True,
                total_files_processed=len(files_to_reindex),
                total_chunks_generated=len(chunks) if files_to_reindex and chunks else 0,
                total_points_stored=total_points_stored,
                collections_used=collections_used,
                processing_time_seconds=processing_time,
                error_count=0,
                warning_count=0,
                change_summary=changes.get_summary(),
                performance_metrics=self._get_performance_metrics(),
            )

        except Exception as e:
            self.logger.error(f"Incremental indexing pipeline failed: {e}")
            self._report_error("pipeline", "Incremental indexing failed", str(e))

            return PipelineResult(
                success=False,
                total_files_processed=0,
                total_chunks_generated=0,
                total_points_stored=0,
                collections_used=[],
                processing_time_seconds=time.time() - start_time,
                error_count=1,
                warning_count=0,
            )
        finally:
            try:
                self.memory_monitor.stop_monitoring()
            except Exception:
                pass

    def _process_chunks_to_storage(self, chunks: list, project_context: dict[str, Any]) -> tuple[int, list[str]]:
        """
        Process chunks into embeddings and store them.

        Args:
            chunks: List of chunks to process
            project_context: Project context information

        Returns:
            Tuple of (total_points_stored, collections_used)
        """
        import os

        # Get embedding model
        model_name = os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

        # Group chunks by collection type
        collection_chunks = defaultdict(list)

        for chunk in chunks:
            file_path = chunk.metadata.get("file_path", "")
            language = chunk.metadata.get("language", "unknown")

            # Determine collection type
            if language in ["python", "javascript", "typescript", "java", "go", "rust"]:
                collection_type = "code"
            elif any(file_path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".toml", ".ini"]):
                collection_type = "config"
            else:
                collection_type = "documentation"

            project_name = project_context.get("project_name", "unknown")
            collection_name = f"project_{project_name}_{collection_type}"
            collection_chunks[collection_name].append(chunk)

        total_points = 0
        collections_used = list(collection_chunks.keys())

        # Process each collection
        for collection_name, collection_chunk_list in collection_chunks.items():
            try:
                # Generate embeddings
                texts = [chunk.content for chunk in collection_chunk_list]
                embeddings = self.embedding_service.generate_embeddings(model_name, texts)

                if embeddings is None:
                    self._report_error("embedding", collection_name, "Failed to generate embeddings", "Check Ollama service availability")
                    continue

                # Create points
                points = []
                for chunk, embedding in zip(collection_chunk_list, embeddings, strict=False):
                    if embedding is not None:
                        point_id = str(uuid4())
                        metadata = chunk.metadata.copy()
                        metadata["collection"] = collection_name
                        # CRITICAL FIX: Include chunk content in payload
                        metadata["content"] = chunk.content

                        point = PointStruct(id=point_id, vector=embedding.tolist(), payload=metadata)
                        points.append(point)

                if points:
                    # Ensure collection exists
                    self._ensure_collection_exists(collection_name)

                    # Store points
                    stats = self.qdrant_service.batch_upsert_with_retry(collection_name, points)
                    total_points += stats.successful_insertions

                    if stats.failed_insertions > 0:
                        self._report_error(
                            "storage",
                            collection_name,
                            f"{stats.failed_insertions} points failed to store",
                            "Check Qdrant connection and disk space",
                        )

            except Exception as e:
                self._report_error("processing", collection_name, f"Collection processing failed: {str(e)}")

        return total_points, collections_used

    def _store_file_metadata(self, directory: str, project_name: str, specific_files: list[str] | None = None):
        """
        Store file metadata for change detection.

        Args:
            directory: Directory being indexed
            project_name: Name of the project
            specific_files: Optional list of specific files to store metadata for.
                           If None, stores metadata for all relevant files.
        """
        try:
            # Use specific files if provided, otherwise get all relevant files
            if specific_files:
                files_to_process = specific_files
            else:
                files_to_process = self.project_analysis.get_relevant_files(directory)

            from models.file_metadata import FileMetadata

            metadata_list = []

            for file_path in files_to_process:
                try:
                    metadata = FileMetadata.from_file_path(file_path, directory)
                    metadata_list.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to create metadata for {file_path}: {e}")

            success = self.metadata_service.store_file_metadata(project_name, metadata_list)

            if not success:
                self._report_error("metadata", directory, "Failed to store file metadata")

        except Exception as e:
            self._report_error("metadata", directory, f"Error storing file metadata: {str(e)}")

    def _ensure_collection_exists(self, collection_name: str):
        """Ensure collection exists before storing data."""
        try:
            if not self.qdrant_service.collection_exists(collection_name):
                from qdrant_client.http.models import Distance, VectorParams

                self.qdrant_service.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )

        except Exception as e:
            self.logger.error(f"Failed to ensure collection {collection_name} exists: {e}")
            raise

    def _get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics."""
        return {
            "memory_usage_mb": self.memory_monitor.get_current_usage(),
            "timestamp": datetime.now().isoformat(),
        }

    def _report_progress(self, message: str):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(message)
        else:
            self.logger.info(message)

    def _report_error(self, error_type: str, location: str, message: str, suggestion: str = ""):
        """Report error if callback is set."""
        if self._error_callback:
            self._error_callback(error_type, location, message, suggestion=suggestion)
        else:
            self.logger.error(f"{error_type.upper()} in {location}: {message}")
