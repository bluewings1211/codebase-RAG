import gc
import logging
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any

import psutil
from git import GitCommandError, Repo
from models.code_chunk import ChunkType
from models.code_chunk import CodeChunk as ParsedCodeChunk
from services.code_parser_service import CodeParserService
from services.project_analysis_service import ProjectAnalysisService
from utils.performance_monitor import MemoryMonitor, ProgressTracker
from utils.stage_logger import (
    get_file_discovery_logger,
    get_file_reading_logger,
    log_batch_summary,
    log_timing,
)


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any]


class IndexingService:
    def __init__(self):
        self.project_analysis_service = ProjectAnalysisService()
        self.code_parser_service = CodeParserService()
        self._lock = Lock()  # For thread-safe operations
        self._error_files = []  # Track files that failed processing
        self._processed_count = 0  # Atomic counter for processed files
        self._reset_counters()  # Reset processing counters
        self._setup_thread_safe_logging()

        # Initialize memory monitoring with configurable threshold
        memory_threshold = float(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))
        self.memory_monitor = MemoryMonitor(warning_threshold_mb=memory_threshold)

        # Initialize stage-specific loggers
        self.file_discovery_logger = get_file_discovery_logger()
        self.file_reading_logger = get_file_reading_logger()

        # Progress tracker for external monitoring
        self.progress_tracker: ProgressTracker | None = None

    def _sanitize_file_path(self, file_path: str, base_directory: str) -> str:
        """Convert absolute file path to relative path for security."""
        try:
            # Convert to absolute paths to handle edge cases
            abs_file_path = os.path.abspath(file_path)
            abs_base_dir = os.path.abspath(base_directory)

            # Get relative path
            relative_path = os.path.relpath(abs_file_path, abs_base_dir)

            # Ensure path doesn't go outside the base directory (security check)
            if relative_path.startswith(".."):
                self.logger.warning(f"File path {file_path} is outside base directory {base_directory}")
                return os.path.basename(file_path)  # Fallback to just filename

            return relative_path
        except Exception as e:
            self.logger.warning(f"Error sanitizing path {file_path}: {e}")
            return os.path.basename(file_path)  # Fallback to just filename

    def process_codebase_for_indexing(
        self,
        source_path: str,
        incremental_mode: bool = False,
        project_name: str | None = None,
    ) -> list[Chunk]:
        self.logger.info(f"Processing codebase from: {source_path}")

        is_git_url = source_path.startswith(("http://", "https://", "git@"))

        if is_git_url:
            temp_dir = tempfile.mkdtemp()
            try:
                self.logger.info(f"Cloning {source_path} into {temp_dir}")
                Repo.clone_from(source_path, temp_dir)
                directory_to_index = temp_dir
            except GitCommandError as e:
                self.logger.error(f"Error cloning repository: {e}")
                shutil.rmtree(temp_dir)
                return []
        else:
            directory_to_index = source_path

        # Stage 1: File Discovery with detailed logging
        with self.file_discovery_logger.stage("file_discovery", directory=directory_to_index) as stage:
            discovery_start = time.time()
            relevant_files = self.project_analysis_service.get_relevant_files(directory_to_index)
            discovery_duration = time.time() - discovery_start

            stage.item_count = len(relevant_files)
            stage.processed_count = len(relevant_files)

            log_timing(
                self.file_discovery_logger,
                "file_discovery",
                discovery_duration,
                files_found=len(relevant_files),
                directory=directory_to_index,
            )

        if not relevant_files:
            self.logger.warning("No relevant files found to process.")
            if is_git_url:
                shutil.rmtree(temp_dir)
            return []

        # Handle incremental mode
        files_to_process = relevant_files
        if incremental_mode and project_name:
            self.logger.info("Incremental mode enabled - detecting file changes...")
            files_to_process = self._get_files_for_incremental_indexing(relevant_files, directory_to_index, project_name)

            if not files_to_process:
                self.logger.info("No changes detected - all files are up to date!")
                if is_git_url:
                    shutil.rmtree(temp_dir)
                return []

            self.logger.info(f"Incremental indexing: processing {len(files_to_process)} changed files out of {len(relevant_files)} total")

        chunks = []
        self._error_files = []  # Reset error tracking

        # Get and validate concurrency settings
        max_workers = self._get_optimal_worker_count()
        batch_size = int(os.getenv("INDEXING_BATCH_SIZE", "20"))

        self.logger.info(f"Processing {len(files_to_process)} files with {max_workers} workers (batch size: {batch_size})...")

        # Initialize progress tracking
        self.progress_tracker = ProgressTracker(len(files_to_process), "Indexing codebase files")

        # Monitor initial memory usage
        initial_memory = self.memory_monitor.check_memory_usage(self.logger)
        self.logger.info(f"Initial memory usage: {initial_memory['memory_mb']} MB")

        # Stage 2: File Reading and Processing with detailed logging
        with self.file_reading_logger.stage("file_processing", item_count=len(files_to_process)) as processing_stage:
            # Process files in batches to manage memory
            for batch_start in range(0, len(files_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(files_to_process))
                batch_files = files_to_process[batch_start:batch_end]
                batch_num = batch_start // batch_size + 1

                # Only log batch start at intervals to reduce noise
                if batch_num % 50 == 1 or batch_num == 1:
                    self.file_reading_logger.info(f"Processing batch {batch_num}: files {batch_start + 1}-{batch_end}")
                batch_start_time = time.time()

                # Use ThreadPoolExecutor with proper resource management
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    try:
                        # Submit batch processing tasks
                        future_to_file = {
                            executor.submit(
                                self._process_single_file,
                                file_path,
                                project_name,
                                directory_to_index,
                            ): file_path
                            for file_path in batch_files
                        }

                        batch_processed = 0
                        batch_failed = 0

                        # Collect results as they complete
                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            thread_id = threading.get_ident()
                            try:
                                file_chunks = future.result()
                                if file_chunks:  # Only add non-None chunks
                                    chunks.extend(file_chunks)  # Extend to add all chunks from the file
                                    batch_processed += 1
                                    self.file_reading_logger.log_item_processed(
                                        "file_processing",
                                        file_path=file_path,
                                        chunks_created=len(file_chunks),
                                    )
                                    with self._lock:
                                        self._processed_count += 1
                                        processing_stage.processed_count = self._processed_count
                                        self.progress_tracker.increment_processed()
                                        self.logger.debug(
                                            f"[Thread-{thread_id}] Successfully processed file {file_path} -> {len(file_chunks)} chunks ({self._processed_count}/{len(relevant_files)})"
                                        )
                            except Exception as e:
                                batch_failed += 1
                                self.file_reading_logger.log_item_failed("file_processing", error=str(e), file_path=file_path)
                                with self._lock:
                                    self._error_files.append((file_path, str(e)))
                                    self.progress_tracker.increment_failed()
                                self.logger.error(f"[Thread-{thread_id}] Error processing file {file_path}: {e}")
                            finally:
                                # Clean up future reference
                                del future_to_file[future]

                    except Exception as e:
                        self.logger.error(f"Error in batch processing: {e}")

                    finally:
                        # Ensure ThreadPoolExecutor cleanup
                        executor.shutdown(wait=True)

                # Log batch completion (only logs if there are failures or at intervals)
                batch_duration = time.time() - batch_start_time
                log_batch_summary(
                    self.file_reading_logger,
                    batch_num,
                    len(batch_files),
                    batch_processed,
                    batch_failed,
                    batch_duration,
                    log_interval=50,  # Log every 50 batches
                )

                # Memory cleanup between batches
                self._cleanup_memory()

                # Monitor memory usage with automatic warnings (only log at intervals or if high)
                memory_stats = self.memory_monitor.check_memory_usage(self.logger)
                # Only log memory if it's high (>80%) or at batch intervals
                if memory_stats["memory_percent"] > 80 or batch_num % 50 == 0:
                    self.logger.info(
                        f"[PID:{os.getpid()}] - Batch completed. Memory usage: {memory_stats['memory_mb']} MB ({memory_stats['memory_percent']}%)"
                    )

        # Report any errors
        if self._error_files:
            self.logger.error(f"Failed to process {len(self._error_files)} files:")
            for file_path, error in self._error_files:
                self.logger.error(f"  - {file_path}: {error}")

        # Final processing summary
        self.logger.info(f"Processing completed: {len(chunks)} chunks created, {len(self._error_files)} errors")

        if is_git_url:
            self.logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        return chunks

    def get_progress_summary(self) -> dict[str, Any] | None:
        """Get current progress summary for external monitoring."""
        if self.progress_tracker is None:
            return None

        return self.progress_tracker.get_progress_summary()

    def _process_single_file(
        self,
        file_path: str,
        project_name: str | None = None,
        base_directory: str | None = None,
    ) -> list[Chunk]:
        """Process a single file using intelligent chunking and return list of Chunks. Thread-safe worker function."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Use intelligent code parsing for supported languages
            parse_result = self.code_parser_service.parse_file(file_path, content)

            # Convert parsed chunks to the existing Chunk format
            chunks = []
            if parse_result.chunks:
                for i, parsed_chunk in enumerate(parse_result.chunks):
                    chunk = self._convert_parsed_chunk_to_chunk(parsed_chunk, i, project_name, base_directory)
                    chunks.append(chunk)
            else:
                # Fallback to whole-file chunk if no intelligent chunks were created
                chunks.append(self._create_fallback_chunk(file_path, content, project_name, base_directory))

            return chunks

        except Exception as e:
            # Log the error and create a fallback chunk
            self.logger.warning(f"Failed to parse {file_path} intelligently, using fallback: {e}")
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                return [self._create_fallback_chunk(file_path, content, project_name, base_directory)]
            except Exception as fallback_error:
                # Let the calling code handle the exception
                raise fallback_error

    def _convert_parsed_chunk_to_chunk(
        self,
        parsed_chunk: ParsedCodeChunk,
        chunk_index: int,
        project_name: str | None = None,
        base_directory: str | None = None,
    ) -> Chunk:
        """Convert a ParsedCodeChunk to the existing Chunk format."""
        # Sanitize file path for security
        sanitized_path = self._sanitize_file_path(parsed_chunk.file_path, base_directory) if base_directory else parsed_chunk.file_path

        return Chunk(
            content=parsed_chunk.content,
            metadata={
                "file_path": sanitized_path,
                "full_path": parsed_chunk.file_path,  # Keep original for internal use
                "chunk_index": chunk_index,
                "line_start": parsed_chunk.start_line,
                "line_end": parsed_chunk.end_line,
                "language": parsed_chunk.language,
                "project": project_name or "unknown",
                # Additional intelligent chunking metadata
                "chunk_id": parsed_chunk.chunk_id,
                "chunk_type": parsed_chunk.chunk_type.value,
                "name": parsed_chunk.name,
                "parent_name": parsed_chunk.parent_name,
                "signature": parsed_chunk.signature,
                "docstring": parsed_chunk.docstring,
                "breadcrumb": parsed_chunk.breadcrumb,
                "content_hash": parsed_chunk.content_hash,
                "embedding_text": parsed_chunk.embedding_text,
                "tags": parsed_chunk.tags,
                "complexity_score": parsed_chunk.complexity_score,
                "dependencies": parsed_chunk.dependencies,
                "context_before": parsed_chunk.context_before,
                "context_after": parsed_chunk.context_after,
            },
        )

    def _create_fallback_chunk(
        self,
        file_path: str,
        content: str,
        project_name: str | None = None,
        base_directory: str | None = None,
    ) -> Chunk:
        """Create a fallback whole-file chunk when intelligent parsing fails."""
        # Sanitize file path for security
        sanitized_path = self._sanitize_file_path(file_path, base_directory) if base_directory else file_path

        return Chunk(
            content=content,
            metadata={
                "file_path": sanitized_path,
                "full_path": file_path,  # Keep original for internal use
                "chunk_index": 0,
                "line_start": 1,
                "line_end": len(content.splitlines()),
                "language": self._detect_language(file_path),
                "project": project_name or "unknown",
                "chunk_type": ChunkType.WHOLE_FILE.value,
                "fallback_used": True,
            },
        )

    def _get_optimal_worker_count(self) -> int:
        """Calculate optimal worker count based on CPU cores and configuration."""
        # Get configured concurrency or use default
        configured_workers = int(os.getenv("INDEXING_CONCURRENCY", "4"))

        # Get CPU count for optimization
        cpu_count = multiprocessing.cpu_count()

        # For I/O-bound operations like file reading, we can use more threads than CPU cores
        # But cap it at 2x CPU count to avoid too much context switching
        max_recommended = min(cpu_count * 2, 8)  # Cap at 8 to be conservative

        # Use the smaller of configured or recommended
        optimal_workers = min(configured_workers, max_recommended)

        # Ensure at least 1 worker
        optimal_workers = max(1, optimal_workers)

        if optimal_workers != configured_workers:
            self.logger.info(f"Adjusted worker count from {configured_workers} to {optimal_workers} based on CPU cores ({cpu_count})")

        return optimal_workers

    def _validate_configuration(self) -> dict[str, Any]:
        """Validate and return configuration settings with safe defaults."""
        config = {}

        # Validate concurrency settings
        try:
            config["concurrency"] = max(1, int(os.getenv("INDEXING_CONCURRENCY", "4")))
        except ValueError:
            self.logger.warning("Invalid INDEXING_CONCURRENCY value, using default: 4")
            config["concurrency"] = 4

        try:
            config["batch_size"] = max(1, int(os.getenv("INDEXING_BATCH_SIZE", "20")))
        except ValueError:
            self.logger.warning("Invalid INDEXING_BATCH_SIZE value, using default: 20")
            config["batch_size"] = 20

        return config

    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _cleanup_memory(self) -> None:
        """Force garbage collection to free memory."""
        try:
            # Force garbage collection
            gc.collect()

            # Additional cleanup for large objects
            if hasattr(gc, "collect"):
                # Run multiple collection cycles for thorough cleanup
                for _ in range(3):
                    collected = gc.collect()
                    if collected == 0:
                        break
        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")

    def _reset_counters(self) -> None:
        """Reset processing counters for new indexing operation."""
        with self._lock:
            self._processed_count = 0
            self._error_files = []

    def _setup_thread_safe_logging(self) -> None:
        """Setup thread-safe logging configuration."""
        # Create logger for this service
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Only configure if not already configured
        if not self.logger.handlers:
            # Set level from environment or default to INFO
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))

            # Create thread-safe formatter with thread ID
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - [PID:%(process)d] - %(message)s")

            # Create console handler (thread-safe by default)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Prevent duplicate logging
            self.logger.propagate = False

    def _detect_language(self, file_path: str) -> str:
        """Enhanced language detection based on file extension."""
        extension = os.path.splitext(file_path)[1].lower()

        # Create comprehensive language mapping
        language_map = {
            # Python
            ".py": "python",
            ".pyw": "python",
            ".pyi": "python",
            # JavaScript/TypeScript
            ".js": "javascript",
            ".jsx": "javascript",
            ".mjs": "javascript",
            ".cjs": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            # C/C++
            ".c": "cpp",
            ".h": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".cc": "cpp",
            ".hpp": "cpp",
            ".hxx": "cpp",
            ".hh": "cpp",
            # Java
            ".java": "java",
            ".class": "java",
            # C#
            ".cs": "csharp",
            ".csx": "csharp",
            # Go
            ".go": "go",
            # Rust
            ".rs": "rust",
            # Ruby
            ".rb": "ruby",
            ".rake": "ruby",
            ".gemspec": "ruby",
            # PHP
            ".php": "php",
            ".phtml": "php",
            ".php3": "php",
            ".php4": "php",
            ".php5": "php",
            # Swift
            ".swift": "swift",
            # Kotlin
            ".kt": "kotlin",
            ".kts": "kotlin",
            # Scala
            ".scala": "scala",
            ".sc": "scala",
            # Other languages
            ".clj": "clojure",
            ".cljs": "clojure",
            ".ex": "elixir",
            ".exs": "elixir",
            ".erl": "erlang",
            ".hrl": "erlang",
            ".hs": "haskell",
            ".lhs": "haskell",
            ".lua": "lua",
            ".pl": "perl",
            ".pm": "perl",
            ".r": "r",
            ".R": "r",
            ".m": "matlab",
            ".dart": "dart",
            # Shell scripts
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".fish": "fish",
            ".ps1": "powershell",
            ".psm1": "powershell",
            # Web technologies
            ".html": "html",
            ".htm": "html",
            ".xhtml": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            # Configuration formats
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "config",
            ".config": "config",
            ".xml": "xml",
            ".plist": "xml",
            # Documentation
            ".md": "markdown",
            ".markdown": "markdown",
            ".rst": "restructuredtext",
            ".txt": "text",
            ".adoc": "asciidoc",
            ".tex": "latex",
            # Database
            ".sql": "sql",
            # Infrastructure
            ".dockerfile": "dockerfile",
            ".tf": "terraform",
            ".tfvars": "terraform",
            # Build files
            ".makefile": "makefile",
            ".cmake": "cmake",
            ".gradle": "gradle",
            # Editor files
            ".vim": "vim",
            ".vimrc": "vim",
        }

        return language_map.get(extension, "unknown")

    def process_specific_files(
        self,
        file_paths: list[str],
        project_name: str | None = None,
        base_directory: str | None = None,
    ) -> list[Chunk]:
        """
        Process a specific list of files for incremental indexing.

        Args:
            file_paths: List of file paths to process
            project_name: Name of the project for metadata
            base_directory: Base directory for path sanitization

        Returns:
            List of Chunk objects
        """
        if not file_paths:
            self.logger.info("No files to process")
            return []

        self.logger.info(f"Processing {len(file_paths)} specific files for incremental indexing")

        chunks = []
        self._reset_counters()

        # Set up progress tracking
        if self.progress_tracker:
            self.progress_tracker.set_total_items(len(file_paths))

        # Process files in batches for memory efficiency
        batch_size = int(os.getenv("INDEXING_BATCH_SIZE", "10"))

        with self.file_reading_logger.stage("file_reading", file_count=len(file_paths)) as stage:
            stage.item_count = len(file_paths)

            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i : i + batch_size]
                batch_num = (i // batch_size) + 1

                # Only log batch start at intervals to reduce noise
                if batch_num % 50 == 1 or batch_num == 1:
                    self.logger.info(f"Processing batch {batch_num}: {len(batch_files)} files")
                batch_start_time = time.time()
                batch_processed = 0
                batch_failed = 0

                # Process batch with threading
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_to_file = {
                        executor.submit(
                            self._process_single_file,
                            file_path,
                            project_name,
                            base_directory,
                        ): file_path
                        for file_path in batch_files
                    }

                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            file_chunks = future.result()
                            if file_chunks:
                                chunks.extend(file_chunks)  # Extend to add all chunks from the file
                                batch_processed += 1

                                # Update progress
                                with self._lock:
                                    self._processed_count += 1
                                    if self.progress_tracker:
                                        self.progress_tracker.update_progress(self._processed_count)

                                self.logger.debug(f"Processed {file_path} -> {len(file_chunks)} chunks")
                        except Exception as e:
                            self.logger.error(f"Failed to process {file_path}: {e}")
                            with self._lock:
                                self._error_files.append((file_path, str(e)))
                                batch_failed += 1

                # Update stage progress
                stage.processed_count = self._processed_count

                # Log batch completion (only logs if there are failures or at intervals)
                batch_duration = time.time() - batch_start_time
                log_batch_summary(
                    self.file_reading_logger,
                    batch_num,
                    len(batch_files),
                    batch_processed,
                    batch_failed,
                    batch_duration,
                    log_interval=50,  # Log every 50 batches
                )

                # Memory cleanup between batches
                self._cleanup_memory()

                # Monitor memory usage (only log at intervals or if high)
                memory_stats = self.memory_monitor.check_memory_usage(self.logger)
                if memory_stats["memory_percent"] > 80 or batch_num % 50 == 0:
                    self.logger.info(f"Batch completed. Memory usage: {memory_stats['memory_mb']} MB ({memory_stats['memory_percent']}%)")

        # Report any errors
        if self._error_files:
            self.logger.error(f"Failed to process {len(self._error_files)} files:")
            for file_path, error in self._error_files:
                self.logger.error(f"  - {file_path}: {error}")

        self.logger.info(f"Specific file processing completed: {len(chunks)} chunks created, {len(self._error_files)} errors")
        return chunks

    def _get_files_for_incremental_indexing(self, relevant_files: list[str], directory: str, project_name: str) -> list[str]:
        """
        Get list of files that need to be processed for incremental indexing.

        Args:
            relevant_files: List of all relevant files
            directory: Project directory
            project_name: Name of the project

        Returns:
            List of files that need to be reindexed
        """
        try:
            from services.change_detector_service import ChangeDetectorService
            from services.file_metadata_service import FileMetadataService

            # Initialize services if not already done
            if not hasattr(self, "_metadata_service"):
                from services.qdrant_service import QdrantService

                self._metadata_service = FileMetadataService(QdrantService())
                self._change_detector = ChangeDetectorService(self._metadata_service)

            # Detect changes
            changes = self._change_detector.detect_changes(
                project_name=project_name,
                current_files=relevant_files,
                project_root=directory,
            )

            if not changes.has_changes:
                return []

            # Log change summary
            summary = changes.get_summary()
            self.logger.info(f"Change detection summary: {summary}")

            # Handle file deletions
            files_to_remove = changes.get_files_to_remove()
            if files_to_remove:
                self.logger.info(f"Removing {len(files_to_remove)} deleted files from index...")
                self._remove_deleted_files_from_index(files_to_remove, project_name)

            # Return files that need reindexing
            return changes.get_files_to_reindex()

        except Exception as e:
            self.logger.error(f"Error in incremental change detection: {e}")
            # Fallback to processing all files
            self.logger.warning("Falling back to full indexing due to change detection error")
            return relevant_files

    def _remove_deleted_files_from_index(self, file_paths: list[str], project_name: str) -> bool:
        """
        Remove deleted files from vector database.

        Args:
            file_paths: List of file paths to remove
            project_name: Name of the project

        Returns:
            True if removal was successful
        """
        try:
            if not file_paths:
                return True

            from services.qdrant_service import QdrantService

            # Initialize Qdrant service if not already done
            if not hasattr(self, "_qdrant_service"):
                self._qdrant_service = QdrantService()

            # Get project collections
            collections = self._qdrant_service.get_collections_by_pattern(f"project_{project_name}")

            if not collections:
                self.logger.warning(f"No collections found for project '{project_name}'")
                return True

            success = True

            # Remove from each collection
            for collection_name in collections:
                try:
                    result = self._qdrant_service.delete_points_by_file_paths(collection_name, file_paths)
                    if not result:
                        success = False
                        self.logger.error(f"Failed to delete files from collection '{collection_name}'")
                except Exception as e:
                    self.logger.error(f"Error deleting from collection '{collection_name}': {e}")
                    success = False

            # Also remove from metadata collection
            if hasattr(self, "_metadata_service"):
                try:
                    self._metadata_service.remove_file_metadata(project_name, file_paths)
                except Exception as e:
                    self.logger.error(f"Error removing file metadata: {e}")
                    success = False

            if success:
                self.logger.info(f"Successfully removed {len(file_paths)} deleted files from index")
            else:
                self.logger.error("Some files could not be removed from index")

            return success

        except Exception as e:
            self.logger.error(f"Error removing deleted files from index: {e}")
            return False
