"""Index directory tool implementation.

This module provides the index_directory tool for indexing codebases.
"""

import hashlib
import logging
import os
import time
import traceback
from collections import defaultdict
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.tools.core.errors import (
    QdrantConnectionError,
)
from src.tools.core.retry_utils import retry_operation
from src.tools.database.qdrant_utils import (
    check_qdrant_health,
    log_database_metrics,
    retry_individual_points,
    retry_qdrant_operation,
)
from src.utils.memory_utils import (
    clear_processing_variables,
    force_memory_cleanup,
    get_adaptive_batch_size,
    get_memory_usage_mb,
    log_memory_usage,
    should_cleanup_memory,
)

from .async_fix import fix_async_generator_issue

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger(__name__)

# Configuration
PROJECT_MARKERS = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]
max_files_in_response = int(os.getenv("max_files_in_response", "50"))

# Global clients (lazy initialization)
_qdrant_client = None
_embeddings_manager = None
_current_project = None


def get_qdrant_client():
    """Get or create Qdrant client with connection validation."""
    global _qdrant_client
    if _qdrant_client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        try:
            _qdrant_client = QdrantClient(host=host, port=port)
            retry_operation(lambda: _qdrant_client.get_collections())
        except Exception as e:
            raise QdrantConnectionError(f"Failed to connect to Qdrant at {host}:{port}", details=str(e))
    return _qdrant_client


def get_embeddings_manager_instance():
    """Get or create embeddings manager instance."""
    global _embeddings_manager
    if _embeddings_manager is None:
        from src.services.embedding_service import EmbeddingService

        _embeddings_manager = EmbeddingService()
    return _embeddings_manager


def get_current_project(
    client_directory: str | None = None,
) -> dict[str, str] | None:
    """Detect current project context based on directory markers."""
    global _current_project
    if client_directory:
        cwd = Path(client_directory).resolve()
    else:
        cwd = Path.cwd()

    for parent in [cwd] + list(cwd.parents):
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                _current_project = {
                    "name": project_name,
                    "root": str(parent),
                    "collection_prefix": f"project_{project_name}",
                }
                return _current_project

    # Fallback to directory name
    _current_project = {
        "name": cwd.name,
        "root": str(cwd),
        "collection_prefix": f"dir_{cwd.name.replace(' ', '_').replace('-', '_')}",
    }
    return _current_project


def get_collection_name(file_path: str, file_type: str = "code") -> str:
    """Generate collection name based on project context and file type."""
    path = Path(file_path).resolve()
    current_project = get_current_project()
    if current_project:
        project_root = Path(current_project["root"])
        try:
            path.relative_to(project_root)
            return f"{current_project['collection_prefix']}_{file_type}"
        except ValueError:
            pass

    # Check if file is in a project
    for parent in path.parents:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                return f"project_{project_name}_{file_type}"

    return f"global_{file_type}"


def ensure_collection(
    collection_name: str,
    embedding_dimension: int | None = None,
    embedding_model_name: str | None = None,
    content_type: str = "general",
):
    """Ensure collection exists in Qdrant with proper configuration."""
    client = get_qdrant_client()

    def check_and_create():
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            final_dimension = embedding_dimension or 768  # Default for nomic-embed-text
            final_model_name = embedding_model_name or os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=final_dimension, distance=Distance.COSINE),
            )

            # Store collection metadata
            metadata_payload = {
                "type": "collection_metadata",
                "embedding_model": final_model_name,
                "embedding_dimension": final_dimension,
                "content_type": content_type,
                "created_at": datetime.now().isoformat(),
            }
            metadata_point_id = hashlib.md5(f"{collection_name}_metadata".encode()).hexdigest()
            metadata_point = PointStruct(
                id=metadata_point_id,
                vector=[0.0] * final_dimension,
                payload=metadata_payload,
            )
            try:
                client.upsert(collection_name=collection_name, points=[metadata_point])
                logger.info(f"Created collection {collection_name} with model {final_model_name}")
            except Exception as e:
                logger.warning(f"Failed to store metadata for collection {collection_name}: {e}")

    retry_operation(check_and_create)


def check_existing_index(project_info: dict[str, str]) -> dict[str, Any]:
    """Check if the project already has indexed data."""
    try:
        client = get_qdrant_client()
        existing_collections = [c.name for c in client.get_collections().collections]

        project_collections = []
        total_points = 0
        collection_info = []

        if project_info:
            prefix = project_info["collection_prefix"]
            for collection_type in ["code", "config", "documentation"]:
                collection_name = f"{prefix}_{collection_type}"
                if collection_name in existing_collections:
                    try:
                        collection_info_response = client.get_collection(collection_name)
                        points_count = collection_info_response.points_count

                        project_collections.append(collection_name)
                        total_points += points_count
                        collection_info.append(
                            {
                                "name": collection_name,
                                "type": collection_type,
                                "points_count": points_count,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Could not get info for collection {collection_name}: {e}")

        return {
            "has_existing_data": len(project_collections) > 0,
            "collections": project_collections,
            "total_points": total_points,
            "collection_details": collection_info,
            "project_name": (project_info.get("name", "unknown") if project_info else "unknown"),
        }

    except Exception as e:
        logger.error(f"Error checking existing index: {e}")
        return {"has_existing_data": False, "error": str(e)}


def estimate_indexing_time(file_count: int, existing_points: int = 0) -> dict[str, Any]:
    """Estimate indexing time and provide recommendations."""
    # Rough estimates based on typical performance
    seconds_per_file = 0.5  # Including file reading, embedding, and DB insertion
    estimated_seconds = file_count * seconds_per_file
    estimated_minutes = estimated_seconds / 60

    if existing_points > 0:
        # Estimate time saved by not re-indexing
        avg_points_per_file = existing_points / max(file_count, 1)
        time_saved_seconds = existing_points / avg_points_per_file * seconds_per_file if avg_points_per_file > 0 else 0
        time_saved_minutes = time_saved_seconds / 60
    else:
        time_saved_minutes = 0

    # Smart recommendation logic
    if existing_points > 0 and estimated_minutes > 5:
        recommendation = "keep_existing_recommended"
    elif estimated_minutes > 30:
        recommendation = "large_operation_confirm"
    elif estimated_minutes > 10:
        recommendation = "medium_operation"
    else:
        recommendation = "quick_operation"

    return {
        "estimated_time_minutes": round(estimated_minutes, 1),
        "time_saved_by_keeping_existing_minutes": round(time_saved_minutes, 1),
        "recommendation": recommendation,
        "file_count": file_count,
        "existing_points": existing_points,
    }


def load_ragignore_patterns(directory: Path) -> tuple[set[str], set[str]]:
    """Load .ragignore patterns for file exclusion."""
    exclude_dirs = set()
    exclude_patterns = set()

    default_exclude_dirs = {
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
        ".tox",
        "data",
        "logs",
        "tmp",
        "temp",
        ".idea",
        ".vscode",
        ".vs",
        "qdrant_storage",
        "models",
        ".cache",
    }
    default_exclude_patterns = {
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.class",
        "*.log",
        "*.lock",
        "*.swp",
        "*.swo",
        "*.bak",
        "*.tmp",
        "*.temp",
        "*.old",
        "*.orig",
        "*.rej",
        ".env*",
        "*.sqlite",
        "*.db",
        "*.pid",
    }

    ragignore_path = None
    for parent in [directory] + list(directory.parents):
        potential_path = parent / ".ragignore"
        if potential_path.exists():
            ragignore_path = potential_path
            break

    if not ragignore_path:
        return default_exclude_dirs, default_exclude_patterns

    try:
        with open(ragignore_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith("/"):
                    exclude_dirs.add(line.rstrip("/"))
                else:
                    exclude_patterns.add(line)
    except Exception as e:
        logger.warning(f"Error reading .ragignore: {e}, using defaults")
        return default_exclude_dirs, default_exclude_patterns

    if not exclude_dirs and not exclude_patterns:
        return default_exclude_dirs, default_exclude_patterns

    return exclude_dirs, exclude_patterns


def clear_project_collections() -> dict[str, Any]:
    """Clear all collections for the current project."""
    current_project = get_current_project()
    if not current_project:
        return {"error": "No project context found", "cleared": []}

    client = get_qdrant_client()
    cleared = []
    errors = []
    existing_collections = [c.name for c in client.get_collections().collections]

    for collection_type in ["code", "config", "documentation", "file_metadata"]:
        collection_name = f"{current_project['collection_prefix']}_{collection_type}"
        if collection_name in existing_collections:
            try:
                client.delete_collection(collection_name)
                cleared.append(collection_name)
                logger.info(f"Cleared collection: {collection_name}")
            except Exception as e:
                error_msg = f"Failed to clear {collection_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

    return {
        "project": current_project["name"],
        "cleared_collections": cleared,
        "errors": errors if errors else None,
    }


async def _process_chunk_batch_for_streaming(
    chunks: list[Any], embeddings_manager, batch_size: int = 20
) -> AsyncGenerator[tuple[str, list[PointStruct]], None]:
    """
    Process chunks in batches and yield (collection_name, points) for streaming insertion.

    Args:
        chunks: List of chunk objects to process
        embeddings_manager: Embedding service instance
        batch_size: Number of chunks to process in each batch

    Yields:
        Tuple of (collection_name, points_list) ready for database insertion
    """
    # Group chunks by collection to enable efficient batch processing
    collection_batches = defaultdict(list)

    for i in range(0, len(chunks), batch_size):
        # Adaptive batch sizing based on memory pressure
        current_memory = get_memory_usage_mb()
        adaptive_batch_size = get_adaptive_batch_size(batch_size, current_memory)

        # Adjust batch bounds if needed
        if adaptive_batch_size != batch_size:
            batch_chunks = chunks[i : i + adaptive_batch_size]
            batch_size = adaptive_batch_size
        else:
            batch_chunks = chunks[i : i + batch_size]

        logger.info(f"Processing chunk batch {i // batch_size + 1}: {len(batch_chunks)} chunks (memory: {current_memory:.1f}MB)")

        # Group chunks by collection within this batch
        batch_collections = defaultdict(list)

        for chunk in batch_chunks:
            file_path = chunk.metadata["file_path"]
            language = chunk.metadata.get("language", "unknown")

            # Determine collection type
            if language in [
                "json",
                "yaml",
                "toml",
                "ini",
                "config",
                "xml",
                "dockerfile",
                "terraform",
            ]:
                group_name = "config"
            elif language in [
                "markdown",
                "restructuredtext",
                "text",
                "asciidoc",
                "latex",
                "documentation",
            ]:
                group_name = "documentation"
            else:
                group_name = "code"

            collection_name = get_collection_name(file_path, group_name)
            batch_collections[collection_name].append(chunk)

        # Process each collection batch
        for collection_name, collection_chunks in batch_collections.items():
            try:
                # Ensure collection exists
                ensure_collection(collection_name, content_type=collection_name.split("_")[-1])

                # Prepare texts for batch embedding generation, filtering empty content
                valid_chunks = []
                texts = []
                for chunk in collection_chunks:
                    if chunk.content and chunk.content.strip():
                        texts.append(chunk.content)
                        valid_chunks.append(chunk)
                    else:
                        logger.warning(
                            f"Skipping empty chunk from {chunk.metadata.get('file_path', 'unknown')} "
                            f"at line {chunk.metadata.get('line_start', 'unknown')}"
                        )

                if not texts:
                    logger.warning(f"No valid content found for collection {collection_name}, skipping batch")
                    continue

                # Generate embeddings in batch
                start_time = time.time()
                embeddings = await embeddings_manager.generate_embeddings(
                    os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text"),
                    texts,
                )
                embedding_time = time.time() - start_time

                if embeddings is None:
                    logger.error(f"Failed to generate embeddings for batch in collection {collection_name}")
                    continue

                # Create points for this collection batch
                points = []
                successful_embeddings = 0

                for chunk, embedding in zip(valid_chunks, embeddings, strict=False):
                    if embedding is None:
                        logger.warning(f"Skipping chunk from {chunk.metadata['file_path']} due to failed embedding")
                        continue

                    # Convert tensor to list if necessary
                    if hasattr(embedding, "tolist"):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = embedding

                    chunk_id = hashlib.md5(f"{chunk.metadata['file_path']}_{chunk.metadata.get('chunk_index', 0)}".encode()).hexdigest()

                    payload = {
                        "file_path": chunk.metadata["file_path"],
                        "content": chunk.content,
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "line_start": chunk.metadata.get("line_start", 0),
                        "line_end": chunk.metadata.get("line_end", 0),
                        "language": chunk.metadata.get("language", "unknown"),
                        "project": chunk.metadata.get("project", "unknown"),
                        "chunk_type": chunk.metadata.get("chunk_type", "file_chunk"),
                        # Enhanced breadcrumb navigation information
                        "breadcrumb": chunk.metadata.get("breadcrumb", ""),
                        "name": chunk.metadata.get("name", ""),
                        "parent_name": chunk.metadata.get("parent_name", ""),
                        "signature": chunk.metadata.get("signature", ""),
                        "docstring": chunk.metadata.get("docstring", ""),
                        # Additional context for better search experience
                        "context_before": chunk.metadata.get("context_before", ""),
                        "context_after": chunk.metadata.get("context_after", ""),
                        "start_byte": chunk.metadata.get("start_byte", 0),
                        "end_byte": chunk.metadata.get("end_byte", 0),
                        "complexity_score": chunk.metadata.get("complexity_score", 0.0),
                        "tags": chunk.metadata.get("tags", []),
                    }

                    points.append(PointStruct(id=chunk_id, vector=embedding_list, payload=payload))
                    successful_embeddings += 1

                if points:
                    logger.info(
                        f"Prepared {len(points)} points for collection {collection_name} " f"(embedding time: {embedding_time:.2f}s)"
                    )
                    yield collection_name, points

                    # Clear points from memory after yielding
                    clear_processing_variables(points)
                else:
                    logger.warning(f"No valid points generated for collection {collection_name}")

                # Clear intermediate variables
                clear_processing_variables(texts, embeddings, collection_chunks)

            except Exception as e:
                logger.error(f"Error processing batch for collection {collection_name}: {e}")
                # Clean up on error
                clear_processing_variables(
                    texts if "texts" in locals() else None,
                    embeddings if "embeddings" in locals() else None,
                    collection_chunks if "collection_chunks" in locals() else None,
                )
                continue

        # Memory cleanup between processing batches
        if should_cleanup_memory(i // batch_size + 1):
            force_memory_cleanup(f"chunk batch {i // batch_size + 1}")
        else:
            import gc

            gc.collect()


def _stream_points_to_qdrant(
    collection_points_generator: Generator[tuple[str, list[PointStruct]], None, None],
    qdrant_batch_size: int = 500,
) -> dict[str, Any]:
    """
    Stream points to Qdrant in configurable batches with comprehensive monitoring.

    Args:
        collection_points_generator: Generator yielding (collection_name, points) tuples
        qdrant_batch_size: Maximum number of points to insert in each Qdrant operation

    Returns:
        Dictionary with detailed insertion statistics, timing metrics, and error information
    """
    qdrant_client = get_qdrant_client()

    # Enhanced statistics tracking
    stats = {
        "total_points": 0,
        "successful_insertions": 0,
        "failed_insertions": 0,
        "collections_used": set(),
        "batch_count": 0,
        "errors": [],
        "timing_metrics": {
            "total_time": 0.0,
            "avg_batch_time": 0.0,
            "min_batch_time": float("inf"),
            "max_batch_time": 0.0,
        },
        "health_checks": [],
        "retry_count": 0,
        "partial_failures": 0,
    }

    operation_start_time = time.time()

    try:
        for collection_name, points in collection_points_generator:
            stats["collections_used"].add(collection_name)

            # Split points into Qdrant-sized batches
            for i in range(0, len(points), qdrant_batch_size):
                batch_points = points[i : i + qdrant_batch_size]
                stats["batch_count"] += 1
                stats["total_points"] += len(batch_points)

                # Periodic health check
                health_check_interval = int(os.getenv("DB_HEALTH_CHECK_INTERVAL", "50"))
                if stats["batch_count"] % health_check_interval == 0:
                    health_status = check_qdrant_health(qdrant_client)
                    stats["health_checks"].append(health_status)
                    if not health_status["healthy"]:
                        logger.warning(f"Qdrant health check failed: {health_status['error']}")
                        # Attempt to reconnect
                        try:
                            global _qdrant_client
                            _qdrant_client = None  # Force reconnection
                            qdrant_client = get_qdrant_client()
                        except Exception as reconnect_error:
                            error_msg = f"Failed to reconnect to Qdrant: {reconnect_error}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)

                # Perform insertion with retry logic
                insertion_start = time.time()

                # Capture variables for closure to avoid B023
                current_batch_points = batch_points
                current_collection_name = collection_name
                current_qdrant_client = qdrant_client

                def insert_batch():
                    return current_qdrant_client.upsert(collection_name=current_collection_name, points=current_batch_points)

                try:
                    retry_qdrant_operation(
                        insert_batch,
                        f"batch insertion to {collection_name}",
                        max_retries=int(os.getenv("DB_RETRY_ATTEMPTS", "3")),
                    )
                    insertion_time = time.time() - insertion_start

                    # Update timing metrics
                    stats["timing_metrics"]["min_batch_time"] = min(stats["timing_metrics"]["min_batch_time"], insertion_time)
                    stats["timing_metrics"]["max_batch_time"] = max(stats["timing_metrics"]["max_batch_time"], insertion_time)

                    stats["successful_insertions"] += len(batch_points)
                    logger.info(
                        f"Inserted {len(batch_points)} points to {collection_name} "
                        f"(batch {stats['batch_count']}, time: {insertion_time:.2f}s)"
                    )

                except Exception as e:
                    # Try individual point insertion for partial recovery
                    if len(batch_points) > 1:
                        logger.info(f"Attempting individual point recovery for failed batch {stats['batch_count']}")
                        individual_successes, individual_failures = retry_individual_points(qdrant_client, collection_name, batch_points)

                        stats["successful_insertions"] += individual_successes
                        stats["failed_insertions"] += individual_failures
                        stats["partial_failures"] += 1

                        if individual_successes > 0:
                            logger.info(f"Recovered {individual_successes}/{len(batch_points)} points individually")
                    else:
                        stats["failed_insertions"] += len(batch_points)

                    error_msg = f"Failed to insert batch {stats['batch_count']} to {collection_name}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    stats["retry_count"] += 1

                # Memory cleanup after each database batch
                del batch_points

                # Check if comprehensive cleanup is needed
                if should_cleanup_memory(stats["batch_count"], force_check=True):
                    force_memory_cleanup(f"database batch {stats['batch_count']}")
                else:
                    import gc

                    gc.collect()

                # Log memory usage and metrics periodically
                if stats["batch_count"] % 10 == 0:
                    log_memory_usage(f"after {stats['batch_count']} database batches")

                    # Log intermediate metrics
                    current_time = time.time() - operation_start_time
                    points_per_second = stats["successful_insertions"] / current_time if current_time > 0 else 0
                    logger.info(
                        f"Progress: {stats['successful_insertions']} points inserted, "
                        f"{points_per_second:.1f} points/sec, "
                        f"{stats['batch_count']} batches processed"
                    )

        # Finalize timing metrics
        total_operation_time = time.time() - operation_start_time
        stats["timing_metrics"]["total_time"] = total_operation_time
        if stats["batch_count"] > 0:
            stats["timing_metrics"]["avg_batch_time"] = total_operation_time / stats["batch_count"]

        # Convert set to list for JSON serialization
        stats["collections_used"] = list(stats["collections_used"])

        # Log comprehensive final metrics
        log_database_metrics(stats, total_operation_time, "streaming insertion")

        # Additional detailed logging
        logger.info(
            f"Streaming insertion complete: {stats['successful_insertions']} points inserted, "
            f"{stats['failed_insertions']} failed, {stats['batch_count']} batches, "
            f"{stats['retry_count']} retries, {stats['partial_failures']} partial failures"
        )

        if stats["health_checks"]:
            healthy_checks = sum(1 for check in stats["health_checks"] if check["healthy"])
            logger.info(f"Health checks: {healthy_checks}/{len(stats['health_checks'])} passed")

        return stats

    except Exception as e:
        total_operation_time = time.time() - operation_start_time
        stats["timing_metrics"]["total_time"] = total_operation_time

        error_msg = f"Critical error during streaming insertion: {e}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        stats["collections_used"] = list(stats["collections_used"])

        # Log metrics even on failure
        if stats["total_points"] > 0:
            log_database_metrics(stats, total_operation_time, "failed streaming insertion")

        return stats


async def index_directory(
    directory: str = ".",
    patterns: list[str] = None,
    recursive: bool = True,
    clear_existing: bool = False,
    incremental: bool = False,
    project_name: str | None = None,
) -> dict[str, Any]:
    """
    Index files in a directory with smart existing data detection and time estimation.

    This is the async wrapper for MCP tool registration.

    Args:
        directory: Directory to index (default: current directory)
        patterns: File patterns to include (default: common code file types)
        recursive: Whether to index subdirectories (default: True)
        clear_existing: Whether to clear existing indexed data (default: False)
                      If False and existing data is found, returns recommendations instead of indexing
        incremental: Whether to use incremental indexing (only process changed files) (default: False)
        project_name: Optional custom project name for collections (default: auto-detect)

    Returns:
        Dictionary with indexing results, time estimates, or recommendations for existing data
    """
    # Create async version that properly handles async generators
    return await index_directory_async_impl(directory, patterns, recursive, clear_existing, incremental, project_name)


async def index_directory_async_impl(
    directory: str = ".",
    patterns: list[str] = None,
    recursive: bool = True,
    clear_existing: bool = False,
    incremental: bool = False,
    project_name: str = None,
) -> dict[str, Any]:
    """
    Async implementation of index_directory that properly handles async generators.
    For now, just wrap the sync version to fix the immediate issue.
    """
    import asyncio
    import concurrent.futures
    from functools import partial

    # Run the sync version in a thread pool to avoid blocking
    sync_func = partial(index_directory_sync, directory, patterns, recursive, clear_existing, incremental, project_name)

    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(executor, sync_func)
            return result
    except RuntimeError:
        # No event loop running
        return sync_func()


def index_directory_sync(
    directory: str = ".",
    patterns: list[str] = None,
    recursive: bool = True,
    clear_existing: bool = False,
    incremental: bool = False,
    project_name: str | None = None,
) -> dict[str, Any]:
    """
    Synchronous implementation of index_directory.

    Args:
        directory: Directory to index (default: current directory)
        patterns: File patterns to include (default: common code file types)
        recursive: Whether to index subdirectories (default: True)
        clear_existing: Whether to clear existing indexed data (default: False)
                      If False and existing data is found, returns recommendations instead of indexing
        incremental: Whether to use incremental indexing (only process changed files) (default: False)
        project_name: Optional custom project name for collections (default: auto-detect)

    Returns:
        Dictionary with indexing results, time estimates, or recommendations for existing data
    """
    try:
        # Initialize memory monitoring
        from src.utils.performance_monitor import MemoryMonitor

        memory_threshold = float(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))
        memory_monitor = MemoryMonitor(warning_threshold_mb=memory_threshold)

        # Initial memory check
        initial_memory = memory_monitor.check_memory_usage(logger)
        logger.info(f"Index operation starting - Initial memory: {initial_memory['memory_mb']} MB")

        if patterns is None:
            patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.jsx",
                "*.tsx",
                "*.java",
                "*.go",
                "*.rs",
                "*.sh",
                "*.bash",
                "*.zsh",
                "*.fish",
                "*.tf",
                "*.tfvars",
                "*.json",
                "*.yaml",
                "*.yml",
                "*.toml",
                "*.ini",
                "*.md",
                "*.markdown",
                "*.rst",
                "*.txt",
                ".gitignore",
                ".dockerignore",
                ".prettierrc*",
                ".eslintrc*",
                ".editorconfig",
                ".npmrc",
                ".yarnrc",
                ".ragignore",
            ]

        if not directory:
            return {"error": "Directory parameter is required"}

        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}

        # Use custom project name if provided, otherwise auto-detect
        if project_name:
            current_project = {
                "name": project_name.replace(" ", "_").replace("-", "_").lower(),
                "root": str(dir_path),
                "collection_prefix": f"project_{project_name.replace(' ', '_').replace('-', '_').lower()}",
            }
            logger.info(f"Using custom project name: {current_project['name']}")
        else:
            current_project = get_current_project(client_directory=str(dir_path))
            logger.info(f"Auto-detected project: {current_project}")

        # Check for existing indexed data BEFORE processing
        existing_index_info = check_existing_index(current_project)

        # Get file count estimation for better decision making
        from src.services.indexing_service import IndexingService

        indexing_service = IndexingService()

        # Quick file count without full processing
        logger.info("Analyzing directory structure...")
        try:
            # Use project analysis service for quick file count
            from src.services.project_analysis_service import ProjectAnalysisService

            analysis_service = ProjectAnalysisService()
            quick_analysis = analysis_service.analyze_repository(str(dir_path))
            estimated_file_count = quick_analysis.get("relevant_files", 0)
        except Exception as e:
            logger.warning(f"Could not get quick file count: {e}, proceeding with full analysis")
            estimated_file_count = 0

        # Generate intelligent time estimates and recommendations
        from src.services.time_estimator_service import TimeEstimatorService

        time_estimator = TimeEstimatorService()

        # Determine mode for estimation
        estimation_mode = "incremental" if incremental else "clear_existing"
        estimate = time_estimator.estimate_indexing_time(str(dir_path), estimation_mode)

        # Legacy time estimates for backward compatibility
        time_estimates = {
            "estimated_time_minutes": estimate.estimated_minutes,
            "time_saved_by_keeping_existing_minutes": (0.0 if not incremental else estimate.estimated_minutes * 0.8),
            "recommendation": estimate.recommendation,
            "file_count": estimate.file_count,
            "existing_points": existing_index_info.get("total_points", 0),
        }

        # Smart decision logic for clear_existing
        if existing_index_info.get("has_existing_data", False) and not clear_existing:
            # Found existing data but clear_existing is False
            return {
                "success": False,
                "action_required": True,
                "message": "Existing indexed data found for this project",
                "existing_data": {
                    "project_name": existing_index_info["project_name"],
                    "collections": existing_index_info["collections"],
                    "total_points": existing_index_info["total_points"],
                    "collection_details": existing_index_info.get("collection_details", []),
                },
                "estimates": time_estimates,
                "recommendations": {
                    "keep_existing": {
                        "description": "Keep existing data to save time",
                        "time_saved_minutes": time_estimates["time_saved_by_keeping_existing_minutes"],
                        "action": "Use existing indexed data for searches",
                    },
                    "incremental_update": {
                        "description": "Add only new/modified files using incremental mode",
                        "action": "Call index_directory again with incremental=true",
                        "estimated_time_minutes": time_estimator.estimate_indexing_time(str(dir_path), "incremental").estimated_minutes,
                    },
                    "full_reindex": {
                        "description": "Clear existing data and reindex everything",
                        "estimated_time_minutes": time_estimates["estimated_time_minutes"],
                        "action": "Call index_directory again with clear_existing=true",
                    },
                    "manual_tool": {
                        "description": "Use standalone manual indexing tool for heavy operations",
                        "command": time_estimator.get_manual_tool_command(str(dir_path), estimation_mode),
                        "recommended": time_estimator.should_recommend_manual_tool(estimate),
                        "reason": "Recommended for operations exceeding 5 minutes",
                    },
                },
                "directory": str(dir_path),
            }

        # Handle clearing existing data if requested
        if clear_existing and existing_index_info.get("has_existing_data", False):
            logger.info(f"Clearing existing data for project {existing_index_info['project_name']}...")
            clear_result = clear_project_collections()
            if clear_result.get("errors"):
                return {
                    "error": "Failed to clear some collections",
                    "clear_errors": clear_result["errors"],
                }
            logger.info(f"Cleared {len(clear_result.get('cleared_collections', []))} collections")

        exclude_dirs, exclude_patterns = load_ragignore_patterns(dir_path)

        indexed_files = []
        errors = []
        collections_used = set()

        # Now proceed with full indexing
        logger.info("Processing codebase for indexing...")

        # Memory check before heavy processing
        pre_index_memory = memory_monitor.check_memory_usage(logger)
        logger.info(f"Pre-indexing memory: {pre_index_memory['memory_mb']} MB")

        # Pass incremental mode and project name to indexing service
        # process_codebase_for_indexing is async, need to run it synchronously
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
            # Running in async context, use thread pool
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    indexing_service.process_codebase_for_indexing(
                        str(dir_path),
                        incremental_mode=incremental,
                        project_name=current_project["name"] if current_project else None,
                    ),
                )
                processed_chunks = future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run directly
            processed_chunks = asyncio.run(
                indexing_service.process_codebase_for_indexing(
                    str(dir_path),
                    incremental_mode=incremental,
                    project_name=current_project["name"] if current_project else None,
                )
            )

        # Get progress summary after processing
        try:
            progress_summary = indexing_service.get_progress_summary()
        except AttributeError as e:
            logger.warning(f"Progress summary unavailable: {e}")
            progress_summary = {"status": "completed", "error": str(e)}

        # Memory check after processing
        post_index_memory = memory_monitor.check_memory_usage(logger)
        logger.info(f"Post-indexing memory: {post_index_memory['memory_mb']} MB")

        if not processed_chunks:
            return {
                "success": False,
                "summary": "No relevant files found to index",
                "total_files": 0,
                "total_points": 0,
                "collections": [],
                "project_context": (current_project["name"] if current_project else "no project"),
                "directory": str(dir_path),
                "existing_data": (existing_index_info if existing_index_info.get("has_existing_data", False) else None),
            }

        # Get configuration for streaming processing
        chunk_batch_size = int(os.getenv("INDEXING_BATCH_SIZE", "20"))
        qdrant_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "500"))

        # Check initial memory and adapt batch sizes if needed
        initial_memory = get_memory_usage_mb()
        chunk_batch_size = get_adaptive_batch_size(chunk_batch_size, initial_memory)
        qdrant_batch_size = get_adaptive_batch_size(qdrant_batch_size, initial_memory)

        logger.info(f"Starting streaming indexing with chunk batch size: {chunk_batch_size}, " f"Qdrant batch size: {qdrant_batch_size}")

        # Log initial memory usage
        log_memory_usage("before indexing")

        embeddings_manager = get_embeddings_manager_instance()

        # Create streaming pipeline: chunks -> embeddings -> database
        # Since _process_chunk_batch_for_streaming is async, we need to run it synchronously
        import asyncio

        def _sync_streaming_generator():
            """Convert async generator to sync generator."""
            try:
                asyncio.get_running_loop()
                # If we're in an async context, use a different approach
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # Create a new event loop in a separate thread
                    def run_async_generator():
                        async def collect_all():
                            result = []
                            async for collection_name, points in _process_chunk_batch_for_streaming(
                                processed_chunks, embeddings_manager, batch_size=chunk_batch_size
                            ):
                                result.append((collection_name, points))
                            return result

                        return asyncio.run(collect_all())

                    future = executor.submit(run_async_generator)
                    results = future.result()
                    for item in results:
                        yield item
            except RuntimeError:
                # No event loop running, we can use asyncio.run directly
                async def collect_all():
                    result = []
                    async for collection_name, points in _process_chunk_batch_for_streaming(
                        processed_chunks, embeddings_manager, batch_size=chunk_batch_size
                    ):
                        result.append((collection_name, points))
                    return result

                results = asyncio.run(collect_all())
                for item in results:
                    yield item

        collection_points_generator = _sync_streaming_generator()

        # Stream points to Qdrant and collect statistics
        streaming_stats = _stream_points_to_qdrant(collection_points_generator, qdrant_batch_size=qdrant_batch_size)

        # Extract results from streaming stats
        collections_used = set(streaming_stats["collections_used"])
        total_indexed_points = streaming_stats["successful_insertions"]

        # Get unique file paths without creating huge lists for large codebases
        unique_files = set()
        for chunk in processed_chunks:
            unique_files.add(chunk.metadata["file_path"])

        indexed_files_count = len(unique_files)

        # For MCP response size management, limit the files list
        if indexed_files_count <= max_files_in_response:
            indexed_files = list(unique_files)
        else:
            # Return only a sample of files for large codebases
            indexed_files = list(unique_files)[:max_files_in_response]
            logger.info(f"Truncating file list in response: showing {max_files_in_response} " f"of {indexed_files_count} files")

        # Add any errors from streaming to existing errors list
        if streaming_stats["errors"]:
            errors.extend(streaming_stats["errors"])

        # Final memory cleanup and logging
        force_memory_cleanup("indexing complete")
        final_memory = log_memory_usage("final")

        # Create a compact result for MCP response
        actual_time = streaming_stats.get("timing_metrics", {}).get("total_time", 0)

        result = {
            "success": True,
            "summary": f"Successfully indexed {indexed_files_count} files into {len(collections_used)} collections",
            "total_files": indexed_files_count,
            "total_points": total_indexed_points,
            "collections": list(collections_used),
            "project_context": (current_project["name"] if current_project else "no project"),
            "directory": str(dir_path),
            "was_reindexed": clear_existing and existing_index_info.get("has_existing_data", False),
            "performance": {
                "batches_processed": streaming_stats["batch_count"],
                "processing_time_seconds": actual_time,
                "estimated_time_minutes": time_estimates.get("estimated_time_minutes", 0),
                "actual_time_minutes": (round(actual_time / 60, 1) if actual_time > 0 else 0),
                "points_per_second": (round(total_indexed_points / actual_time, 1) if actual_time > 0 else 0),
                "memory_efficient": True,
                "final_memory_mb": final_memory if "final_memory" in locals() else None,
                "progress_tracking": progress_summary,
            },
            "errors": {
                "count": len(errors) if errors else 0,
                "failed_insertions": streaming_stats["failed_insertions"],
                "retry_count": streaming_stats.get("retry_count", 0),
                "details": errors[:3] if errors else None,  # Only first 3 errors
            },
        }

        # Add change summary for incremental mode
        if incremental and hasattr(indexing_service, "_change_summary"):
            result["change_summary"] = indexing_service._change_summary
        elif incremental:
            # If no change summary available, add basic info
            result["change_summary"] = {
                "mode": "incremental",
                "files_processed": indexed_files_count,
                "note": "Change detection completed successfully",
            }
        else:
            result["change_summary"] = {
                "mode": "full_reindex",
                "files_processed": indexed_files_count,
            }

        # Add sample files only if the list is manageable
        if indexed_files_count <= max_files_in_response:
            result["indexed_files"] = indexed_files
        else:
            result["sample_files"] = indexed_files  # Sample of files
            result["note"] = f"Showing {len(indexed_files)} sample files out of {indexed_files_count} total files indexed"

        return result

    except Exception as e:
        error_msg = str(e)
        # Truncate very long error messages
        if len(error_msg) > 1000:
            error_msg = error_msg[:1000] + "... (truncated)"

        logger.error(f"Index directory failed: {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")

        return {
            "success": False,
            "error": error_msg,
            "directory": directory,
            "summary": "Indexing failed due to error",
        }
