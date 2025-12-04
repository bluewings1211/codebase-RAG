"""Search tools implementation for MCP server.

This module provides the search tool for querying indexed codebases with
natural language, supporting multiple search modes, context expansion,
and function-level precision results.
"""

import logging
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
)
from tools.core.errors import (
    EmbeddingError,
    QdrantConnectionError,
    SearchError,
    ValidationError,
)
from tools.core.retry_utils import retry_operation
from utils.output_control import filter_search_results

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

logger = logging.getLogger(__name__)

# Configuration
PROJECT_MARKERS = [".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"]

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
        from services.embedding_service import EmbeddingService

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


def _truncate_content(content: str, max_length: int = 1500) -> str:
    """Truncate content to specified length with indicator.

    Args:
        content: Text content to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated content with indicator if needed
    """
    if len(content) <= max_length:
        return content
    return content[:max_length] + "\n... (truncated)"


def _is_valid_content(content: str) -> bool:
    """Check if content is valid for search results.

    Args:
        content: Content to validate

    Returns:
        True if content is valid, False if it should be filtered out
    """
    if not content:
        return False

    # Remove whitespace and check if anything meaningful remains
    stripped_content = content.strip()
    if not stripped_content:
        return False

    # Filter out content that's too short to be meaningful (less than 3 characters)
    if len(stripped_content) < 3:
        return False

    # Filter out content that's only punctuation, brackets, or special characters
    meaningful_chars = sum(1 for c in stripped_content if c.isalnum() or c in ["_", "-", "."])
    if meaningful_chars == 0:
        return False

    # Filter out content that's mostly whitespace (more than 80% whitespace)
    if meaningful_chars / len(stripped_content) < 0.2:
        return False

    return True


def _expand_search_context(
    results: list[dict[str, Any]],
    qdrant_client,
    search_collections: list[str],
    context_chunks: int = 1,
    embedding_dimension: int = 384,
) -> list[dict[str, Any]]:
    """Expand search results with surrounding context.

    This function attempts to find and include code context around search results
    by looking for chunks from the same file with adjacent line numbers.

    Args:
        results: List of search results to expand
        qdrant_client: Qdrant client instance
        search_collections: Collections to search for context
        context_chunks: Number of context chunks to include before/after
        embedding_dimension: Dimension of embedding vectors

    Returns:
        List of results with expanded context where available
    """
    if not results or context_chunks <= 0:
        return results

    try:
        expanded_results = []

        for result in results:
            expanded_result = result.copy()

            # Extract key information for context search
            file_path = result.get("file_path", "")
            line_start = result.get("line_start", 0)
            line_end = result.get("line_end", 0)
            collection = result.get("collection", "")

            if not file_path or not collection or collection not in search_collections:
                expanded_results.append(expanded_result)
                continue

            # Search for context chunks in the same file
            try:
                # Create filter for same file
                file_filter = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))])

                # Search for chunks in the same file
                context_results = qdrant_client.search(
                    collection_name=collection,
                    query_vector=[0.0] * embedding_dimension,  # Dummy vector for filtering
                    query_filter=file_filter,
                    limit=50,  # Get many chunks to find context
                    score_threshold=0.0,  # Accept all results since we're filtering by file
                )

                # Find chunks that provide context (adjacent or overlapping line ranges)
                context_before = []
                context_after = []

                for ctx_result in context_results:
                    if not isinstance(ctx_result.payload, dict):
                        continue

                    ctx_line_start = ctx_result.payload.get("line_start", 0)
                    ctx_line_end = ctx_result.payload.get("line_end", 0)
                    ctx_content = ctx_result.payload.get("content", "")

                    # Skip the original result chunk
                    if ctx_line_start == line_start and ctx_line_end == line_end:
                        continue

                    # Check if this chunk provides context before the result
                    if ctx_line_end <= line_start and (line_start - ctx_line_end) <= 10:
                        context_before.append(
                            {
                                "content": ctx_content,
                                "line_start": ctx_line_start,
                                "line_end": ctx_line_end,
                                "distance_from_result": line_start - ctx_line_end,
                            }
                        )

                    # Check if this chunk provides context after the result
                    elif ctx_line_start >= line_end and (ctx_line_start - line_end) <= 10:
                        context_after.append(
                            {
                                "content": ctx_content,
                                "line_start": ctx_line_start,
                                "line_end": ctx_line_end,
                                "distance_from_result": ctx_line_start - line_end,
                            }
                        )

                # Sort context chunks and limit to requested number
                context_before.sort(key=lambda x: x["line_start"], reverse=True)
                context_after.sort(key=lambda x: x["line_start"])

                context_before = context_before[:context_chunks]
                context_after = context_after[:context_chunks]

                # Build expanded content
                expanded_content_parts = []

                # Add context before
                for ctx in reversed(context_before):  # Reverse to maintain file order
                    expanded_content_parts.append(f"// Context (lines {ctx['line_start']}-{ctx['line_end']}):")
                    expanded_content_parts.append(ctx["content"])
                    expanded_content_parts.append("")

                # Add original content
                expanded_content_parts.append(f"// Main result (lines {line_start}-{line_end}):")
                expanded_content_parts.append(result.get("content", ""))
                expanded_content_parts.append("")

                # Add context after
                for ctx in context_after:
                    expanded_content_parts.append(f"// Context (lines {ctx['line_start']}-{ctx['line_end']}):")
                    expanded_content_parts.append(ctx["content"])
                    expanded_content_parts.append("")

                if context_before or context_after:
                    expanded_result["expanded_content"] = "\n".join(expanded_content_parts)
                    expanded_result["context_info"] = {
                        "chunks_before": len(context_before),
                        "chunks_after": len(context_after),
                        "total_context_lines": sum(ctx["line_end"] - ctx["line_start"] + 1 for ctx in context_before + context_after),
                    }

            except Exception as e:
                logger.debug(f"Failed to expand context for result in {file_path}: {e}")
                # Continue without context expansion

            expanded_results.append(expanded_result)

        return expanded_results

    except Exception as e:
        logger.warning(f"Context expansion failed: {e}")
        return results  # Return original results if expansion fails


def _perform_hybrid_search(
    qdrant_client,
    embedding_model,
    query: str,
    query_embedding: list[float],
    search_collections: list[str],
    n_results: int,
    search_mode: str = "hybrid",
    collection_filter: dict | None = None,
    result_processor: Callable | None = None,
    metadata_extractor: Callable | None = None,
) -> list[dict[str, Any]]:
    """Perform hybrid search across multiple collections.

    This function implements semantic search using vector similarity.
    In the future, it can be extended to include keyword search and
    combine results using different ranking strategies.

    Args:
        qdrant_client: Qdrant client instance
        embedding_model: Embedding service instance (for future keyword search)
        query: Original search query
        query_embedding: Query vector embedding
        search_collections: List of collections to search
        n_results: Total number of results to return across all collections
        search_mode: Search mode (semantic, keyword, hybrid)
        collection_filter: Optional filter for collections
        result_processor: Optional function to process each result
        metadata_extractor: Optional function to extract metadata from payload

    Returns:
        List of search results with scores and metadata
    """
    all_results = []
    empty_content_stats = {"total_found": 0, "empty_skipped": 0, "by_collection": {}}

    # Calculate how many results to retrieve from each collection
    # We want to retrieve enough results to ensure we get the best results
    # across all collections, but not excessively more than needed
    num_collections = len(search_collections)
    if num_collections == 1:
        # Single collection: request exactly what we need plus a buffer
        per_collection_limit = max(n_results, 10)
    else:
        # Multiple collections: calculate optimal distribution
        # Base allocation per collection, with minimum buffer
        base_per_collection = max(1, n_results // num_collections)
        # Add buffer to ensure we get good results, but cap it reasonably
        buffer_per_collection = min(max(5, n_results // 4), 20)
        per_collection_limit = base_per_collection + buffer_per_collection

    for collection in search_collections:
        empty_content_stats["by_collection"][collection] = {"found": 0, "empty": 0}
        try:
            # Perform vector similarity search
            search_results = qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                query_filter=collection_filter,
                limit=per_collection_limit,
                score_threshold=0.1,  # Minimum similarity threshold
            )

            for result in search_results:
                empty_content_stats["total_found"] += 1
                empty_content_stats["by_collection"][collection]["found"] += 1

                if not isinstance(result.payload, dict):
                    continue

                # Enhanced content filtering - skip results with empty, whitespace-only, or minimal content
                content = result.payload.get("content", "")
                if not _is_valid_content(content):
                    empty_content_stats["empty_skipped"] += 1
                    empty_content_stats["by_collection"][collection]["empty"] += 1
                    file_path = result.payload.get("file_path", "unknown")
                    chunk_type = result.payload.get("chunk_type", "unknown")
                    chunk_name = result.payload.get("name", "unknown")
                    logger.debug(f"Skipping result with invalid content from {collection}: {file_path} ({chunk_type}:{chunk_name})")
                    continue

                # Build base result structure
                base_result = {
                    "score": float(result.score),
                    "collection": collection,
                    "search_mode": search_mode,
                    "file_path": result.payload.get("file_path", ""),
                    "content": content,
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "project": result.payload.get("project", "unknown"),
                    "line_start": result.payload.get("line_start", 0),
                    "line_end": result.payload.get("line_end", 0),
                    "language": result.payload.get("language", ""),
                    "chunk_type": result.payload.get("chunk_type", ""),
                    "breadcrumb": result.payload.get("breadcrumb", ""),
                    "name": result.payload.get("name", ""),
                    "parent_name": result.payload.get("parent_name", ""),
                    "signature": result.payload.get("signature", ""),
                    "docstring": result.payload.get("docstring", ""),
                }

                # Apply metadata extractor if provided
                if metadata_extractor:
                    extracted_metadata = metadata_extractor(result.payload)
                    base_result.update(extracted_metadata)

                # Apply result processor if provided
                if result_processor:
                    base_result = result_processor(base_result)

                all_results.append(base_result)

        except Exception as e:
            logger.debug(f"Error searching collection {collection}: {e}")
            # Continue with other collections
            continue

    # Sort results by score in descending order, with secondary sort by collection type
    # to ensure stable ordering for results with same score
    all_results.sort(key=lambda x: (x.get("score", 0), x.get("collection", ""), x.get("line_start", 0)), reverse=True)

    # Log collection-wise result statistics
    collection_stats = {}
    for result in all_results:
        coll = result.get("collection", "unknown")
        if coll not in collection_stats:
            collection_stats[coll] = {"count": 0, "top_score": 0, "avg_score": 0, "scores": []}
        collection_stats[coll]["count"] += 1
        collection_stats[coll]["scores"].append(result.get("score", 0))
        collection_stats[coll]["top_score"] = max(collection_stats[coll]["top_score"], result.get("score", 0))

    # Calculate average scores
    for coll, stats in collection_stats.items():
        if stats["scores"]:
            stats["avg_score"] = sum(stats["scores"]) / len(stats["scores"])

    logger.debug(f"Cross-collection aggregation: {len(all_results)} total results from {len(search_collections)} collections")
    for coll, stats in collection_stats.items():
        logger.debug(f"  {coll}: {stats['count']} results, top_score={stats['top_score']:.3f}, avg_score={stats['avg_score']:.3f}")

    # Log empty content statistics if any were found
    if empty_content_stats["empty_skipped"] > 0:
        empty_rate = (empty_content_stats["empty_skipped"] / empty_content_stats["total_found"]) * 100
        logger.warning(
            f"Filtered out {empty_content_stats['empty_skipped']} results with empty content "
            f"({empty_rate:.1f}% of {empty_content_stats['total_found']} total found)"
        )

        # Log per-collection breakdown
        for coll, stats in empty_content_stats["by_collection"].items():
            if stats["empty"] > 0:
                coll_rate = (stats["empty"] / stats["found"]) * 100 if stats["found"] > 0 else 0
                logger.debug(f"Collection {coll}: {stats['empty']}/{stats['found']} empty ({coll_rate:.1f}%)")

    # Limit total results to n_results (properly aggregated across collections)
    final_results = all_results[:n_results]

    # Log final result distribution
    if final_results:
        final_collection_counts = {}
        for result in final_results:
            coll = result.get("collection", "unknown")
            final_collection_counts[coll] = final_collection_counts.get(coll, 0) + 1

        logger.debug(f"Final result distribution ({len(final_results)} results): {final_collection_counts}")

    return final_results


def _create_general_metadata_extractor() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Create a metadata extractor for general search results.

    Returns:
        Function that extracts and formats metadata from result payload
    """

    def extractor(payload: dict[str, Any]) -> dict[str, Any]:
        collection = payload.get("collection", "")

        # Determine content type from collection name
        if "_code" in collection:
            content_type = "code"
        elif "_config" in collection:
            content_type = "config"
        elif "_documentation" in collection:
            content_type = "docs"
        else:
            content_type = "unknown"

        return {
            "display_path": payload.get("file_path", ""),
            "type": content_type,
            "language": payload.get("language", ""),
            "line_start": payload.get("line_start", 0),
            "line_end": payload.get("line_end", 0),
            "breadcrumb": payload.get("breadcrumb", ""),
            "chunk_name": payload.get("name", ""),
            "chunk_type": payload.get("chunk_type", ""),
            "parent_name": payload.get("parent_name", ""),
            "signature": payload.get("signature", ""),
            "docstring": payload.get("docstring", ""),
            # Additional metadata for rich results
            "complexity_score": payload.get("complexity_score", 0.0),
            "tags": payload.get("tags", []),
            "context_before": payload.get("context_before", ""),
            "context_after": payload.get("context_after", ""),
        }

    return extractor


def filter_collections_by_type(collections: list[str], collection_types: list[str] | None) -> list[str]:
    """Filter collections based on specified collection types.

    Args:
        collections: List of all available collection names
        collection_types: List of collection types to include ("code", "config", "documentation")
                         None means include all types

    Returns:
        Filtered list of collection names
    """
    if not collection_types:
        return collections

    # Normalize collection types to lowercase
    normalized_types = [t.lower() for t in collection_types]
    filtered_collections = []

    for collection in collections:
        # Skip metadata collections
        if collection.endswith("_file_metadata"):
            continue

        # Check each requested type
        for collection_type in normalized_types:
            if collection_type == "code" and collection.endswith("_code"):
                filtered_collections.append(collection)
                break
            elif collection_type == "config" and collection.endswith("_config"):
                filtered_collections.append(collection)
                break
            elif collection_type == "documentation" and collection.endswith("_documentation"):
                filtered_collections.append(collection)
                break

    return filtered_collections


async def search(
    query: str,
    n_results: int = 5,
    cross_project: bool = False,
    search_mode: str = "hybrid",
    include_context: bool = True,
    context_chunks: int = 1,
    target_projects: list[str] | None = None,
    collection_types: list[str] | None = None,
    minimal_output: bool | None = None,
) -> dict[str, Any]:
    """
    Search indexed content using natural language queries.

    This tool provides function-level precision search with intelligent chunking,
    supporting multiple search modes and context expansion for better code understanding.

    Args:
        query: Natural language search query
        n_results: Number of results to return (1-100)
        cross_project: Whether to search across all projects (default: current project only)
        search_mode: Search strategy - "semantic", "keyword", or "hybrid" (default: "hybrid")
        include_context: Whether to include surrounding code context (default: True)
        context_chunks: Number of context chunks to include before/after results (default: 1)
        target_projects: List of specific project names to search in (optional)
        collection_types: List of collection types to search in (optional)
                        - ["code"] - Only search source code files
                        - ["config"] - Only search configuration files (JSON, YAML, TOML, etc.)
                        - ["documentation"] - Only search documentation files (Markdown, text, etc.)
                        - ["code", "config"] - Search both code and configuration files
                        - None - Search all collection types (default behavior)
        minimal_output: Return simplified output for AI agents (default: controlled by MCP_MINIMAL_OUTPUT env)
                       - True: Returns only essential fields (file_path, content, line numbers, breadcrumb)
                       - False: Returns full results with all metadata and scores
                       - None: Uses MCP_MINIMAL_OUTPUT environment variable (default: true)

    Returns:
        Dictionary containing search results with metadata, scores, and context.
        Output verbosity controlled by minimal_output parameter or MCP_MINIMAL_OUTPUT env var.
    """
    # Use the synchronous implementation
    results = search_sync(
        query,
        n_results,
        cross_project,
        search_mode,
        include_context,
        context_chunks,
        target_projects,
        collection_types,
    )
    # Apply output filtering
    return filter_search_results(results, minimal_output)


def search_sync(
    query: str,
    n_results: int = 5,
    cross_project: bool = False,
    search_mode: str = "hybrid",
    include_context: bool = True,
    context_chunks: int = 1,
    target_projects: list[str] | None = None,
    collection_types: list[str] | None = None,
) -> dict[str, Any]:
    """
    Synchronous implementation of search functionality.

    Args:
        query: Natural language search query
        n_results: Number of results to return (1-100)
        cross_project: Whether to search across all projects (default: current project only)
        search_mode: Search strategy - "semantic", "keyword", or "hybrid" (default: "hybrid")
        include_context: Whether to include surrounding code context (default: True)
        context_chunks: Number of context chunks to include before/after results (default: 1)
        target_projects: List of specific project names to search in (optional)
        collection_types: List of collection types to search in (optional)
                        - ["code"] - Only search source code files
                        - ["config"] - Only search configuration files
                        - ["documentation"] - Only search documentation files
                        - None - Search all collection types (default)

    Returns:
        Dictionary containing search results with metadata, scores, and context
    """
    try:
        # Input validation
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string", field_name="query", value=str(query))

        if not isinstance(n_results, int) or n_results < 1 or n_results > 100:
            raise ValidationError(
                "n_results must be between 1 and 100",
                field_name="n_results",
                value=str(n_results),
            )

        if search_mode not in ["semantic", "keyword", "hybrid"]:
            raise ValidationError(
                "search_mode must be 'semantic', 'keyword', or 'hybrid'",
                field_name="search_mode",
                value=search_mode,
            )

        if not isinstance(context_chunks, int) or context_chunks < 0 or context_chunks > 5:
            raise ValidationError(
                "context_chunks must be between 0 and 5",
                field_name="context_chunks",
                value=str(context_chunks),
            )

        # Validate target_projects parameter
        if target_projects is not None:
            if not isinstance(target_projects, list):
                raise ValidationError(
                    "target_projects must be a list of project names",
                    field_name="target_projects",
                    value=str(target_projects),
                )
            if not all(isinstance(p, str) for p in target_projects):
                raise ValidationError(
                    "All project names in target_projects must be strings",
                    field_name="target_projects",
                    value=str(target_projects),
                )
            if len(target_projects) == 0:
                raise ValidationError(
                    "target_projects cannot be empty if specified",
                    field_name="target_projects",
                    value=str(target_projects),
                )

        # Initialize services
        embeddings_manager = get_embeddings_manager_instance()
        qdrant_client = get_qdrant_client()

        # Generate query embedding
        embedding_model = os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
        try:
            query_embedding_tensor = embeddings_manager.generate_embeddings(embedding_model, query)

            if query_embedding_tensor is None:
                raise EmbeddingError(
                    "Failed to generate embedding for query (empty query or embedding service error)",
                    model_name=embedding_model,
                )

            # Convert tensor to list if necessary
            if hasattr(query_embedding_tensor, "tolist"):
                query_embedding = query_embedding_tensor.tolist()
            else:
                query_embedding = query_embedding_tensor

        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {str(e)}", model_name=embedding_model)

        # Determine search collections
        all_collections = [c.name for c in qdrant_client.get_collections().collections]

        # Handle target_projects logic
        if target_projects:
            search_collections = get_target_project_collections(target_projects, all_collections)

            # Check if any collections were found
            if not search_collections:
                from ..project.project_utils import get_available_project_names

                return {
                    "error": f"No indexed collections found for projects: {target_projects}",
                    "available_projects": get_available_project_names(all_collections),
                    "query": query,
                    "results": [],
                    "total": 0,
                }

        elif cross_project:
            # Search across all collections
            search_collections = [c for c in all_collections if not c.endswith("_file_metadata")]
        else:
            # Search only current project collections
            current_project = get_current_project()
            if current_project:
                search_collections = [
                    c for c in all_collections if c.startswith(current_project["collection_prefix"]) and not c.endswith("_file_metadata")
                ]
            else:
                # Fallback to global collections
                search_collections = [c for c in all_collections if c.startswith("global_") and not c.endswith("_file_metadata")]

        # Apply collection type filtering if specified
        if collection_types:
            search_collections = filter_collections_by_type(search_collections, collection_types)
            logger.debug(f"Filtered to collection types {collection_types}: {len(search_collections)} collections")

        if not search_collections:
            return {
                "results": [],
                "query": query,
                "total": 0,
                "project_context": (current_project.get("name", "no project") if current_project else "no project"),
                "search_scope": "all projects" if cross_project else "current project",
                "search_mode": search_mode,
                "collections_searched": [],
                "message": "No indexed collections found. Please index some content first.",
            }

        # Create metadata extractor
        metadata_extractor = _create_general_metadata_extractor()

        # Perform search
        logger.info(f"Searching {len(search_collections)} collections for query: '{query[:50]}...'")

        search_results = _perform_hybrid_search(
            qdrant_client=qdrant_client,
            embedding_model=embeddings_manager,
            query=query,
            query_embedding=query_embedding,
            search_collections=search_collections,
            n_results=n_results,
            search_mode=search_mode,
            metadata_extractor=metadata_extractor,
        )

        # Expand context if requested
        if include_context and search_results and context_chunks > 0:
            logger.debug(f"Expanding context for {len(search_results)} results with {context_chunks} chunks")
            search_results = _expand_search_context(
                search_results,
                qdrant_client,
                search_collections,
                context_chunks,
                embedding_dimension=768,  # Default for nomic-embed-text
            )

        # Truncate content for response size management
        for result in search_results:
            if "content" in result:
                result["content"] = _truncate_content(result["content"], max_length=1500)
            if "expanded_content" in result:
                result["expanded_content"] = _truncate_content(result["expanded_content"], max_length=2000)

        # Get current project info
        current_project = get_current_project()

        # Determine search scope description
        if target_projects:
            search_scope = f"specific projects: {', '.join(target_projects)}"
        elif cross_project:
            search_scope = "all projects"
        else:
            search_scope = "current project"

        # Build response
        response = {
            "results": search_results,
            "query": query,
            "total": len(search_results),
            "project_context": (current_project.get("name", "no project") if current_project else "no project"),
            "search_scope": search_scope,
            "search_mode": search_mode,
            "collections_searched": search_collections,
            "target_projects": target_projects if target_projects else None,
            "performance": {
                "collections_count": len(search_collections),
                "context_expanded": include_context and context_chunks > 0,
                "context_chunks": context_chunks if include_context else 0,
            },
        }

        # Add search tips for empty results
        if not search_results:
            response["suggestions"] = [
                "Try different keywords or synonyms",
                "Check if the content has been indexed using index_directory",
                "Try cross-project search if looking across multiple projects",
                "Use more general terms instead of very specific ones",
            ]

        logger.info(f"Search completed: {len(search_results)} results found for query '{query[:30]}...'")

        return response

    except (ValidationError, EmbeddingError, QdrantConnectionError, SearchError) as e:
        # Known errors with specific handling
        logger.error(f"Search failed with known error: {e}")
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "query": query,
            "results": [],
            "total": 0,
        }
    except Exception as e:
        # Unexpected errors
        error_msg = f"Search failed with unexpected error: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")

        return {
            "error": error_msg,
            "error_type": "UnexpectedError",
            "query": query,
            "results": [],
            "total": 0,
            "debug_info": {
                "search_mode": search_mode,
                "n_results": n_results,
                "cross_project": cross_project,
                "include_context": include_context,
            },
        }


# Utility functions for collection management and search optimization


def get_target_project_collections(target_projects: list[str], all_collections: list[str]) -> list[str]:
    """Get collections for specified target projects.

    Args:
        target_projects: List of project names to search
        all_collections: List of all available collections

    Returns:
        List of collection names for the target projects
    """
    # Do not normalize case - preserve original project names for exact matching first
    search_collections = []

    # Find collections for specified projects
    for original_name in target_projects:
        # Try exact match first (preserving original case)
        exact_match_collections = [
            c
            for c in all_collections
            if (c.startswith(f"project_{original_name}_") or c.startswith(f"dir_{original_name}_")) and not c.endswith("_file_metadata")
        ]

        if exact_match_collections:
            search_collections.extend(exact_match_collections)
        else:
            # Try normalized case matching (replace spaces/hyphens with underscores, convert to lowercase)
            normalized_name = original_name.replace(" ", "_").replace("-", "_").lower()
            normalized_match_collections = [
                c
                for c in all_collections
                if (c.startswith(f"project_{normalized_name}_") or c.startswith(f"dir_{normalized_name}_"))
                and not c.endswith("_file_metadata")
            ]

            if normalized_match_collections:
                search_collections.extend(normalized_match_collections)
            else:
                # Finally, try partial case-insensitive matching
                partial_match_collections = [
                    c
                    for c in all_collections
                    if (
                        (c.startswith("project_") and original_name.lower() in c.lower())
                        or (c.startswith("dir_") and original_name.lower() in c.lower())
                    )
                    and not c.endswith("_file_metadata")
                ]
                search_collections.extend(partial_match_collections)
    return search_collections


def get_search_collections(cross_project: bool = False, project_context: str | None = None) -> list[str]:
    """Get list of collections to search based on scope and project context.

    Args:
        cross_project: Whether to search across all projects
        project_context: Specific project context (optional)

    Returns:
        List of collection names to search
    """
    try:
        qdrant_client = get_qdrant_client()
        all_collections = [c.name for c in qdrant_client.get_collections().collections]

        if cross_project:
            # Return all non-metadata collections
            return [c for c in all_collections if not c.endswith("_file_metadata")]

        # Determine project context
        if project_context:
            project_prefix = f"project_{project_context}"
        else:
            current_project = get_current_project()
            project_prefix = current_project.get("collection_prefix", "global") if current_project else "global"

        # Return project-specific collections
        return [c for c in all_collections if c.startswith(project_prefix) and not c.endswith("_file_metadata")]

    except Exception as e:
        logger.error(f"Failed to get search collections: {e}")
        return []


def validate_search_parameters(query: str, n_results: int, search_mode: str, context_chunks: int) -> list[str]:
    """Validate search parameters and return list of validation errors.

    Args:
        query: Search query
        n_results: Number of results requested
        search_mode: Search mode
        context_chunks: Number of context chunks

    Returns:
        List of validation error messages (empty if all valid)
    """
    errors = []

    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        errors.append("Query must be a non-empty string")
    elif len(query) > 1000:
        errors.append("Query must be less than 1000 characters")

    if not isinstance(n_results, int) or n_results < 1 or n_results > 100:
        errors.append("n_results must be an integer between 1 and 100")

    if search_mode not in ["semantic", "keyword", "hybrid"]:
        errors.append("search_mode must be 'semantic', 'keyword', or 'hybrid'")

    if not isinstance(context_chunks, int) or context_chunks < 0 or context_chunks > 5:
        errors.append("context_chunks must be an integer between 0 and 5")

    return errors


def format_search_result_summary(results: list[dict[str, Any]], query: str) -> str:
    """Format a human-readable summary of search results.

    Args:
        results: List of search results
        query: Original search query

    Returns:
        Formatted summary string
    """
    if not results:
        return f"No results found for query: '{query}'"

    total = len(results)

    # Count results by type
    type_counts = {}
    file_counts = {}

    for result in results:
        result_type = result.get("type", "unknown")
        type_counts[result_type] = type_counts.get(result_type, 0) + 1

        file_path = result.get("file_path", "")
        if file_path:
            file_counts[file_path] = file_counts.get(file_path, 0) + 1

    summary_parts = [f"Found {total} results for query: '{query}'"]

    # Add type breakdown
    if type_counts:
        type_summary = ", ".join([f"{count} {type_name}" for type_name, count in type_counts.items()])
        summary_parts.append(f"Types: {type_summary}")

    # Add file count
    unique_files = len(file_counts)
    if unique_files > 0:
        summary_parts.append(f"Across {unique_files} files")

    # Add top score
    if results:
        top_score = max(result.get("score", 0) for result in results)
        summary_parts.append(f"Top relevance score: {top_score:.3f}")

    return " | ".join(summary_parts)


# Additional search tools for the MCP registry


async def analyze_repository_tool(directory: str = ".") -> dict[str, Any]:
    """
    Analyze repository structure and provide detailed statistics for indexing planning.

    This tool helps assess repository complexity, file distribution, and provides
    recommendations for optimal indexing strategies.

    Args:
        directory: Path to the directory to analyze (default: current directory)

    Returns:
        Detailed analysis including file counts, size distribution, language breakdown,
        complexity assessment, and indexing recommendations.
    """
    try:
        from services.project_analysis_service import ProjectAnalysisService

        logger.info(f"Analyzing repository: {directory}")

        analysis_service = ProjectAnalysisService()
        analysis = analysis_service.analyze_repository(directory)

        if "error" in analysis:
            logger.error(f"Repository analysis failed: {analysis['error']}")
            return analysis

        # Log key statistics
        logger.info("Repository analysis complete:")
        logger.info(f"  - Total files: {analysis['total_files']:,}")
        logger.info(f"  - Relevant files: {analysis['relevant_files']:,}")
        logger.info(f"  - Exclusion rate: {analysis['exclusion_rate']}%")
        logger.info(f"  - Repository size: {analysis['size_analysis']['total_size_mb']}MB")
        logger.info(f"  - Complexity level: {analysis['indexing_complexity']['level']}")

        return analysis

    except Exception as e:
        error_msg = f"Repository analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg, "directory": directory}


async def get_file_filtering_stats_tool(directory: str = ".") -> dict[str, Any]:
    """
    Get detailed statistics about file filtering for debugging and optimization.

    This tool shows how many files are excluded by different criteria,
    helping users understand and optimize their .ragignore patterns.

    Args:
        directory: Path to the directory to analyze (default: current directory)

    Returns:
        Detailed breakdown of file filtering statistics including exclusion reasons,
        configuration settings, and recommendations.
    """
    try:
        from services.project_analysis_service import ProjectAnalysisService

        logger.info(f"Analyzing file filtering for: {directory}")

        analysis_service = ProjectAnalysisService()
        stats = analysis_service.get_file_filtering_stats(directory)

        if "error" in stats:
            logger.error(f"File filtering analysis failed: {stats['error']}")
            return stats

        # Log filtering statistics
        logger.info("File filtering analysis complete:")
        logger.info(f"  - Total examined: {stats['total_examined']:,}")
        logger.info(f"  - Included: {stats['included']:,}")
        logger.info(f"  - Excluded by size: {stats['excluded_by_size']:,}")
        logger.info(f"  - Excluded by binary detection: {stats['excluded_by_binary_extension'] + stats['excluded_by_binary_header']:,}")
        logger.info(f"  - Excluded by .ragignore: {stats['excluded_by_ragignore']:,}")

        # Calculate percentages
        total = stats["total_examined"]
        if total > 0:
            stats["inclusion_rate"] = round(stats["included"] / total * 100, 1)
            stats["size_exclusion_rate"] = round(stats["excluded_by_size"] / total * 100, 1)
            stats["binary_exclusion_rate"] = round(
                (stats["excluded_by_binary_extension"] + stats["excluded_by_binary_header"]) / total * 100,
                1,
            )
            stats["ragignore_exclusion_rate"] = round(stats["excluded_by_ragignore"] / total * 100, 1)

        return stats

    except Exception as e:
        error_msg = f"File filtering analysis failed: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": error_msg, "directory": directory}


async def check_index_status_tool(directory: str = ".") -> dict[str, Any]:
    """
    Check if a directory already has indexed data and provide recommendations.

    This tool helps users understand the current indexing state and make informed
    decisions about whether to reindex or use existing data.

    Args:
        directory: Path to the directory to check (default: current directory)

    Returns:
        Status information and recommendations for the indexed data
    """
    try:
        from .index_tools import (
            check_existing_index,
            estimate_indexing_time,
            get_current_project,
        )

        dir_path = Path(directory).resolve()
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}

        current_project = get_current_project(client_directory=str(dir_path))
        existing_index_info = check_existing_index(current_project)

        if not existing_index_info.get("has_existing_data", False):
            return {
                "has_existing_data": False,
                "message": "No existing indexed data found for this project",
                "project_context": (current_project.get("name", "unknown") if current_project else "unknown"),
                "directory": str(dir_path),
                "recommendation": "Ready for initial indexing",
            }

        # Get file count estimation for recommendations
        try:
            from services.project_analysis_service import ProjectAnalysisService

            analysis_service = ProjectAnalysisService()
            quick_analysis = analysis_service.analyze_repository(str(dir_path))
            estimated_file_count = quick_analysis.get("relevant_files", 0)
        except Exception:
            estimated_file_count = 0

        time_estimates = estimate_indexing_time(estimated_file_count, existing_index_info.get("total_points", 0))

        return {
            "has_existing_data": True,
            "project_context": existing_index_info["project_name"],
            "directory": str(dir_path),
            "existing_data": {
                "collections": existing_index_info["collections"],
                "total_points": existing_index_info["total_points"],
                "collection_details": existing_index_info.get("collection_details", []),
            },
            "estimates": time_estimates,
            "recommendations": {
                "current_status": "Data already indexed and ready for search",
                "reindex_time_estimate_minutes": time_estimates["estimated_time_minutes"],
                "time_saved_by_keeping_existing_minutes": time_estimates["time_saved_by_keeping_existing_minutes"],
                "actions": {
                    "use_existing": "Data is ready for search operations",
                    "full_reindex": "Call index_directory with clear_existing=true to reindex everything",
                    "incremental_update": "Call index_directory with incremental=true to update only changed files",
                },
            },
        }

    except Exception as e:
        return {
            "error": f"Failed to check index status: {str(e)}",
            "directory": directory,
        }
