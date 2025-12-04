"""Output control utilities for MCP tools.

This module provides utilities to control the level of detail in MCP tool outputs
based on environment variables and user preferences. Designed to reduce output
verbosity for AI agents while preserving essential information.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def get_output_config() -> dict[str, bool]:
    """Get output configuration based on environment variables.

    Environment Variables:
        MCP_MINIMAL_OUTPUT: 'true' or 'false' (default: 'true' for agent-friendly output)
        MCP_DEBUG_LEVEL: 'DEBUG', 'INFO', 'WARNING', 'ERROR' (default: 'INFO')

    Returns:
        Dictionary with configuration flags:
        - minimal_output: Return minimal output suitable for AI agents
        - include_scores: Include relevance scores
        - include_metadata: Include chunk metadata (name, signature, docstring)
        - include_performance: Include performance metrics
        - include_debug_info: Include debug-level information
    """
    minimal_output = os.getenv("MCP_MINIMAL_OUTPUT", "true").lower() == "true"
    debug_level = os.getenv("MCP_DEBUG_LEVEL", "INFO").upper()
    is_debug = debug_level == "DEBUG"

    config = {
        "minimal_output": minimal_output and not is_debug,
        "include_scores": not minimal_output or is_debug,
        "include_metadata": not minimal_output or is_debug,
        "include_performance": is_debug,
        "include_debug_info": is_debug,
    }

    logger.debug(f"Output config: {config} (minimal={minimal_output}, debug={debug_level})")
    return config


def filter_search_results(
    results: dict[str, Any],
    minimal_output: bool | None = None,
) -> dict[str, Any]:
    """Filter search results based on output configuration.

    Args:
        results: Raw search results dictionary
        minimal_output: Force minimal output mode (None = use env config)

    Returns:
        Filtered results dictionary with appropriate level of detail
    """
    config = get_output_config()

    # Override config if minimal_output is explicitly specified
    if minimal_output is not None:
        config["minimal_output"] = minimal_output
        config["include_scores"] = not minimal_output
        config["include_metadata"] = not minimal_output
        config["include_performance"] = False

    # If not minimal, return original results
    if not config["minimal_output"]:
        return results

    # Build minimal response
    filtered_results = {
        "query": results.get("query"),
        "total": results.get("total", 0),
        "results": [],
    }

    # Process individual results - keep only essential fields
    for result in results.get("results", []):
        filtered_result = {
            # Essential fields for code navigation
            "file_path": result.get("file_path"),
            "line_start": result.get("line_start", 0),
            "line_end": result.get("line_end", 0),
            # Content - truncated for minimal output
            "content": _truncate_for_minimal(result.get("content", ""), max_length=800),
            # Context for understanding
            "breadcrumb": result.get("breadcrumb"),
            "chunk_type": result.get("chunk_type"),
            "language": result.get("language"),
        }

        # Include score only if configured
        if config["include_scores"]:
            filtered_result["score"] = result.get("score")

        # Include metadata only if configured
        if config["include_metadata"]:
            if result.get("name"):
                filtered_result["name"] = result["name"]
            if result.get("signature"):
                filtered_result["signature"] = result["signature"]

        filtered_results["results"].append(filtered_result)

    # Add minimal top-level info
    if results.get("search_scope"):
        filtered_results["search_scope"] = results["search_scope"]

    # Add error info if present
    if results.get("error"):
        filtered_results["error"] = results["error"]

    if results.get("suggestions"):
        filtered_results["suggestions"] = results["suggestions"]

    return filtered_results


def _truncate_for_minimal(content: str, max_length: int = 800) -> str:
    """Truncate content for minimal output mode.

    Args:
        content: Text content to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated content with indicator if needed
    """
    if not content or len(content) <= max_length:
        return content
    return content[:max_length] + "\n... (truncated)"


def get_environment_info() -> dict[str, str]:
    """Get current environment configuration for debugging.

    Returns:
        Dictionary with current environment variable values
    """
    return {
        "MCP_MINIMAL_OUTPUT": os.getenv("MCP_MINIMAL_OUTPUT", "true"),
        "MCP_DEBUG_LEVEL": os.getenv("MCP_DEBUG_LEVEL", "INFO"),
    }
