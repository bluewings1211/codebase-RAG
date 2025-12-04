"""Output control utilities for MCP tools.

This module provides utilities to control the level of detail in MCP tool outputs
based on environment variables and user preferences.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def get_output_config() -> dict[str, bool]:
    """Get output configuration based on environment variables.

    This function reads MCP_ENV and MCP_DEBUG_LEVEL environment variables
    to determine the appropriate level of detail for MCP tool outputs.

    Environment Variables:
        MCP_ENV: 'production' or 'development' (default: 'development')
        MCP_DEBUG_LEVEL: 'DEBUG', 'INFO', 'WARNING', 'ERROR' (default: 'INFO')
        CACHE_DEBUG_MODE: 'true' or 'false' (default: 'false')

    Returns:
        Dictionary with configuration flags:
        - include_performance: Include performance metrics and timing data
        - include_technical_details: Include technical implementation details
        - include_internal_metadata: Include internal system metadata
        - include_debug_info: Include debug-level information
    """
    # Get environment variables with defaults
    mcp_env = os.getenv("MCP_ENV", "development").lower()
    debug_level = os.getenv("MCP_DEBUG_LEVEL", "INFO").upper()
    cache_debug = os.getenv("CACHE_DEBUG_MODE", "false").lower() == "true"

    # Determine flags based on environment
    is_production = mcp_env == "production"
    is_debug = debug_level == "DEBUG"
    is_development = mcp_env == "development"

    config = {
        # Performance metrics - show in debug mode or when cache debugging
        "include_performance": is_debug or cache_debug,
        # Technical details - show in development or debug mode
        "include_technical_details": is_development or is_debug,
        # Internal metadata - only in debug mode
        "include_internal_metadata": is_debug,
        # Debug info - only in debug mode
        "include_debug_info": is_debug,
        # Minimal output for production
        "minimal_output": is_production and not is_debug,
    }

    logger.debug(f"Output config: {config} (env={mcp_env}, debug={debug_level})")
    return config


def filter_search_results(
    results: dict[str, Any],
    minimal_output: bool = False,
    config: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Filter search results based on output configuration.

    Args:
        results: Raw search results dictionary
        minimal_output: Force minimal output mode (overrides config)
        config: Output configuration (uses get_output_config() if None)

    Returns:
        Filtered results dictionary with appropriate level of detail
    """
    if config is None:
        config = get_output_config()

    # Override config if minimal_output is explicitly requested
    if minimal_output:
        config = {
            "include_performance": False,
            "include_technical_details": False,
            "include_internal_metadata": False,
            "include_debug_info": False,
            "minimal_output": True,
        }

    # Always keep essential fields
    filtered_results = {
        "query": results.get("query"),
        "results": [],
        "total": results.get("total", 0),
    }

    # Process individual results
    for result in results.get("results", []):
        filtered_result = {
            # Essential fields always included
            "file_path": result.get("file_path"),
            "content": result.get("content"),
            "breadcrumb": result.get("breadcrumb"),
            "chunk_type": result.get("chunk_type"),
            "language": result.get("language"),
            "line_start": result.get("line_start", 0),
            "line_end": result.get("line_end", 0),
        }

        # Add optional fields based on configuration
        if not config.get("minimal_output", False):
            # Add basic metadata
            if "name" in result:
                filtered_result["name"] = result["name"]
            if "signature" in result:
                filtered_result["signature"] = result["signature"]
            if "docstring" in result:
                filtered_result["docstring"] = result["docstring"]

        if config.get("include_technical_details", True):
            # Add technical scoring details
            filtered_result.update(
                {
                    "local_score": result.get("local_score"),
                    "global_score": result.get("global_score"),
                    "combined_score": result.get("combined_score"),
                    "confidence_level": result.get("confidence_level"),
                    "retrieval_mode": result.get("retrieval_mode"),
                    "retrieval_source": result.get("retrieval_source"),
                }
            )

        if config.get("include_internal_metadata", False):
            # Add internal system metadata
            filtered_result.update(
                {
                    "local_context": result.get("local_context", []),
                    "global_context": result.get("global_context", []),
                    "relationship_paths": result.get("relationship_paths", []),
                    "rank": result.get("rank"),
                    "project": result.get("project"),
                }
            )

        filtered_results["results"].append(filtered_result)

    # Add optional top-level metadata
    if config.get("include_technical_details", True):
        filtered_results.update(
            {
                "mode_used": results.get("mode_used"),
                "projects_searched": results.get("projects_searched"),
                "search_method": results.get("search_method"),
            }
        )

    if config.get("include_performance", False):
        # Add performance data
        if "performance" in results:
            filtered_results["performance"] = results["performance"]
        if "performance_context" in results:
            filtered_results["performance_context"] = results["performance_context"]
        if "_performance" in results:
            filtered_results["_performance"] = results["_performance"]

    if config.get("include_internal_metadata", False):
        # Add internal metadata
        if "multi_modal_metadata" in results:
            filtered_results["multi_modal_metadata"] = results["multi_modal_metadata"]
        if "query_analysis" in results:
            filtered_results["query_analysis"] = results["query_analysis"]

    return filtered_results


def get_environment_info() -> dict[str, str]:
    """Get current environment configuration for debugging.

    Returns:
        Dictionary with current environment variable values
    """
    return {
        "MCP_ENV": os.getenv("MCP_ENV", "development"),
        "MCP_DEBUG_LEVEL": os.getenv("MCP_DEBUG_LEVEL", "INFO"),
        "CACHE_DEBUG_MODE": os.getenv("CACHE_DEBUG_MODE", "false"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }
