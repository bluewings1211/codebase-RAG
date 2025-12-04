"""MCP Tools Registry

This module manages the registration of all MCP tools.
"""

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_tools(mcp_app: FastMCP) -> None:
    """
    Register all MCP tools with the FastMCP application.

    Args:
        mcp_app: The FastMCP application instance
    """
    # Register MCP Prompts system first
    try:
        from prompts import register_prompts

        register_prompts(mcp_app)
        logger.info("MCP Prompts system registered successfully")
    except Exception as e:
        logger.error(f"Failed to register MCP Prompts system: {e}")
        # Continue without prompts if there's an error

    logger.info("Registering MCP Tools...")

    # Register core tools
    from .core.health import health_check

    @mcp_app.tool()
    async def health_check_tool():
        """Check the health of the MCP server and its dependencies.

        Checks the status of:
        - Qdrant database connectivity
        - Ollama service availability
        - Memory usage and system resources

        Returns detailed health information with warnings and issues.
        """
        return await health_check()

    # Register indexing tools
    from .indexing.index_tools import index_directory as index_directory_impl

    @mcp_app.tool()
    async def index_directory(
        directory: str = ".",
        patterns: list[str] = None,
        recursive: bool = True,
        clear_existing: bool = False,
        incremental: bool = False,
        project_name: str = None,
    ):
        """Index files in a directory with smart existing data detection and time estimation.

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
        return await index_directory_impl(directory, patterns, recursive, clear_existing, incremental, project_name)

    # Register search tools
    from .indexing.search_tools import (
        analyze_repository_tool as analyze_repository_impl,
    )
    from .indexing.search_tools import (
        check_index_status_tool as check_index_status_impl,
    )
    from .indexing.search_tools import (
        get_file_filtering_stats_tool as get_file_filtering_stats_impl,
    )
    from .indexing.search_tools import search as search_impl

    @mcp_app.tool()
    async def search(
        query: str,
        n_results: int = 5,
        cross_project: bool = False,
        search_mode: str = "hybrid",
        include_context: bool = True,
        context_chunks: int = 1,
        target_projects: list[str] = None,
        collection_types: list[str] = None,
    ):
        """Search indexed content using natural language queries.

        This tool provides function-level precision search with intelligent chunking,
        supporting multiple search modes and context expansion for better code understanding.

        Args:
            query: Natural language search query
            n_results: Number of results to return (1-100, default: 5)
            cross_project: Whether to search across all projects (default: False - current project only)
            search_mode: Search strategy - "semantic", "keyword", or "hybrid" (default: "hybrid")
            include_context: Whether to include surrounding code context (default: True)
            context_chunks: Number of context chunks to include before/after results (0-5, default: 1)
            target_projects: List of specific project names to search in (optional)
            collection_types: List of collection types to search in (optional)
                            - ["code"] - Only search source code files
                            - ["config"] - Only search configuration files
                            - ["documentation"] - Only search documentation files
                            - None - Search all collection types (default)

        Returns:
            Dictionary containing search results with metadata, scores, and context
        """
        return await search_impl(
            query,
            n_results,
            cross_project,
            search_mode,
            include_context,
            context_chunks,
            target_projects,
            collection_types,
        )

    @mcp_app.tool()
    async def analyze_repository_tool(directory: str = "."):
        """Analyze repository structure and provide detailed statistics for indexing planning.

        This tool helps assess repository complexity, file distribution, and provides
        recommendations for optimal indexing strategies.

        Args:
            directory: Path to the directory to analyze (default: current directory)

        Returns:
            Detailed analysis including file counts, size distribution, language breakdown,
            complexity assessment, and indexing recommendations.
        """
        return await analyze_repository_impl(directory)

    @mcp_app.tool()
    async def get_file_filtering_stats_tool(directory: str = "."):
        """Get detailed statistics about file filtering for debugging and optimization.

        This tool shows how many files are excluded by different criteria,
        helping users understand and optimize their .ragignore patterns.

        Args:
            directory: Path to the directory to analyze (default: current directory)

        Returns:
            Detailed breakdown of file filtering statistics including exclusion reasons,
            configuration settings, and recommendations.
        """
        return await get_file_filtering_stats_impl(directory)

    @mcp_app.tool()
    async def check_index_status(directory: str = "."):
        """Check if a directory already has indexed data and provide recommendations.

        This tool helps users understand the current indexing state and make informed
        decisions about whether to reindex or use existing data.

        Args:
            directory: Path to the directory to check (default: current directory)

        Returns:
            Status information and recommendations for the indexed data
        """
        return await check_index_status_impl(directory)

    # Register chunking tools
    from .indexing.chunking_tools import register_chunking_tools

    register_chunking_tools(mcp_app)

    # Register parser tools
    from .indexing.parser_tools import register_parser_tools

    register_parser_tools(mcp_app)

    # Register progress tools
    from .indexing.progress_tools import register_progress_tools

    register_progress_tools(mcp_app)

    # Register project tools
    from .project.project_tools import register_project_tools

    register_project_tools(mcp_app)

    # Register file tools
    from .project.file_tools import register_file_tools

    register_file_tools(mcp_app)

    logger.info("All MCP Tools registered successfully")
