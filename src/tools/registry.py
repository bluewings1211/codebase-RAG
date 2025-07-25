"""MCP Tools Registry

This module manages the registration of all MCP tools.
"""

import logging
import os  # Add this import for environment variables
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_tools(mcp_app: FastMCP) -> None:
    """
    Register all MCP tools with the FastMCP application.

    Args:
        mcp_app: The FastMCP application instance
    """
    # Check environment variable to determine registration mode
    env = os.getenv("MCP_ENV", "development").lower()
    logger.info(f"Registering tools in {env} mode")

    # Register MCP Prompts system first
    try:
        from src.prompts.registry import register_prompts

        register_prompts(mcp_app)
        logger.info("MCP Prompts registered successfully")
    except ImportError as e:
        logger.warning(f"Skipping MCP Prompts registration due to import issues: {e}")
    except Exception as e:
        logger.error(f"Failed to register MCP Prompts: {e}")

    logger.info("Registering MCP Tools...")

    # Register core tools (always registered)
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
        patterns: list[str] | None = None,
        recursive: bool = True,
        clear_existing: bool = False,
        incremental: bool = False,
        project_name: str | None = None,
    ):
        """Index files in a directory with smart existing data detection
        and time estimation.

        Args:
            directory: Directory to index (default: current directory)
            patterns: File patterns to include
                      (default: common code file types)
            recursive: Whether to index subdirectories (default: True)
            clear_existing: Whether to clear existing indexed data
                            (default: False)
            If False and existing data is found, returns
            recommendations instead of indexing
            incremental: Whether to use incremental indexing
                         (only process changed files) (default: False)
            project_name: Optional custom project name for collections
                          (default: auto-detect)

        Returns:
            Dictionary with indexing results, time estimates, or
            recommendations for existing data
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
        target_projects: list[str] | None = None,
    ):
        """Search indexed content using natural language queries.

        This tool provides function-level precision search with intelligent
        chunking, supporting multiple search modes and context expansion for
        better code understanding.

        Args:
            query: Natural language search query
            n_results: Number of results to return (1-100, default: 5)
            cross_project: Whether to search across all projects
                           (default: False - current project only)
            search_mode: Search strategy - "semantic", "keyword", or
                         "hybrid" (default: "hybrid")
            include_context: Whether to include surrounding code context
                             (default: True)
            context_chunks: Number of context chunks to include before/after
                            results (0-5, default: 1)
            target_projects: List of specific project names to search in
                             (optional)

        Returns:
            Dictionary containing search results with metadata, scores,
            and context
        """
        return await search_impl(
            query,
            n_results,
            cross_project,
            search_mode,
            include_context,
            context_chunks,
            target_projects,
        )

    @mcp_app.tool()
    async def analyze_repository_tool(directory: str = "."):
        """Analyze repository structure and provide detailed statistics
        for indexing planning.

        This tool helps assess repository complexity, file distribution,
        and provides recommendations for optimal indexing strategies.

        Args:
            directory: Path to the directory to analyze
                       (default: current directory)

        Returns:
            Detailed analysis including file counts, size distribution,
            language breakdown, complexity assessment, and indexing
            recommendations.
        """
        return await analyze_repository_impl(directory)

    @mcp_app.tool()
    async def get_file_filtering_stats_tool(directory: str = "."):
        """Get detailed statistics about file filtering for debugging
        and optimization.

        This tool shows how many files are excluded by different criteria,
        helping users understand and optimize their .ragignore patterns.

        Args:
            directory: Path to the directory to analyze
                       (default: current directory)

        Returns:
            Detailed breakdown of file filtering statistics including
            exclusion reasons, configuration settings, and recommendations.
        """
        return await get_file_filtering_stats_impl(directory)

    @mcp_app.tool()
    async def check_index_status(directory: str = "."):
        """Check if a directory already has indexed data and provide
        recommendations.

        This tool helps users understand the current indexing state and
        make informed decisions about whether to reindex or use existing data.

        Args:
            directory: Path to the directory to check
                       (default: current directory)

        Returns:
            Status information and recommendations for the indexed data
        """
        return await check_index_status_impl(directory)

    # Register project tools (core)
    from .project.project_tools import register_project_tools

    register_project_tools(mcp_app)

    # Register file tools (core)
    from .project.file_tools import register_file_tools

    register_file_tools(mcp_app)

    # Register specific cache tools for production
    from .cache.cache_control import get_cache_health_status
    from .cache.cache_management import (
        clear_all_caches,
        warm_cache_for_project,
    )
    from .cache.cache_optimization import optimize_cache_performance

    @mcp_app.tool()
    async def clear_all_caches_tool(
        reason: str = "manual_invalidation",
        confirm: bool = False,
    ):
        """Clear all caches across all services (DESTRUCTIVE OPERATION).

        Args:
            reason: Reason for clearing all caches
                    (manual_invalidation, system_upgrade, cache_corruption)
            confirm: Must be True to confirm this destructive operation

        Returns:
            Dictionary with clearing results and statistics
        """
        return await clear_all_caches(reason, confirm)

    @mcp_app.tool()
    async def warm_cache_for_project_tool(
        project_name: str,
        cache_types: list[str] | None = None,
        warmup_strategy: str = "comprehensive",
        max_concurrent: int = 5,
    ):
        """Warm up caches for a specific project with comprehensive
        preloading.

        Args:
            project_name: Name of the project to warm up
            cache_types: Types of caches to warm
                         (embedding, search, project, file, all)
            warmup_strategy: Warmup strategy (comprehensive, selective,
                             recent, critical)
            max_concurrent: Maximum concurrent warming operations

        Returns:
            Dictionary with cache warming results and statistics
        """
        return await warm_cache_for_project(project_name, cache_types, warmup_strategy, max_concurrent)

    @mcp_app.tool()
    async def get_cache_health_status_tool(
        include_detailed_checks: bool = True,
        check_connectivity: bool = True,
        check_performance: bool = True,
    ):
        """Get comprehensive cache health status across all services.

        Args:
            include_detailed_checks: Whether to include detailed health checks
            check_connectivity: Whether to check cache connectivity
            check_performance: Whether to check performance metrics

        Returns:
            Dictionary with comprehensive health status and alerts
        """
        return await get_cache_health_status(include_detailed_checks, check_connectivity, check_performance)

    @mcp_app.tool()
    async def optimize_cache_performance_tool(
        optimization_type: str = "comprehensive",
        apply_changes: bool = False,
        project_name: str | None = None,
    ):
        """Analyze cache performance and provide optimization
        recommendations.

        Args:
            optimization_type: Type of optimization (comprehensive, memory,
                               ttl, connections, hit_rate)
            apply_changes: Whether to automatically apply safe optimizations
            project_name: Optional project name for scoped optimization

        Returns:
            Dictionary with optimization analysis and recommendations
        """
        return await optimize_cache_performance(optimization_type, apply_changes, project_name)

    if env == "development":
        # Register additional tools for development mode
        # Register chunking tools
        from .indexing.chunking_tools import register_chunking_tools

        register_chunking_tools(mcp_app)

        # Register parser tools
        from .indexing.parser_tools import register_parser_tools

        register_parser_tools(mcp_app)

        # Register progress tools
        from .indexing.progress_tools import register_progress_tools

        register_progress_tools(mcp_app)

        # Register all cache management tools
        from .cache.cache_management import (
            debug_cache_key,
            generate_cache_report,
            get_cache_invalidation_stats,
            get_comprehensive_cache_stats,
            get_project_invalidation_policy,
            inspect_cache_state,
            invalidate_chunks,
            manual_invalidate_cache_keys,
            manual_invalidate_cache_pattern,
            manual_invalidate_file_cache,
            manual_invalidate_project_cache,
            preload_embedding_cache,
            preload_search_cache,
            set_project_invalidation_policy,
        )

        @mcp_app.tool()
        async def manual_invalidate_file_cache_tool(
            file_path: str,
            reason: str = "manual_invalidation",
            cascade: bool = True,
            use_partial: bool = True,
            old_content: str | None = None,
            new_content: str | None = None,
            project_name: str | None = None,
        ):
            """Manually invalidate cache entries for a specific file.

            Args:
                file_path: Path to the file to invalidate
                reason: Reason for invalidation
                       (manual_invalidation, file_modified, file_deleted,
                       content_changed, metadata_changed)
                cascade: Whether to cascade invalidation to dependent caches
                use_partial: Whether to use partial invalidation if content
                            is provided
                old_content: Previous content of the file
                            (for partial invalidation)
                new_content: New content of the file
                            (for partial invalidation)
                project_name: Project name for scoped invalidation

            Returns:
                Dictionary with invalidation results and optimization
                statistics
            """
            return await manual_invalidate_file_cache(file_path, reason, cascade, use_partial, old_content, new_content, project_name)

        @mcp_app.tool()
        async def manual_invalidate_project_cache_tool(
            project_name: str,
            reason: str = "manual_invalidation",
            invalidation_scope: str = "cascade",
            strategy: str = "immediate",
        ):
            """Manually invalidate all cache entries for a project.

            Args:
                project_name: Name of the project to invalidate
                reason: Reason for invalidation
                       (manual_invalidation, project_changed,
                       dependency_changed, system_upgrade)
                invalidation_scope: Scope of invalidation
                                   (file_only, project_wide, cascade,
                                   conservative, aggressive)
                strategy: Invalidation strategy (immediate, lazy, batch,
                          scheduled)

            Returns:
                Dictionary with invalidation results and statistics
            """
            return await manual_invalidate_project_cache(project_name, reason, invalidation_scope, strategy)

        @mcp_app.tool()
        async def manual_invalidate_cache_keys_tool(
            cache_keys: list[str],
            reason: str = "manual_invalidation",
            cascade: bool = False,
        ):
            """Manually invalidate specific cache keys.

            Args:
                cache_keys: List of cache keys to invalidate
                reason: Reason for invalidation
                       (manual_invalidation, dependency_changed,
                       cache_corruption, ttl_expired)
                cascade: Whether to cascade invalidation to dependent caches

            Returns:
                Dictionary with invalidation results and statistics
            """
            return await manual_invalidate_cache_keys(cache_keys, reason, cascade)

        @mcp_app.tool()
        async def manual_invalidate_cache_pattern_tool(
            pattern: str,
            reason: str = "manual_invalidation",
        ):
            """Manually invalidate cache keys matching a pattern.

            Args:
                pattern: Pattern to match cache keys (supports wildcards)
                reason: Reason for invalidation
                       (manual_invalidation, dependency_changed,
                       cache_corruption, system_upgrade)

            Returns:
                Dictionary with invalidation results and statistics
            """
            return await manual_invalidate_cache_pattern(pattern, reason)

        @mcp_app.tool()
        async def get_cache_invalidation_stats_tool():
            """Get comprehensive cache invalidation statistics and metrics.

            Returns:
                Dictionary with detailed invalidation statistics, recent
                events, and monitoring info
            """
            return await get_cache_invalidation_stats()

        @mcp_app.tool()
        async def get_project_invalidation_policy_tool(project_name: str):
            """Get invalidation policy for a specific project.

            Args:
                project_name: Name of the project

            Returns:
                Dictionary with project invalidation policy details and
                monitoring status
            """
            return await get_project_invalidation_policy(project_name)

        @mcp_app.tool()
        async def set_project_invalidation_policy_tool(
            project_name: str,
            scope: str = "cascade",
            strategy: str = "immediate",
            batch_threshold: int = 5,
            delay_seconds: float = 0.0,
            file_patterns: list[str] | None = None,
            exclude_patterns: list[str] | None = None,
            invalidate_embeddings: bool = True,
            invalidate_search: bool = True,
            invalidate_project: bool = True,
            invalidate_file: bool = True,
            max_concurrent_invalidations: int = 10,
            cascade_depth_limit: int = 3,
        ):
            """Set or update invalidation policy for a specific project.

            Args:
                project_name: Name of the project
                scope: Invalidation scope (file_only, project_wide, cascade,
                       conservative, aggressive)
                strategy: Invalidation strategy (immediate, lazy, batch,
                          scheduled)
                batch_threshold: Number of changes to trigger batch processing
                delay_seconds: Delay before processing invalidation
                file_patterns: File patterns to monitor
                               (default: common code files)
                exclude_patterns: Patterns to exclude from monitoring
                                  (default: temp/cache files)
                invalidate_embeddings: Whether to invalidate embedding caches
                invalidate_search: Whether to invalidate search caches
                invalidate_project: Whether to invalidate project caches
                invalidate_file: Whether to invalidate file caches
                max_concurrent_invalidations: Maximum concurrent invalidations
                cascade_depth_limit: Maximum cascade depth

            Returns:
                Dictionary with policy creation/update results
            """
            return await set_project_invalidation_policy(
                project_name,
                scope,
                strategy,
                batch_threshold,
                delay_seconds,
                file_patterns,
                exclude_patterns,
                invalidate_embeddings,
                invalidate_search,
                invalidate_project,
                invalidate_file,
                max_concurrent_invalidations,
                cascade_depth_limit,
            )

        @mcp_app.tool()
        async def invalidate_chunks_tool(
            file_path: str,
            chunk_ids: list[str],
            reason: str = "chunk_modified",
        ):
            """Invalidate specific chunks within a file.

            Args:
                file_path: Path to the file containing the chunks
                chunk_ids: List of chunk IDs to invalidate
                reason: Reason for chunk invalidation (chunk_modified,
                       manual_invalidation, content_changed)

            Returns:
                Dictionary with chunk invalidation results and statistics
            """
            return await invalidate_chunks(file_path, chunk_ids, reason)

        @mcp_app.tool()
        async def inspect_cache_state_tool(
            cache_type: str = "all",
            include_content: bool = False,
            max_entries: int = 100,
        ):
            """Inspect the current state of cache services with detailed
            debugging information.

            Args:
                cache_type: Type of cache to inspect (all, embedding, search,
                            project, file, l1, l2)
                include_content: Whether to include actual cache content in
                                 response
                max_entries: Maximum number of cache entries to include

            Returns:
                Dictionary with detailed cache state information and service
                health
            """
            return await inspect_cache_state(cache_type, include_content, max_entries)

        @mcp_app.tool()
        async def debug_cache_key_tool(
            cache_key: str,
            cache_type: str = "all",
        ):
            """Debug a specific cache key across all cache services.

            Args:
                cache_key: The cache key to debug
                cache_type: Type of cache to check (all, embedding, search,
                            project, file)

            Returns:
                Dictionary with debugging information for the cache key
            """
            return await debug_cache_key(cache_key, cache_type)

        @mcp_app.tool()
        async def preload_embedding_cache_tool(
            queries: list[str],
            project_name: str | None = None,
            model_name: str | None = None,
        ):
            """Preload embedding cache with specific queries or content.

            Args:
                queries: List of queries/content to preload embeddings for
                project_name: Optional project name for scoped preloading
                model_name: Optional specific embedding model name

            Returns:
                Dictionary with preloading results and statistics
            """
            return await preload_embedding_cache(queries, project_name, model_name)

        @mcp_app.tool()
        async def preload_search_cache_tool(
            search_queries: list[dict],
            project_name: str | None = None,
        ):
            """Preload search result cache with specific search queries.

            Args:
                search_queries: List of search query dictionaries with
                               parameters
                project_name: Optional project name for scoped preloading

            Returns:
                Dictionary with search cache preloading results
            """
            return await preload_search_cache(search_queries, project_name)

        @mcp_app.tool()
        async def get_comprehensive_cache_stats_tool(
            project_name: str | None = None,
            include_historical: bool = False,
            time_range_hours: int = 24,
        ):
            """Get comprehensive cache statistics across all cache services.

            Args:
                project_name: Optional project name for scoped statistics
                include_historical: Whether to include historical data
                time_range_hours: Time range for historical data (hours)

            Returns:
                Dictionary with comprehensive cache statistics and aggregated
                metrics
            """
            return await get_comprehensive_cache_stats(project_name, include_historical, time_range_hours)

        @mcp_app.tool()
        async def generate_cache_report_tool(
            report_type: str = "comprehensive",
            project_name: str | None = None,
            export_format: str = "json",
        ):
            """Generate a comprehensive cache performance report.

            Args:
                report_type: Type of report (comprehensive, performance,
                             health, optimization)
                project_name: Optional project name for scoped reporting
                export_format: Format for export (json, markdown, csv)

            Returns:
                Dictionary with cache report data and optimization
                recommendations
            """
            return await generate_cache_report(report_type, project_name, export_format)

        # Register cache control interface tools
        from .cache.cache_control import (
            configure_cache_alerts,
            export_cache_configuration,
            get_cache_alerts,
            get_cache_configuration,
            import_cache_configuration,
            update_cache_configuration,
        )

        @mcp_app.tool()
        async def get_cache_configuration_tool(
            config_type: str = "all",
            export_format: str = "json",
        ):
            """Get current cache configuration across all services.

            Args:
                config_type: Type of configuration to retrieve (all, redis,
                             memory, ttl, limits, security)
                export_format: Format for export (json, yaml, env)

            Returns:
                Dictionary with cache configuration details and formatted
                output
            """
            return await get_cache_configuration(config_type, export_format)

        @mcp_app.tool()
        async def update_cache_configuration_tool(
            config_updates: dict[str, Any],
            validate_only: bool = False,
            restart_services: bool = False,
        ):
            """Update cache configuration settings.

            Args:
                config_updates: Dictionary of configuration updates to apply
                validate_only: Only validate changes without applying them
                restart_services: Whether to restart cache services after
                                  updates

            Returns:
                Dictionary with update results and validation status
            """
            return await update_cache_configuration(config_updates, validate_only, restart_services)

        @mcp_app.tool()
        async def export_cache_configuration_tool(
            export_path: str,
            config_type: str = "all",
            include_sensitive: bool = False,
        ):
            """Export cache configuration to file.

            Args:
                export_path: Path to export configuration file
                config_type: Type of configuration to export (all, redis,
                             memory, ttl, limits)
                include_sensitive: Whether to include sensitive information
                                   (passwords, keys)

            Returns:
                Dictionary with export results and file information
            """
            return await export_cache_configuration(export_path, config_type, include_sensitive)

        @mcp_app.tool()
        async def import_cache_configuration_tool(
            import_path: str,
            config_type: str = "all",
            validate_only: bool = True,
            backup_current: bool = True,
        ):
            """Import cache configuration from file.

            Args:
                import_path: Path to configuration file to import
                config_type: Type of configuration to import (all, redis,
                             memory, ttl, limits)
                validate_only: Only validate without applying changes
                backup_current: Create backup of current configuration

            Returns:
                Dictionary with import results and validation status
            """
            return await import_cache_configuration(import_path, config_type, validate_only, backup_current)

        @mcp_app.tool()
        async def configure_cache_alerts_tool(
            alert_config: dict[str, Any],
            enable_alerts: bool = True,
        ):
            """Configure cache monitoring alerts and thresholds.

            Args:
                alert_config: Alert configuration including thresholds and
                              notification settings
                enable_alerts: Whether to enable alert monitoring

            Returns:
                Dictionary with alert configuration results
            """
            return await configure_cache_alerts(alert_config, enable_alerts)

        @mcp_app.tool()
        async def get_cache_alerts_tool(
            severity_filter: str = "all",
            time_range_hours: int = 24,
            service_filter: str = "all",
        ):
            """Get recent cache alerts and notifications.

            Args:
                severity_filter: Filter by severity (all, error, warning, info)
                time_range_hours: Time range for alert history (hours)
                service_filter: Filter by service (all, embedding, search,
                                project, file)

            Returns:
                Dictionary with recent alerts and statistics
            """
            return await get_cache_alerts(severity_filter, time_range_hours, service_filter)

        # Register cache optimization tools
        from .cache.cache_optimization import (
            backup_cache_data,
            get_migration_status,
            migrate_cache_data,
            restore_cache_data,
        )

        @mcp_app.tool()
        async def backup_cache_data_tool(
            backup_path: str,
            backup_type: str = "incremental",
            include_services: list[str] = None,
            compression: bool = True,
        ):
            """Create a backup of cache data and configuration.

            Args:
                backup_path: Path where backup will be created
                backup_type: Type of backup (full, incremental,
                             configuration_only)
                include_services: List of services to backup (default: all)
                compression: Whether to compress backup data

            Returns:
                Dictionary with backup operation results
            """
            return await backup_cache_data(backup_path, backup_type, include_services, compression)

        @mcp_app.tool()
        async def restore_cache_data_tool(
            backup_path: str,
            restore_type: str = "full",
            target_services: list[str] = None,
            validate_only: bool = False,
        ):
            """Restore cache data from backup.

            Args:
                backup_path: Path to backup file or directory
                restore_type: Type of restore (full, configuration_only,
                              data_only)
                target_services: List of services to restore (default: all
                                 from backup)
                validate_only: Only validate backup without restoring

            Returns:
                Dictionary with restore operation results
            """
            return await restore_cache_data(backup_path, restore_type, target_services, validate_only)

        @mcp_app.tool()
        async def migrate_cache_data_tool(
            migration_type: str,
            source_config: dict[str, Any] = None,
            target_config: dict[str, Any] = None,
            dry_run: bool = True,
        ):
            """Migrate cache data between different configurations or
            versions.

            Args:
                migration_type: Type of migration (redis_upgrade,
                                schema_migration, configuration_migration)
                source_config: Source configuration for migration
                target_config: Target configuration for migration
                dry_run: Whether to perform a dry run without making changes

            Returns:
                Dictionary with migration operation results
            """
            return await migrate_cache_data(migration_type, source_config, target_config, dry_run)

        @mcp_app.tool()
        async def get_migration_status_tool(
            migration_id: str = None,
        ):
            """Get status of ongoing or completed migrations.

            Args:
                migration_id: Optional specific migration ID to check

            Returns:
                Dictionary with migration status information
            """
            return await get_migration_status(migration_id)

        # Register file monitoring tools
        from .cache.file_monitoring_tools import register_file_monitoring_tools

        register_file_monitoring_tools(mcp_app)

        # Register cascade invalidation tools
        from .cache.cascade_invalidation_tools import register_cascade_invalidation_tools

        register_cascade_invalidation_tools(mcp_app)

    # Register Graph RAG tools (production mode)
    from .graph_rag.function_chain_analysis import trace_function_chain
    from .graph_rag.function_path_finding import find_function_path
    from .graph_rag.pattern_identification import graph_identify_patterns
    from .graph_rag.project_chain_analysis import analyze_project_chains
    from .graph_rag.similar_implementations import graph_find_similar_implementations
    from .graph_rag.structure_analysis import graph_analyze_structure

    @mcp_app.tool()
    async def graph_analyze_structure_tool(
        breadcrumb: str,
        project_name: str,
        analysis_type: str = "comprehensive",
        max_depth: int = 3,
        include_siblings: bool = False,
        include_connectivity: bool = True,
        force_rebuild_graph: bool = False,
        generate_report: bool = False,
        include_recommendations: bool = True,
        enable_performance_optimization: bool = True,
    ):
        """Analyze the structural relationships of a specific breadcrumb in the codebase.

        This tool leverages Graph RAG capabilities to provide deep structural analysis
        of code components, including hierarchical relationships, connectivity patterns,
        and related components within the codebase structure.

        Args:
            breadcrumb: The breadcrumb path to analyze (e.g., "MyClass.method_name")
            project_name: Name of the project to analyze within
            analysis_type: Type of analysis ("comprehensive", "hierarchy", "connectivity", "overview")
            max_depth: Maximum depth for relationship traversal (1-10, default: 3)
            include_siblings: Whether to include sibling components in the analysis
            include_connectivity: Whether to analyze component connectivity patterns
            force_rebuild_graph: Whether to force rebuild the structure graph
            generate_report: Whether to generate a comprehensive analysis report with recommendations
            include_recommendations: Whether to include optimization recommendations in the report
            enable_performance_optimization: Whether to enable performance optimizations for large projects

        Returns:
            Dictionary containing structural analysis results with hierarchical relationships,
            connectivity patterns, and related components. If generate_report=True, includes
            a comprehensive report with statistics and recommendations.
        """
        return await graph_analyze_structure(
            breadcrumb,
            project_name,
            analysis_type,
            max_depth,
            include_siblings,
            include_connectivity,
            force_rebuild_graph,
            generate_report,
            include_recommendations,
            enable_performance_optimization,
        )

    @mcp_app.tool()
    async def graph_find_similar_implementations_tool(
        query: str,
        source_breadcrumb: str = None,
        source_project: str = None,
        target_projects: list[str] = None,
        exclude_projects: list[str] = None,
        chunk_types: list[str] = None,
        languages: list[str] = None,
        similarity_threshold: float = 0.7,
        structural_weight: float = 0.5,
        max_results: int = 10,
        include_implementation_chains: bool = False,
        include_architectural_context: bool = True,
    ):
        """Find similar implementations across projects using Graph RAG capabilities.

        This tool leverages cross-project search and implementation chain analysis
        to find similar code implementations, patterns, and architectural solutions
        across multiple indexed projects.

        Args:
            query: Natural language description of what to search for
            source_breadcrumb: Optional specific breadcrumb to find similar implementations for
            source_project: Optional source project name (used with source_breadcrumb)
            target_projects: List of specific projects to search in (default: all projects)
            exclude_projects: List of projects to exclude from search
            chunk_types: List of chunk types to include ("function", "class", "method", etc.)
            languages: List of programming languages to include
            similarity_threshold: Minimum semantic similarity score (0.0-1.0, default: 0.7)
            structural_weight: Weight for structural vs semantic similarity (0.0-1.0, default: 0.5)
            max_results: Maximum number of similar implementations to return (1-50, default: 10)
            include_implementation_chains: Whether to include implementation chain analysis
            include_architectural_context: Whether to include architectural context analysis

        Returns:
            Dictionary containing similar implementations with similarity scores,
            architectural context, and optional implementation chains
        """
        return await graph_find_similar_implementations(
            query,
            source_breadcrumb,
            source_project,
            target_projects,
            exclude_projects,
            chunk_types,
            languages,
            similarity_threshold,
            structural_weight,
            max_results,
            include_implementation_chains,
            include_architectural_context,
        )

    @mcp_app.tool()
    async def graph_identify_patterns_tool(
        project_name: str,
        pattern_types: list[str] = None,
        scope_breadcrumb: str = None,
        min_confidence: float = 0.6,
        include_comparisons: bool = True,
        include_improvements: bool = False,
        max_patterns: int = 20,
        analysis_depth: str = "comprehensive",
    ):
        """Identify architectural patterns in a codebase using Graph RAG capabilities.

        This tool leverages pattern recognition algorithms to detect common
        architectural patterns, design patterns, and code organization structures
        within the analyzed codebase.

        Args:
            project_name: Name of the project to analyze
            pattern_types: List of specific pattern types to look for
                          ("structural", "behavioral", "creational", "naming", "architectural")
            scope_breadcrumb: Optional breadcrumb to limit analysis scope (e.g., "MyClass")
            min_confidence: Minimum confidence threshold for pattern detection (0.0-1.0, default: 0.6)
            include_comparisons: Whether to include pattern comparison analysis
            include_improvements: Whether to suggest pattern improvements
            max_patterns: Maximum number of patterns to return (1-50, default: 20)
            analysis_depth: Depth of analysis ("basic", "comprehensive", "detailed")

        Returns:
            Dictionary containing identified patterns with confidence scores,
            pattern types, architectural context, and optional improvement suggestions
        """
        return await graph_identify_patterns(
            project_name,
            pattern_types,
            scope_breadcrumb,
            min_confidence,
            include_comparisons,
            include_improvements,
            max_patterns,
            analysis_depth,
        )

    @mcp_app.tool()
    async def trace_function_chain_tool(
        entry_point: str,
        project_name: str,
        direction: str = "forward",
        max_depth: int = 10,
        output_format: str = "arrow",
        include_mermaid: bool = False,
        chain_type: str = "execution_flow",
        min_link_strength: float = 0.3,
        identify_branch_points: bool = True,
        identify_terminal_points: bool = True,
        performance_monitoring: bool = True,
    ):
        """Trace a complete function chain from an entry point with various analysis options.

        This tool provides comprehensive function chain tracing capabilities, supporting
        multiple directions (forward/backward/bidirectional), output formats, and
        detailed analysis of execution flows, data flows, and dependencies.

        Args:
            entry_point: Function/class identifier (breadcrumb or natural language)
            project_name: Name of the project to analyze
            direction: Tracing direction ("forward", "backward", "bidirectional")
            max_depth: Maximum depth for chain traversal (default: 10)
            output_format: Output format ("arrow", "mermaid", "both")
            include_mermaid: Whether to include Mermaid diagram output
            chain_type: Type of chain to trace ("execution_flow", "data_flow", "dependency_chain")
            min_link_strength: Minimum link strength threshold (0.0-1.0)
            identify_branch_points: Whether to identify branch points in the chain
            identify_terminal_points: Whether to identify terminal points in the chain
            performance_monitoring: Whether to include performance monitoring

        Returns:
            Dictionary containing chain analysis results with formatted output
        """
        return await trace_function_chain(
            entry_point,
            project_name,
            direction,
            max_depth,
            output_format,
            include_mermaid,
            chain_type,
            min_link_strength,
            identify_branch_points,
            identify_terminal_points,
            performance_monitoring,
        )

    @mcp_app.tool()
    async def find_function_path_tool(
        start_function: str,
        end_function: str,
        project_name: str,
        strategy: str = "optimal",
        max_paths: int = 3,
        max_depth: int = 15,
        include_quality_metrics: bool = True,
        output_format: str = "arrow",
        include_mermaid: bool = False,
        min_link_strength: float = 0.3,
        optimize_for: str = "reliability",
    ):
        """Find the most efficient path between two functions in a codebase.

        This tool discovers how functions are connected and identifies the optimal
        navigation paths between them, supporting various path-finding strategies
        and quality metrics.

        Args:
            start_function: Starting function identifier (breadcrumb or natural language)
            end_function: Target function identifier (breadcrumb or natural language)
            project_name: Name of the project to search within
            strategy: Path finding strategy ("shortest", "optimal", "all")
            max_paths: Maximum number of paths to return (1-10, default: 3)
            max_depth: Maximum search depth for path finding (default: 15)
            include_quality_metrics: Whether to calculate path quality metrics
            output_format: Output format ("arrow", "mermaid", "both")
            include_mermaid: Whether to include Mermaid diagram output
            min_link_strength: Minimum link strength for path inclusion (0.0-1.0)
            optimize_for: Optimization criteria ("reliability", "directness", "simplicity")

        Returns:
            Dictionary containing found paths with quality metrics and formatted output
        """
        return await find_function_path(
            start_function,
            end_function,
            project_name,
            strategy,
            max_paths,
            max_depth,
            include_quality_metrics,
            output_format,
            include_mermaid,
            min_link_strength,
            optimize_for,
        )

    @mcp_app.tool()
    async def analyze_project_chains_tool(
        project_name: str,
        analysis_scope: str = "full_project",
        breadcrumb_patterns: list[str] = None,
        analysis_types: list[str] = None,
        max_functions_per_chain: int = 50,
        complexity_threshold: float = 0.7,
        output_format: str = "comprehensive",
        include_mermaid: bool = True,
        include_hotspot_analysis: bool = True,
        include_refactoring_suggestions: bool = False,
        enable_complexity_weighting: bool = True,
        complexity_weights: dict[str, float] = None,
    ):
        """Analyze function chains across an entire project with comprehensive insights.

        This tool provides project-wide analysis of function chains, patterns, complexity,
        and architecture, supporting pattern matching, hotspot identification, and
        refactoring recommendations.

        Args:
            project_name: Name of the project to analyze
            analysis_scope: Scope of analysis ("full_project", "scoped_breadcrumbs",
                          "specific_modules", "function_patterns")
            breadcrumb_patterns: List of breadcrumb patterns to focus analysis on
            analysis_types: Types of analysis to perform ("complexity_analysis",
                          "hotspot_identification", "pattern_detection", "architectural_analysis")
            max_functions_per_chain: Maximum functions to include per chain (default: 50)
            complexity_threshold: Complexity threshold for highlighting (0.0-1.0, default: 0.7)
            output_format: Output format ("comprehensive", "summary", "detailed")
            include_mermaid: Whether to include Mermaid diagram outputs
            include_hotspot_analysis: Whether to identify complexity hotspots
            include_refactoring_suggestions: Whether to provide refactoring recommendations
            enable_complexity_weighting: Whether to use weighted complexity calculations
            complexity_weights: Custom complexity weights (branching_factor, cyclomatic_complexity, etc.)

        Returns:
            Dictionary containing comprehensive project chain analysis with patterns,
            complexity metrics, hotspots, and optional refactoring suggestions
        """
        return await analyze_project_chains(
            project_name,
            analysis_types,
            "*",  # scope_pattern
            complexity_weights,
            None,  # chain_types
            complexity_threshold,
            max_functions_per_chain,
            include_refactoring_suggestions,
            output_format,
            True,  # performance_monitoring
            50,  # batch_size
        )

    logger.info("All MCP Tools registered successfully")
