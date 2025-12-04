#!/usr/bin/env python3
"""
Manual Indexing Tool for Codebase RAG MCP Server with Intelligent Code Chunking

This standalone script performs intelligent code indexing operations independently
from the MCP server, using Tree-sitter parsers for syntax-aware code analysis.
Particularly useful for large codebases that require function-level granular indexing.

Key Features:
- 🎯 Intelligent Code Chunking: Function, class, and method-level granular indexing
- 🌐 Multi-Language Support: Python, JavaScript, TypeScript, Go, Rust, Java, and more
- 🛡️ Syntax Error Tolerance: Graceful handling with detailed error reporting
- ⚡ Performance Optimized: Parallel processing with memory monitoring
- 📊 Comprehensive Reporting: Detailed syntax error statistics and recommendations

Usage Examples:
    # Full indexing with intelligent chunking (recommended for first-time indexing)
    python manual_indexing.py -d /path/to/repo -m clear_existing

    # Incremental indexing (only process changed files)
    python manual_indexing.py -d /path/to/repo -m incremental

    # Verbose output with detailed syntax error reporting
    python manual_indexing.py -d ./src --mode incremental --verbose

    # Skip confirmation prompts for automated workflows
    python manual_indexing.py -d /large/codebase -m clear_existing --no-confirm

    # Custom error report directory
    python manual_indexing.py -d ./src -m clear_existing --error-report-dir ./reports
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.file_discovery_service import FileDiscoveryService
from services.indexing_pipeline import IndexingPipeline
from services.indexing_reporter import IndexingReporter
from utils import format_duration, format_memory_size


class ManualIndexingTool:
    """
    Manual indexing tool coordinator for standalone indexing operations.

    This tool coordinates the indexing pipeline, file discovery, and reporting
    services to provide comprehensive indexing capabilities outside the MCP server.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the manual indexing tool.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.setup_logging()

        # Initialize core services
        self.file_discovery = FileDiscoveryService()
        self.pipeline = IndexingPipeline()
        self.reporter = IndexingReporter()

        # Set up service callbacks
        self.pipeline.set_error_callback(self._handle_pipeline_error)
        self.pipeline.set_progress_callback(self._handle_progress)

        # Configuration
        self.error_report_dir = None

        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO

        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Reduce noise from third-party libraries
        if not self.verbose:
            logging.getLogger("qdrant_client").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)

    def validate_arguments(self, directory: str, mode: str) -> tuple[bool, str]:
        """
        Validate command-line arguments.

        Args:
            directory: Target directory path
            mode: Indexing mode

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate mode
        valid_modes = ["clear_existing", "incremental"]
        if mode not in valid_modes:
            return (
                False,
                f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}",
            )

        # Validate directory using file discovery service
        return self.file_discovery.validate_directory(directory)

    def check_dependencies(self) -> tuple[bool, list[str]]:
        """
        Check if required dependencies are available.

        Returns:
            Tuple of (all_available, missing_services)
        """
        missing = []

        # Check services through pipeline
        try:
            # Test Qdrant connection
            collections = self.pipeline.qdrant_service.client.get_collections()
            self.logger.debug(f"Qdrant connection successful, found {len(collections.collections)} collections")
        except Exception as e:
            missing.append(f"Qdrant (Error: {e})")

        # Check embedding service
        try:
            if not hasattr(self.pipeline.embedding_service, "generate_embeddings"):
                missing.append("Embedding service not properly initialized")
        except Exception as e:
            missing.append(f"Embedding service (Error: {e})")

        return len(missing) == 0, missing

    def show_pre_indexing_summary(self, directory: str, mode: str):
        """
        Show summary before starting indexing.

        Args:
            directory: Target directory
            mode: Indexing mode
        """
        print("\n" + "=" * 60)
        print("MANUAL INDEXING TOOL - PRE-INDEXING SUMMARY")
        print("=" * 60)

        print(f"📁 Directory: {directory}")
        print(f"⚙️  Mode: {mode}")

        # Get processing estimates
        estimates = self.file_discovery.estimate_processing_requirements(directory, mode)

        file_count = estimates.get("file_count", 0)
        size_mb = estimates.get("total_size_mb", 0)
        estimated_minutes = estimates.get("estimated_minutes", 0)

        if file_count > 0:
            if mode == "incremental":
                print(f"📊 Files to process: {file_count:,} (changed files only)")
            else:
                print(f"📊 Files to process: {file_count:,}")
            print(f"💾 Total size: {size_mb:.1f} MB")
            print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes")

            if estimated_minutes > 5:
                print("\n⚠️  WARNING: This operation may take several minutes.")
                print("   Consider running this in a separate terminal.")
        elif mode == "incremental":
            print("✅ No changes detected - all files are up to date!")
            print("⏱️  Estimated time: 0 minutes (no processing needed)")
        else:
            print("⚠️  No files found to process")

        print("\n" + "-" * 60)

    async def perform_indexing(self, directory: str, mode: str) -> bool:
        """
        Perform the actual indexing operation.

        Args:
            directory: Target directory
            mode: Indexing mode ('clear_existing' or 'incremental')

        Returns:
            True if indexing was successful
        """
        import time

        start_time = time.time()

        try:
            # Discover project information
            discovery_result = self.file_discovery.discover_project_files(directory)

            if "error" in discovery_result:
                print(f"\n❌ File discovery failed: {discovery_result['error']}")
                return False

            project_context = discovery_result["project_context"]
            project_name = project_context.get("project_name", "unknown")

            # Start error reporting
            self.reporter.start_report(mode, directory, project_name)

            print(f"\n🚀 Starting {mode} indexing for project: {project_name}")

            # Execute appropriate indexing pipeline
            if mode == "clear_existing":
                result = self.pipeline.execute_full_indexing(directory, project_name, clear_existing=True)
            elif mode == "incremental":
                result = self.pipeline.execute_incremental_indexing(directory, project_name)
            else:
                self.logger.error(f"Unknown indexing mode: {mode}")
                return False

            # Finalize reporting
            duration = time.time() - start_time

            # Update file counts for reporting - use actual counts based on embedding success
            self.reporter.update_file_counts(
                processed=result.total_files_processed,
                successful=result.total_files_processed,  # Files are processed, chunks may fail
            )

            # Finalize report with embedding failure information
            self.reporter.finalize_report(
                total_chunks=result.total_chunks_generated,
                total_points=result.total_points_stored,
                collections_used=result.collections_used,
                performance_metrics=result.performance_metrics,
                embedding_failures=result.embedding_failures,
                embedding_success_rate=result.embedding_success_rate,
                failed_chunks=result.failed_chunks,
            )

            # Extract any syntax errors from the indexing service
            self.reporter.extract_syntax_errors_from_indexing_service(self.pipeline.indexing_service)

            # Print summary
            self.reporter.print_summary()

            # Save error report if there were issues or verbose mode
            has_issues = self.reporter.current_report and (
                self.reporter.current_report.get_error_summary() or self.reporter.current_report.embedding_failures > 0 or self.verbose
            )
            if has_issues:
                try:
                    report_path = self.reporter.save_report(self.error_report_dir)
                    print(f"\n📋 Report saved to: {report_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save report: {e}")

            # Print final status
            if result.success:
                if result.embedding_failures > 0:
                    print(f"\n⚠️  Indexing completed with {result.embedding_failures} embedding failures")
                    print(f"   Embedding success rate: {result.embedding_success_rate:.1f}%")
                else:
                    print("\n✅ Indexing completed successfully!")
                print(f"⏱️  Total time: {format_duration(duration)}")
                if result.performance_metrics:
                    memory_mb = result.performance_metrics.get("memory_usage_mb", 0)
                    print(f"💾 Memory usage: {format_memory_size(memory_mb)}")
            else:
                print("\n❌ Indexing failed!")
                print(f"⏱️  Time elapsed: {format_duration(duration)}")
                if self.reporter.current_report and self.reporter.current_report.has_critical_errors():
                    print("🚨 Critical errors prevented successful completion")

            return result.success

        except Exception as e:
            self.logger.error(f"Error during indexing: {e}", exc_info=True)
            print(f"\n❌ Indexing failed with error: {e}")
            return False

    def _handle_pipeline_error(self, error_type: str, location: str, message: str, suggestion: str = ""):
        """Handle errors reported by the pipeline."""
        self.reporter.add_error(
            error_type=error_type,
            file_path=location,
            error_message=message,
            suggestion=suggestion,
        )

    def _handle_progress(self, message: str):
        """Handle progress updates from the pipeline."""
        if self.verbose:
            print(f"🔄 {message}")


def main():
    """Main entry point for the manual indexing tool."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manual Indexing Tool with Intelligent Code Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Intelligent Code Chunking Examples:

Basic Usage:
  %(prog)s -d /path/to/repo -m clear_existing
                        Full indexing with intelligent chunking (first-time setup)
  %(prog)s -d ./src -m incremental
                        Process only changed files (fast updates)

Advanced Usage:
  %(prog)s -d /large/codebase -m clear_existing --verbose
                        Verbose output with detailed syntax error reporting
  %(prog)s -d ./project -m incremental --no-confirm
                        Skip prompts for automated CI/CD workflows
  %(prog)s -d ./src -m clear_existing --error-report-dir ./reports
                        Save detailed error reports to custom directory
        """,
    )

    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Target directory path to index with intelligent code chunking",
    )

    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["clear_existing", "incremental"],
        help="Indexing mode: clear_existing (full reindex) or incremental (only changed files)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging with detailed error reporting and performance metrics",
    )

    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip confirmation prompts - useful for automated CI/CD workflows",
    )

    parser.add_argument(
        "--error-report-dir",
        help="Directory to save comprehensive error reports (default: current directory)",
    )

    args = parser.parse_args()

    # Initialize tool
    tool = ManualIndexingTool(verbose=args.verbose)

    # Set error report directory if provided
    if args.error_report_dir:
        tool.error_report_dir = args.error_report_dir

    # Validate arguments
    is_valid, error_msg = tool.validate_arguments(args.directory, args.mode)
    if not is_valid:
        print(f"❌ Error: {error_msg}")
        sys.exit(1)

    # Check dependencies
    deps_ok, missing = tool.check_dependencies()
    if not deps_ok:
        print("❌ Missing dependencies:")
        for service in missing:
            print(f"   - {service}")
        print("\nPlease ensure Qdrant is running and services are properly configured.")
        sys.exit(1)

    # Show pre-indexing summary
    tool.show_pre_indexing_summary(args.directory, args.mode)

    # Confirmation prompt (unless --no-confirm)
    if not args.no_confirm:
        response = input("\nProceed with indexing? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Indexing cancelled.")
            sys.exit(0)

    # Run indexing
    try:
        success = asyncio.run(tool.perform_indexing(args.directory, args.mode))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
