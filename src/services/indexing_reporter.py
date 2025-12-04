"""
Indexing Reporter Service for comprehensive error reporting and analytics.

This service provides detailed error tracking, performance reporting, and
actionable recommendations for indexing operations.
"""

# ruff: noqa: T201

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class IndexingError:
    """Represents an error that occurred during indexing."""

    error_type: str  # Type of error (syntax, processing, embedding, storage, etc.)
    file_path: str  # File where error occurred
    line_number: int | None = None  # Line number if applicable
    error_message: str = ""  # Detailed error message
    severity: str = "error"  # Severity: error, warning, info
    context: str = ""  # Additional context or code snippet
    timestamp: str = ""  # When the error occurred
    suggestion: str = ""  # Suggested fix or workaround

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class IndexingReport:
    """Comprehensive report for indexing operations."""

    operation_type: str  # Type of operation (full_indexing, incremental, etc.)
    directory: str  # Directory being processed
    project_name: str  # Project name
    start_time: str  # Operation start time
    end_time: str = ""  # Operation end time
    total_files: int = 0  # Total files processed
    successful_files: int = 0  # Successfully processed files
    failed_files: int = 0  # Files that failed to process
    total_chunks: int = 0  # Total chunks generated
    total_points: int = 0  # Total points stored
    collections_used: list[str] = field(default_factory=list)  # Collections used
    errors: list[IndexingError] = field(default_factory=list)  # List of all errors
    warnings: list[IndexingError] = field(default_factory=list)  # List of warnings
    syntax_errors: list[IndexingError] = field(default_factory=list)  # Syntax-specific errors
    performance_metrics: dict[str, Any] = field(default_factory=dict)  # Performance data
    recommendations: list[str] = field(default_factory=list)  # Actionable recommendations
    # Embedding failure tracking
    embedding_failures: int = 0  # Number of chunks that failed embedding
    embedding_success_rate: float = 100.0  # Embedding success rate percentage
    failed_chunks: list[dict[str, Any]] = field(default_factory=list)  # List of failed chunk details

    def add_error(self, error: IndexingError):
        """Add an error to the appropriate category."""
        if error.severity == "warning":
            self.warnings.append(error)
        elif error.error_type == "syntax":
            self.syntax_errors.append(error)
        else:
            self.errors.append(error)

    def get_error_summary(self) -> dict[str, int]:
        """Get a summary of error counts by type."""
        error_types = {}
        for error in self.errors + self.warnings + self.syntax_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        return error_types

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors that prevent successful completion."""
        critical_types = ["embedding", "storage", "qdrant_connection", "pipeline"]
        return any(error.error_type in critical_types for error in self.errors)

    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_files == 0:
            return 100.0
        return (self.successful_files / self.total_files) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "operation_type": self.operation_type,
            "directory": self.directory,
            "project_name": self.project_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "total_chunks": self.total_chunks,
            "total_points": self.total_points,
            "collections_used": self.collections_used,
            "success_rate": self.get_success_rate(),
            "error_summary": self.get_error_summary(),
            "has_critical_errors": self.has_critical_errors(),
            "errors": [asdict(error) for error in self.errors],
            "warnings": [asdict(error) for error in self.warnings],
            "syntax_errors": [asdict(error) for error in self.syntax_errors],
            "performance_metrics": self.performance_metrics,
            "recommendations": self.recommendations,
            # Embedding failure details
            "embedding_failures": self.embedding_failures,
            "embedding_success_rate": self.embedding_success_rate,
            "failed_chunks": self.failed_chunks,
        }


class IndexingReporter:
    """
    Service for comprehensive indexing error reporting and analytics.

    This service tracks errors, generates reports, and provides actionable
    recommendations for improving indexing operations.
    """

    def __init__(self):
        """Initialize the indexing reporter."""
        self.logger = logger
        self.current_report: IndexingReport | None = None
        self.processed_files_count = 0
        self.successful_files_count = 0

    def start_report(self, operation_type: str, directory: str, project_name: str) -> IndexingReport:
        """
        Start a new indexing report.

        Args:
            operation_type: Type of indexing operation
            directory: Directory being indexed
            project_name: Name of the project

        Returns:
            New IndexingReport instance
        """
        self.current_report = IndexingReport(
            operation_type=operation_type,
            directory=directory,
            project_name=project_name,
            start_time=datetime.now().isoformat(),
        )

        self.processed_files_count = 0
        self.successful_files_count = 0

        return self.current_report

    def add_error(
        self,
        error_type: str,
        file_path: str,
        error_message: str,
        line_number: int | None = None,
        severity: str = "error",
        context: str = "",
        suggestion: str = "",
    ):
        """Add an error to the current report."""
        if not self.current_report:
            self.logger.warning("No active report - creating a default one")
            self.current_report = IndexingReport(
                operation_type="unknown",
                directory="unknown",
                project_name="unknown",
                start_time=datetime.now().isoformat(),
            )

        error = IndexingError(
            error_type=error_type,
            file_path=file_path,
            line_number=line_number,
            error_message=error_message,
            severity=severity,
            context=context,
            suggestion=suggestion,
        )

        self.current_report.add_error(error)

        # Log the error as well
        log_method = self.logger.warning if severity == "warning" else self.logger.error
        log_method(f"{error_type.upper()} in {file_path}: {error_message}")

    def update_file_counts(self, processed: int, successful: int):
        """Update file processing counts."""
        self.processed_files_count = processed
        self.successful_files_count = successful

    def finalize_report(
        self,
        total_chunks: int = 0,
        total_points: int = 0,
        collections_used: list[str] | None = None,
        performance_metrics: dict[str, Any] | None = None,
        embedding_failures: int = 0,
        embedding_success_rate: float = 100.0,
        failed_chunks: list[dict[str, Any]] | None = None,
    ) -> IndexingReport:
        """
        Finalize the current report with final statistics.

        Args:
            total_chunks: Total chunks generated
            total_points: Total points stored in vector database
            collections_used: List of collections used
            performance_metrics: Performance metrics dictionary
            embedding_failures: Number of chunks that failed embedding
            embedding_success_rate: Embedding success rate percentage
            failed_chunks: List of failed chunk details

        Returns:
            Finalized IndexingReport
        """
        if not self.current_report:
            raise ValueError("No active report to finalize")

        # Update final statistics
        self.current_report.end_time = datetime.now().isoformat()
        self.current_report.total_files = self.processed_files_count
        self.current_report.successful_files = self.successful_files_count
        self.current_report.failed_files = self.processed_files_count - self.successful_files_count
        self.current_report.total_chunks = total_chunks
        self.current_report.total_points = total_points
        self.current_report.collections_used = collections_used or []
        self.current_report.performance_metrics = performance_metrics or {}

        # Update embedding failure statistics
        self.current_report.embedding_failures = embedding_failures
        self.current_report.embedding_success_rate = embedding_success_rate
        self.current_report.failed_chunks = failed_chunks or []

        # Generate recommendations
        self._generate_recommendations()

        return self.current_report

    def save_report(self, output_dir: str | None = None) -> str:
        """
        Save the current report to a JSON file.

        Args:
            output_dir: Directory to save the report (defaults to current directory)

        Returns:
            Path to the saved report file
        """
        if not self.current_report:
            raise ValueError("No report to save")

        # Create output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = self.current_report.project_name.replace(" ", "_").replace("/", "_")
        filename = f"indexing_report_{project_name}_{timestamp}.json"
        report_path = output_dir / filename

        # Save report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.current_report.to_dict(), f, indent=2, ensure_ascii=False)

        return str(report_path)

    def print_summary(self):
        """Print a summary of the current report to console."""
        if not self.current_report:
            self.logger.warning("No report available to summarize")
            return

        error_summary = self.current_report.get_error_summary()
        total_errors = sum(error_summary.values())

        print("\n📊 INDEXING SUMMARY")
        print("=" * 50)
        print(f"Project: {self.current_report.project_name}")
        print(f"Operation: {self.current_report.operation_type}")
        print(f"Files processed: {self.current_report.successful_files}/{self.current_report.total_files}")
        print(f"Success rate: {self.current_report.get_success_rate():.1f}%")
        print(f"Chunks generated: {self.current_report.total_chunks:,}")
        print(f"Points stored: {self.current_report.total_points:,}")
        print(f"Collections used: {len(self.current_report.collections_used)}")

        # Show embedding statistics
        if self.current_report.embedding_failures > 0:
            print(f"\n⚠️  EMBEDDING FAILURES: {self.current_report.embedding_failures} chunks")
            print(f"   Embedding success rate: {self.current_report.embedding_success_rate:.1f}%")

            # Group failed chunks by file for summary
            if self.current_report.failed_chunks:
                failures_by_file: dict[str, list[dict]] = {}
                for chunk in self.current_report.failed_chunks:
                    file_path = chunk.get("file_path", "unknown")
                    if file_path not in failures_by_file:
                        failures_by_file[file_path] = []
                    failures_by_file[file_path].append(chunk)

                print(f"   Failed chunks by file ({len(failures_by_file)} files affected):")
                for file_path, chunks in list(failures_by_file.items())[:5]:  # Show first 5 files
                    print(f"      📁 {file_path}: {len(chunks)} chunks")
                if len(failures_by_file) > 5:
                    print(f"      ... and {len(failures_by_file) - 5} more files")

        if total_errors == 0 and self.current_report.embedding_failures == 0:
            print("✅ No errors or warnings encountered!")
        elif total_errors > 0:
            print(f"\n📋 ISSUES SUMMARY ({total_errors} total)")
            print("-" * 30)

            for error_type, count in sorted(error_summary.items()):
                severity_icon = (
                    "⚠️"
                    if any(
                        e.severity == "warning"
                        for e in (self.current_report.errors + self.current_report.warnings + self.current_report.syntax_errors)
                        if e.error_type == error_type
                    )
                    else "❌"
                )
                print(f"{severity_icon} {error_type}: {count}")

            # Show critical errors
            if self.current_report.has_critical_errors():
                print("\n🚨 CRITICAL ERRORS DETECTED")

        # Show top recommendations
        if self.current_report.recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(self.current_report.recommendations[:3], 1):
                print(f"   {i}. {rec}")

        print("=" * 50)

    def extract_syntax_errors_from_indexing_service(self, indexing_service):
        """Extract syntax errors from indexing service parse results."""
        try:
            if hasattr(indexing_service, "_parse_results"):
                for parse_result in indexing_service._parse_results:
                    if parse_result.syntax_errors:
                        for syntax_error in parse_result.syntax_errors:
                            self.add_error(
                                error_type="syntax",
                                file_path=parse_result.file_path,
                                line_number=syntax_error.start_line,
                                error_message=f"{syntax_error.error_type}: {syntax_error.context}",
                                severity="warning" if syntax_error.severity == "warning" else "error",
                                context=syntax_error.context,
                                suggestion="Fix syntax errors using a linter or IDE",
                            )

                    # Report fallback usage
                    if parse_result.fallback_used:
                        self.add_error(
                            error_type="parsing",
                            file_path=parse_result.file_path,
                            error_message="Intelligent parsing failed, used fallback to whole-file chunking",
                            severity="warning",
                            suggestion="Check file syntax and Tree-sitter parser support",
                        )
        except Exception as e:
            self.logger.debug(f"Could not extract syntax errors: {e}")

    def _generate_recommendations(self):
        """Generate actionable recommendations based on collected errors."""
        if not self.current_report:
            return

        recommendations = []
        error_summary = self.current_report.get_error_summary()

        # Syntax error recommendations
        if error_summary.get("syntax", 0) > 0:
            recommendations.append(
                "Fix syntax errors in your source files before indexing. "
                "Consider using a linter or IDE to identify and resolve syntax issues."
            )

        # Embedding error recommendations
        if error_summary.get("embedding", 0) > 0:
            recommendations.append(
                "Embedding generation failed for some files. "
                "Check if Ollama is running and the model is available. "
                "Try: ollama pull nomic-embed-text"
            )

        # Storage error recommendations
        if error_summary.get("storage", 0) > 0:
            recommendations.append(
                "Database storage errors occurred. Check Qdrant connection and "
                "disk space. Ensure Qdrant container is running: "
                "docker run -p 6333:6333 qdrant/qdrant"
            )

        # Processing error recommendations
        if error_summary.get("processing", 0) > 0:
            recommendations.append(
                "Some files failed to process. Check file permissions and encoding. " "Large or binary files may cause processing issues."
            )

        # Performance recommendations
        success_rate = self.current_report.get_success_rate()
        if success_rate < 80:
            recommendations.append(
                "Low success rate detected. Consider reducing batch sizes, "
                "checking system resources (memory, disk space), or reviewing file quality."
            )

        # Pipeline error recommendations
        if error_summary.get("pipeline", 0) > 0:
            recommendations.append(
                "Pipeline errors occurred. Check service dependencies and configuration. "
                "Ensure all required services (Qdrant, Ollama) are running and accessible."
            )

        self.current_report.recommendations = recommendations
