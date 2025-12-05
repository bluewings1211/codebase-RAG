"""
Search diagnostics utility module for debugging and monitoring search operations.

This module provides comprehensive diagnostic tools for analyzing search performance,
content quality, vector database consistency, and identifying potential issues.
"""

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)


@dataclass
class SearchDiagnosticResult:
    """Result of a search diagnostic operation."""

    status: str  # 'healthy', 'warning', 'error'
    message: str
    details: dict[str, Any]
    timestamp: float
    recommendations: list[str]


class SearchDiagnostics:
    """Comprehensive search diagnostics and monitoring utilities."""

    def __init__(self, qdrant_client: QdrantClient | None = None):
        """Initialize search diagnostics.

        Args:
            qdrant_client: Optional Qdrant client instance
        """
        self.logger = logging.getLogger(__name__)
        self.qdrant_client = qdrant_client

        if not self.qdrant_client:
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            self.qdrant_client = QdrantClient(host=host, port=port)

    def run_comprehensive_diagnostics(self, project_name: str | None = None) -> dict[str, SearchDiagnosticResult]:
        """Run all diagnostic tests and return comprehensive results.

        Args:
            project_name: Optional project name to focus diagnostics on

        Returns:
            Dictionary of diagnostic results by test name
        """
        results = {}

        self.logger.info("Starting comprehensive search diagnostics")

        # Test 1: Vector database connectivity
        results["connectivity"] = self.test_vector_db_connectivity()

        # Test 2: Collection health
        results["collections"] = self.test_collection_health(project_name)

        # Test 3: Content quality analysis
        results["content_quality"] = self.analyze_content_quality(project_name)

        # Test 4: Search performance analysis
        results["search_performance"] = self.analyze_search_performance(project_name)

        # Test 5: Vector database consistency
        results["db_consistency"] = self.check_vector_db_consistency(project_name)

        # Test 6: Empty content detection
        results["empty_content"] = self.detect_empty_content_issues(project_name)

        self.logger.info("Comprehensive diagnostics completed")

        return results

    def test_vector_db_connectivity(self) -> SearchDiagnosticResult:
        """Test vector database connectivity and basic operations."""
        try:
            start_time = time.time()

            # Test basic connectivity
            collections = self.qdrant_client.get_collections()

            # Test health endpoint if available
            health_info = {}
            try:
                health_info = {"collections_count": len(collections.collections)}
            except Exception as e:
                self.logger.debug(f"Could not get health info: {e}")

            response_time = time.time() - start_time

            if response_time > 2.0:
                status = "warning"
                message = f"Vector database responding slowly ({response_time:.2f}s)"
                recommendations = ["Check network connectivity", "Consider Qdrant server performance"]
            else:
                status = "healthy"
                message = f"Vector database connectivity is healthy ({response_time:.3f}s)"
                recommendations = []

            return SearchDiagnosticResult(
                status=status,
                message=message,
                details={
                    "response_time_seconds": response_time,
                    "collections_found": len(collections.collections),
                    "health_info": health_info,
                },
                timestamp=time.time(),
                recommendations=recommendations,
            )

        except Exception as e:
            return SearchDiagnosticResult(
                status="error",
                message=f"Vector database connectivity failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                recommendations=[
                    "Check if Qdrant server is running",
                    "Verify connection parameters (host, port)",
                    "Check network connectivity",
                ],
            )

    def test_collection_health(self, project_name: str | None = None) -> SearchDiagnosticResult:
        """Test health of collections in the vector database."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_details = []
            total_points = 0
            empty_collections = []

            for collection in collections.collections:
                if project_name and not collection.name.startswith(f"project_{project_name}"):
                    continue

                try:
                    info = self.qdrant_client.get_collection(collection.name)
                    points_count = info.points_count or 0
                    total_points += points_count

                    collection_detail = {
                        "name": collection.name,
                        "points_count": points_count,
                        "status": "healthy" if points_count > 0 else "empty",
                    }

                    if points_count == 0:
                        empty_collections.append(collection.name)

                    collection_details.append(collection_detail)

                except Exception as e:
                    collection_details.append({"name": collection.name, "points_count": 0, "status": "error", "error": str(e)})

            # Determine overall status
            if not collection_details:
                status = "warning"
                message = "No collections found"
                recommendations = ["Index some content first using index_directory"]
            elif empty_collections:
                status = "warning"
                message = f"Found {len(empty_collections)} empty collections out of {len(collection_details)}"
                recommendations = ["Consider reindexing empty collections", "Check if content was properly processed during indexing"]
            else:
                status = "healthy"
                message = f"All {len(collection_details)} collections are healthy"
                recommendations = []

            return SearchDiagnosticResult(
                status=status,
                message=message,
                details={
                    "total_collections": len(collection_details),
                    "total_points": total_points,
                    "empty_collections": empty_collections,
                    "collection_details": collection_details,
                },
                timestamp=time.time(),
                recommendations=recommendations,
            )

        except Exception as e:
            return SearchDiagnosticResult(
                status="error",
                message=f"Collection health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                recommendations=["Check vector database connectivity"],
            )

    def analyze_content_quality(self, project_name: str | None = None) -> SearchDiagnosticResult:
        """Analyze quality of indexed content."""
        try:
            collections = self.qdrant_client.get_collections()
            content_stats = {
                "total_chunks": 0,
                "empty_content_chunks": 0,
                "short_content_chunks": 0,
                "average_content_length": 0,
                "by_collection": {},
            }

            total_content_length = 0

            for collection in collections.collections:
                if collection.name.endswith("_file_metadata"):
                    continue

                if project_name and not collection.name.startswith(f"project_{project_name}"):
                    continue

                try:
                    # Sample some points to analyze content quality
                    sample_points = self.qdrant_client.scroll(
                        collection_name=collection.name, limit=100, with_payload=True, with_vectors=False
                    )[0]

                    collection_stats = {"total": len(sample_points), "empty": 0, "short": 0, "avg_length": 0}

                    collection_content_length = 0

                    for point in sample_points:
                        content = point.payload.get("content", "") if point.payload else ""
                        content_length = len(content.strip())

                        content_stats["total_chunks"] += 1
                        total_content_length += content_length
                        collection_content_length += content_length

                        if not content.strip():
                            content_stats["empty_content_chunks"] += 1
                            collection_stats["empty"] += 1
                        elif content_length < 10:
                            content_stats["short_content_chunks"] += 1
                            collection_stats["short"] += 1

                    if collection_stats["total"] > 0:
                        collection_stats["avg_length"] = collection_content_length / collection_stats["total"]

                    content_stats["by_collection"][collection.name] = collection_stats

                except Exception as e:
                    self.logger.debug(f"Error analyzing collection {collection.name}: {e}")
                    content_stats["by_collection"][collection.name] = {"error": str(e)}

            # Calculate overall statistics
            if content_stats["total_chunks"] > 0:
                content_stats["average_content_length"] = total_content_length / content_stats["total_chunks"]
                empty_rate = (content_stats["empty_content_chunks"] / content_stats["total_chunks"]) * 100
                short_rate = (content_stats["short_content_chunks"] / content_stats["total_chunks"]) * 100
            else:
                empty_rate = 0
                short_rate = 0

            # Determine status and recommendations
            recommendations = []
            if empty_rate > 10:
                status = "warning"
                recommendations.append("High rate of empty content chunks detected")
                recommendations.append("Consider reindexing with improved content extraction")
            elif empty_rate > 5:
                status = "warning"
                recommendations.append("Some empty content chunks detected")
            else:
                status = "healthy"

            if short_rate > 20:
                recommendations.append("High rate of very short content chunks")
                recommendations.append("Review chunking strategy for better content extraction")

            message = f"Content quality: {empty_rate:.1f}% empty, {short_rate:.1f}% short chunks"

            return SearchDiagnosticResult(
                status=status,
                message=message,
                details={**content_stats, "empty_content_rate": empty_rate, "short_content_rate": short_rate},
                timestamp=time.time(),
                recommendations=recommendations,
            )

        except Exception as e:
            return SearchDiagnosticResult(
                status="error",
                message=f"Content quality analysis failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                recommendations=["Check vector database connectivity"],
            )

    def analyze_search_performance(self, project_name: str | None = None) -> SearchDiagnosticResult:
        """Analyze search performance with sample queries."""
        try:
            collections = self.qdrant_client.get_collections()
            target_collections = [
                c.name
                for c in collections.collections
                if not c.name.endswith("_file_metadata") and (not project_name or c.name.startswith(f"project_{project_name}"))
            ]

            if not target_collections:
                return SearchDiagnosticResult(
                    status="warning",
                    message="No collections available for performance testing",
                    details={},
                    timestamp=time.time(),
                    recommendations=["Index some content first"],
                )

            # Test search performance with dummy queries
            performance_stats = []
            total_time = 0

            for collection_name in target_collections[:3]:  # Test up to 3 collections
                try:
                    # Get vector dimension from collection config
                    collection_info = self.qdrant_client.get_collection(collection_name)
                    vector_size = 768  # Default fallback
                    if hasattr(collection_info.config, "params") and hasattr(collection_info.config.params, "vectors"):
                        vectors_config = collection_info.config.params.vectors
                        if hasattr(vectors_config, "size"):
                            vector_size = vectors_config.size
                        elif isinstance(vectors_config, dict) and "" in vectors_config:
                            vector_size = vectors_config[""].size

                    # Create dummy vector with correct dimension for this collection
                    dummy_vector = [0.1] * vector_size

                    start_time = time.time()

                    # Perform a simple vector search
                    results = self.qdrant_client.search(
                        collection_name=collection_name, query_vector=dummy_vector, limit=5, score_threshold=0.0
                    )

                    search_time = time.time() - start_time
                    total_time += search_time

                    performance_stats.append(
                        {
                            "collection": collection_name,
                            "search_time_ms": search_time * 1000,
                            "results_returned": len(results),
                            "vector_dimension": vector_size,
                            "status": "success",
                        }
                    )

                except Exception as e:
                    performance_stats.append(
                        {"collection": collection_name, "search_time_ms": 0, "results_returned": 0, "status": "error", "error": str(e)}
                    )

            # Calculate average performance
            successful_tests = [s for s in performance_stats if s["status"] == "success"]
            if successful_tests:
                avg_time_ms = (total_time / len(successful_tests)) * 1000

                if avg_time_ms > 1000:
                    status = "warning"
                    message = f"Search performance is slow (avg: {avg_time_ms:.0f}ms)"
                    recommendations = ["Consider optimizing vector database configuration", "Check if collections are properly indexed"]
                elif avg_time_ms > 500:
                    status = "warning"
                    message = f"Search performance is acceptable (avg: {avg_time_ms:.0f}ms)"
                    recommendations = ["Monitor search performance over time"]
                else:
                    status = "healthy"
                    message = f"Search performance is good (avg: {avg_time_ms:.0f}ms)"
                    recommendations = []
            else:
                status = "error"
                message = "All search performance tests failed"
                recommendations = ["Check vector database and collection health"]
                avg_time_ms = 0

            return SearchDiagnosticResult(
                status=status,
                message=message,
                details={
                    "average_search_time_ms": avg_time_ms,
                    "total_collections_tested": len(performance_stats),
                    "successful_tests": len(successful_tests),
                    "performance_details": performance_stats,
                },
                timestamp=time.time(),
                recommendations=recommendations,
            )

        except Exception as e:
            return SearchDiagnosticResult(
                status="error",
                message=f"Search performance analysis failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                recommendations=["Check vector database connectivity"],
            )

    def check_vector_db_consistency(self, project_name: str | None = None) -> SearchDiagnosticResult:
        """Check vector database consistency and detect potential issues."""
        try:
            collections = self.qdrant_client.get_collections()
            consistency_issues = []

            content_collections = []
            metadata_collections = []

            for collection in collections.collections:
                if project_name and not collection.name.startswith(f"project_{project_name}"):
                    continue

                if collection.name.endswith("_file_metadata"):
                    metadata_collections.append(collection.name)
                else:
                    content_collections.append(collection.name)

            # Check for orphaned metadata collections
            for metadata_coll in metadata_collections:
                expected_prefix = metadata_coll.replace("_file_metadata", "")
                matching_content_colls = [c for c in content_collections if c.startswith(expected_prefix)]

                if not matching_content_colls:
                    consistency_issues.append(
                        {
                            "type": "orphaned_metadata",
                            "collection": metadata_coll,
                            "description": "Metadata collection without corresponding content collections",
                        }
                    )

            # Check for missing metadata collections
            for content_coll in content_collections:
                # Extract project prefix
                if content_coll.startswith("project_"):
                    project_prefix = "_".join(content_coll.split("_")[:2])  # e.g., "project_myproject"
                    expected_metadata = f"{project_prefix}_file_metadata"

                    if expected_metadata not in [c.name for c in collections.collections]:
                        consistency_issues.append(
                            {
                                "type": "missing_metadata",
                                "collection": content_coll,
                                "description": f"Content collection missing metadata collection: {expected_metadata}",
                            }
                        )

            # Check for vector dimension consistency
            dimension_issues = []
            for collection_name in content_collections[:5]:  # Check first 5 collections
                try:
                    collection_info = self.qdrant_client.get_collection(collection_name)
                    vector_size = collection_info.config.params.vectors.size

                    # Check if vector size matches expected dimensions
                    if vector_size not in [384, 768, 1024, 1536]:  # Common embedding dimensions
                        dimension_issues.append(
                            {
                                "collection": collection_name,
                                "vector_size": vector_size,
                                "description": f"Unusual vector dimension: {vector_size}",
                            }
                        )

                except Exception as e:
                    dimension_issues.append(
                        {"collection": collection_name, "vector_size": "unknown", "description": f"Could not check vector dimensions: {e}"}
                    )

            # Determine overall status
            total_issues = len(consistency_issues) + len(dimension_issues)

            if total_issues == 0:
                status = "healthy"
                message = "Vector database consistency is good"
                recommendations = []
            elif total_issues <= 2:
                status = "warning"
                message = f"Found {total_issues} minor consistency issues"
                recommendations = ["Review and clean up orphaned collections if needed"]
            else:
                status = "warning"
                message = f"Found {total_issues} consistency issues"
                recommendations = ["Consider reindexing to fix consistency issues", "Review collection management practices"]

            return SearchDiagnosticResult(
                status=status,
                message=message,
                details={
                    "total_collections": len(collections.collections),
                    "content_collections": len(content_collections),
                    "metadata_collections": len(metadata_collections),
                    "consistency_issues": consistency_issues,
                    "dimension_issues": dimension_issues,
                    "total_issues": total_issues,
                },
                timestamp=time.time(),
                recommendations=recommendations,
            )

        except Exception as e:
            return SearchDiagnosticResult(
                status="error",
                message=f"Vector database consistency check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                recommendations=["Check vector database connectivity"],
            )

    def detect_empty_content_issues(self, project_name: str | None = None) -> SearchDiagnosticResult:
        """Detect and analyze empty content issues in indexed data."""
        try:
            collections = self.qdrant_client.get_collections()
            empty_content_analysis = {
                "collections_analyzed": 0,
                "total_points_checked": 0,
                "empty_content_points": [],
                "problematic_files": defaultdict(int),
                "by_collection": {},
            }

            for collection in collections.collections:
                if collection.name.endswith("_file_metadata"):
                    continue

                if project_name and not collection.name.startswith(f"project_{project_name}"):
                    continue

                try:
                    # Sample points to check for empty content
                    points, _ = self.qdrant_client.scroll(
                        collection_name=collection.name,
                        limit=200,  # Check more points for better analysis
                        with_payload=True,
                        with_vectors=False,
                    )

                    collection_stats = {"total_points": len(points), "empty_content": 0, "empty_points": []}

                    empty_content_analysis["collections_analyzed"] += 1
                    empty_content_analysis["total_points_checked"] += len(points)

                    for point in points:
                        if not point.payload:
                            continue

                        content = point.payload.get("content", "")
                        file_path = point.payload.get("file_path", "unknown")
                        chunk_type = point.payload.get("chunk_type", "unknown")
                        chunk_name = point.payload.get("name", "unknown")

                        # Check for various types of empty/problematic content
                        is_empty = False
                        issue_type = None

                        if not content:
                            is_empty = True
                            issue_type = "no_content"
                        elif not content.strip():
                            is_empty = True
                            issue_type = "whitespace_only"
                        elif len(content.strip()) < 3:
                            is_empty = True
                            issue_type = "too_short"
                        elif content.strip() in ["{}", "[]", "()", "null", "undefined", "None"]:
                            is_empty = True
                            issue_type = "placeholder_content"

                        if is_empty:
                            collection_stats["empty_content"] += 1
                            empty_point_info = {
                                "point_id": str(point.id),
                                "file_path": file_path,
                                "chunk_type": chunk_type,
                                "chunk_name": chunk_name,
                                "issue_type": issue_type,
                                "content_preview": content[:50] if content else "",
                            }
                            collection_stats["empty_points"].append(empty_point_info)
                            empty_content_analysis["empty_content_points"].append(empty_point_info)
                            empty_content_analysis["problematic_files"][file_path] += 1

                    empty_content_analysis["by_collection"][collection.name] = collection_stats

                except Exception as e:
                    self.logger.debug(f"Error analyzing collection {collection.name} for empty content: {e}")
                    empty_content_analysis["by_collection"][collection.name] = {"error": str(e)}

            # Analyze results
            total_empty = len(empty_content_analysis["empty_content_points"])
            total_checked = empty_content_analysis["total_points_checked"]

            if total_checked == 0:
                empty_rate = 0
            else:
                empty_rate = (total_empty / total_checked) * 100

            # Determine status and recommendations
            recommendations = []
            if empty_rate > 15:
                status = "error"
                message = f"High rate of empty content: {empty_rate:.1f}% ({total_empty}/{total_checked})"
                recommendations.extend(
                    [
                        "Significant content quality issues detected",
                        "Consider reindexing with improved content extraction",
                        "Review parsing logic for problematic file types",
                    ]
                )
            elif empty_rate > 5:
                status = "warning"
                message = f"Moderate empty content rate: {empty_rate:.1f}% ({total_empty}/{total_checked})"
                recommendations.extend(["Some content quality issues detected", "Review files with multiple empty chunks"])
            elif empty_rate > 0:
                status = "warning"
                message = f"Low empty content rate: {empty_rate:.1f}% ({total_empty}/{total_checked})"
                recommendations.append("Monitor content quality during future indexing")
            else:
                status = "healthy"
                message = "No empty content issues detected"

            # Add specific recommendations for problematic files
            if empty_content_analysis["problematic_files"]:
                frequent_problems = sorted(empty_content_analysis["problematic_files"].items(), key=lambda x: x[1], reverse=True)[:5]

                if frequent_problems[0][1] > 3:  # File with more than 3 empty chunks
                    recommendations.append(f"Review file {frequent_problems[0][0]} (has {frequent_problems[0][1]} empty chunks)")

            return SearchDiagnosticResult(
                status=status,
                message=message,
                details={
                    **empty_content_analysis,
                    "empty_content_rate": empty_rate,
                    "frequent_problem_files": dict(
                        sorted(empty_content_analysis["problematic_files"].items(), key=lambda x: x[1], reverse=True)[:10]
                    ),
                },
                timestamp=time.time(),
                recommendations=recommendations,
            )

        except Exception as e:
            return SearchDiagnosticResult(
                status="error",
                message=f"Empty content detection failed: {str(e)}",
                details={"error": str(e)},
                timestamp=time.time(),
                recommendations=["Check vector database connectivity"],
            )

    def generate_diagnostic_report(self, results: dict[str, SearchDiagnosticResult]) -> str:
        """Generate a human-readable diagnostic report.

        Args:
            results: Dictionary of diagnostic results

        Returns:
            Formatted diagnostic report as string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("SEARCH FUNCTIONALITY DIAGNOSTIC REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated at: {time.ctime()}")
        report_lines.append("")

        # Summary
        total_tests = len(results)
        healthy_tests = sum(1 for r in results.values() if r.status == "healthy")
        warning_tests = sum(1 for r in results.values() if r.status == "warning")
        error_tests = sum(1 for r in results.values() if r.status == "error")

        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total tests: {total_tests}")
        report_lines.append(f"  Healthy: {healthy_tests}")
        report_lines.append(f"  Warnings: {warning_tests}")
        report_lines.append(f"  Errors: {error_tests}")
        report_lines.append("")

        # Overall status
        if error_tests > 0:
            overall_status = "CRITICAL - Immediate attention required"
        elif warning_tests > 0:
            overall_status = "WARNING - Issues detected"
        else:
            overall_status = "HEALTHY - All systems operational"

        report_lines.append(f"Overall Status: {overall_status}")
        report_lines.append("")

        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)

        for test_name, result in results.items():
            status_indicator = {"healthy": "✓", "warning": "⚠", "error": "✗"}.get(result.status, "?")

            report_lines.append(f"{status_indicator} {test_name.upper()}: {result.message}")

            if result.recommendations:
                report_lines.append("  Recommendations:")
                for rec in result.recommendations:
                    report_lines.append(f"    - {rec}")

            report_lines.append("")

        # All recommendations summary
        all_recommendations = []
        for result in results.values():
            all_recommendations.extend(result.recommendations)

        if all_recommendations:
            report_lines.append("ALL RECOMMENDATIONS:")
            report_lines.append("-" * 40)
            for i, rec in enumerate(set(all_recommendations), 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)


# Convenience functions for easy usage


def run_quick_diagnostics(project_name: str | None = None) -> dict[str, SearchDiagnosticResult]:
    """Run quick diagnostic tests and return results.

    Args:
        project_name: Optional project name to focus on

    Returns:
        Dictionary of diagnostic results
    """
    diagnostics = SearchDiagnostics()

    # Run essential tests only
    results = {}
    results["connectivity"] = diagnostics.test_vector_db_connectivity()
    results["collections"] = diagnostics.test_collection_health(project_name)
    results["empty_content"] = diagnostics.detect_empty_content_issues(project_name)

    return results


def run_full_diagnostics(project_name: str | None = None) -> str:
    """Run comprehensive diagnostics and return formatted report.

    Args:
        project_name: Optional project name to focus on

    Returns:
        Formatted diagnostic report
    """
    diagnostics = SearchDiagnostics()
    results = diagnostics.run_comprehensive_diagnostics(project_name)
    return diagnostics.generate_diagnostic_report(results)
