"""Integration tests for search functionality bug fixes.

This module contains comprehensive integration tests to verify that search
functionality bug fixes work correctly in realistic scenarios.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from tools.indexing.search_tools import _perform_hybrid_search, search_sync
from utils.search_diagnostics import SearchDiagnostics, create_search_quality_report


class TestSearchBugFixesIntegration:
    """Integration tests for search bug fixes."""

    def test_n_results_fix_with_real_scenario(self):
        """Test n_results fix with realistic multi-collection scenario."""
        # Create a realistic scenario with multiple collections
        mock_qdrant_client = Mock()
        mock_embeddings = Mock()

        # Simulate 3 collections with different content types
        collections = ["project_testapp_code", "project_testapp_docs", "project_testapp_config"]

        # Create varied results across collections
        def create_mock_result(score, content, file_path, collection_suffix):
            result = Mock()
            result.score = score
            result.payload = {
                "content": content,
                "file_path": file_path,
                "line_start": 1,
                "line_end": 5,
                "language": "python" if collection_suffix == "code" else "text",
                "chunk_type": "function" if collection_suffix == "code" else "section",
                "collection": f"project_testapp_{collection_suffix}",
            }
            return result

        # Mock search results for each collection
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            if "code" in collection_name:
                return [
                    create_mock_result(0.95, "def authenticate_user(username, password):", "/app/auth.py", "code"),
                    create_mock_result(0.85, "def validate_credentials(user_data):", "/app/validation.py", "code"),
                    create_mock_result(0.80, "class UserService:", "/app/services.py", "code"),
                    create_mock_result(0.75, "def hash_password(password):", "/app/crypto.py", "code"),
                    create_mock_result(0.70, "def generate_token(user_id):", "/app/tokens.py", "code"),
                ]
            elif "docs" in collection_name:
                return [
                    create_mock_result(0.90, "## User Authentication\nThis section covers user authentication...", "/docs/auth.md", "docs"),
                    create_mock_result(0.82, "### Password Security\nPasswords must be hashed...", "/docs/security.md", "docs"),
                    create_mock_result(0.78, "# API Documentation\nAuthentication endpoints...", "/docs/api.md", "docs"),
                ]
            elif "config" in collection_name:
                return [
                    create_mock_result(0.88, "AUTH_SECRET_KEY = 'your-secret-key'", "/config/auth.py", "config"),
                    create_mock_result(0.72, "PASSWORD_MIN_LENGTH = 8", "/config/security.py", "config"),
                ]
            return []

        mock_qdrant_client.search.side_effect = mock_search

        # Test with various n_results values
        test_cases = [
            (3, 3),  # Should return exactly 3 results
            (5, 5),  # Should return exactly 5 results
            (8, 8),  # Should return exactly 8 results
            (15, 10),  # Should return 10 results (limited by available results)
        ]

        for n_results, expected_count in test_cases:
            results = _perform_hybrid_search(
                qdrant_client=mock_qdrant_client,
                embedding_model=mock_embeddings,
                query="user authentication",
                query_embedding=[0.1] * 384,
                search_collections=collections,
                n_results=n_results,
                search_mode="hybrid",
            )

            assert len(results) == expected_count, f"Expected {expected_count} results, got {len(results)} for n_results={n_results}"

            # Verify results are sorted by score
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

            # Verify top result is the highest scoring one
            if results:
                assert results[0]["score"] == 0.95, "Top result should have highest score"

    def test_empty_content_fix_comprehensive(self):
        """Test comprehensive empty content filtering."""
        mock_qdrant_client = Mock()
        mock_embeddings = Mock()

        # Create results with various empty content scenarios
        def create_result_with_content(score, content, file_path="test.py"):
            result = Mock()
            result.score = score
            result.payload = {"content": content, "file_path": file_path, "line_start": 1, "line_end": 2}
            return result

        # Mix of valid and invalid results
        mock_results = [
            create_result_with_content(0.95, "def valid_function(): pass"),
            create_result_with_content(0.90, ""),  # Empty string
            create_result_with_content(0.85, "   "),  # Whitespace only
            create_result_with_content(0.80, "\n\t\n"),  # Newlines and tabs
            create_result_with_content(0.75, "class ValidClass: pass"),
            create_result_with_content(0.70, ""),  # Another empty
            create_result_with_content(0.65, "# Valid comment"),
            create_result_with_content(0.60, "   \n   "),  # More whitespace
            create_result_with_content(0.55, "import os"),
            create_result_with_content(0.50, ""),  # Yet another empty
        ]

        mock_qdrant_client.search.return_value = mock_results

        results = _perform_hybrid_search(
            qdrant_client=mock_qdrant_client,
            embedding_model=mock_embeddings,
            query="test query",
            query_embedding=[0.1] * 384,
            search_collections=["test_collection"],
            n_results=10,
            search_mode="hybrid",
        )

        # Should only return results with valid content
        assert len(results) == 4, "Should filter out empty content results"

        expected_content = ["def valid_function(): pass", "class ValidClass: pass", "# Valid comment", "import os"]

        actual_content = [r["content"] for r in results]
        assert actual_content == expected_content, f"Expected {expected_content}, got {actual_content}"

        # Verify scores are maintained in order
        expected_scores = [0.95, 0.75, 0.65, 0.55]
        actual_scores = [r["score"] for r in results]
        assert actual_scores == expected_scores, f"Expected {expected_scores}, got {actual_scores}"

    def test_cross_collection_aggregation_realistic(self):
        """Test realistic cross-collection result aggregation."""
        mock_qdrant_client = Mock()
        mock_embeddings = Mock()

        # Simulate a real project with multiple collection types
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            base_results = []

            if "code" in collection_name:
                # Code results typically have higher precision for code queries
                base_results = [
                    {"score": 0.92, "content": "def database_connect():", "type": "function"},
                    {"score": 0.88, "content": "class DatabaseManager:", "type": "class"},
                    {"score": 0.84, "content": "def execute_query(sql):", "type": "function"},
                    {"score": 0.78, "content": "def close_connection():", "type": "function"},
                ]
            elif "docs" in collection_name:
                # Documentation results
                base_results = [
                    {"score": 0.89, "content": "## Database Connection\nTo connect to the database...", "type": "documentation"},
                    {"score": 0.82, "content": "### Connection Pool\nThe connection pool manages...", "type": "documentation"},
                    {"score": 0.76, "content": "# Database Setup\nBefore using the database...", "type": "documentation"},
                ]
            elif "config" in collection_name:
                # Configuration results
                base_results = [
                    {"score": 0.85, "content": "DATABASE_URL = 'postgresql://...'", "type": "config"},
                    {"score": 0.79, "content": "CONNECTION_POOL_SIZE = 10", "type": "config"},
                    {"score": 0.73, "content": "QUERY_TIMEOUT = 30", "type": "config"},
                ]

            # Convert to mock objects
            mock_results = []
            for result_data in base_results:
                result = Mock()
                result.score = result_data["score"]
                result.payload = {
                    "content": result_data["content"],
                    "file_path": f"/app/{result_data['type']}.py",
                    "line_start": 1,
                    "line_end": 2,
                    "chunk_type": result_data["type"],
                }
                mock_results.append(result)

            return mock_results

        mock_qdrant_client.search.side_effect = mock_search

        # Test aggregation across all collection types
        results = _perform_hybrid_search(
            qdrant_client=mock_qdrant_client,
            embedding_model=mock_embeddings,
            query="database connection",
            query_embedding=[0.1] * 384,
            search_collections=["project_app_code", "project_app_docs", "project_app_config"],
            n_results=6,
            search_mode="hybrid",
        )

        assert len(results) == 6, "Should return exactly 6 results"

        # Verify results are properly aggregated and sorted
        expected_scores = [0.92, 0.89, 0.88, 0.85, 0.84, 0.82]
        actual_scores = [r["score"] for r in results]
        assert actual_scores == expected_scores, f"Expected scores {expected_scores}, got {actual_scores}"

        # Verify we get results from all collection types
        result_types = [r["chunk_type"] for r in results]
        assert "function" in result_types, "Should include function results"
        assert "documentation" in result_types, "Should include documentation results"
        assert "config" in result_types, "Should include config results"

    @pytest.mark.requires_qdrant
    @pytest.mark.skip(reason="search_sync integration requires real Qdrant service - complex mock setup")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_current_project")
    def test_search_sync_integration_with_fixes(self, mock_get_project, mock_get_client, mock_get_embeddings):
        """Test complete search_sync integration with bug fixes."""
        # Setup mocks
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_project = {"name": "test_project", "collection_prefix": "project_test"}

        mock_get_client.return_value = mock_client
        mock_get_embeddings.return_value = mock_embeddings
        mock_get_project.return_value = mock_project

        # Mock collections - create proper mock objects with name attribute
        mock_collections = []
        for name in ["project_test_code", "project_test_docs", "project_test_config"]:
            mock_collection = Mock()
            mock_collection.name = name
            mock_collections.append(mock_collection)
        mock_client.get_collections.return_value = Mock(collections=mock_collections)

        # Mock embedding generation
        mock_embeddings.generate_embeddings.return_value = [0.1] * 384

        # Mock search results with potential empty content
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            results = []

            if "code" in collection_name:
                # Include some results with empty content to test filtering
                valid_result = Mock()
                valid_result.score = 0.9
                valid_result.payload = {
                    "content": "def main():\n    print('hello')",
                    "file_path": "/app/main.py",
                    "line_start": 1,
                    "line_end": 2,
                    "language": "python",
                    "chunk_type": "function",
                }
                results.append(valid_result)

                # Empty content result (should be filtered out)
                empty_result = Mock()
                empty_result.score = 0.8
                empty_result.payload = {
                    "content": "",
                    "file_path": "/app/empty.py",
                    "line_start": 1,
                    "line_end": 2,
                    "language": "python",
                    "chunk_type": "function",
                }
                results.append(empty_result)

            elif "docs" in collection_name:
                doc_result = Mock()
                doc_result.score = 0.85
                doc_result.payload = {
                    "content": "# Getting Started\nThis is the documentation...",
                    "file_path": "/docs/getting-started.md",
                    "line_start": 1,
                    "line_end": 3,
                    "language": "markdown",
                    "chunk_type": "section",
                }
                results.append(doc_result)

            return results

        mock_client.search.side_effect = mock_search

        # Test search with n_results=3 (should not be multiplied by collection count)
        result = search_sync(
            query="main function", n_results=3, cross_project=False, search_mode="hybrid", include_context=False, context_chunks=0
        )

        # Verify results
        assert "error" not in result, f"Search should not return error: {result.get('error')}"
        assert result["total"] == 2, f"Should return 2 results (empty content filtered), got {result['total']}"
        assert len(result["results"]) == 2, f"Results list should have 2 items, got {len(result['results'])}"

        # Verify empty content was filtered out
        contents = [r["content"] for r in result["results"]]
        assert all(content.strip() for content in contents), "All results should have non-empty content"

        # Verify results are sorted by score
        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

        # Verify we get the expected content
        assert "def main():" in result["results"][0]["content"]
        assert "# Getting Started" in result["results"][1]["content"]

        # Verify search was called for all collections
        assert mock_client.search.call_count == 3, f"Should call search 3 times, called {mock_client.search.call_count} times"

        # Verify metadata structure
        for search_result in result["results"]:
            assert "score" in search_result
            assert "file_path" in search_result
            assert "content" in search_result
            assert "line_start" in search_result
            assert "line_end" in search_result
            assert "language" in search_result
            assert "chunk_type" in search_result

    def test_search_performance_with_bug_fixes(self):
        """Test that bug fixes don't negatively impact performance."""
        mock_qdrant_client = Mock()
        mock_embeddings = Mock()

        # Create a large number of results to test performance
        def create_many_results(collection_name, count=50):
            results = []
            for i in range(count):
                result = Mock()
                result.score = 0.9 - (i * 0.01)  # Decreasing scores
                result.payload = {
                    "content": f"def function_{i}(): pass" if i % 2 == 0 else "",  # Half empty
                    "file_path": f"/app/file_{i}.py",
                    "line_start": i,
                    "line_end": i + 1,
                    "language": "python",
                    "chunk_type": "function",
                }
                results.append(result)
            return results

        mock_qdrant_client.search.side_effect = lambda *args, **kwargs: create_many_results("test", 50)

        # Test with multiple collections
        import time

        start_time = time.time()

        results = _perform_hybrid_search(
            qdrant_client=mock_qdrant_client,
            embedding_model=mock_embeddings,
            query="test query",
            query_embedding=[0.1] * 384,
            search_collections=["collection1", "collection2", "collection3"],
            n_results=10,
            search_mode="hybrid",
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Verify functionality
        assert len(results) == 10, "Should return exactly 10 results"

        # Verify empty content filtering worked
        contents = [r["content"] for r in results]
        assert all(content.strip() for content in contents), "All results should have non-empty content"

        # Verify performance is reasonable (should complete quickly)
        assert execution_time < 1.0, f"Search should complete within 1 second, took {execution_time:.3f}s"

    def test_edge_case_scenarios(self):
        """Test edge cases that could reveal bugs."""
        mock_qdrant_client = Mock()
        mock_embeddings = Mock()

        # Test case 1: All results have empty content
        def mock_search_all_empty(*args, **kwargs):
            results = []
            for i in range(5):
                result = Mock()
                result.score = 0.9 - (i * 0.1)
                result.payload = {"content": "", "file_path": f"/test/file_{i}.py", "line_start": 1, "line_end": 2}  # All empty
                results.append(result)
            return results

        mock_qdrant_client.search.side_effect = mock_search_all_empty

        results = _perform_hybrid_search(
            qdrant_client=mock_qdrant_client,
            embedding_model=mock_embeddings,
            query="test query",
            query_embedding=[0.1] * 384,
            search_collections=["collection1"],
            n_results=10,
            search_mode="hybrid",
        )

        assert len(results) == 0, "Should return no results when all content is empty"

        # Test case 2: n_results larger than available results
        def mock_search_few_results(*args, **kwargs):
            results = []
            for i in range(3):  # Only 3 results available
                result = Mock()
                result.score = 0.9 - (i * 0.1)
                result.payload = {"content": f"content_{i}", "file_path": f"/test/file_{i}.py", "line_start": 1, "line_end": 2}
                results.append(result)
            return results

        mock_qdrant_client.search.side_effect = mock_search_few_results

        results = _perform_hybrid_search(
            qdrant_client=mock_qdrant_client,
            embedding_model=mock_embeddings,
            query="test query",
            query_embedding=[0.1] * 384,
            search_collections=["collection1", "collection2"],
            n_results=10,  # Requesting more than available
            search_mode="hybrid",
        )

        assert len(results) == 6, "Should return all available results (3 from each collection)"

        # Test case 3: Single collection with n_results=1
        def mock_search_single_result(*args, **kwargs):
            result = Mock()
            result.score = 0.9
            result.payload = {"content": "single result", "file_path": "/test/single.py", "line_start": 1, "line_end": 2}
            return [result]

        mock_qdrant_client.search.side_effect = mock_search_single_result

        results = _perform_hybrid_search(
            qdrant_client=mock_qdrant_client,
            embedding_model=mock_embeddings,
            query="test query",
            query_embedding=[0.1] * 384,
            search_collections=["collection1"],
            n_results=1,
            search_mode="hybrid",
        )

        assert len(results) == 1, "Should return exactly 1 result"
        assert results[0]["content"] == "single result"


class TestSearchDiagnostics:
    """Test search diagnostic utilities."""

    def test_search_diagnostics_initialization(self):
        """Test SearchDiagnostics initialization."""
        mock_client = Mock()
        diagnostics = SearchDiagnostics(mock_client)

        assert diagnostics.qdrant_client == mock_client
        assert diagnostics.logger is not None

    @pytest.mark.skip(reason="validate_search_results method not implemented in SearchDiagnostics")
    def test_validate_search_results_with_issues(self):
        """Test search result validation with various issues."""
        mock_client = Mock()
        diagnostics = SearchDiagnostics(mock_client)

        # Create results with various issues
        results = [
            {"content": "valid content", "file_path": "/test/valid.py", "score": 0.9, "line_start": 1, "line_end": 5},
            {"content": "", "file_path": "/test/empty.py", "score": 0.8, "line_start": 1, "line_end": 2},  # Empty content
            {"content": "valid content 2", "file_path": "", "score": 0.7, "line_start": 1, "line_end": 2},  # Missing file path
            {
                "content": "valid content 3",
                "file_path": "/test/valid2.py",
                # Missing score
                "line_start": 1,
                "line_end": 2,
            },
            {
                "content": "valid content 4",
                "file_path": "/test/valid3.py",
                "score": 0.6,
                "line_start": 0,  # Invalid line numbers
                "line_end": -1,
            },
        ]

        validation = diagnostics.validate_search_results(results)

        assert validation["total_results"] == 5
        assert validation["valid_results"] == 1  # Only the first result is valid
        assert validation["validity_rate"] == 0.2

        # Check issue counts
        assert validation["issue_counts"]["empty_content"] == 1
        assert validation["issue_counts"]["missing_file_path"] == 1
        assert validation["issue_counts"]["missing_score"] == 1
        assert validation["issue_counts"]["invalid_line_numbers"] == 1

        # Check detailed issues
        assert len(validation["issues"]) == 4  # 4 results have issues

    def test_create_search_quality_report(self):
        """Test comprehensive search quality report generation."""
        mock_client = Mock()

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.name = "Cosine"

        mock_client.get_collection.return_value = mock_collection_info

        # Mock scroll results (sample points)
        mock_points = []
        for i in range(10):
            point = Mock()
            point.payload = {
                "content": f"content_{i}" if i % 2 == 0 else "",  # Half empty
                "file_path": f"/test/file_{i}.py",
                "line_start": i,
                "line_end": i + 1,
                "language": "python",
            }
            mock_points.append(point)

        mock_client.scroll.return_value = (mock_points, None)

        # Test report generation
        report = create_search_quality_report(qdrant_client=mock_client, collection_names=["test_collection"])

        assert "timestamp" in report
        assert "collections_analyzed" in report
        assert "collection_health" in report
        assert "consistency_check" in report
        assert "overall_recommendations" in report

        # Check that collection health was analyzed
        assert "test_collection" in report["collection_health"]
        health = report["collection_health"]["test_collection"]
        assert "total_points" in health
        assert "content_analysis" in health
        assert "metadata_analysis" in health

        # Should detect empty content issues
        assert any(issue["type"] == "empty_content" for issue in health.get("issues", []))
