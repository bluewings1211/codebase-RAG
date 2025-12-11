"""Unit tests for search diagnostics utilities."""

from collections import defaultdict
from unittest.mock import Mock, patch

import pytest
from utils.search_diagnostics import SearchDiagnostics, create_search_quality_report


class TestSearchDiagnostics:
    """Test suite for SearchDiagnostics class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.diagnostics = SearchDiagnostics(self.mock_client)

    def test_initialization(self):
        """Test SearchDiagnostics initialization."""
        assert self.diagnostics.qdrant_client == self.mock_client
        assert self.diagnostics.logger is not None

    def test_analyze_collection_health_success(self):
        """Test successful collection health analysis."""
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 150
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.name = "Cosine"

        self.mock_client.get_collection.return_value = mock_collection_info

        # Mock sample points
        mock_points = []
        for i in range(10):
            point = Mock()
            point.payload = {
                "content": f"content_{i}" if i % 2 == 0 else "",  # Half have empty content
                "file_path": f"/test/file_{i}.py" if i % 3 != 0 else "",  # Some missing file paths
                "line_start": i,
                "line_end": i + 2,
                "language": "python" if i % 2 == 0 else "",  # Some missing language
            }
            mock_points.append(point)

        self.mock_client.scroll.return_value = (mock_points, None)

        # Perform analysis
        analysis = self.diagnostics.analyze_collection_health("test_collection")

        # Verify basic info
        assert analysis["collection_name"] == "test_collection"
        assert analysis["total_points"] == 150
        assert analysis["vector_size"] == 384
        assert analysis["distance_metric"] == "Cosine"

        # Verify content analysis
        content_analysis = analysis["content_analysis"]
        assert content_analysis["total_sampled"] == 10
        assert content_analysis["empty_content_count"] == 5  # Half are empty
        assert content_analysis["valid_content_rate"] == 0.5

        # Verify metadata analysis
        metadata_analysis = analysis["metadata_analysis"]
        assert metadata_analysis["total_sampled"] == 10
        assert metadata_analysis["missing_file_paths"] > 0
        assert metadata_analysis["missing_language"] > 0

        # Verify issues were detected
        assert len(analysis["issues"]) > 0
        issue_types = [issue["type"] for issue in analysis["issues"]]
        assert "empty_content" in issue_types
        assert "missing_metadata" in issue_types

        # Verify recommendations
        assert len(analysis["recommendations"]) > 0

    def test_analyze_collection_health_error(self):
        """Test collection health analysis with error."""
        self.mock_client.get_collection.side_effect = Exception("Collection not found")

        analysis = self.diagnostics.analyze_collection_health("nonexistent_collection")

        assert analysis["collection_name"] == "nonexistent_collection"
        assert "error" in analysis
        assert analysis["error"] == "Collection not found"
        assert len(analysis["issues"]) == 1
        assert analysis["issues"][0]["type"] == "analysis_error"

    def test_analyze_content_quality(self):
        """Test content quality analysis."""
        # Create test points with various content scenarios
        points = []

        # Valid content
        for i in range(3):
            point = Mock()
            point.payload = {"content": f"def function_{i}(): pass"}
            points.append(point)

        # Empty content
        for i in range(2):
            point = Mock()
            point.payload = {"content": ""}
            points.append(point)

        # Whitespace only
        point = Mock()
        point.payload = {"content": "   \n\t  "}
        points.append(point)

        # Missing content field
        point = Mock()
        point.payload = {}
        points.append(point)

        # No payload
        point = Mock()
        point.payload = None
        points.append(point)

        analysis = self.diagnostics._analyze_content_quality(points)

        assert analysis["total_sampled"] == 8
        assert analysis["empty_content_count"] == 5  # 2 empty + 1 whitespace + 1 missing + 1 no payload
        assert analysis["valid_content_rate"] == 3 / 8  # 3 valid out of 8
        assert analysis["min_content_length"] > 0
        assert analysis["max_content_length"] > 0
        assert analysis["average_content_length"] > 0

    def test_analyze_metadata_quality(self):
        """Test metadata quality analysis."""
        points = []

        # Complete metadata
        for i in range(3):
            point = Mock()
            point.payload = {"file_path": f"/test/file_{i}.py", "line_start": i, "line_end": i + 5, "language": "python"}
            points.append(point)

        # Missing file path
        point = Mock()
        point.payload = {"line_start": 1, "line_end": 5, "language": "python"}
        points.append(point)

        # Missing line info
        point = Mock()
        point.payload = {"file_path": "/test/file.py", "language": "python"}
        points.append(point)

        # Missing language
        point = Mock()
        point.payload = {"file_path": "/test/file.js", "line_start": 1, "line_end": 5}
        points.append(point)

        # No payload
        point = Mock()
        point.payload = None
        points.append(point)

        analysis = self.diagnostics._analyze_metadata_quality(points)

        assert analysis["total_sampled"] == 7
        assert analysis["missing_file_paths"] == 2  # 1 missing + 1 no payload
        assert analysis["missing_line_info"] == 2  # 1 missing + 1 no payload
        assert analysis["missing_language"] == 2  # 1 missing + 1 no payload
        assert "py" in analysis["file_types"]
        assert "js" in analysis["file_types"]
        assert "python" in analysis["languages"]

    def test_validate_search_results_all_valid(self):
        """Test validation with all valid results."""
        results = [
            {"content": "def function1(): pass", "file_path": "/test/file1.py", "score": 0.9, "line_start": 1, "line_end": 5},
            {"content": "class TestClass: pass", "file_path": "/test/file2.py", "score": 0.8, "line_start": 10, "line_end": 15},
        ]

        validation = self.diagnostics.validate_search_results(results)

        assert validation["total_results"] == 2
        assert validation["valid_results"] == 2
        assert validation["validity_rate"] == 1.0
        assert len(validation["issues"]) == 0
        assert all(count == 0 for count in validation["issue_counts"].values())

    def test_validate_search_results_with_issues(self):
        """Test validation with various issues."""
        results = [
            {"content": "", "file_path": "/test/file1.py", "score": 0.9, "line_start": 1, "line_end": 5},  # Empty content
            {"content": "valid content", "file_path": "", "score": 0.8, "line_start": 1, "line_end": 5},  # Missing file path
            {
                "content": "valid content",
                "file_path": "/test/file3.py",
                # Missing score
                "line_start": 1,
                "line_end": 5,
            },
            {
                "content": "valid content",
                "file_path": "/test/file4.py",
                "score": 0.7,
                "line_start": 0,  # Invalid line numbers
                "line_end": -1,
            },
        ]

        validation = self.diagnostics.validate_search_results(results)

        assert validation["total_results"] == 4
        assert validation["valid_results"] == 0  # All have issues
        assert validation["validity_rate"] == 0.0
        assert len(validation["issues"]) == 4

        # Check issue counts
        assert validation["issue_counts"]["empty_content"] == 1
        assert validation["issue_counts"]["missing_file_path"] == 1
        assert validation["issue_counts"]["missing_score"] == 1
        assert validation["issue_counts"]["invalid_line_numbers"] == 1

        # Check issue details
        issues = validation["issues"]
        assert any("empty_content" in issue["issues"] for issue in issues)
        assert any("missing_file_path" in issue["issues"] for issue in issues)
        assert any("missing_score" in issue["issues"] for issue in issues)
        assert any("invalid_line_numbers" in issue["issues"] for issue in issues)

    def test_check_vector_database_consistency_success(self):
        """Test vector database consistency check with healthy collections."""
        collection_names = ["collection1", "collection2", "collection3"]

        # Mock collection info
        def mock_get_collection(name):
            info = Mock()
            info.points_count = 100
            info.config.params.vectors.size = 384
            info.config.params.vectors.distance.name = "Cosine"
            return info

        self.mock_client.get_collection.side_effect = mock_get_collection

        consistency = self.diagnostics.check_vector_database_consistency(collection_names)

        assert consistency["collections_checked"] == collection_names
        assert consistency["total_collections"] == 3
        assert len(consistency["issues"]) == 0
        assert len(consistency["collection_stats"]) == 3

        # Verify all collections have same vector size (consistent)
        for stats in consistency["collection_stats"].values():
            assert stats["vector_size"] == 384
            assert stats["points_count"] == 100

    def test_check_vector_database_consistency_with_issues(self):
        """Test consistency check with inconsistent collections."""
        collection_names = ["collection1", "collection2", "collection3"]

        # Mock collection info with inconsistencies
        def mock_get_collection(name):
            info = Mock()
            if name == "collection1":
                info.points_count = 100
                info.config.params.vectors.size = 384
            elif name == "collection2":
                info.points_count = 0  # Empty collection
                info.config.params.vectors.size = 768  # Different vector size
            else:  # collection3
                raise Exception("Collection access error")

            info.config.params.vectors.distance.name = "Cosine"
            return info

        self.mock_client.get_collection.side_effect = mock_get_collection

        consistency = self.diagnostics.check_vector_database_consistency(collection_names)

        assert len(consistency["issues"]) >= 2  # Should have multiple issues

        issue_types = [issue["type"] for issue in consistency["issues"]]
        assert "inconsistent_vector_sizes" in issue_types
        assert "empty_collections" in issue_types
        assert "collection_access_error" in issue_types

        assert len(consistency["recommendations"]) > 0

    def test_measure_search_performance(self):
        """Test search performance measurement."""
        query = "test query"
        query_embedding = [0.1] * 384
        collection_names = ["collection1", "collection2"]

        # Mock search results
        def mock_search(collection_name, query_vector, limit, score_threshold):
            results = []
            for i in range(3):
                result = Mock()
                result.score = 0.9 - (i * 0.1)
                result.payload = {"content": f"content_{i}" if i % 2 == 0 else "", "file_path": f"/test/file_{i}.py"}  # Some empty
                results.append(result)
            return results

        self.mock_client.search.side_effect = mock_search

        performance = self.diagnostics.measure_search_performance(
            query=query, query_embedding=query_embedding, collection_names=collection_names, n_results=5
        )

        assert performance["query"] == query
        assert performance["collections_tested"] == collection_names
        assert performance["n_results"] == 5
        assert performance["total_time"] > 0
        assert performance["total_results"] == 6  # 3 from each collection
        assert len(performance["collection_performance"]) == 2

        # Check per-collection metrics
        for collection_name in collection_names:
            perf = performance["collection_performance"][collection_name]
            assert "search_time" in perf
            assert "result_count" in perf
            assert "average_score" in perf
            assert "max_score" in perf
            assert "min_score" in perf
            assert "empty_content_count" in perf
            assert perf["empty_content_count"] >= 0  # Should detect empty content

    def test_diagnose_empty_content_issue(self):
        """Test empty content issue diagnosis."""
        collection_name = "test_collection"

        # Mock scroll results with various empty content scenarios
        points = []

        # Valid content
        for i in range(3):
            point = Mock()
            point.payload = {"content": f"def function_{i}(): pass", "file_path": f"/test/file_{i}.py", "chunk_type": "function"}
            points.append(point)

        # Empty content scenarios
        scenarios = [
            {"content": "", "type": "completely_empty"},  # Empty string
            {"content": "   ", "type": "whitespace_only"},  # Whitespace
            {"content": None, "type": "null_content"},  # Null
            {"type": "missing_content_field"},  # Missing field
        ]

        for i, scenario in enumerate(scenarios):
            point = Mock()
            payload = {"file_path": f"/test/problem_{i}.py", "chunk_type": "function"}
            if "content" in scenario:
                payload["content"] = scenario["content"]
            elif scenario["type"] != "missing_content_field":
                payload["content"] = ""

            point.payload = payload
            points.append(point)

        self.mock_client.scroll.return_value = (points, None)

        diagnosis = self.diagnostics.diagnose_empty_content_issue(collection_name)

        assert diagnosis["collection_name"] == collection_name
        assert diagnosis["total_sampled"] == 7

        # Check empty content analysis
        empty_analysis = diagnosis["empty_content_analysis"]
        assert empty_analysis["completely_empty"] >= 1
        assert empty_analysis["whitespace_only"] >= 1
        assert empty_analysis["null_content"] >= 1

        assert diagnosis["empty_content_rate"] > 0
        assert len(diagnosis["recommendations"]) > 0

        # Check file and chunk type analysis
        assert len(diagnosis["file_pattern_analysis"]) > 0
        assert len(diagnosis["chunk_type_analysis"]) > 0

    def test_diagnose_empty_content_issue_error(self):
        """Test empty content diagnosis with error."""
        self.mock_client.scroll.side_effect = Exception("Scroll failed")

        diagnosis = self.diagnostics.diagnose_empty_content_issue("test_collection")

        assert diagnosis["collection_name"] == "test_collection"
        assert "error" in diagnosis
        assert "recommendations" in diagnosis
        assert "Unable to diagnose" in diagnosis["recommendations"][0]


class TestCreateSearchQualityReport:
    """Test suite for create_search_quality_report function."""

    def test_create_search_quality_report_success(self):
        """Test successful search quality report creation."""
        mock_client = Mock()
        collection_names = ["collection1", "collection2"]

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.name = "Cosine"
        mock_client.get_collection.return_value = mock_collection_info

        # Mock scroll for health analysis
        mock_points = []
        for i in range(5):
            point = Mock()
            point.payload = {
                "content": f"content_{i}" if i % 2 == 0 else "",
                "file_path": f"/test/file_{i}.py",
                "line_start": i,
                "line_end": i + 2,
                "language": "python",
            }
            mock_points.append(point)

        mock_client.scroll.return_value = (mock_points, None)

        # Mock collections for consistency check
        mock_collections = [Mock(name=name) for name in collection_names]
        mock_client.get_collections.return_value = Mock(collections=mock_collections)

        with patch("utils.search_diagnostics.logger") as mock_logger:
            report = create_search_quality_report(qdrant_client=mock_client, collection_names=collection_names)

        # Verify report structure
        assert "timestamp" in report
        assert "collections_analyzed" in report
        assert "collection_health" in report
        assert "consistency_check" in report
        assert "overall_recommendations" in report

        assert report["collections_analyzed"] == collection_names

        # Verify collection health analysis
        for collection_name in collection_names:
            assert collection_name in report["collection_health"]
            health = report["collection_health"][collection_name]
            assert "total_points" in health
            assert "content_analysis" in health
            assert "metadata_analysis" in health

            # Should detect empty content and include diagnosis
            if any(issue["type"] == "empty_content" for issue in health.get("issues", [])):
                assert "empty_content_diagnosis" in health

        # Verify consistency check
        consistency = report["consistency_check"]
        assert "collections_checked" in consistency
        assert "total_collections" in consistency
        assert "collection_stats" in consistency

        # Verify recommendations
        assert isinstance(report["overall_recommendations"], list)

        # Verify logging
        mock_logger.info.assert_called()

    def test_create_search_quality_report_with_minimal_params(self):
        """Test report creation with minimal parameters."""
        mock_client = Mock()
        collection_names = ["test_collection"]

        # Mock minimal setup
        mock_collection_info = Mock()
        mock_collection_info.points_count = 50
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.name = "Cosine"
        mock_client.get_collection.return_value = mock_collection_info

        mock_client.scroll.return_value = ([], None)  # Empty points

        report = create_search_quality_report(
            qdrant_client=mock_client,
            collection_names=collection_names,
        )

        assert "timestamp" in report
        assert report["collections_analyzed"] == collection_names
        assert "test_collection" in report["collection_health"]

    def test_create_search_quality_report_with_issues(self):
        """Test report creation when issues are detected."""
        mock_client = Mock()
        collection_names = ["problematic_collection"]

        # Mock collection with issues
        mock_collection_info = Mock()
        mock_collection_info.points_count = 0  # Empty collection
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.name = "Cosine"
        mock_client.get_collection.return_value = mock_collection_info

        # Mock points with empty content
        mock_points = []
        for i in range(3):
            point = Mock()
            point.payload = {"content": "", "file_path": f"/test/file_{i}.py"}  # All empty content
            mock_points.append(point)

        mock_client.scroll.return_value = (mock_points, None)

        report = create_search_quality_report(qdrant_client=mock_client, collection_names=collection_names)

        # Should detect and report issues
        health = report["collection_health"]["problematic_collection"]
        assert len(health.get("issues", [])) > 0

        # Should include empty content diagnosis
        assert "empty_content_diagnosis" in health

        # Should have overall recommendations about issues
        assert len(report["overall_recommendations"]) > 0
        assert any("issues" in rec for rec in report["overall_recommendations"])

    @patch("utils.search_diagnostics.logger")
    def test_create_search_quality_report_logging(self, mock_logger):
        """Test that report creation includes proper logging."""
        mock_client = Mock()
        collection_names = ["collection1", "collection2"]

        # Mock basic setup
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.name = "Cosine"
        mock_client.get_collection.return_value = mock_collection_info

        mock_client.scroll.return_value = ([], None)

        create_search_quality_report(qdrant_client=mock_client, collection_names=collection_names)

        # Verify logging calls
        info_calls = [call.args[0] for call in mock_logger.info.call_args_list]

        # Should log analysis of each collection
        assert any("Analyzing collection: collection1" in call for call in info_calls)
        assert any("Analyzing collection: collection2" in call for call in info_calls)

        # Should log consistency check
        assert any("Checking vector database consistency" in call for call in info_calls)
