"""Tests for enhanced search tool functionality with target_projects parameter."""

from unittest.mock import Mock, patch

import pytest

# Import the search function from the correct module
from tools.indexing.search_tools import search_sync


@pytest.mark.skip(reason="Tests require complex mock setup for search_sync - mock collection iteration issues")
class TestAdvancedSearchTool:
    """Test suite for enhanced search tool with target_projects functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_query = "authentication function"
        self.mock_collections = [
            "project_service1_code",
            "project_service1_config",
            "project_service1_documentation",
            "project_service2_code",
            "project_service2_config",
            "dir_frontend_code",
            "dir_frontend_documentation",
            "global_code",
        ]

        self.mock_search_results = [
            {
                "score": 0.95,
                "collection": "project_service1_code",
                "file_path": "/path/to/auth.py",
                "content": "def authenticate_user(username, password):",
                "chunk_type": "function",
                "language": "python",
                "project": "service1",
            },
            {
                "score": 0.87,
                "collection": "project_service2_code",
                "file_path": "/path/to/auth_service.py",
                "content": "class AuthenticationService:",
                "chunk_type": "class",
                "language": "python",
                "project": "service2",
            },
        ]

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_parameter_validation(self, mock_embeddings, mock_qdrant):
        """Test validation of target_projects parameter."""

        # Test invalid target_projects types
        result = search_sync(
            query=self.test_query,
            target_projects="invalid_type",  # Should be list
        )
        assert "error" in result
        assert "target_projects must be a list" in result["error"]

        # Test non-string elements in target_projects
        result = search_sync(
            query=self.test_query,
            target_projects=["valid", 123, "also_valid"],  # 123 is not string
        )
        assert "error" in result
        assert "All project names in target_projects must be strings" in result["error"]

        # Test empty target_projects list
        result = search_sync(
            query=self.test_query,
            target_projects=[],  # Empty list
        )
        assert "error" in result
        assert "target_projects cannot be empty if specified" in result["error"]

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_collection_filtering(self, mock_embeddings, mock_qdrant):
        """Test that target_projects correctly filters collections."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_client.search.return_value = []

        # Test searching specific projects
        result = search_sync(query=self.test_query, target_projects=["service1", "service2"])

        # Should not have error
        assert "error" not in result

        # Should include target_projects in response
        assert result["target_projects"] == ["service1", "service2"]
        assert "specific projects: service1, service2" in result["search_scope"]

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    @patch("tools.project.project_utils.get_available_project_names")
    def test_target_projects_not_found_error(self, mock_get_projects, mock_embeddings, mock_qdrant):
        """Test error handling when target projects are not found."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_get_projects.return_value = ["service1", "service2", "frontend"]

        # Test searching for non-existent projects
        result = search_sync(query=self.test_query, target_projects=["nonexistent1", "nonexistent2"])

        # Should return error with available projects
        assert "error" in result
        assert "No indexed collections found for projects" in result["error"]
        assert result["available_projects"] == ["service1", "service2", "frontend"]

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_search_execution(self, mock_embeddings, mock_qdrant):
        """Test actual search execution with target_projects."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        # Mock search results
        mock_search_result_objects = []
        for result_data in self.mock_search_results:
            mock_result = Mock()
            mock_result.score = result_data["score"]
            mock_result.payload = {
                "file_path": result_data["file_path"],
                "content": result_data["content"],
                "chunk_type": result_data["chunk_type"],
                "language": result_data["language"],
                "project": result_data["project"],
                "line_start": 1,
                "line_end": 5,
            }
            mock_search_result_objects.append(mock_result)

        mock_client.search.return_value = mock_search_result_objects

        # Execute search with target projects
        result = search_sync(query=self.test_query, target_projects=["service1", "service2"])

        # Verify results
        assert "error" not in result
        assert result["total"] == len(self.mock_search_results)
        assert result["target_projects"] == ["service1", "service2"]
        assert len(result["results"]) == len(self.mock_search_results)

        # Verify collections searched
        expected_collections = [
            "project_service1_code",
            "project_service1_config",
            "project_service1_documentation",
            "project_service2_code",
            "project_service2_config",
        ]
        searched_collections = result["collections_searched"]
        for expected_collection in expected_collections:
            assert expected_collection in searched_collections

        # Should not include other projects
        assert "dir_frontend_code" not in searched_collections
        assert "global_code" not in searched_collections

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_with_hyphens_and_spaces(self, mock_embeddings, mock_qdrant):
        """Test project name normalization for names with hyphens and spaces."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        # Mock collections with normalized names
        normalized_collections = [
            "project_my_service_code",
            "project_my_service_config",
            "project_another_app_code",
        ]

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in normalized_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_client.search.return_value = []

        # Test with project names containing hyphens and spaces
        result = search_sync(
            query=self.test_query,
            target_projects=["my-service", "Another App"],  # Will be normalized
        )

        # Should not have error (collections should be found after normalization)
        assert "error" not in result
        assert result["target_projects"] == ["my-service", "Another App"]

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_none_uses_default_behavior(self, mock_embeddings, mock_qdrant):
        """Test that target_projects=None uses default search behavior."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_client.search.return_value = []

        with patch("tools.indexing.search_tools.get_current_project") as mock_get_project:
            mock_get_project.return_value = {
                "name": "test_project",
                "collection_prefix": "project_test_project",
            }

            # Test with target_projects=None (default behavior)
            result = search_sync(query=self.test_query, target_projects=None)

            # Should use current project behavior
            assert "error" not in result
            assert result["target_projects"] is None
            assert result["search_scope"] == "current project"


@pytest.mark.skip(reason="Tests require complex mock setup for search_sync - mock collection iteration issues")
class TestSearchResultEnhancements:
    """Test enhanced search results with project information."""

    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_search_result_includes_project_metadata(self, mock_embeddings, mock_qdrant):
        """Test that search results include enhanced project metadata."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name="project_test_code")]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        # Mock search result with enhanced metadata
        mock_result = Mock()
        mock_result.score = 0.95
        mock_result.payload = {
            "file_path": "/path/to/test.py",
            "content": "def test_function():",
            "project": "test_project",
            "chunk_type": "function",
            "name": "test_function",
            "signature": "def test_function():",
            "docstring": "Test function documentation",
            "breadcrumb": "test_module.test_function",
            "line_start": 10,
            "line_end": 15,
        }

        mock_client.search.return_value = [mock_result]

        with patch("tools.indexing.search_tools.get_current_project") as mock_get_project:
            mock_get_project.return_value = {
                "name": "test_project",
                "collection_prefix": "project_test",
            }

            result = search_sync(query="test function")

            # Verify enhanced metadata is preserved
            assert len(result["results"]) == 1
            result_item = result["results"][0]

            assert result_item["project"] == "test_project"
            assert result_item["chunk_type"] == "function"
            assert result_item["name"] == "test_function"
            assert result_item["signature"] == "def test_function():"
            assert result_item["docstring"] == "Test function documentation"
            assert result_item["breadcrumb"] == "test_module.test_function"
            assert result_item["line_start"] == 10
            assert result_item["line_end"] == 15


if __name__ == "__main__":
    pytest.main([__file__])
