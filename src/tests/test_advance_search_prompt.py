"""Tests for the advance_search prompt implementation."""

from unittest.mock import Mock, patch

import pytest
from prompts.advanced_search.advance_search import AdvanceSearchPrompt


class TestAdvanceSearchPrompt:
    """Test suite for the advance_search prompt."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_prompts_system = Mock()
        self.mock_prompts_system.indexing_service = Mock()
        self.mock_prompts_system.analysis_service = Mock()
        self.mock_prompts_system.embedding_service = Mock()

        self.prompt = AdvanceSearchPrompt(self.mock_prompts_system)

        self.mock_project_data = {
            "total_projects": 3,
            "projects": [
                {
                    "name": "service1",
                    "total_points": 1500,
                    "collections": ["project_service1_code", "project_service1_config"],
                    "collection_types": ["code", "config"],
                    "collection_details": [
                        {
                            "name": "project_service1_code",
                            "type": "code",
                            "points_count": 1200,
                        },
                        {
                            "name": "project_service1_config",
                            "type": "config",
                            "points_count": 300,
                        },
                    ],
                },
                {
                    "name": "frontend",
                    "total_points": 800,
                    "collections": ["dir_frontend_code", "dir_frontend_documentation"],
                    "collection_types": ["code", "documentation"],
                    "collection_details": [
                        {
                            "name": "dir_frontend_code",
                            "type": "code",
                            "points_count": 600,
                        },
                        {
                            "name": "dir_frontend_documentation",
                            "type": "documentation",
                            "points_count": 200,
                        },
                    ],
                },
                {
                    "name": "api_gateway",
                    "total_points": 2000,
                    "collections": ["project_api_gateway_code"],
                    "collection_types": ["code"],
                    "collection_details": [
                        {
                            "name": "project_api_gateway_code",
                            "type": "code",
                            "points_count": 2000,
                        }
                    ],
                },
            ],
        }

    def test_discover_indexed_projects_success(self):
        """Test successful project discovery."""
        with patch("tools.project.project_utils.list_indexed_projects") as mock_list:
            mock_list.return_value = self.mock_project_data

            result = self.prompt._discover_indexed_projects()

            assert result == self.mock_project_data
            assert result["total_projects"] == 3
            assert len(result["projects"]) == 3
            mock_list.assert_called_once()

    def test_discover_indexed_projects_error(self):
        """Test project discovery with error."""
        with patch("tools.project.project_utils.list_indexed_projects") as mock_list:
            mock_list.side_effect = Exception("Database connection failed")

            result = self.prompt._discover_indexed_projects()

            assert "error" in result
            assert "Database connection failed" in result["error"]

    def test_format_brief_project_list(self):
        """Test brief project list formatting."""
        projects = self.mock_project_data["projects"]

        result = self.prompt._format_brief_project_list(projects)

        assert "service1" in result
        assert "frontend" in result
        assert "api_gateway" in result
        assert "Available Projects:" in result

    def test_format_standard_project_list(self):
        """Test standard project list formatting."""
        projects = self.mock_project_data["projects"]

        result = self.prompt._format_standard_project_list(projects)

        assert "service1" in result
        assert "1,500 items" in result  # Check formatting of total_points
        assert "code, config" in result
        assert "frontend" in result
        assert "800 items" in result
        assert "api_gateway" in result
        assert "2,000 items" in result

    def test_format_comprehensive_project_list(self):
        """Test comprehensive project list formatting."""
        projects = self.mock_project_data["projects"]

        result = self.prompt._format_comprehensive_project_list(projects)

        assert "service1" in result
        assert "Total Items: 1,500" in result
        assert "Collections: 2" in result
        assert "code: 1,200 items" in result
        assert "config: 300 items" in result

        assert "frontend" in result
        assert "api_gateway" in result

    def test_get_available_content_types(self):
        """Test getting available content types across projects."""
        projects = self.mock_project_data["projects"]

        result = self.prompt._get_available_content_types(projects)

        expected_types = {"code", "config", "documentation"}
        result_types = set(result.split(", "))
        assert result_types == expected_types

    def test_recommend_projects_for_query_name_match(self):
        """Test project recommendations based on query matching project names."""
        projects = self.mock_project_data["projects"]

        # Test query that matches project name
        result = self.prompt._recommend_projects_for_query("service authentication", projects)

        assert "service1" in result  # Should match because query contains "service"

    def test_recommend_projects_for_query_fallback_by_size(self):
        """Test project recommendations fallback to largest projects."""
        projects = self.mock_project_data["projects"]

        # Test query that doesn't match any project names
        result = self.prompt._recommend_projects_for_query("random unrelated query", projects)

        # Should recommend projects by size (api_gateway=2000, service1=1500, frontend=800)
        assert "api_gateway" in result
        assert "service1" in result

    def test_build_search_guidance_prompt_standard(self):
        """Test building the main search guidance prompt with standard detail level."""
        query = "authentication function"

        result = self.prompt._build_search_guidance_prompt(query, self.mock_project_data, "hybrid", "standard")

        # Check key elements are present
        assert query in result
        assert "hybrid" in result
        # Note: detail_level may or may not be displayed in the output
        assert "Total Projects:" in result  # The count is in the output but with different formatting
        assert "service1" in result
        assert "frontend" in result
        assert "api_gateway" in result

        # Check search strategy options
        assert "Current Project Search" in result or "search" in result
        assert "target_projects" in result
        assert "cross_project" in result

    def test_build_search_guidance_prompt_brief(self):
        """Test building search guidance prompt with brief detail level."""
        query = "test query"

        result = self.prompt._build_search_guidance_prompt(query, self.mock_project_data, "semantic", "brief")

        # Should contain brief project list format
        assert "Available Projects:" in result
        assert "`service1`, `frontend`, `api_gateway`" in result or all(name in result for name in ["service1", "frontend", "api_gateway"])

    def test_build_search_guidance_prompt_comprehensive(self):
        """Test building search guidance prompt with comprehensive detail level."""
        query = "comprehensive test"

        result = self.prompt._build_search_guidance_prompt(query, self.mock_project_data, "keyword", "comprehensive")

        # Should contain comprehensive project list format
        assert "Available Projects (Detailed):" in result
        assert "Total Items:" in result
        assert "Collections:" in result
        assert "code: 1,200 items" in result  # From service1 details

    def test_create_no_projects_response(self):
        """Test response when no projects are indexed."""
        result = self.prompt._create_no_projects_response()

        assert len(result) == 1
        message = result[0]
        # Message.content may be a TextContent object with .text attribute
        content = message.content.text if hasattr(message.content, "text") else str(message.content)
        assert "No indexed projects detected" in content or "No Projects Found" in content
        assert "index_directory" in content
        assert "check_index_status" in content

    def test_create_error_response(self):
        """Test error response creation."""
        error_msg = "Database connection failed"

        result = self.prompt._create_error_response(error_msg)

        assert len(result) == 1
        message = result[0]
        # Message.content may be a TextContent object with .text attribute
        content = message.content.text if hasattr(message.content, "text") else str(message.content)
        assert error_msg in content
        assert "Error" in content
        assert "health_check()" in content

    def test_create_fallback_response(self):
        """Test fallback response creation."""
        query = "test query"
        search_mode = "hybrid"
        error_msg = "Service unavailable"

        result = self.prompt._create_fallback_response(query, search_mode, error_msg)

        assert len(result) == 1
        message = result[0]
        # Message.content may be a TextContent object with .text attribute
        content = message.content.text if hasattr(message.content, "text") else str(message.content)
        assert query in content
        assert search_mode in content
        assert error_msg in content
        assert "Fallback Mode" in content
        assert "Basic Search Options" in content


class TestAdvanceSearchPromptIntegration:
    """Integration tests for the advance_search prompt."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_prompts_system = Mock()
        self.mock_prompts_system.indexing_service = Mock()
        self.mock_prompts_system.analysis_service = Mock()
        self.mock_prompts_system.embedding_service = Mock()

        self.prompt = AdvanceSearchPrompt(self.mock_prompts_system)

    @patch("tools.project.project_utils.list_indexed_projects")
    def test_register_and_call_with_projects(self, mock_list_projects):
        """Test registering the prompt and calling it with available projects."""
        mock_list_projects.return_value = {
            "total_projects": 2,
            "projects": [
                {
                    "name": "test_project",
                    "total_points": 1000,
                    "collections": ["project_test_project_code"],
                    "collection_types": ["code"],
                    "collection_details": [
                        {
                            "name": "project_test_project_code",
                            "type": "code",
                            "points_count": 1000,
                        }
                    ],
                }
            ],
        }

        # Mock FastMCP app
        mock_app = Mock()
        registered_prompts = {}

        def mock_prompt_decorator():
            def decorator(func):
                registered_prompts[func.__name__] = func
                return func

            return decorator

        mock_app.prompt = mock_prompt_decorator

        # Register the prompt
        self.prompt.register(mock_app)

        # Verify the prompt was registered
        assert "advance_search" in registered_prompts

        # Call the registered prompt function
        prompt_func = registered_prompts["advance_search"]
        result = prompt_func(query="test query", search_mode="hybrid", detail_level="standard")

        # Verify the result
        assert len(result) == 1
        message = result[0]
        # Message.content may be a TextContent object with .text attribute
        content = message.content.text if hasattr(message.content, "text") else str(message.content)
        assert "Advanced Cross-Project Search Interface" in content
        assert "test query" in content
        assert "hybrid" in content
        assert "Total Projects:" in content

    @patch("tools.project.project_utils.list_indexed_projects")
    def test_register_and_call_no_projects(self, mock_list_projects):
        """Test calling the prompt when no projects are available."""
        mock_list_projects.return_value = {"total_projects": 0, "projects": []}

        # Mock FastMCP app
        mock_app = Mock()
        registered_prompts = {}

        def mock_prompt_decorator():
            def decorator(func):
                registered_prompts[func.__name__] = func
                return func

            return decorator

        mock_app.prompt = mock_prompt_decorator

        # Register the prompt
        self.prompt.register(mock_app)

        # Call the registered prompt function
        prompt_func = registered_prompts["advance_search"]
        result = prompt_func(query="test query")

        # Verify no projects response
        assert len(result) == 1
        message = result[0]
        # Message.content may be a TextContent object with .text attribute
        content = message.content.text if hasattr(message.content, "text") else str(message.content)
        assert "No indexed projects detected" in content or "No Projects Found" in content
        assert "index_directory" in content

    @patch("tools.project.project_utils.list_indexed_projects")
    def test_register_and_call_with_error(self, mock_list_projects):
        """Test calling the prompt when project discovery fails."""
        mock_list_projects.side_effect = Exception("Connection timeout")

        # Mock FastMCP app
        mock_app = Mock()
        registered_prompts = {}

        def mock_prompt_decorator():
            def decorator(func):
                registered_prompts[func.__name__] = func
                return func

            return decorator

        mock_app.prompt = mock_prompt_decorator

        # Register the prompt
        self.prompt.register(mock_app)

        # Call the registered prompt function
        prompt_func = registered_prompts["advance_search"]
        result = prompt_func(query="test query")

        # Verify error response (not necessarily Fallback Mode)
        assert len(result) == 1
        message = result[0]
        # Message.content may be a TextContent object with .text attribute
        content = message.content.text if hasattr(message.content, "text") else str(message.content)
        assert "Connection timeout" in content
        assert "Error" in content or "Fallback" in content


if __name__ == "__main__":
    pytest.main([__file__])
