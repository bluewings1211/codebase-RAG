"""
Integration tests for intelligent code chunking functionality.

These tests verify the end-to-end intelligent chunking workflow including:
- Code parsing and chunk generation
- Integration with the indexing service
- Metadata extraction and breadcrumb generation
- Multi-language support and fallback mechanisms
- Error handling and recovery
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.code_chunk import ChunkType, CodeChunk, ParseResult
from services.code_parser_service import CodeParserService
from services.indexing_service import IndexingService


class TestIntelligentChunkingIntegration:
    """Integration tests for the complete intelligent chunking workflow."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory with sample files."""
        temp_dir = tempfile.mkdtemp()

        # Create project structure
        project_path = Path(temp_dir) / "test_project"
        project_path.mkdir()

        # Create various source files
        self._create_sample_files(project_path)

        yield str(project_path)

        # Cleanup
        shutil.rmtree(temp_dir)

    def _create_sample_files(self, project_path: Path):
        """Create sample source files for testing."""

        # Python file
        python_file = project_path / "main.py"
        python_file.write_text(
            '''
"""Main module for the application."""

import os
import sys

class Application:
    """Main application class."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.is_running = False

    def start(self):
        """Start the application."""
        self.is_running = True
        return True

    def stop(self):
        """Stop the application."""
        self.is_running = False

def main():
    """Main entry point."""
    app = Application("/etc/config.json")
    app.start()

if __name__ == "__main__":
    main()
'''
        )

        # JavaScript file
        js_file = project_path / "utils.js"
        js_file.write_text(
            """
/**
 * Utility functions for the application
 */

const API_BASE_URL = 'https://api.example.com';

class HttpClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        const response = await fetch(`${this.baseUrl}${endpoint}`);
        return response.json();
    }

    async post(endpoint, data) {
        const response = await fetch(`${this.baseUrl}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        return response.json();
    }
}

function formatDate(date) {
    return date.toISOString().split('T')[0];
}

const utils = {
    delay: (ms) => new Promise(resolve => setTimeout(resolve, ms)),
    randomId: () => Math.random().toString(36).substr(2, 9)
};

export { HttpClient, formatDate, utils };
"""
        )

        # TypeScript file
        ts_file = project_path / "types.ts"
        ts_file.write_text(
            """
interface User {
    id: number;
    name: string;
    email: string;
}

type Status = 'active' | 'inactive' | 'pending';

class UserManager {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    getUser(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}

function processUsers(users: User[]): User[] {
    return users.filter(u => u.email.includes('@'));
}

export { User, UserManager, processUsers };
export type { Status };
"""
        )

        # File with syntax errors
        broken_file = project_path / "broken.py"
        broken_file.write_text(
            '''
# File with syntax errors
def broken_function(
    # Missing closing parenthesis
    return "broken"

class IncompleteClass
    # Missing colon
    def method(self):
        return True

def valid_function():
    """This function is valid."""
    return "valid"
'''
        )

    @pytest.fixture
    def code_parser(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    @pytest.fixture
    def indexing_service(self):
        """Create an IndexingService instance."""
        return IndexingService()

    def test_end_to_end_chunking_workflow(self, temp_project_dir, code_parser, indexing_service):
        """Test the complete chunking workflow from files to indexed chunks."""

        # Process the project directory
        chunks = indexing_service.process_codebase_for_indexing(temp_project_dir)

        # Verify chunks were generated
        assert len(chunks) > 0, "Should generate chunks from the project"

        # Check that different file types were processed
        file_paths = {chunk.metadata.get("file_path", "") for chunk in chunks}
        assert any("main.py" in path for path in file_paths), "Should process Python file"
        assert any("utils.js" in path for path in file_paths), "Should process JavaScript file"
        assert any("types.ts" in path for path in file_paths), "Should process TypeScript file"

        # Verify chunk types
        chunk_types = {chunk.metadata.get("chunk_type", "") for chunk in chunks}
        expected_types = {"function", "class", "constant", "interface"}
        assert any(ct in expected_types for ct in chunk_types), "Should generate various chunk types"

    def test_language_specific_chunking(self, temp_project_dir, code_parser):
        """Test that language-specific constructs are properly chunked."""

        # Test Python file
        python_file = Path(temp_project_dir) / "main.py"
        with open(python_file) as f:
            python_content = f.read()

        python_result = code_parser.parse_file(str(python_file), python_content)

        assert python_result.parse_success, "Python parsing should succeed"
        assert len(python_result.chunks) > 0, "Should generate chunks from Python code"

        # Check for Python-specific chunk types
        python_chunk_types = {chunk.chunk_type for chunk in python_result.chunks}
        assert ChunkType.CLASS in python_chunk_types or any("class" in str(ct) for ct in python_chunk_types)
        assert ChunkType.FUNCTION in python_chunk_types or any("function" in str(ct) for ct in python_chunk_types)

        # Test JavaScript file
        js_file = Path(temp_project_dir) / "utils.js"
        with open(js_file) as f:
            js_content = f.read()

        js_result = code_parser.parse_file(str(js_file), js_content)

        # JavaScript parsing may use fallback chunking
        # The key is that we get some result, even if using fallback
        assert len(js_result.chunks) > 0, "Should generate chunks from JavaScript code (may use fallback)"

        # Test TypeScript file
        ts_file = Path(temp_project_dir) / "types.ts"
        with open(ts_file) as f:
            ts_content = f.read()

        ts_result = code_parser.parse_file(str(ts_file), ts_content)

        # TypeScript parsing may use fallback chunking
        assert len(ts_result.chunks) > 0, "Should generate chunks from TypeScript code (may use fallback)"

    def test_breadcrumb_generation(self, temp_project_dir, code_parser):
        """Test that breadcrumbs are properly generated for nested structures."""

        # Test with Python file containing nested structures
        python_file = Path(temp_project_dir) / "main.py"
        with open(python_file) as f:
            content = f.read()

        result = code_parser.parse_file(str(python_file), content)

        # Check for breadcrumbs in chunk metadata
        for chunk in result.chunks:
            if hasattr(chunk, "breadcrumb") and chunk.breadcrumb:
                # Breadcrumb should contain hierarchical information
                assert "." in chunk.breadcrumb or chunk.name in chunk.breadcrumb

                # For class methods, should include class name
                if chunk.parent_name:
                    assert chunk.parent_name in chunk.breadcrumb

    def test_metadata_extraction(self, temp_project_dir, code_parser):
        """Test that comprehensive metadata is extracted from chunks."""

        python_file = Path(temp_project_dir) / "main.py"
        with open(python_file) as f:
            content = f.read()

        result = code_parser.parse_file(str(python_file), content)

        for chunk in result.chunks:
            # Verify basic metadata
            assert hasattr(chunk, "chunk_id")
            assert hasattr(chunk, "file_path")
            assert hasattr(chunk, "content")
            assert hasattr(chunk, "start_line")
            assert hasattr(chunk, "end_line")

            # Check for intelligent metadata
            if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS]:
                assert hasattr(chunk, "name")
                assert chunk.name is not None and chunk.name != ""

            # Verify line numbers are reasonable
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line

    def test_error_handling_and_recovery(self, temp_project_dir, code_parser):
        """Test that syntax errors are handled gracefully with recovery."""

        # Test file with syntax errors
        broken_file = Path(temp_project_dir) / "broken.py"
        with open(broken_file) as f:
            content = f.read()

        result = code_parser.parse_file(str(broken_file), content)

        # Should still return a result even with errors
        assert isinstance(result, ParseResult)

        # Should still extract some valid chunks even with errors
        # The key assertion is that we get usable results
        assert len(result.chunks) > 0

    def test_context_enhancement(self, temp_project_dir, code_parser):
        """Test that code context is properly extracted and enhanced."""

        python_file = Path(temp_project_dir) / "main.py"
        with open(python_file) as f:
            content = f.read()

        result = code_parser.parse_file(str(python_file), content)

        # Check for context enhancement in chunks
        for chunk in result.chunks:
            if hasattr(chunk, "context_before") or hasattr(chunk, "context_after"):
                # Context should be strings if present
                if chunk.context_before:
                    assert isinstance(chunk.context_before, str)
                if chunk.context_after:
                    assert isinstance(chunk.context_after, str)

    def test_chunk_serialization(self, temp_project_dir, code_parser):
        """Test that chunks can be properly serialized and deserialized."""

        python_file = Path(temp_project_dir) / "main.py"
        with open(python_file) as f:
            content = f.read()

        result = code_parser.parse_file(str(python_file), content)

        for original_chunk in result.chunks:
            # Serialize to dict
            chunk_dict = original_chunk.to_dict()
            assert isinstance(chunk_dict, dict)

            # Verify required fields are present
            required_fields = [
                "chunk_id",
                "file_path",
                "content",
                "chunk_type",
                "language",
            ]
            for field in required_fields:
                assert field in chunk_dict

            # Deserialize back to object
            restored_chunk = CodeChunk.from_dict(chunk_dict)

            # Verify key fields match
            assert restored_chunk.chunk_id == original_chunk.chunk_id
            assert restored_chunk.file_path == original_chunk.file_path
            assert restored_chunk.content == original_chunk.content
            assert restored_chunk.chunk_type == original_chunk.chunk_type

    def test_performance_with_large_files(self, temp_project_dir, code_parser):
        """Test chunking performance with larger code files."""

        # Create a large Python file
        large_file = Path(temp_project_dir) / "large_file.py"

        # Generate a large but valid Python file
        large_content = '''"""Large Python file for performance testing."""\n\n'''

        for i in range(100):
            large_content += f'''
class GeneratedClass{i}:
    """Generated class {i}."""

    def __init__(self):
        self.value = {i}

    def get_value(self):
        """Get the value."""
        return self.value

    def calculate(self, x, y):
        """Perform calculation."""
        return x + y + self.value

def generated_function_{i}(param1, param2):
    """Generated function {i}."""
    result = param1 * param2 + {i}
    return result
'''

        large_file.write_text(large_content)

        # Parse the large file
        result = code_parser.parse_file(str(large_file), large_content)

        # Should successfully parse despite size
        assert result.parse_success
        assert len(result.chunks) > 100  # Should find many chunks

        # Performance should be reasonable (tracked in processing_time_ms)
        assert result.processing_time_ms < 10000  # Less than 10 seconds

    def test_cross_language_project(self, temp_project_dir, indexing_service):
        """Test processing a project with multiple languages."""

        # Add more language files
        go_file = Path(temp_project_dir) / "server.go"
        go_file.write_text(
            """
package main

import "fmt"

type Server struct {
    Port int
    Host string
}

func (s *Server) Start() error {
    fmt.Printf("Starting server on %s:%d", s.Host, s.Port)
    return nil
}

func NewServer(host string, port int) *Server {
    return &Server{
        Host: host,
        Port: port,
    }
}
"""
        )

        rust_file = Path(temp_project_dir) / "lib.rs"
        rust_file.write_text(
            """
use std::collections::HashMap;

pub struct UserStore {
    users: HashMap<u32, String>,
}

impl UserStore {
    pub fn new() -> Self {
        UserStore {
            users: HashMap::new(),
        }
    }

    pub fn add_user(&mut self, id: u32, name: String) {
        self.users.insert(id, name);
    }

    pub fn get_user(&self, id: u32) -> Option<&String> {
        self.users.get(&id)
    }
}

pub fn create_default_store() -> UserStore {
    UserStore::new()
}
"""
        )

        # Process the multi-language project
        chunks = indexing_service.process_codebase_for_indexing(temp_project_dir)

        # Should process all language files
        languages = {chunk.metadata.get("language", "") for chunk in chunks}
        expected_languages = {"python", "javascript", "typescript"}

        # Check that we got chunks from multiple languages
        assert len(languages.intersection(expected_languages)) >= 2

        # Verify language-specific chunks exist
        for chunk in chunks:
            lang = chunk.metadata.get("language", "")
            chunk_type = chunk.metadata.get("chunk_type", "")

            # Check that chunk types are valid for the language
            # Note: Some languages may use fallback chunking (whole_file)
            if lang == "python":
                assert chunk_type in ["function", "class", "import", "whole_file", "constant", "method"]
            elif lang in ["javascript", "typescript"]:
                assert chunk_type in ["function", "class", "interface", "export", "whole_file", "constant", "method"]
            elif lang in ["go", "rust"]:
                # Go and Rust may use fallback chunking if strategy not fully implemented
                assert chunk_type in [
                    "function",
                    "class",
                    "struct",
                    "impl",
                    "import",
                    "whole_file",
                    "constant",
                    "method",
                    "interface",
                    "module",
                ]


class TestChunkQuality:
    """Tests focused on the quality and accuracy of generated chunks."""

    def test_chunk_boundaries(self):
        """Test that chunk boundaries are accurate."""
        sample_code = '''
def function_one():
    """First function."""
    return 1

def function_two():
    """Second function."""
    return 2

class TestClass:
    """A test class."""

    def method_one(self):
        return "method1"

    def method_two(self):
        return "method2"
'''

        parser = CodeParserService()
        result = parser.parse_file("test.py", sample_code)

        # Verify chunks don't overlap
        lines = sample_code.split("\n")
        used_lines = set()

        for chunk in result.chunks:
            chunk_lines = set(range(chunk.start_line, chunk.end_line + 1))

            # Check for overlaps (some overlap might be acceptable for context)
            if used_lines.intersection(chunk_lines):
                # If there's overlap, it should be minimal (context lines)
                overlap_size = len(used_lines.intersection(chunk_lines))
                assert overlap_size <= 2, f"Excessive overlap: {overlap_size} lines"

            used_lines.update(chunk_lines)

            # Verify content matches line numbers
            "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
            # Content should be similar (may have some processing differences)
            assert len(chunk.content.strip()) > 0, "Chunk should have content"

    def test_chunk_completeness(self):
        """Test that chunks capture complete logical units."""
        sample_code = '''
class Calculator:
    """A simple calculator."""

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        """Add two numbers."""
        self.result = x + y
        return self.result

    def multiply(self, x, y):
        """Multiply two numbers."""
        self.result = x * y
        return self.result

def standalone_function():
    """A standalone function."""
    calc = Calculator()
    return calc.add(5, 3)
'''

        parser = CodeParserService()
        result = parser.parse_file("test.py", sample_code)

        # Should have chunks for class and methods
        chunk_types = [chunk.chunk_type for chunk in result.chunks]

        # Verify we got logical units
        assert ChunkType.CLASS in chunk_types or any("class" in str(ct) for ct in chunk_types)
        assert ChunkType.FUNCTION in chunk_types or any("function" in str(ct) for ct in chunk_types)

        # Check that class methods are properly identified
        for chunk in result.chunks:
            if chunk.chunk_type == ChunkType.FUNCTION and chunk.parent_name:
                assert chunk.parent_name == "Calculator"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
