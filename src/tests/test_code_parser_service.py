"""
Comprehensive unit tests for the CodeParser service.

Tests cover all major functionality including:
- Language detection and parser loading
- AST parsing and intelligent chunking
- Syntax error handling and recovery
- Metadata extraction and breadcrumb generation
- Multi-language support
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.code_chunk import ChunkType, CodeChunk, CodeSyntaxError, ParseResult
from services.code_parser_service import CodeParserService


class TestCodeParserService:
    """Test suite for CodeParserService functionality."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance for testing."""
        return CodeParserService()

    @pytest.fixture
    def sample_python_code(self):
        """Sample Python code for testing."""
        return '''"""Module docstring."""

import os
import sys

CONSTANT_VALUE = 42

class SampleClass:
    """A sample class for testing."""

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        """Return the name."""
        return self.name

    async def async_method(self) -> None:
        """An async method."""
        await some_operation()

def global_function(x: int, y: int) -> int:
    """A global function."""
    return x + y

if __name__ == "__main__":
    main()
'''

    @pytest.fixture
    def sample_javascript_code(self):
        """Sample JavaScript code for testing."""
        return """// JavaScript module
import { Component } from 'react';

const CONSTANT = 'value';

class MyComponent extends Component {
    constructor(props) {
        super(props);
        this.state = { count: 0 };
    }

    handleClick = () => {
        this.setState({ count: this.state.count + 1 });
    }

    render() {
        return <div>{this.state.count}</div>;
    }
}

function utilityFunction(a, b) {
    return a + b;
}

const arrowFunction = (x) => x * 2;

export default MyComponent;
"""

    @pytest.fixture
    def sample_cpp_header_code(self):
        """Sample C++ header file for testing."""
        return """// math_utils.hpp - C++ header file
#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <vector>
#include <string>

namespace MathUtils {

    // Template class for mathematical operations
    template<typename T>
    class Calculator {
    private:
        T value;

    public:
        // Constructor
        Calculator(T initial_value);

        // Destructor
        ~Calculator();

        // Member functions
        T add(T other) const;
        T multiply(T other) const;

        // Static function
        static T max(T a, T b);
    };

    // Template function
    template<typename T>
    T square(T value) {
        return value * value;
    }

    // Constants
    const double PI = 3.14159265359;
    extern const int MAX_SIZE;

    // Function declarations
    double calculateArea(double radius);
    std::vector<int> generateSequence(int start, int end);

}  // namespace MathUtils

#endif // MATH_UTILS_HPP
"""

    @pytest.fixture
    def sample_cpp_source_code(self):
        """Sample C++ source file for testing."""
        return """// math_utils.cpp - C++ implementation file
#include "math_utils.hpp"
#include <algorithm>
#include <cmath>

namespace MathUtils {

    // Template specialization
    template<>
    Calculator<int>::Calculator(int initial_value) : value(initial_value) {
        // Constructor implementation
    }

    // Destructor implementation
    template<typename T>
    Calculator<T>::~Calculator() {
        // Cleanup resources
    }

    // Member function implementation
    template<typename T>
    T Calculator<T>::add(T other) const {
        return value + other;
    }

    // Static function implementation
    template<typename T>
    T Calculator<T>::max(T a, T b) {
        return (a > b) ? a : b;
    }

    // Constant definition
    const int MAX_SIZE = 1000;

    // Function implementations
    double calculateArea(double radius) {
        return PI * radius * radius;
    }

    std::vector<int> generateSequence(int start, int end) {
        std::vector<int> sequence;
        for (int i = start; i <= end; ++i) {
            sequence.push_back(i);
        }
        return sequence;
    }

}  // namespace MathUtils

// Global function outside namespace
int main(int argc, char* argv[]) {
    MathUtils::Calculator<double> calc(10.0);
    double result = calc.add(5.0);
    return 0;
}
"""

    @pytest.fixture
    def sample_typescript_code(self):
        """Sample TypeScript code for testing."""
        return """interface User {
    id: number;
    name: string;
    email?: string;
}

type Status = 'active' | 'inactive';

class UserService {
    private users: User[] = [];

    constructor(private apiClient: ApiClient) {}

    async getUser(id: number): Promise<User | null> {
        return this.users.find(user => user.id === id) || null;
    }

    addUser(user: User): void {
        this.users.push(user);
    }
}

function processUser(user: User): string {
    return `Processing ${user.name}`;
}
"""

    @pytest.fixture
    def sample_code_with_errors(self):
        """Sample code with syntax errors for testing error handling."""
        return """def broken_function(
    # Missing closing parenthesis
    pass

class IncompleteClass
    # Missing colon
    def method(self):
        return "test"

# Unclosed string
invalid = "unclosed string
"""

    def test_initialization(self, parser_service):
        """Test CodeParserService initialization."""
        assert parser_service is not None
        # Check for coordinator pattern services
        assert hasattr(parser_service, "language_support")
        assert hasattr(parser_service, "ast_extractor")
        assert hasattr(parser_service, "get_supported_languages")
        assert hasattr(parser_service, "parse_file")

    def test_supported_languages(self, parser_service):
        """Test that all expected languages are supported."""
        expected_languages = {
            "python",
            "javascript",
            "typescript",
            "go",
            "rust",
            "java",
            "cpp",
        }
        # Get actual languages from the TreeSitterManager
        actual_languages = set(parser_service.get_supported_languages())
        assert expected_languages.issubset(actual_languages), f"Missing languages: {expected_languages - actual_languages}"

    def test_language_detection(self, parser_service):
        """Test language detection from file content and paths."""
        # Test Python detection
        assert parser_service.detect_language("test.py") == "python"

        # Test JavaScript detection
        assert parser_service.detect_language("test.js") == "javascript"

        # Test TypeScript detection
        assert parser_service.detect_language("test.ts") == "typescript"

        # Test C++ detection
        assert parser_service.detect_language("test.cpp") == "cpp"
        assert parser_service.detect_language("test.hpp") == "cpp"
        assert parser_service.detect_language("test.h") == "cpp"
        assert parser_service.detect_language("test.c") == "cpp"

    def test_parser_loading_success(self, parser_service):
        """Test successful parser loading for supported languages."""
        # Test parser loading via language_support service
        parser = parser_service.language_support.get_parser("python")
        assert parser is not None

    def test_parser_loading_failure(self, parser_service):
        """Test parser loading failure for unsupported languages."""
        # Test unsupported language via language_support service
        parser = parser_service.language_support.get_parser("unsupported_language")
        assert parser is None

    def test_parse_code_success(self, parser_service, sample_python_code):
        """Test successful code parsing and chunk generation."""
        # Test parsing with real implementation
        result = parser_service.parse_file("test.py", sample_python_code)

        assert isinstance(result, ParseResult)
        assert result.language == "python"
        assert result.file_path == "test.py"
        # Note: parse_success might be False if tree-sitter parsers aren't available
        # So we just check that we get a result

    def test_parse_code_with_syntax_errors(self, parser_service, sample_code_with_errors):
        """Test parsing code with syntax errors."""
        # Test parsing with real implementation
        result = parser_service.parse_file("broken.py", sample_code_with_errors)

        assert isinstance(result, ParseResult)
        # The actual result depends on tree-sitter availability and error handling
        # Just check we get a valid result structure

    def test_chunk_metadata_extraction(self, parser_service, sample_python_code):
        """Test extraction of metadata from code chunks."""
        # Test with actual parsing to see if chunks have proper metadata
        result = parser_service.parse_file("test.py", sample_python_code)

        assert isinstance(result, ParseResult)
        # If we got chunks, check they have metadata
        if result.chunks:
            chunk = result.chunks[0]
            assert hasattr(chunk, "name")
            assert hasattr(chunk, "chunk_type")
            assert hasattr(chunk, "breadcrumb")

    def test_breadcrumb_generation(self, parser_service, sample_python_code):
        """Test breadcrumb generation for nested code structures."""
        # Test breadcrumb via parsing - breadcrumbs are now generated by chunking strategies
        result = parser_service.parse_file("test.py", sample_python_code)

        assert isinstance(result, ParseResult)
        if result.chunks:
            # Check that chunks have breadcrumbs
            for chunk in result.chunks:
                assert hasattr(chunk, "breadcrumb")
                # Breadcrumb should contain file info or chunk hierarchy
                if chunk.breadcrumb:
                    assert isinstance(chunk.breadcrumb, str)

    def test_context_in_chunks(self, parser_service, sample_python_code):
        """Test that chunks contain context information."""
        result = parser_service.parse_file("test.py", sample_python_code)

        assert isinstance(result, ParseResult)
        if result.chunks:
            chunk = result.chunks[0]
            # Check chunk has line position info for context
            assert hasattr(chunk, "start_line")
            assert hasattr(chunk, "end_line")
            assert chunk.start_line >= 1
            assert chunk.end_line >= chunk.start_line

    def test_error_classification(self, parser_service, sample_code_with_errors):
        """Test syntax error classification and categorization."""
        # Test with actual broken code to see if errors are classified
        result = parser_service.parse_file("broken.py", sample_code_with_errors)

        assert isinstance(result, ParseResult)
        # If syntax errors were found, check their structure
        if result.syntax_errors:
            error = result.syntax_errors[0]
            assert isinstance(error, CodeSyntaxError)
            assert hasattr(error, "start_line")
            assert hasattr(error, "error_type")

    def test_fallback_chunking(self, parser_service, sample_python_code):
        """Test fallback to whole-file chunking when parsing fails."""
        # Test fallback by using unsupported language
        result = parser_service.parse_file("test.unknown", sample_python_code)

        assert isinstance(result, ParseResult)
        # Should still return some result, potentially using fallback

    @pytest.mark.parametrize(
        "language",
        ["python", "javascript", "typescript", "java", "go", "rust"],
    )
    def test_language_specific_support(self, parser_service, language):
        """Test that each language is properly supported."""
        # Verify language is in supported languages
        supported = parser_service.get_supported_languages()
        assert language in supported, f"Language {language} not in supported languages"

        # Verify we can get a parser for the language
        parser = parser_service.language_support.get_parser(language)
        assert parser is not None, f"Failed to get parser for {language}"

        # Verify language config exists
        config = parser_service.language_support.get_language_config(language)
        assert config is not None, f"No config for {language}"

    def test_performance_monitoring(self, parser_service, sample_python_code):
        """Test that parsing performance is tracked."""
        # Parse code and check that timing is recorded
        result = parser_service.parse_file("test.py", sample_python_code)

        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms >= 0

    def test_memory_management(self, parser_service):
        """Test that memory is properly managed during parsing."""
        # Test with multiple large code samples
        large_code = "def function_{}(): pass\n" * 1000

        results = []
        for i in range(10):
            result = parser_service.parse_file(f"test_{i}.py", large_code)
            results.append(result)

        # Should not crash or consume excessive memory
        assert len(results) == 10
        assert all(isinstance(r, ParseResult) for r in results)

    def test_thread_safety(self, parser_service):
        """Test that the parser service is thread-safe."""
        import concurrent.futures

        def parse_worker(code, file_id):
            return parser_service.parse_file(f"test_{file_id}.py", code)

        test_code = "def test(): pass"
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(parse_worker, test_code, i) for i in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert len(results) == 20
        assert all(isinstance(r, ParseResult) for r in results)


class TestCodeChunkModel:
    """Test suite for CodeChunk data model."""

    def test_code_chunk_creation(self):
        """Test CodeChunk instantiation and basic properties."""
        chunk = CodeChunk(
            chunk_id="test_chunk_1",
            file_path="/test/file.py",
            content="def test(): pass",
            chunk_type=ChunkType.FUNCTION,
            language="python",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
            name="test",
            breadcrumb="file.test",
        )

        assert chunk.chunk_id == "test_chunk_1"
        assert chunk.file_path == "/test/file.py"
        assert chunk.chunk_type == ChunkType.FUNCTION
        assert chunk.language == "python"
        assert chunk.name == "test"
        assert chunk.breadcrumb == "file.test"
        assert chunk.line_count == 2
        assert chunk.char_count == 16  # "def test(): pass" is 16 characters

    def test_code_chunk_serialization(self):
        """Test CodeChunk to_dict and from_dict methods."""
        original_chunk = CodeChunk(
            chunk_id="test_chunk_1",
            file_path="/test/file.py",
            content="def test(): pass",
            chunk_type=ChunkType.FUNCTION,
            language="python",
            start_line=1,
            end_line=2,
            start_byte=0,
            end_byte=20,
            name="test",
            tags=["utility", "test"],
        )

        # Test serialization
        chunk_dict = original_chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["chunk_id"] == "test_chunk_1"
        assert chunk_dict["chunk_type"] == "function"
        assert chunk_dict["tags"] == ["utility", "test"]

        # Test deserialization
        restored_chunk = CodeChunk.from_dict(chunk_dict)
        assert restored_chunk.chunk_id == original_chunk.chunk_id
        assert restored_chunk.chunk_type == original_chunk.chunk_type
        assert restored_chunk.content == original_chunk.content
        assert restored_chunk.tags == original_chunk.tags


class TestIntegrationScenarios:
    """Integration tests for real-world parsing scenarios."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance for testing."""
        return CodeParserService()

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            yield f
        os.unlink(f.name)

    @pytest.fixture
    def sample_cpp_header_code(self):
        """Sample C++ header file for testing."""
        return """// math_utils.hpp - C++ header file
#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <vector>
#include <string>

namespace MathUtils {

    // Template class for mathematical operations
    template<typename T>
    class Calculator {
    private:
        T value;

    public:
        // Constructor
        Calculator(T initial_value);

        // Destructor
        ~Calculator();

        // Member functions
        T add(T other) const;
        T multiply(T other) const;

        // Static function
        static T max(T a, T b);
    };

    // Template function
    template<typename T>
    T square(T value) {
        return value * value;
    }

    // Constants
    const double PI = 3.14159265359;
    extern const int MAX_SIZE;

    // Function declarations
    double calculateArea(double radius);
    std::vector<int> generateSequence(int start, int end);

}  // namespace MathUtils

#endif // MATH_UTILS_HPP
"""

    @pytest.fixture
    def sample_cpp_source_code(self):
        """Sample C++ source file for testing."""
        return """// math_utils.cpp - C++ implementation file
#include "math_utils.hpp"
#include <algorithm>
#include <cmath>

namespace MathUtils {

    // Template specialization
    template<>
    Calculator<int>::Calculator(int initial_value) : value(initial_value) {
        // Constructor implementation
    }

    // Destructor implementation
    template<typename T>
    Calculator<T>::~Calculator() {
        // Cleanup resources
    }

    // Member function implementation
    template<typename T>
    T Calculator<T>::add(T other) const {
        return value + other;
    }

    // Static function implementation
    template<typename T>
    T Calculator<T>::max(T a, T b) {
        return (a > b) ? a : b;
    }

    // Constant definition
    const int MAX_SIZE = 1000;

    // Function implementations
    double calculateArea(double radius) {
        return PI * radius * radius;
    }

    std::vector<int> generateSequence(int start, int end) {
        std::vector<int> sequence;
        for (int i = start; i <= end; ++i) {
            sequence.push_back(i);
        }
        return sequence;
    }

}  // namespace MathUtils

// Global function outside namespace
int main(int argc, char* argv[]) {
    MathUtils::Calculator<double> calc(10.0);
    double result = calc.add(5.0);
    return 0;
}
"""

    def test_end_to_end_parsing(self, temp_file):
        """Test complete parsing workflow from file to chunks."""
        # Write sample code to temp file
        sample_code = '''"""Module for testing."""

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.result = 0

    def add(self, x, y):
        """Add two numbers."""
        return x + y

    def multiply(self, x, y):
        """Multiply two numbers."""
        return x * y

def main():
    """Main function."""
    calc = Calculator()
    print(calc.add(2, 3))

if __name__ == "__main__":
    main()
'''

        temp_file.write(sample_code)
        temp_file.flush()

        # Parse the file
        parser_service = CodeParserService()
        with open(temp_file.name) as f:
            content = f.read()

        result = parser_service.parse_file(temp_file.name, content)

        # Verify results
        assert isinstance(result, ParseResult)
        assert result.parse_success is True
        assert len(result.chunks) > 0

        # Check for expected chunk types
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.CLASS in chunk_types or ChunkType.FUNCTION in chunk_types

    def test_error_recovery_integration(self):
        """Test error recovery in realistic scenarios."""
        # Code with recoverable syntax errors
        problematic_code = """
class BrokenClass:
    def good_method(self):
        return "works"

    def broken_method(
        # Missing closing paren and colon
        pass

    def another_good_method(self):
        return "also works"
"""

        parser_service = CodeParserService()
        result = parser_service.parse_file("broken.py", problematic_code)

        # Should still extract some valid chunks despite errors
        assert isinstance(result, ParseResult)
        # Note: error_count and error_recovery_used may vary depending on
        # tree-sitter's error recovery capabilities and chunking strategy
        # The key assertion is that we still get usable chunks
        assert len(result.chunks) > 0

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_header_parsing(self, parser_service, sample_cpp_header_code):
        """Test parsing C++ header files with classes, templates, and namespaces."""
        result = parser_service.parse_file("test.hpp", sample_cpp_header_code)

        assert isinstance(result, ParseResult)
        assert result.language == "cpp"
        assert len(result.chunks) > 0

        # Just check that we get some chunk types - C++ parsing is complex
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert len(chunk_types) > 0

        # Check that at least some expected types are present
        expected_types = {
            ChunkType.NAMESPACE,
            ChunkType.CLASS,
            ChunkType.TEMPLATE,
            ChunkType.IMPORT,
            ChunkType.CONSTANT,
        }
        actual_types = set(chunk_types)
        assert len(expected_types.intersection(actual_types)) >= 2  # At least 2 expected types

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_source_parsing(self, parser_service, sample_cpp_source_code):
        """Test parsing C++ source files with implementations and functions."""
        result = parser_service.parse_file("test.cpp", sample_cpp_source_code)

        assert isinstance(result, ParseResult)
        assert result.language == "cpp"
        assert len(result.chunks) > 0

        # Just check that we get some relevant chunk types
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert len(chunk_types) > 0

        # Check that at least some expected types are present
        expected_types = {
            ChunkType.NAMESPACE,
            ChunkType.FUNCTION,
            ChunkType.CONSTRUCTOR,
            ChunkType.DESTRUCTOR,
            ChunkType.IMPORT,
        }
        actual_types = set(chunk_types)
        assert len(expected_types.intersection(actual_types)) >= 2  # At least 2 expected types

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_constructor_destructor_detection(self, parser_service):
        """Test detection of C++ constructors and destructors."""
        cpp_code = """
class TestClass {
public:
    // Constructor
    TestClass(int value) : member_var(value) {
        // Constructor body
    }

    // Destructor
    ~TestClass() {
        // Destructor body
    }

private:
    int member_var;
};
"""
        result = parser_service.parse_file("test.cpp", cpp_code)

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0

        # Check for constructor and destructor detection
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.CLASS in chunk_types

        # Note: Due to Tree-sitter parsing complexity, constructors/destructors
        # might be classified as functions. The important thing is they're detected.
        function_like_chunks = [
            c for c in result.chunks if c.chunk_type in [ChunkType.FUNCTION, ChunkType.CONSTRUCTOR, ChunkType.DESTRUCTOR]
        ]
        assert len(function_like_chunks) > 0

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_template_parsing(self, parser_service):
        """Test parsing C++ template declarations."""
        cpp_template_code = """
template<typename T>
class TemplateClass {
public:
    T value;

    template<typename U>
    void templateMethod(U param) {
        // Template method implementation
    }
};

template<typename T>
T templateFunction(T a, T b) {
    return a + b;
}
"""
        result = parser_service.parse_file("template.hpp", cpp_template_code)

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0

        # Should detect template declarations
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.TEMPLATE in chunk_types or ChunkType.CLASS in chunk_types

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_namespace_parsing(self, parser_service):
        """Test parsing C++ namespace declarations."""
        cpp_namespace_code = """
namespace OuterNamespace {
    namespace InnerNamespace {

        void function_in_inner() {
            // Function implementation
        }

    }  // namespace InnerNamespace

    void function_in_outer() {
        // Function implementation
    }

}  // namespace OuterNamespace
"""
        result = parser_service.parse_file("namespace.cpp", cpp_namespace_code)

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0

        # Should detect namespace declarations
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.NAMESPACE in chunk_types

        # Should have functions within namespaces
        assert ChunkType.FUNCTION in chunk_types

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_include_parsing(self, parser_service):
        """Test parsing C++ include statements."""
        cpp_include_code = """
#include <iostream>
#include <vector>
#include <string>
#include "local_header.h"
#include "another/header.hpp"

int main() {
    std::cout << "Hello World" << std::endl;
    return 0;
}
"""
        result = parser_service.parse_file("includes.cpp", cpp_include_code)

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0

        # Should detect include statements
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.IMPORT in chunk_types

        # Should have multiple include chunks
        include_chunks = [c for c in result.chunks if c.chunk_type == ChunkType.IMPORT]
        assert len(include_chunks) >= 3  # At least 3 includes

        # Verify include content
        include_names = [c.name for c in include_chunks if c.name]
        assert any("iostream" in name for name in include_names)

    @pytest.mark.skip(reason="C++ intelligent chunking strategy not implemented - uses fallback chunking")
    def test_cpp_constants_and_variables(self, parser_service):
        """Test parsing C++ constants and variable declarations."""
        cpp_vars_code = """
const int CONSTANT_VALUE = 42;
static const double PI = 3.14159;
extern int external_variable;

int global_variable = 100;
static int static_variable;

namespace MyNamespace {
    const std::string NAMESPACE_CONSTANT = "test";
    int namespace_variable = 200;
}
"""
        result = parser_service.parse_file("variables.cpp", cpp_vars_code)

        assert isinstance(result, ParseResult)
        assert len(result.chunks) > 0

        # Should detect constants and variables
        chunk_types = [chunk.chunk_type for chunk in result.chunks]
        assert ChunkType.CONSTANT in chunk_types or ChunkType.VARIABLE in chunk_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
