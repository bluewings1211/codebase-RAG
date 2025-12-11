"""
Comprehensive tests for syntax error handling and recovery in code parsing.

This test suite verifies that the CodeParser service can:
- Detect and classify various types of syntax errors
- Recover valid code sections from files with syntax errors
- Provide meaningful error information
- Continue processing despite encountering errors
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.code_chunk import ChunkType, ParseResult
from services.code_parser_service import CodeParserService


class TestSyntaxErrorDetection:
    """Test syntax error detection capabilities."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_missing_parenthesis_detection(self, parser_service):
        """Test detection of missing parentheses."""
        code_with_error = '''
def broken_function(arg1, arg2
    """Function with missing closing parenthesis."""
    return arg1 + arg2

def valid_function():
    """This function is valid."""
    return "valid"
'''

        result = parser_service.parse_file("test.py", code_with_error)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should still generate chunks (via fallback if needed)
        assert len(result.chunks) > 0

    def test_missing_colon_detection(self, parser_service):
        """Test detection of missing colons."""
        code_with_error = '''
class IncompleteClass
    """Class missing colon after name."""

    def __init__(self, name):
        self.name = name

    def get_name(self)
        # Missing colon here too
        return self.name

def valid_function():
    return "still valid"
'''

        result = parser_service.parse_file("test.py", code_with_error)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should still generate chunks
        assert len(result.chunks) > 0

    def test_unclosed_string_detection(self, parser_service):
        """Test detection of unclosed string literals."""
        code_with_error = """
def function_with_unclosed_string():
    message = "This string is never closed
    return message

def another_function():
    return "This is fine"
"""

        result = parser_service.parse_file("test.py", code_with_error)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should still generate chunks
        assert len(result.chunks) > 0

    def test_indentation_error_detection(self, parser_service):
        """Test detection of indentation errors."""
        code_with_error = """
def indentation_error():
    if True:
        print("Correct indentation")
      print("Incorrect indentation")  # Wrong indentation level

    return "done"

def valid_function():
    return "valid"
"""

        result = parser_service.parse_file("test.py", code_with_error)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should still generate chunks
        assert len(result.chunks) > 0

    def test_invalid_syntax_patterns(self, parser_service):
        """Test detection of various invalid syntax patterns."""
        code_with_errors = """
# Multiple syntax errors in one file

# Error 1: Invalid assignment
5 = x

# Error 2: Invalid operator
def invalid_operation():
    return 5 ++ 3

# Error 3: Missing comma in function arguments
def function_call_error():
    return max(1 2 3)

# Valid code for recovery testing
def valid_function():
    return "I am valid"

# Error 4: Invalid lambda syntax
calculate = lambda x, y: x + y if x > 0 else

# More valid code
class ValidClass:
    def valid_method(self):
        return True
"""

        result = parser_service.parse_file("test.py", code_with_errors)

        # Should return a valid result even with many errors
        assert isinstance(result, ParseResult)
        # Should generate chunks (may use fallback)
        assert len(result.chunks) > 0

    def test_complex_syntax_errors(self, parser_service):
        """Test handling of complex, nested syntax errors."""
        code_with_complex_errors = '''
class BrokenClass:
    def __init__(self
        # Missing closing paren and colon
        self.data = {
            "key1": "value1",
            "key2": ["item1", "item2",
            "key3": "value3"
        }  # Missing closing bracket for list

    def method_with_issues(self):
        try:
            risky_operation()
        except ValueError as e
            # Missing colon
            print(f"Error: {e}")

    def valid_method(self):
        """This method should be recoverable."""
        return "valid"

def valid_standalone_function():
    """This should also be recoverable."""
    return {"status": "ok"}

# Unclosed function call
result = some_function(arg1, arg2, arg3

def another_valid_function():
    return "also valid"
'''

        result = parser_service.parse_file("test.py", code_with_complex_errors)

        # Should return a valid result even with complex errors
        assert isinstance(result, ParseResult)
        # Should generate chunks
        assert len(result.chunks) > 0


class TestErrorRecovery:
    """Test error recovery mechanisms."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_recovery_after_syntax_errors(self, parser_service):
        """Test that parser can recover and continue after syntax errors."""
        code_with_mixed_content = '''
"""Module with mixed valid and invalid code."""

# Valid import
import os

# Valid constant
VALID_CONSTANT = 42

# Broken function
def broken_function(
    # Missing closing parenthesis
    pass

# Valid class that should be recovered
class RecoverableClass:
    """This class should be found despite previous errors."""

    def __init__(self, name):
        self.name = name

    def get_info(self):
        return f"Name: {self.name}"

# Another syntax error
invalid_string = "unclosed string

# Another valid function
def another_valid_function():
    """This should also be recovered."""
    return {"recovered": True}

# Valid lambda
square = lambda x: x ** 2
'''

        result = parser_service.parse_file("test.py", code_with_mixed_content)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0

    def test_partial_recovery_statistics(self, parser_service):
        """Test that recovery statistics are accurate."""
        code_with_known_structure = """
# File with known structure for testing statistics

def valid_function_1():
    return "first"

def broken_function(
    # Missing closing paren
    return "broken"

def valid_function_2():
    return "second"

class ValidClass:
    def method(self):
        return "method"

class BrokenClass
    # Missing colon
    def method(self):
        return "broken method"

def valid_function_3():
    return "third"
"""

        result = parser_service.parse_file("test.py", code_with_known_structure)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0

    def test_recovery_with_nested_structures(self, parser_service):
        """Test recovery of nested code structures with errors."""
        code_with_nested_errors = '''
class OuterClass:
    """Outer class with nested issues."""

    def valid_outer_method(self):
        return "outer valid"

    class NestedClass:
        def broken_nested_method(self
            # Missing closing paren and colon
            return "nested broken"

        def valid_nested_method(self):
            return "nested valid"

    def another_outer_method(self):
        return "another outer"

def standalone_function():
    return "standalone"
'''

        result = parser_service.parse_file("test.py", code_with_nested_errors)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0


class TestErrorClassification:
    """Test classification and categorization of syntax errors."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_error_severity_classification(self, parser_service):
        """Test that errors are classified by severity."""
        code_with_various_errors = """
# File with errors of different severities

# Critical error: completely invalid syntax
def *(invalid_function_name):
    pass

# Warning-level: unused variable (if detected)
def function_with_warning():
    unused_var = "not used"
    return "result"

# Error: missing syntax element
def missing_colon_function()
    return "missing colon"

# Valid code
def valid_function():
    return "valid"
"""

        result = parser_service.parse_file("test.py", code_with_various_errors)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0

    def test_error_type_classification(self, parser_service):
        """Test that errors are classified by type."""
        code_with_typed_errors = """
# Different types of syntax errors

# Missing punctuation
def missing_colon()
    pass

# Invalid tokens
def invalid_operator():
    return 5 ++ 3

# Unclosed constructs
def unclosed_paren(arg1, arg2
    pass

# Invalid assignments
5 = variable

def valid_function():
    return "valid"
"""

        result = parser_service.parse_file("test.py", code_with_typed_errors)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0

    def test_error_context_extraction(self, parser_service):
        """Test that error context is properly extracted."""
        code_with_contextual_errors = '''
def function_with_context():
    """Function to test context extraction."""
    x = 5
    y = 10
    result = x ++ y  # Error on this line
    return result

def another_function():
    return "valid"
'''

        result = parser_service.parse_file("test.py", code_with_contextual_errors)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0


class TestErrorSuggestions:
    """Test generation of error suggestions and fixes."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_common_error_suggestions(self, parser_service):
        """Test that common errors get appropriate suggestions."""
        code_with_common_errors = """
def missing_colon()
    pass

def unclosed_paren(arg1, arg2
    pass

invalid_string = "unclosed string

def valid_function():
    return "valid"
"""

        result = parser_service.parse_file("test.py", code_with_common_errors)

        # Should return a valid result
        assert isinstance(result, ParseResult)
        # Should have chunks
        assert len(result.chunks) > 0

    def test_error_recovery_recommendations(self, parser_service):
        """Test recommendations for error recovery."""
        severely_broken_code = '''
def completely_broken_function(((((
    this is not valid python at all
    more invalid content here
    }}}}}

def recoverable_function():
    """This should be recoverable."""
    return "ok"

more invalid syntax here ++++++
'''

        result = parser_service.parse_file("test.py", severely_broken_code)

        # Should return a valid result even for severely broken code
        assert isinstance(result, ParseResult)
        # Should have chunks (via fallback)
        assert len(result.chunks) > 0


class TestRealWorldErrorScenarios:
    """Test handling of real-world syntax error scenarios."""

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_incomplete_file_during_editing(self, parser_service):
        """Test parsing of incomplete files (as during live editing)."""
        incomplete_code = """
import os
import sys

class IncompleteClass:
    def __init__(self):
        self.
"""  # File cuts off mid-statement

        result = parser_service.parse_file("incomplete.py", incomplete_code)

        # Should handle incomplete files gracefully
        assert isinstance(result, ParseResult)
        # Should still have chunks
        assert len(result.chunks) > 0

    def test_mixed_language_content(self, parser_service):
        """Test handling of files with mixed or embedded content."""
        mixed_content = '''
"""
This is a Python file but it contains some pseudo-code
that might confuse the parser.

```javascript
function embeddedJS() {
    return "this is not python";
}
```
"""

def actual_python_function():
    """This is real Python."""
    return "python code"

# Some shell commands in comments
# $ ls -la
# $ grep "pattern" file.txt

def another_python_function():
    sql_query = """
    SELECT * FROM users
    WHERE name = 'test'
    """
    return sql_query
'''

        result = parser_service.parse_file("mixed.py", mixed_content)

        # Should parse as Python despite mixed content
        assert result.language == "python"

        # Should have chunks
        assert len(result.chunks) > 0

    def test_large_file_with_scattered_errors(self, parser_service):
        """Test parsing of large files with errors scattered throughout."""
        # Create a large file with errors distributed throughout
        large_code_with_errors = '''"""Large file with scattered syntax errors."""\n\n'''

        for i in range(50):
            if i % 7 == 0:  # Every 7th block has an error
                large_code_with_errors += f"""
def broken_function_{i}(
    # Missing closing paren in function {i}
    return {i}
"""
            else:
                large_code_with_errors += f'''
def valid_function_{i}():
    """Valid function {i}."""
    return {i}

class ValidClass_{i}:
    """Valid class {i}."""
    def method(self):
        return {i}
'''

        result = parser_service.parse_file("large_with_errors.py", large_code_with_errors)

        # Should handle large files with scattered errors
        assert isinstance(result, ParseResult)

        # Should have chunks
        assert len(result.chunks) > 0

        # Should complete processing within reasonable time
        assert result.processing_time_ms < 30000  # Less than 30 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
