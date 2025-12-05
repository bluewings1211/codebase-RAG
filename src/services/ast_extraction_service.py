"""
AST extraction service for intelligent code analysis using Tree-sitter.

This service provides a comprehensive set of utilities for extracting meaningful
information from Abstract Syntax Trees (AST), including node traversal, property
extraction, and semantic analysis.
"""

import logging

from models.code_chunk import ChunkType, CodeChunk, CodeSyntaxError
from tree_sitter import Node


class AstExtractionService:
    """
    Service for extracting information from Tree-sitter AST nodes.

    This service encapsulates all logic related to AST traversal, node analysis,
    property extraction, and semantic code chunk creation.
    """

    def __init__(self):
        """Initialize the AST extraction service."""
        self.logger = logging.getLogger(__name__)

    # =================== Core AST Traversal Methods ===================

    def extract_chunks(
        self,
        root_node: Node,
        file_path: str,
        content: str,
        language: str,
        node_mappings: dict[ChunkType, list[str]],
    ) -> list[CodeChunk]:
        """
        Extract code chunks from parsed AST with intelligent error detection.

        Args:
            root_node: Root node of the parsed AST
            file_path: Path to the source file
            content: Original file content
            language: Programming language
            node_mappings: Language-specific node type mappings

        Returns:
            List of extracted code chunks
        """
        chunks = []
        content_lines = content.split("\n")

        # Check for syntax errors and use appropriate traversal strategy
        error_count = self.count_errors(root_node)

        if error_count == 0:
            # Clean AST - use normal traversal
            self._traverse_ast(
                root_node,
                chunks,
                file_path,
                content,
                content_lines,
                language,
                node_mappings,
            )
        else:
            # AST has errors - use error-aware traversal
            error_lines = self._collect_error_lines(root_node)
            self._traverse_ast_with_error_filtering(
                root_node,
                chunks,
                file_path,
                content,
                content_lines,
                language,
                node_mappings,
                error_lines,
            )

        return chunks

    def traverse_ast(
        self,
        node: Node,
        chunks: list[CodeChunk],
        file_path: str,
        content: str,
        content_lines: list[str],
        language: str,
        node_mappings: dict[ChunkType, list[str]],
    ) -> None:
        """
        Recursively traverse AST nodes to extract meaningful chunks.

        Args:
            node: Current AST node
            chunks: List to collect extracted chunks
            file_path: Path to the source file
            content: Original file content
            content_lines: Content split into lines
            language: Programming language
            node_mappings: Language-specific node type mappings
        """
        self._traverse_ast(node, chunks, file_path, content, content_lines, language, node_mappings)

    def _traverse_ast(
        self,
        node: Node,
        chunks: list[CodeChunk],
        file_path: str,
        content: str,
        content_lines: list[str],
        language: str,
        node_mappings: dict[ChunkType, list[str]],
    ) -> None:
        """Core AST traversal implementation for normal parsing."""
        # Check if this node represents a meaningful code construct
        chunk_type = self.get_chunk_type(node, node_mappings, language)

        if chunk_type:
            # Create chunk from this node
            chunk = self.create_chunk_from_node(node, chunk_type, file_path, content, content_lines, language)
            if chunk:
                chunks.append(chunk)

        # Recursively process children
        for child in node.children:
            self._traverse_ast(
                child,
                chunks,
                file_path,
                content,
                content_lines,
                language,
                node_mappings,
            )

    def _traverse_ast_with_error_filtering(
        self,
        node: Node,
        chunks: list[CodeChunk],
        file_path: str,
        content: str,
        content_lines: list[str],
        language: str,
        node_mappings: dict[ChunkType, list[str]],
        error_lines: set[int],
    ) -> None:
        """Enhanced AST traversal that filters out nodes overlapping with syntax errors."""
        # Skip nodes that overlap with error lines
        node_start_line = node.start_point[0] + 1  # Tree-sitter is 0-indexed
        node_end_line = node.end_point[0] + 1

        # Check if this node overlaps with any error lines
        node_lines = set(range(node_start_line, node_end_line + 1))
        if node_lines.intersection(error_lines):
            # Node overlaps with errors, skip extraction but continue traversal
            pass
        else:
            # Node is clean, process normally
            chunk_type = self.get_chunk_type(node, node_mappings, language)
            if chunk_type:
                chunk = self.create_chunk_from_node(node, chunk_type, file_path, content, content_lines, language)
                if chunk:
                    chunks.append(chunk)

        # Recursively process children
        for child in node.children:
            self._traverse_ast_with_error_filtering(
                child,
                chunks,
                file_path,
                content,
                content_lines,
                language,
                node_mappings,
                error_lines,
            )

    def traverse_for_errors(self, node: Node, content_lines: list[str], language: str) -> list[CodeSyntaxError]:
        """
        Traverse AST to find and classify ERROR nodes.

        Args:
            node: Root node to traverse
            content_lines: Content split into lines
            language: Programming language

        Returns:
            List of detailed syntax errors
        """
        errors = []
        self._traverse_for_errors(node, content_lines, language, errors)
        return errors

    def _traverse_for_errors(
        self,
        node: Node,
        content_lines: list[str],
        language: str,
        errors: list[CodeSyntaxError],
    ) -> None:
        """Recursive implementation for error detection."""
        if node.type == "ERROR":
            # Create detailed error information
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            start_col = node.start_point[1] + 1
            end_col = node.end_point[1] + 1

            # Extract error context
            error_text = node.text.decode("utf-8") if node.text else ""
            context_before = self.extract_context_before(content_lines, start_line - 1, 2)
            context_after = self.extract_context_after(content_lines, end_line - 1, 2)

            # Combine context into a single string
            context_parts = []
            if context_before:
                context_parts.append(f"Before: {context_before}")
            context_parts.append(f"Error: {error_text}")
            if context_after:
                context_parts.append(f"After: {context_after}")

            error = CodeSyntaxError(
                start_line=start_line,
                start_column=start_col,
                end_line=end_line,
                end_column=end_col,
                error_type="syntax_error",
                context=" | ".join(context_parts),
            )
            errors.append(error)

        # Recursively check children
        for child in node.children:
            self._traverse_for_errors(child, content_lines, language, errors)

    # =================== Node Classification Methods ===================

    def get_chunk_type(self, node: Node, node_mappings: dict[ChunkType, list[str]], language: str) -> ChunkType | None:
        """
        Determine the semantic chunk type for a given AST node.

        Args:
            node: AST node to classify
            node_mappings: Language-specific node type mappings
            language: Programming language

        Returns:
            ChunkType if node represents a meaningful construct, None otherwise
        """
        node_type = node.type

        # Check each chunk type mapping
        for chunk_type, node_types in node_mappings.items():
            if node_type in node_types:
                # Additional validation for special cases
                if chunk_type == ChunkType.CONSTANT:
                    # Special handling for constants
                    if language == "python" and not self.is_module_level_assignment(node):
                        continue
                    elif language in [
                        "javascript",
                        "typescript",
                    ] and not self.is_module_level_js_declaration(node):
                        continue
                    elif language == "cpp" and not self.is_cpp_const_declaration(node):
                        continue

                elif chunk_type == ChunkType.ASYNC_FUNCTION:
                    # Verify it's actually async
                    if not self.is_async_function(node):
                        continue

                elif chunk_type == ChunkType.CONSTRUCTOR:
                    # Verify it's actually a constructor
                    if language == "java" and not self.is_java_constructor(node):
                        continue
                    elif language == "cpp":
                        func_name = self.extract_name(node, language)
                        if not func_name or not self.is_cpp_constructor(node, func_name):
                            continue

                return chunk_type

        return None

    def count_errors(self, node: Node) -> int:
        """Recursively count ERROR nodes in the AST tree."""
        count = 1 if node.type == "ERROR" else 0

        for child in node.children:
            count += self.count_errors(child)

        return count

    def count_valid_nodes(self, node: Node) -> int:
        """Count valid (non-ERROR) nodes in AST subtree."""
        count = 0 if node.type == "ERROR" else 1

        for child in node.children:
            count += self.count_valid_nodes(child)

        return count

    # =================== Property Extraction Methods ===================

    def extract_name(self, node: Node, language: str) -> str | None:
        """
        Extract the name/identifier from code constructs.

        Args:
            node: AST node
            language: Programming language

        Returns:
            Extracted name or None if not found
        """
        if language == "python":
            return self._extract_python_name(node)
        elif language in ["javascript", "typescript", "tsx"]:
            return self._extract_js_name(node)
        elif language == "go":
            return self._extract_go_name(node)
        elif language == "rust":
            return self._extract_rust_name(node)
        elif language == "java":
            return self._extract_java_name(node)
        elif language == "cpp":
            return self._extract_cpp_name(node)

        return None

    def extract_signature(self, node: Node, language: str) -> str | None:
        """
        Extract function/method signatures from AST nodes.

        Args:
            node: AST node
            language: Programming language

        Returns:
            Extracted signature or None if not applicable
        """
        if language == "python":
            return self._extract_python_signature(node)
        elif language in ["javascript", "typescript", "tsx"]:
            return self._extract_js_signature(node)
        elif language == "go":
            return self._extract_go_signature(node)
        elif language == "rust":
            return self._extract_rust_signature(node)
        elif language == "java":
            return self._extract_java_signature(node)
        elif language == "cpp":
            return self._extract_cpp_signature(node)

        return None

    def extract_docstring(self, node: Node, content_lines: list[str], language: str) -> str | None:
        """
        Extract documentation strings from code constructs.

        Args:
            node: AST node
            content_lines: Content split into lines
            language: Programming language

        Returns:
            Extracted docstring or None if not found
        """
        if language == "python":
            return self._extract_python_docstring(node, content_lines)
        elif language in ["javascript", "typescript", "tsx"]:
            return self._extract_js_docstring(node, content_lines)
        # Other languages can be added as needed

        return None

    # =================== Language-Specific Name Extraction ===================

    def _extract_python_name(self, node: Node) -> str | None:
        """Extract name from Python AST nodes."""
        if node.type == "function_definition":
            # Find the identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        elif node.type == "class_definition":
            # Find the identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        elif node.type == "assignment":
            # Get the left side of assignment
            if node.children and node.children[0].type == "identifier":
                return node.children[0].text.decode("utf-8")

        return None

    def _extract_js_name(self, node: Node) -> str | None:
        """Extract name from JavaScript/TypeScript AST nodes."""
        if node.type in ["function_declaration", "async_function_declaration"]:
            # Find the identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        elif node.type == "class_declaration":
            # Find the identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        elif node.type == "method_definition":
            # Find the property_identifier child
            for child in node.children:
                if child.type in ["identifier", "property_identifier"]:
                    return child.text.decode("utf-8")

        elif node.type in ["lexical_declaration", "variable_declaration"]:
            # Find the first identifier in variable_declarator
            for child in node.children:
                if child.type == "variable_declarator":
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            return subchild.text.decode("utf-8")

        return None

    def _extract_go_name(self, node: Node) -> str | None:
        """Extract name from Go AST nodes."""
        if node.type == "function_declaration":
            # Find the identifier child (function name)
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        elif node.type == "method_declaration":
            # In method declarations, name is field_identifier (after receiver)
            for child in node.children:
                if child.type == "field_identifier":
                    return child.text.decode("utf-8")

        elif node.type == "type_declaration":
            # Find type_spec which contains the type name
            for child in node.children:
                if child.type == "type_spec":
                    for subchild in child.children:
                        if subchild.type == "type_identifier":
                            return subchild.text.decode("utf-8")

        elif node.type in ["const_declaration", "var_declaration"]:
            # Find const_spec or var_spec which contains the identifier
            for child in node.children:
                if child.type in ["const_spec", "var_spec"]:
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            return subchild.text.decode("utf-8")

        elif node.type == "short_var_declaration":
            # Left side has identifier list
            for child in node.children:
                if child.type == "expression_list":
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            return subchild.text.decode("utf-8")

        elif node.type == "import_declaration":
            # For imports, extract package path or alias
            for child in node.children:
                if child.type == "import_spec":
                    for subchild in child.children:
                        if subchild.type == "interpreted_string_literal":
                            # Return the import path without quotes
                            path = subchild.text.decode("utf-8").strip('"')
                            return path.split("/")[-1]  # Return package name
                elif child.type == "import_spec_list":
                    # Multiple imports - return "imports"
                    return "imports"

        return None

    def _extract_rust_name(self, node: Node) -> str | None:
        """Extract name from Rust AST nodes."""
        if node.type in ["function_item", "struct_item", "enum_item", "impl_item"]:
            # Find the identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        return None

    def _extract_java_name(self, node: Node) -> str | None:
        """Extract name from Java AST nodes."""
        if node.type in [
            "method_declaration",
            "constructor_declaration",
            "class_declaration",
        ]:
            # Find the identifier child
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf-8")

        return None

    def _extract_cpp_name(self, node: Node) -> str | None:
        """Extract name from C++ AST nodes."""
        if node.type == "function_definition":
            return self._extract_cpp_function_name(node)

        elif node.type in ["class_specifier", "struct_specifier"]:
            # Find the type_identifier child
            for child in node.children:
                if child.type in ["identifier", "type_identifier"]:
                    return child.text.decode("utf-8")

        return None

    def _extract_cpp_function_name(self, node: Node) -> str | None:
        """Extract function name from C++ function definition."""
        # Find function_declarator
        for child in node.children:
            if child.type == "function_declarator":
                # Find the identifier in the declarator
                for subchild in child.children:
                    if subchild.type == "identifier":
                        return subchild.text.decode("utf-8")

        return None

    # =================== Signature Extraction Methods ===================

    def _extract_python_signature(self, node: Node) -> str | None:
        """Extract Python function/method signature."""
        if node.type == "function_definition":
            # Find the parameters child
            for child in node.children:
                if child.type == "parameters":
                    return child.text.decode("utf-8")

        return None

    def _extract_js_signature(self, node: Node) -> str | None:
        """Extract JavaScript/TypeScript function signature."""
        if node.type in [
            "function_declaration",
            "async_function_declaration",
            "method_definition",
        ]:
            # Find the formal_parameters child
            for child in node.children:
                if child.type == "formal_parameters":
                    return child.text.decode("utf-8")

        return None

    def _extract_go_signature(self, node: Node) -> str | None:
        """Extract Go function signature."""
        if node.type in ["function_declaration", "method_declaration"]:
            # Find the parameter_list child
            for child in node.children:
                if child.type == "parameter_list":
                    return child.text.decode("utf-8")

        return None

    def _extract_rust_signature(self, node: Node) -> str | None:
        """Extract Rust function signature."""
        if node.type == "function_item":
            # Find the parameters child
            for child in node.children:
                if child.type == "parameters":
                    return child.text.decode("utf-8")

        return None

    def _extract_java_signature(self, node: Node) -> str | None:
        """Extract Java method signature."""
        if node.type in ["method_declaration", "constructor_declaration"]:
            # Find the formal_parameters child
            for child in node.children:
                if child.type == "formal_parameters":
                    return child.text.decode("utf-8")

        return None

    def _extract_cpp_signature(self, node: Node) -> str | None:
        """Extract C++ function signature."""
        if node.type == "function_definition":
            # Find function_declarator and extract parameter_list
            for child in node.children:
                if child.type == "function_declarator":
                    for subchild in child.children:
                        if subchild.type == "parameter_list":
                            return subchild.text.decode("utf-8")

        return None

    # =================== Docstring Extraction Methods ===================

    def _extract_python_docstring(self, node: Node, content_lines: list[str]) -> str | None:
        """Extract Python docstring from function or class."""
        if node.type in ["function_definition", "class_definition"]:
            # Look for the first string literal in the body
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr in stmt.children:
                                if expr.type == "string" and expr.text:
                                    docstring = expr.text.decode("utf-8")
                                    # Clean up the docstring
                                    if docstring.startswith('"""') and docstring.endswith('"""'):
                                        return docstring[3:-3].strip()
                                    elif docstring.startswith("'''") and docstring.endswith("'''"):
                                        return docstring[3:-3].strip()
                                    elif docstring.startswith('"') and docstring.endswith('"'):
                                        return docstring[1:-1].strip()
                                    elif docstring.startswith("'") and docstring.endswith("'"):
                                        return docstring[1:-1].strip()
                            break  # Only check first statement

        return None

    def _extract_js_docstring(self, node: Node, content_lines: list[str]) -> str | None:
        """Extract JSDoc comment from JavaScript/TypeScript function."""
        # Look for preceding comment
        start_line = node.start_point[0]

        # Check lines before the function for JSDoc comments
        for i in range(max(0, start_line - 10), start_line):
            line = content_lines[i].strip()
            if line.startswith("/**") or "/**" in line:
                # Found JSDoc start, collect until */
                docstring_lines = []
                for j in range(i, min(len(content_lines), start_line)):
                    doc_line = content_lines[j].strip()
                    docstring_lines.append(doc_line)
                    if "*/" in doc_line:
                        break

                if docstring_lines:
                    return "\n".join(docstring_lines)

        return None

    # =================== Node Property Check Methods ===================

    def is_async_function(self, node: Node) -> bool:
        """Check if a function definition node represents an async function."""
        if node.type == "async_function_declaration":
            return True

        # Check for 'async' keyword in children
        for child in node.children:
            if child.type == "async" or (child.text and child.text.decode("utf-8") == "async"):
                return True

        return False

    def is_java_constructor(self, node: Node) -> bool:
        """Determine if a Java method declaration is a constructor."""
        return node.type == "constructor_declaration"

    def is_module_level_assignment(self, node: Node) -> bool:
        """Check if a Python assignment is at module level (for constant detection)."""
        # Simple heuristic: check if parent is module or if assignment is ALL_CAPS
        if node.type == "assignment" and node.children:
            identifier = node.children[0]
            if identifier.type == "identifier":
                name = identifier.text.decode("utf-8")
                # Consider uppercase names as constants
                return name.isupper() and "_" in name

        return False

    def is_module_level_js_declaration(self, node: Node) -> bool:
        """Check if a JavaScript declaration is at module level."""
        # Check for const declarations (simple heuristic)
        for child in node.children:
            if child.type == "const" or (child.text and child.text.decode("utf-8") == "const"):
                return True

        return False

    def is_cpp_constructor(self, node: Node, func_name: str) -> bool:
        """Determine if a C++ function is a constructor."""
        # Constructor name matches class name (simplified check)
        # This would need more sophisticated logic in practice
        return func_name and func_name[0].isupper()

    def is_cpp_const_declaration(self, node: Node) -> bool:
        """Check if a C++ declaration is a const declaration."""
        # Look for 'const' keyword in children
        for child in node.children:
            if child.type == "const" or (child.text and child.text.decode("utf-8") == "const"):
                return True

        return False

    # =================== Chunk Creation Methods ===================

    def create_chunk_from_node(
        self,
        node: Node,
        chunk_type: ChunkType,
        file_path: str,
        content: str,
        content_lines: list[str],
        language: str,
    ) -> CodeChunk | None:
        """
        Create a CodeChunk object from an AST node with all metadata.

        Args:
            node: AST node
            chunk_type: Type of chunk to create
            file_path: Path to the source file
            content: Original file content
            content_lines: Content split into lines
            language: Programming language

        Returns:
            CodeChunk object or None if creation failed
        """
        try:
            # Extract basic information
            name = self.extract_name(node, language)
            signature = self.extract_signature(node, language)
            docstring = self.extract_docstring(node, content_lines, language)

            # Calculate line numbers (Tree-sitter is 0-indexed)
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            # Extract content
            chunk_content = node.text.decode("utf-8") if node.text else ""

            # Create content hash
            import hashlib

            content_hash = hashlib.md5(chunk_content.encode("utf-8")).hexdigest()

            # Generate chunk_id using file path and content hash
            chunk_id = f"{file_path}:{content_hash[:8]}"

            # Create chunk
            chunk = CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=chunk_content,
                chunk_type=chunk_type,
                language=language,
                start_line=start_line,
                end_line=end_line,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                name=name or f"unnamed_{chunk_type.value}",
                signature=signature,
                docstring=docstring,
                content_hash=content_hash,
            )

            return chunk

        except Exception as e:
            self.logger.error(f"Failed to create chunk from node: {e}")
            return None

    # =================== Context and Utility Methods ===================

    def extract_context_before(self, lines: list[str], start_idx: int, num_lines: int) -> str | None:
        """Extract context lines before a code construct."""
        if start_idx < 0 or not lines:
            return None

        start = max(0, start_idx - num_lines)
        context_lines = lines[start:start_idx]

        return "\n".join(context_lines) if context_lines else None

    def extract_context_after(self, lines: list[str], end_idx: int, num_lines: int) -> str | None:
        """Extract context lines after a code construct."""
        if end_idx >= len(lines) or not lines:
            return None

        end = min(len(lines), end_idx + 1 + num_lines)
        context_lines = lines[end_idx + 1 : end]

        return "\n".join(context_lines) if context_lines else None

    def find_section_start(self, content_lines: list[str], min_line: int, max_line: int) -> int:
        """Find the logical start of a code section."""
        # Look backwards for comments or decorators
        for i in range(min(max_line, len(content_lines)) - 1, max(0, min_line - 5) - 1, -1):
            line = content_lines[i].strip()

            # Skip empty lines
            if not line:
                continue

            # If we find a comment or decorator, this might be the start
            if line.startswith("#") or line.startswith("@") or line.startswith("//"):
                return i + 1  # Return 1-indexed line number

            # If we find code that's not related, stop searching
            break

        return min_line

    def _collect_error_lines(self, node: Node) -> set[int]:
        """Collect all line numbers that contain ERROR nodes."""
        error_lines = set()
        self._collect_error_lines_recursive(node, error_lines)
        return error_lines

    def _collect_error_lines_recursive(self, node: Node, error_lines: set[int]) -> None:
        """Recursively collect error line numbers."""
        if node.type == "ERROR":
            # Add all lines spanned by this error
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            error_lines.update(range(start_line, end_line + 1))

        # Recursively check children
        for child in node.children:
            self._collect_error_lines_recursive(child, error_lines)
