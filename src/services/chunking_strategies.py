"""
Chunking strategies for language-specific code parsing and chunk extraction.

This module implements the Strategy pattern for handling different programming
languages' specific chunking logic, providing a flexible and extensible way
to support multiple languages with their unique characteristics.
"""

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod

from models.code_chunk import ChunkType, CodeChunk
from services.ast_extraction_service import AstExtractionService
from tree_sitter import Node


class BaseChunkingStrategy(ABC):
    """
    Abstract base class for language-specific chunking strategies.

    This class defines the interface that all chunking strategies must implement,
    providing a consistent way to handle different programming languages while
    allowing for language-specific customizations.
    """

    def __init__(self, language: str):
        """
        Initialize the chunking strategy.

        Args:
            language: Programming language this strategy handles
        """
        self.language = language
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.ast_extractor = AstExtractionService()

    @abstractmethod
    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """
        Get language-specific AST node type mappings.

        Returns:
            Dictionary mapping chunk types to AST node types for this language
        """
        pass

    @abstractmethod
    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract code chunks from the AST for this language.

        Args:
            root_node: Root node of the parsed AST
            file_path: Path to the source file
            content: Original file content

        Returns:
            List of extracted code chunks
        """
        pass

    @abstractmethod
    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """
        Determine if a node should be included as a chunk for this language.

        Args:
            node: AST node to evaluate
            chunk_type: Proposed chunk type

        Returns:
            True if the chunk should be included, False otherwise
        """
        pass

    def get_language(self) -> str:
        """Get the language this strategy handles."""
        return self.language

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """
        Extract additional language-specific metadata for a chunk.

        Args:
            node: AST node
            chunk: Code chunk being processed

        Returns:
            Dictionary with additional metadata (empty by default)
        """
        return {}

    def validate_chunk(self, chunk: CodeChunk) -> bool:
        """
        Validate that a chunk meets language-specific requirements.

        Args:
            chunk: Code chunk to validate

        Returns:
            True if chunk is valid, False otherwise
        """
        # Basic validation - can be overridden by subclasses
        return chunk.content and chunk.content.strip() and chunk.start_line > 0 and chunk.end_line >= chunk.start_line

    def post_process_chunks(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """
        Post-process extracted chunks for language-specific optimizations.

        Args:
            chunks: List of extracted chunks

        Returns:
            Processed list of chunks
        """
        # Filter out invalid chunks
        valid_chunks = [chunk for chunk in chunks if self.validate_chunk(chunk)]

        # Sort by start line
        valid_chunks.sort(key=lambda c: c.start_line)

        return valid_chunks


class ChunkingStrategyRegistry:
    """
    Registry for managing and discovering chunking strategies.

    This class provides a centralized way to register and retrieve
    chunking strategies for different programming languages.
    """

    def __init__(self):
        """Initialize the strategy registry."""
        self.logger = logging.getLogger(__name__)
        self._strategies: dict[str, BaseChunkingStrategy] = {}

    def register_strategy(self, language: str, strategy: BaseChunkingStrategy) -> None:
        """
        Register a chunking strategy for a language.

        Args:
            language: Programming language name
            strategy: Chunking strategy instance
        """
        self._strategies[language] = strategy
        self.logger.info(f"Registered chunking strategy for {language}")

    def get_strategy(self, language: str) -> BaseChunkingStrategy | None:
        """
        Get the chunking strategy for a language.

        Args:
            language: Programming language name

        Returns:
            Chunking strategy instance or None if not found
        """
        return self._strategies.get(language)

    def has_strategy(self, language: str) -> bool:
        """
        Check if a strategy is registered for a language.

        Args:
            language: Programming language name

        Returns:
            True if strategy exists, False otherwise
        """
        return language in self._strategies

    def get_supported_languages(self) -> list[str]:
        """Get list of languages with registered strategies."""
        return list(self._strategies.keys())

    def unregister_strategy(self, language: str) -> bool:
        """
        Unregister a chunking strategy.

        Args:
            language: Programming language name

        Returns:
            True if strategy was removed, False if not found
        """
        if language in self._strategies:
            del self._strategies[language]
            self.logger.info(f"Unregistered chunking strategy for {language}")
            return True
        return False


# Global registry instance
chunking_strategy_registry = ChunkingStrategyRegistry()


def register_chunking_strategy(language: str):
    """
    Decorator for automatically registering chunking strategies.

    Args:
        language: Programming language name

    Returns:
        Decorator function
    """

    def decorator(strategy_class):
        """Register the strategy class."""
        strategy_instance = strategy_class(language)
        chunking_strategy_registry.register_strategy(language, strategy_instance)
        return strategy_class

    return decorator


class FallbackChunkingStrategy(BaseChunkingStrategy):
    """
    Fallback chunking strategy for unsupported languages.

    This strategy provides smart chunking when no specific language strategy
    is available. For small files, it creates a single whole-file chunk.
    For large files, it intelligently splits by lines with context overlap.
    """

    # Configuration for smart splitting (can be overridden via environment variables)
    # nomic-embed-text has 2048 token context limit
    # ~4 chars per token, so 2048 tokens ≈ 8192 chars
    # Default: 6000 chars (~1500 tokens) for safety margin
    DEFAULT_MAX_CHUNK_CHARS = 6000
    # Lines per chunk when splitting (approximately 60-80 lines for code)
    DEFAULT_TARGET_LINES_PER_CHUNK = 75
    # Overlap lines for context continuity
    DEFAULT_OVERLAP_LINES = 15

    def __init__(self, language: str, reason: str = "language_not_supported"):
        """Initialize the fallback strategy.

        Args:
            language: The language identifier
            reason: Reason for using fallback (language_not_supported, parser_unavailable, parse_error)
        """
        super().__init__(language)
        self.reason = reason
        # Load configuration from environment or use defaults
        self.max_chunk_chars = int(os.getenv("FALLBACK_MAX_CHUNK_CHARS", str(self.DEFAULT_MAX_CHUNK_CHARS)))
        self.target_lines_per_chunk = int(os.getenv("FALLBACK_TARGET_LINES", str(self.DEFAULT_TARGET_LINES_PER_CHUNK)))
        self.overlap_lines = int(os.getenv("FALLBACK_OVERLAP_LINES", str(self.DEFAULT_OVERLAP_LINES)))

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Return empty mappings for fallback strategy."""
        return {}

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract chunks from file content with smart splitting for large files.

        For small files (under threshold), creates a single whole-file chunk.
        For large files, splits into multiple chunks with context overlap.

        Args:
            root_node: Root node of the parsed AST (not used in fallback)
            file_path: Path to the source file
            content: Original file content

        Returns:
            List of code chunks
        """
        from utils.fallback_tracker import fallback_tracker

        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        char_count = len(content)

        # Check if smart splitting is needed
        needs_splitting = char_count > self.max_chunk_chars

        # Record fallback usage for tracking
        fallback_tracker.record_fallback(
            file_path=file_path,
            content=content,
            reason=self.reason,
            was_truncated=needs_splitting,
        )

        if needs_splitting:
            return self._extract_split_chunks(file_path, content, content_hash)
        else:
            return self._extract_single_chunk(file_path, content, content_hash)

    def _extract_single_chunk(self, file_path: str, content: str, content_hash: str) -> list[CodeChunk]:
        """Extract a single whole-file chunk for small files."""
        content_lines = content.split("\n")
        chunk_id = f"{file_path}:{content_hash[:8]}"

        chunk = CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=self.language,
            start_line=1,
            end_line=len(content_lines),
            start_byte=0,
            end_byte=len(content.encode("utf-8")),
            name=f"file_{self.language}",
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )

        return [chunk]

    def _extract_split_chunks(self, file_path: str, content: str, content_hash: str) -> list[CodeChunk]:
        """
        Split large files into multiple chunks with context overlap.

        Uses line-based splitting to preserve some semantic structure,
        with overlap to maintain context between chunks.
        """
        lines = content.split("\n")
        total_lines = len(lines)
        chunks = []

        self.logger.info(
            f"Smart splitting {file_path}: {len(content):,} chars, {total_lines} lines "
            f"into ~{(total_lines // self.target_lines_per_chunk) + 1} chunks"
        )

        chunk_index = 0
        current_line = 0

        while current_line < total_lines:
            # Calculate chunk boundaries
            start_line = current_line
            end_line = min(current_line + self.target_lines_per_chunk, total_lines)

            # Try to find a better split point (empty line, comment, etc.)
            end_line = self._find_split_point(lines, end_line, total_lines)

            # Extract chunk content
            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines)

            # Calculate byte positions
            start_byte = sum(len(line.encode("utf-8")) + 1 for line in lines[:start_line])
            end_byte = start_byte + len(chunk_content.encode("utf-8"))

            # Generate unique chunk ID
            chunk_content_hash = hashlib.md5(chunk_content.encode("utf-8")).hexdigest()
            chunk_id = f"{file_path}:part{chunk_index}:{chunk_content_hash[:8]}"

            chunk = CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=chunk_content,
                chunk_type=ChunkType.RAW_CODE,  # Use RAW_CODE for split file parts
                language=self.language,
                start_line=start_line + 1,  # 1-indexed
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                name=f"file_{self.language}_part{chunk_index + 1}",
                signature=f"Part {chunk_index + 1} of {file_path}",
                docstring=None,
                content_hash=chunk_content_hash,
            )

            # Add metadata about splitting
            chunk.metadata = {
                "is_split_chunk": True,
                "chunk_part": chunk_index + 1,
                "total_file_lines": total_lines,
                "split_reason": "exceeded_token_threshold",
            }

            chunks.append(chunk)
            chunk_index += 1

            # Move to next chunk with overlap
            if end_line < total_lines:
                current_line = max(end_line - self.overlap_lines, current_line + 1)
            else:
                break

        self.logger.info(f"Split {file_path} into {len(chunks)} chunks")
        return chunks

    def _find_split_point(self, lines: list[str], target_end: int, total_lines: int) -> int:
        """
        Find a better split point near the target end line.

        Looks for natural breakpoints like:
        - Empty lines
        - Lines starting with comments
        - Lines with closing braces/brackets

        Args:
            lines: All lines in the file
            target_end: Target end line
            total_lines: Total number of lines

        Returns:
            Adjusted end line for the split
        """
        if target_end >= total_lines:
            return total_lines

        # Search window: look up to 20 lines before target for a better split point
        search_start = max(target_end - 20, 0)

        best_split = target_end

        for i in range(target_end - 1, search_start - 1, -1):
            line = lines[i].strip()

            # Empty line is a great split point
            if not line:
                best_split = i + 1
                break

            # Line with only closing brace/bracket
            if line in ["}", "]", ")", "};", "];", ");", "end", "fi", "done", "esac"]:
                best_split = i + 1
                break

            # Comment line (potential section separator)
            if line.startswith("#") or line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                # Check if previous line is empty (section break)
                if i > 0 and not lines[i - 1].strip():
                    best_split = i
                    break

        return best_split

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Always include chunks in fallback strategy."""
        return True


class StructuredFileChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy for structured files (JSON, YAML, etc.).

    This strategy handles non-code files that have structured content
    that can be meaningfully chunked.
    """

    def __init__(self, language: str):
        """Initialize the structured file strategy."""
        super().__init__(language)

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Return empty mappings as structured files don't use AST."""
        return {}

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract chunks from structured files.

        Args:
            root_node: Root node (not used for structured files)
            file_path: Path to the source file
            content: Original file content

        Returns:
            List of extracted chunks based on file structure
        """
        if self.language == "json":
            return self._extract_json_chunks(file_path, content)
        elif self.language == "yaml":
            return self._extract_yaml_chunks(file_path, content)
        elif self.language == "markdown":
            return self._extract_markdown_chunks(file_path, content)

        # Fallback to single chunk
        return self._extract_single_chunk(file_path, content)

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Include all chunks for structured files."""
        return True

    def _extract_json_chunks(self, file_path: str, content: str) -> list[CodeChunk]:
        """Extract chunks from JSON files."""
        try:
            import json

            data = json.loads(content)
            chunks = []

            if isinstance(data, dict):
                for i, (key, value) in enumerate(data.items()):
                    chunk_content = json.dumps({key: value}, indent=2)
                    chunk = self._create_structured_chunk(
                        file_path,
                        chunk_content,
                        ChunkType.OBJECT,
                        name=key,
                        start_line=i + 1,
                        end_line=i + 1,
                    )
                    chunks.append(chunk)

            return chunks if chunks else [self._extract_single_chunk(file_path, content)]

        except json.JSONDecodeError:
            return [self._extract_single_chunk(file_path, content)]

    def _extract_yaml_chunks(self, file_path: str, content: str) -> list[CodeChunk]:
        """Extract chunks from YAML files."""
        try:
            import yaml

            data = yaml.safe_load(content)
            chunks = []

            if isinstance(data, dict):
                for i, (key, value) in enumerate(data.items()):
                    chunk_content = yaml.dump({key: value}, default_flow_style=False)
                    chunk = self._create_structured_chunk(
                        file_path,
                        chunk_content,
                        ChunkType.OBJECT,
                        name=key,
                        start_line=i + 1,
                        end_line=i + 1,
                    )
                    chunks.append(chunk)

            return chunks if chunks else [self._extract_single_chunk(file_path, content)]

        except yaml.YAMLError:
            return [self._extract_single_chunk(file_path, content)]

    def _extract_markdown_chunks(self, file_path: str, content: str) -> list[CodeChunk]:
        """Extract chunks from Markdown files based on headers."""
        chunks = []
        lines = content.split("\n")
        current_section = []
        current_header = None
        section_start = 1

        for i, line in enumerate(lines):
            if line.startswith("#"):
                # New header found
                if current_section and current_header:
                    # Save previous section
                    section_content = "\n".join(current_section)
                    chunk = self._create_structured_chunk(
                        file_path,
                        section_content,
                        ChunkType.SECTION,
                        name=current_header.strip("#").strip(),
                        start_line=section_start,
                        end_line=i,
                    )
                    chunks.append(chunk)

                # Start new section
                current_header = line
                current_section = [line]
                section_start = i + 1
            else:
                current_section.append(line)

        # Add final section
        if current_section and current_header:
            section_content = "\n".join(current_section)
            chunk = self._create_structured_chunk(
                file_path,
                section_content,
                ChunkType.SECTION,
                name=current_header.strip("#").strip(),
                start_line=section_start,
                end_line=len(lines),
            )
            chunks.append(chunk)

        return chunks if chunks else [self._extract_single_chunk(file_path, content)]

    def _create_structured_chunk(
        self,
        file_path: str,
        content: str,
        chunk_type: ChunkType,
        name: str,
        start_line: int,
        end_line: int,
    ) -> CodeChunk:
        """Create a structured file chunk."""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

        # Generate chunk_id using file path and content hash
        chunk_id = f"{file_path}:{content_hash[:8]}"

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=chunk_type,
            language=self.language,
            start_line=start_line,
            end_line=end_line,
            start_byte=0,  # Structured files start at beginning
            end_byte=len(content.encode("utf-8")),
            name=name,
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )

    def _extract_single_chunk(self, file_path: str, content: str) -> CodeChunk:
        """Create a single chunk for the entire file."""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        content_lines = content.split("\n")

        # Generate chunk_id using file path and content hash
        chunk_id = f"{file_path}:{content_hash[:8]}"

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=self.language,
            start_line=1,
            end_line=len(content_lines),
            start_byte=0,
            end_byte=len(content.encode("utf-8")),
            name=f"file_{self.language}",
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )


# Register structured file strategies
structured_json_strategy = StructuredFileChunkingStrategy("json")
structured_yaml_strategy = StructuredFileChunkingStrategy("yaml")
structured_markdown_strategy = StructuredFileChunkingStrategy("markdown")

chunking_strategy_registry.register_strategy("json", structured_json_strategy)
chunking_strategy_registry.register_strategy("yaml", structured_yaml_strategy)
chunking_strategy_registry.register_strategy("markdown", structured_markdown_strategy)


@register_chunking_strategy("python")
class PythonChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy specifically designed for Python code.

    This strategy handles Python-specific constructs like functions, classes,
    decorators, async functions, and module-level constants.

    For classes, this strategy uses "skeleton mode" - extracting only the class
    definition header, docstring, and class-level attributes, while methods are
    extracted as separate chunks. This avoids duplicate content and keeps chunks
    within reasonable size limits.
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get Python-specific AST node type mappings."""
        return {
            ChunkType.FUNCTION: ["function_definition"],
            ChunkType.CLASS: ["class_definition"],
            ChunkType.CONSTANT: ["assignment"],  # Filtered by context
            ChunkType.VARIABLE: ["assignment"],
            ChunkType.IMPORT: ["import_statement", "import_from_statement"],
        }

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract Python-specific chunks from the AST.

        Uses skeleton mode for classes: extracts class definition + docstring +
        class attributes as a compact chunk, while methods are extracted separately.
        """
        chunks = []
        content_lines = content.split("\n")

        # Use custom traversal that handles class skeleton extraction
        self._traverse_python_ast(
            root_node,
            chunks,
            file_path,
            content,
            content_lines,
        )

        # Python-specific post-processing
        processed_chunks = []
        for chunk in chunks:
            # Enhanced validation for Python
            if self._is_valid_python_chunk(chunk):
                # Add Python-specific metadata
                additional_metadata = self._extract_chunk_metadata(chunk)
                if additional_metadata:
                    chunk.metadata = getattr(chunk, "metadata", {})
                    chunk.metadata.update(additional_metadata)

                processed_chunks.append(chunk)

        return self.post_process_chunks(processed_chunks)

    def _traverse_python_ast(
        self,
        node: Node,
        chunks: list[CodeChunk],
        file_path: str,
        content: str,
        content_lines: list[str],
        inside_class: bool = False,
    ) -> None:
        """
        Custom AST traversal for Python with class skeleton extraction.

        Args:
            node: Current AST node
            chunks: List to collect extracted chunks
            file_path: Path to the source file
            content: Original file content
            content_lines: Content split into lines
            inside_class: Whether we're currently inside a class body
        """
        if node.type == "class_definition":
            # Extract class skeleton (definition + docstring + class attributes)
            skeleton_chunk = self._extract_class_skeleton(node, file_path, content_lines)
            if skeleton_chunk:
                chunks.append(skeleton_chunk)

            # Continue traversing to extract methods as separate chunks
            for child in node.children:
                if child.type == "block":
                    # Traverse the class body to find methods
                    for body_child in child.children:
                        self._traverse_python_ast(
                            body_child,
                            chunks,
                            file_path,
                            content,
                            content_lines,
                            inside_class=True,
                        )
            return

        elif node.type == "decorated_definition":
            # Handle decorated classes and functions
            # Find the actual definition inside
            for child in node.children:
                if child.type == "class_definition":
                    # Decorated class - extract skeleton
                    skeleton_chunk = self._extract_class_skeleton(child, file_path, content_lines)
                    if skeleton_chunk:
                        # Include decorators in the skeleton
                        decorators = self._extract_decorators_from_decorated(node, content_lines)
                        if decorators:
                            skeleton_chunk.content = decorators + "\n" + skeleton_chunk.content
                            skeleton_chunk.start_line = node.start_point[0] + 1
                            if skeleton_chunk.metadata is None:
                                skeleton_chunk.metadata = {}
                            skeleton_chunk.metadata["decorators"] = decorators.split("\n")
                        chunks.append(skeleton_chunk)

                    # Continue traversing to extract methods
                    for class_child in child.children:
                        if class_child.type == "block":
                            for body_child in class_child.children:
                                self._traverse_python_ast(
                                    body_child,
                                    chunks,
                                    file_path,
                                    content,
                                    content_lines,
                                    inside_class=True,
                                )
                    return

                elif child.type == "function_definition":
                    # Decorated function/method - extract the whole decorated definition
                    chunk = self.ast_extractor.create_chunk_from_node(
                        node,  # Use the decorated_definition node to include decorators
                        ChunkType.METHOD if inside_class else ChunkType.FUNCTION,
                        file_path,
                        content,
                        content_lines,
                        self.language,
                    )
                    if chunk:
                        chunks.append(chunk)
                    return
            return

        elif node.type == "function_definition":
            # Extract function/method as a chunk
            chunk = self.ast_extractor.create_chunk_from_node(
                node,
                ChunkType.METHOD if inside_class else ChunkType.FUNCTION,
                file_path,
                content,
                content_lines,
                self.language,
            )
            if chunk:
                chunks.append(chunk)
            # Don't traverse into function body for nested functions (optional)
            return

        elif node.type == "assignment":
            # Only extract module-level constants
            if not inside_class and self._is_python_constant(node):
                chunk = self.ast_extractor.create_chunk_from_node(
                    node,
                    ChunkType.CONSTANT,
                    file_path,
                    content,
                    content_lines,
                    self.language,
                )
                if chunk:
                    chunks.append(chunk)

        # Recursively process children
        for child in node.children:
            self._traverse_python_ast(
                child,
                chunks,
                file_path,
                content,
                content_lines,
                inside_class=inside_class,
            )

    def _extract_decorators_from_decorated(self, decorated_node: Node, content_lines: list[str]) -> str:
        """Extract decorator lines from a decorated_definition node."""
        decorators = []
        for child in decorated_node.children:
            if child.type == "decorator":
                start_line = child.start_point[0]
                end_line = child.end_point[0]
                for i in range(start_line, end_line + 1):
                    if i < len(content_lines):
                        decorators.append(content_lines[i])
        return "\n".join(decorators)

    def _extract_class_skeleton(
        self,
        node: Node,
        file_path: str,
        content_lines: list[str],
    ) -> CodeChunk | None:
        """
        Extract a class skeleton chunk containing only:
        - Class definition line (with decorators and inheritance)
        - Docstring
        - Class-level attributes (not inside methods)

        Args:
            node: Class definition AST node
            file_path: Path to the source file
            content_lines: Content split into lines

        Returns:
            CodeChunk with class skeleton or None if extraction failed
        """
        try:
            # Get class name
            class_name = None
            for child in node.children:
                if child.type == "identifier":
                    class_name = child.text.decode("utf-8")
                    break

            if not class_name:
                return None

            # Calculate line numbers (Tree-sitter is 0-indexed)
            start_line = node.start_point[0] + 1

            # Build skeleton content
            skeleton_parts = []

            # 1. Extract decorators (if any, they appear before the class)
            decorators = self._extract_decorators(node)

            # 2. Extract class definition line
            class_def_line = content_lines[node.start_point[0]]
            skeleton_parts.append(class_def_line)

            # 3. Find and extract docstring and class attributes from the block
            docstring = None
            class_attributes = []
            skeleton_end_line = start_line

            for child in node.children:
                if child.type == "block":
                    first_stmt = True
                    for stmt in child.children:
                        # Skip newlines and other non-statement nodes
                        if stmt.type in [":", "newline", "indent", "dedent", "NEWLINE"]:
                            continue

                        # Check for docstring (first expression_statement with string)
                        if first_stmt and stmt.type == "expression_statement":
                            for expr in stmt.children:
                                if expr.type == "string":
                                    docstring = expr.text.decode("utf-8")
                                    # Add docstring to skeleton
                                    indent = "    "  # Standard Python indent
                                    skeleton_parts.append(f"{indent}{docstring}")
                                    skeleton_end_line = stmt.end_point[0] + 1
                                    break
                            first_stmt = False
                            continue

                        first_stmt = False

                        # Check for class-level assignments (attributes)
                        if stmt.type == "expression_statement":
                            # Check if it's a simple assignment (class attribute)
                            for expr in stmt.children:
                                if expr.type == "assignment":
                                    attr_text = expr.text.decode("utf-8")
                                    indent = "    "
                                    skeleton_parts.append(f"{indent}{attr_text}")
                                    class_attributes.append(attr_text)
                                    skeleton_end_line = stmt.end_point[0] + 1

                        # Check for annotated assignments (type-hinted class attributes)
                        elif stmt.type == "annotated_assignment":
                            attr_text = stmt.text.decode("utf-8")
                            indent = "    "
                            skeleton_parts.append(f"{indent}{attr_text}")
                            class_attributes.append(attr_text)
                            skeleton_end_line = stmt.end_point[0] + 1

                        # Stop when we hit a function definition (method)
                        elif stmt.type == "function_definition":
                            break

                        # Also stop at decorated definitions
                        elif stmt.type == "decorated_definition":
                            break

            # Add a placeholder comment for methods
            method_count = self._count_methods(node)
            if method_count > 0:
                skeleton_parts.append(f"    # ... {method_count} method(s)")

            # Join skeleton content
            skeleton_content = "\n".join(skeleton_parts)

            # Create content hash
            content_hash = hashlib.md5(skeleton_content.encode("utf-8")).hexdigest()

            # Generate chunk_id
            chunk_id = f"{file_path}:class:{class_name}:{content_hash[:8]}"

            # Extract inheritance info for signature
            inheritance = self._extract_class_inheritance(node)
            signature = f"({', '.join(inheritance)})" if inheritance else None

            # Create the skeleton chunk
            chunk = CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=skeleton_content,
                chunk_type=ChunkType.CLASS,
                language=self.language,
                start_line=start_line,
                end_line=skeleton_end_line,
                start_byte=node.start_byte,
                end_byte=node.start_byte + len(skeleton_content.encode("utf-8")),
                name=class_name,
                signature=signature,
                docstring=self._clean_docstring(docstring) if docstring else None,
                content_hash=content_hash,
            )

            # Add metadata
            chunk.metadata = {
                "is_skeleton": True,
                "method_count": method_count,
                "class_attributes": class_attributes,
                "original_end_line": node.end_point[0] + 1,
            }

            if decorators:
                chunk.metadata["decorators"] = decorators

            if inheritance:
                chunk.metadata["inheritance"] = inheritance

            return chunk

        except Exception as e:
            self.logger.error(f"Failed to extract class skeleton: {e}")
            return None

    def _count_methods(self, class_node: Node) -> int:
        """Count the number of methods in a class."""
        count = 0
        for child in class_node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "function_definition":
                        count += 1
                    elif stmt.type == "decorated_definition":
                        # Check if the decorated item is a function
                        for subchild in stmt.children:
                            if subchild.type == "function_definition":
                                count += 1
                                break
        return count

    def _clean_docstring(self, docstring: str) -> str:
        """Clean docstring by removing triple quotes and extra whitespace."""
        result = docstring
        # Remove triple double quotes
        if result.startswith('"""'):
            result = result[3:]
        if result.endswith('"""'):
            result = result[:-3]
        # Remove triple single quotes
        if result.startswith("'''"):
            result = result[3:]
        if result.endswith("'''"):
            result = result[:-3]
        return result.strip()

    def _extract_chunk_metadata(self, chunk: CodeChunk) -> dict[str, any]:
        """Extract Python-specific metadata for a chunk."""
        metadata = {}

        # For methods/functions, check for async
        if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
            if "async def" in chunk.content:
                metadata["is_async"] = True

            # Check for decorators in the content
            lines = chunk.content.split("\n")
            decorators = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("@"):
                    decorators.append(stripped)
                elif stripped.startswith("def ") or stripped.startswith("async def"):
                    break
            if decorators:
                metadata["decorators"] = decorators

        return metadata

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Determine if a Python node should be included as a chunk."""
        if chunk_type == ChunkType.CONSTANT:
            # Only include module-level assignments that look like constants
            return self._is_python_constant(node)

        elif chunk_type == ChunkType.FUNCTION:
            # Include all function definitions, including async
            return True

        elif chunk_type == ChunkType.CLASS:
            # Include all class definitions
            return True

        elif chunk_type == ChunkType.IMPORT:
            # Include import statements for dependency tracking
            return True

        return True

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """Extract Python-specific metadata."""
        metadata = {}

        # Check for decorators
        decorators = self._extract_decorators(node)
        if decorators:
            metadata["decorators"] = decorators

        # Check for async functions
        if chunk.chunk_type == ChunkType.FUNCTION:
            metadata["is_async"] = self.ast_extractor.is_async_function(node)

        # Check for class inheritance
        if chunk.chunk_type == ChunkType.CLASS:
            inheritance = self._extract_class_inheritance(node)
            if inheritance:
                metadata["inheritance"] = inheritance

        # Extract type hints if present
        type_hints = self._extract_type_hints(node)
        if type_hints:
            metadata["type_hints"] = type_hints

        return metadata

    def _is_valid_python_chunk(self, chunk: CodeChunk) -> bool:
        """Validate Python-specific chunk requirements."""
        if not self.validate_chunk(chunk):
            return False

        # Python-specific validations
        if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
            # Function/method should have a name and some content
            return chunk.name and chunk.name != "unnamed_function" and chunk.name != "unnamed_method"

        elif chunk.chunk_type == ChunkType.CLASS:
            # Class should have a name
            return chunk.name and chunk.name != "unnamed_class"

        elif chunk.chunk_type == ChunkType.CONSTANT:
            # Constants should be uppercase with underscores
            return chunk.name and chunk.name.isupper() and "_" in chunk.name and len(chunk.content.strip()) < 200  # Avoid huge constants

        return True

    def _is_python_constant(self, node: Node) -> bool:
        """Check if a Python assignment represents a constant."""
        if node.type != "assignment":
            return False

        # Check if assignment target is an identifier
        if not node.children or node.children[0].type != "identifier":
            return False

        # Get the variable name
        name = node.children[0].text.decode("utf-8")

        # Consider uppercase names with underscores as constants
        return name.isupper() and "_" in name and not name.startswith("_")

    def _extract_decorators(self, node: Node) -> list[str]:
        """Extract decorator names from a Python function or class."""
        decorators = []

        # Look for decorator nodes before the function/class
        for child in node.children:
            if child.type == "decorator":
                # Extract decorator name
                for subchild in child.children:
                    if subchild.type == "identifier":
                        decorators.append(f"@{subchild.text.decode('utf-8')}")
                        break
                    elif subchild.type == "attribute":
                        # Handle complex decorators like @dataclass.decorator
                        decorators.append(f"@{subchild.text.decode('utf-8')}")
                        break

        return decorators

    def _extract_class_inheritance(self, node: Node) -> list[str]:
        """Extract base classes from a Python class definition."""
        inheritance = []

        if node.type == "class_definition":
            # Look for argument_list which contains base classes
            for child in node.children:
                if child.type == "argument_list":
                    for arg in child.children:
                        if arg.type == "identifier":
                            inheritance.append(arg.text.decode("utf-8"))

        return inheritance

    def _extract_type_hints(self, node: Node) -> dict[str, str]:
        """Extract type hints from Python functions."""
        type_hints = {}

        if node.type == "function_definition":
            # Look for parameters with type annotations
            for child in node.children:
                if child.type == "parameters":
                    # Extract parameter type hints
                    # This is a simplified implementation
                    pass

        return type_hints


@register_chunking_strategy("javascript")
class JavaScriptChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy specifically designed for JavaScript/TypeScript code.

    This strategy handles JavaScript-specific constructs like functions, classes,
    arrow functions, async functions, and ES6+ features.

    Uses custom AST traversal to maintain node-to-chunk mapping for accurate
    metadata extraction (e.g., arrow function detection, class features).
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get JavaScript-specific AST node type mappings."""
        return {
            ChunkType.FUNCTION: [
                "function_declaration",
                "arrow_function",
                "method_definition",
            ],
            ChunkType.ASYNC_FUNCTION: ["async_function_declaration"],
            ChunkType.CLASS: ["class_declaration"],
            ChunkType.CONSTANT: ["lexical_declaration"],  # const declarations
            ChunkType.VARIABLE: ["variable_declaration"],
            ChunkType.IMPORT: ["import_statement"],
            ChunkType.EXPORT: ["export_statement"],
        }

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract JavaScript-specific chunks from the AST.

        Uses custom AST traversal to maintain node references for accurate
        metadata extraction (arrow functions, methods, class features, etc.).
        """
        chunks = []
        content_lines = content.split("\n")

        # Use custom traversal that maintains node-to-chunk mapping
        self._traverse_js_ast(
            root_node,
            chunks,
            file_path,
            content,
            content_lines,
        )

        # JavaScript-specific post-processing
        processed_chunks = []
        for chunk in chunks:
            if self._is_valid_javascript_chunk(chunk):
                processed_chunks.append(chunk)

        return self.post_process_chunks(processed_chunks)

    def _traverse_js_ast(
        self,
        node: Node,
        chunks: list[CodeChunk],
        file_path: str,
        content: str,
        content_lines: list[str],
    ) -> None:
        """
        Custom AST traversal for JavaScript with proper node-to-chunk mapping.

        Args:
            node: Current AST node
            chunks: List to collect extracted chunks
            file_path: Path to the source file
            content: Original file content
            content_lines: Content split into lines
        """
        node_type = node.type

        if node_type == "function_declaration":
            chunk = self._extract_function_chunk(node, file_path, content, content_lines, is_arrow=False)
            if chunk:
                chunks.append(chunk)
            return  # Don't traverse into function body

        elif node_type == "arrow_function":
            # Only extract top-level arrow functions assigned to variables
            # Arrow functions inside other expressions are skipped
            if self._is_top_level_arrow_function(node):
                chunk = self._extract_function_chunk(node, file_path, content, content_lines, is_arrow=True)
                if chunk:
                    chunks.append(chunk)
            return

        elif node_type == "method_definition":
            chunk = self._extract_method_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return

        elif node_type == "class_declaration":
            chunk = self._extract_class_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            # Continue traversing to extract methods inside class
            for child in node.children:
                if child.type == "class_body":
                    for body_child in child.children:
                        self._traverse_js_ast(body_child, chunks, file_path, content, content_lines)
            return

        elif node_type == "lexical_declaration":
            if self._is_javascript_constant(node):
                chunk = self._extract_const_chunk(node, file_path, content, content_lines)
                if chunk:
                    chunks.append(chunk)
            return

        elif node_type == "variable_declaration":
            chunk = self._extract_variable_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return

        elif node_type == "import_statement":
            chunk = self._extract_import_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return

        elif node_type == "export_statement":
            chunk = self._extract_export_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return

        # Recursively process children
        for child in node.children:
            self._traverse_js_ast(child, chunks, file_path, content, content_lines)

    def _extract_function_chunk(
        self, node: Node, file_path: str, content: str, content_lines: list[str], is_arrow: bool = False
    ) -> CodeChunk | None:
        """Extract a function declaration as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.FUNCTION, file_path, content, content_lines, self.language)
        if chunk:
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["is_arrow_function"] = is_arrow
            chunk.metadata["is_async"] = self.ast_extractor.is_async_function(node)
            chunk.metadata["is_method"] = False
            # Extract ES6 features from this specific node
            es6_features = self._extract_es6_features(node)
            if es6_features:
                chunk.metadata["es6_features"] = es6_features
        return chunk

    def _extract_method_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a method definition as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.FUNCTION, file_path, content, content_lines, self.language)
        if chunk:
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["is_arrow_function"] = False
            chunk.metadata["is_async"] = self.ast_extractor.is_async_function(node)
            chunk.metadata["is_method"] = True
            es6_features = self._extract_es6_features(node)
            if es6_features:
                chunk.metadata["es6_features"] = es6_features
        return chunk

    def _extract_class_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a class declaration as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.CLASS, file_path, content, content_lines, self.language)
        if chunk:
            chunk.metadata = chunk.metadata or {}
            # Extract class features from the actual class node
            class_features = self._extract_class_features(node)
            chunk.metadata.update(class_features)
            es6_features = self._extract_es6_features(node)
            if es6_features:
                chunk.metadata["es6_features"] = es6_features
        return chunk

    def _extract_const_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a const declaration as a chunk."""
        return self.ast_extractor.create_chunk_from_node(node, ChunkType.CONSTANT, file_path, content, content_lines, self.language)

    def _extract_variable_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a variable declaration as a chunk."""
        return self.ast_extractor.create_chunk_from_node(node, ChunkType.VARIABLE, file_path, content, content_lines, self.language)

    def _extract_import_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract an import statement as a chunk."""
        return self.ast_extractor.create_chunk_from_node(node, ChunkType.IMPORT, file_path, content, content_lines, self.language)

    def _extract_export_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract an export statement as a chunk."""
        return self.ast_extractor.create_chunk_from_node(node, ChunkType.EXPORT, file_path, content, content_lines, self.language)

    def _is_top_level_arrow_function(self, node: Node) -> bool:
        """Check if an arrow function is at top-level (assigned to a variable)."""
        # Arrow function should be part of a variable declarator or assignment
        parent = node.parent
        if parent and parent.type in ["variable_declarator", "assignment_expression"]:
            return True
        return False

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Determine if a JavaScript node should be included as a chunk."""
        if chunk_type == ChunkType.CONSTANT:
            # Only include const declarations at module level
            return self._is_javascript_constant(node)

        elif chunk_type == ChunkType.FUNCTION:
            # Include all function types
            return True

        elif chunk_type == ChunkType.CLASS:
            # Include class declarations
            return True

        elif chunk_type in [ChunkType.IMPORT, ChunkType.EXPORT]:
            # Include import/export statements
            return True

        return True

    def _is_valid_javascript_chunk(self, chunk: CodeChunk) -> bool:
        """Validate JavaScript-specific chunk requirements."""
        if not self.validate_chunk(chunk):
            return False

        # JavaScript-specific validations
        if chunk.chunk_type == ChunkType.FUNCTION:
            # Function should have meaningful content
            return len(chunk.content.strip()) > 10

        elif chunk.chunk_type == ChunkType.CLASS:
            # Class should have a name and body
            return chunk.name and "{" in chunk.content

        elif chunk.chunk_type == ChunkType.CONSTANT:
            # Constants should be const declarations
            return "const" in chunk.content and "=" in chunk.content

        return True

    def _is_javascript_constant(self, node: Node) -> bool:
        """Check if a JavaScript declaration represents a constant."""
        if node.type != "lexical_declaration":
            return False

        # Check for 'const' keyword
        for child in node.children:
            if child.type == "const" or (child.text and child.text.decode("utf-8") == "const"):
                return True

        return False

    def _extract_class_features(self, node: Node) -> dict[str, any]:
        """Extract JavaScript class features."""
        features = {}

        if node.type == "class_declaration":
            # Check for extends clause
            for child in node.children:
                if child.type == "class_heritage":
                    # Has inheritance
                    features["has_inheritance"] = True
                    # Extract parent class name
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            features["extends"] = subchild.text.decode("utf-8")
                            break

        return features

    def _extract_es6_features(self, node: Node) -> list[str]:
        """Extract ES6+ features used in the code."""
        features = []

        # Check for various ES6+ features
        node_text = node.text.decode("utf-8") if node.text else ""

        if "=>" in node_text:
            features.append("arrow_functions")

        if "async" in node_text:
            features.append("async_await")

        if "const " in node_text or "let " in node_text:
            features.append("block_scoping")

        if "`" in node_text:
            features.append("template_literals")

        if "class " in node_text:
            features.append("classes")

        return features


@register_chunking_strategy("tsx")
class TsxChunkingStrategy(JavaScriptChunkingStrategy):
    """
    Chunking strategy for TypeScript JSX (TSX) code.

    Extends JavaScript strategy with TypeScript-specific features like
    interfaces, type aliases, and React component patterns.
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get TSX-specific AST node type mappings."""
        base_mappings = super().get_node_mappings()

        # Add TypeScript-specific mappings
        tsx_mappings = {
            ChunkType.INTERFACE: ["interface_declaration"],
            ChunkType.TYPE_ALIAS: ["type_alias_declaration"],
            ChunkType.ENUM: ["enum_declaration"],
        }

        base_mappings.update(tsx_mappings)
        return base_mappings

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """TSX-specific chunk inclusion logic."""
        if chunk_type in [ChunkType.INTERFACE, ChunkType.TYPE_ALIAS, ChunkType.ENUM]:
            return True
        return super().should_include_chunk(node, chunk_type)


@register_chunking_strategy("typescript")
class TypeScriptChunkingStrategy(JavaScriptChunkingStrategy):
    """
    Chunking strategy for TypeScript code.

    Extends JavaScript strategy with TypeScript-specific features like
    interfaces, type aliases, and enhanced type annotations.
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get TypeScript-specific AST node type mappings."""
        base_mappings = super().get_node_mappings()

        # Add TypeScript-specific mappings
        typescript_mappings = {
            ChunkType.INTERFACE: ["interface_declaration"],
            ChunkType.TYPE_ALIAS: ["type_alias_declaration"],
            ChunkType.ENUM: ["enum_declaration"],
        }

        base_mappings.update(typescript_mappings)
        return base_mappings

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """TypeScript-specific chunk inclusion logic."""
        if chunk_type in [ChunkType.INTERFACE, ChunkType.TYPE_ALIAS, ChunkType.ENUM]:
            # Always include TypeScript type definitions
            return True

        # Fall back to JavaScript logic for other types
        return super().should_include_chunk(node, chunk_type)

    def extract_additional_metadata(self, node: Node, chunk: CodeChunk) -> dict[str, any]:
        """Extract TypeScript-specific metadata."""
        metadata = super().extract_additional_metadata(node, chunk)

        # Add TypeScript-specific metadata
        if chunk.chunk_type == ChunkType.INTERFACE:
            interface_features = self._extract_interface_features(node)
            metadata.update(interface_features)

        elif chunk.chunk_type == ChunkType.TYPE_ALIAS:
            type_features = self._extract_type_alias_features(node)
            metadata.update(type_features)

        # Extract generic type parameters
        generic_params = self._extract_generic_parameters(node)
        if generic_params:
            metadata["generic_parameters"] = generic_params

        return metadata

    def _extract_interface_features(self, node: Node) -> dict[str, any]:
        """Extract TypeScript interface features."""
        features = {}

        if node.type == "interface_declaration":
            # Check for interface inheritance
            for child in node.children:
                if child.type == "extends_clause":
                    features["has_inheritance"] = True
                    # Extract parent interfaces
                    extends_list = []
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            extends_list.append(subchild.text.decode("utf-8"))
                    features["extends"] = extends_list

        return features

    def _extract_type_alias_features(self, node: Node) -> dict[str, any]:
        """Extract TypeScript type alias features."""
        features = {}

        if node.type == "type_alias_declaration":
            # Analyze the type being aliased
            node_text = node.text.decode("utf-8") if node.text else ""

            if "union" in node_text or "|" in node_text:
                features["is_union_type"] = True

            if "intersection" in node_text or "&" in node_text:
                features["is_intersection_type"] = True

            if "Record<" in node_text or "Partial<" in node_text:
                features["uses_utility_types"] = True

        return features

    def _extract_generic_parameters(self, node: Node) -> list[str]:
        """Extract generic type parameters from TypeScript constructs."""
        generics = []

        # Look for type_parameters child
        for child in node.children:
            if child.type == "type_parameters":
                for param in child.children:
                    if param.type == "type_parameter":
                        for subchild in param.children:
                            if subchild.type == "type_identifier":
                                generics.append(subchild.text.decode("utf-8"))

        return generics


@register_chunking_strategy("go")
class GoChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy specifically designed for Go code.

    This strategy handles Go-specific constructs like functions, methods,
    structs, interfaces, constants, and package-level declarations.

    Uses custom AST traversal to maintain node-to-chunk mapping for accurate
    metadata extraction (e.g., method receivers, struct features).
    """

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Get Go-specific AST node type mappings."""
        return {
            ChunkType.FUNCTION: ["function_declaration", "method_declaration"],
            ChunkType.STRUCT: ["type_declaration"],  # Handles struct and interface types
            ChunkType.CONSTANT: ["const_declaration"],
            ChunkType.VARIABLE: ["var_declaration", "short_var_declaration"],
            ChunkType.IMPORT: ["import_declaration"],
        }

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract Go-specific chunks from the AST.

        Uses custom AST traversal to maintain node references for accurate
        metadata extraction (method receivers, struct features, etc.).
        """
        chunks = []
        content_lines = content.split("\n")

        # Use custom traversal that maintains node-to-chunk mapping
        self._traverse_go_ast(
            root_node,
            chunks,
            file_path,
            content,
            content_lines,
        )

        # Go-specific post-processing
        processed_chunks = []
        for chunk in chunks:
            if self._is_valid_go_chunk(chunk):
                processed_chunks.append(chunk)

        return self.post_process_chunks(processed_chunks)

    def _traverse_go_ast(
        self,
        node: Node,
        chunks: list[CodeChunk],
        file_path: str,
        content: str,
        content_lines: list[str],
    ) -> None:
        """
        Custom AST traversal for Go with proper node-to-chunk mapping.

        Args:
            node: Current AST node
            chunks: List to collect extracted chunks
            file_path: Path to the source file
            content: Original file content
            content_lines: Content split into lines
        """
        # Check if this node should be extracted as a chunk
        node_type = node.type

        if node_type == "function_declaration":
            chunk = self._extract_function_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return  # Don't traverse into function body

        elif node_type == "method_declaration":
            chunk = self._extract_method_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return  # Don't traverse into method body

        elif node_type == "type_declaration":
            if self._is_go_type_declaration(node):
                chunk = self._extract_type_chunk(node, file_path, content, content_lines)
                if chunk:
                    chunks.append(chunk)
            return

        elif node_type == "const_declaration":
            chunk = self._extract_const_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return

        elif node_type in ["var_declaration", "short_var_declaration"]:
            if self._is_package_level_var(node):
                chunk = self._extract_var_chunk(node, file_path, content, content_lines)
                if chunk:
                    chunks.append(chunk)
            return

        elif node_type == "import_declaration":
            chunk = self._extract_import_chunk(node, file_path, content, content_lines)
            if chunk:
                chunks.append(chunk)
            return

        # Recursively process children
        for child in node.children:
            self._traverse_go_ast(child, chunks, file_path, content, content_lines)

    def _extract_function_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a function declaration as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.FUNCTION, file_path, content, content_lines, self.language)
        if chunk:
            # Add Go-specific metadata with correct node reference
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["is_method"] = False
            if chunk.name:
                chunk.metadata["is_exported"] = chunk.name[0].isupper()
        return chunk

    def _extract_method_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a method declaration as a chunk with receiver info."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.FUNCTION, file_path, content, content_lines, self.language)
        if chunk:
            # Add Go-specific metadata with correct node reference
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["is_method"] = True

            # Extract receiver from the actual method node
            receiver = self._extract_receiver(node)
            if receiver:
                chunk.metadata["receiver"] = receiver

            if chunk.name:
                chunk.metadata["is_exported"] = chunk.name[0].isupper()
        return chunk

    def _extract_type_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a type declaration (struct/interface) as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.STRUCT, file_path, content, content_lines, self.language)
        if chunk:
            # Add Go-specific metadata with correct node reference
            chunk.metadata = chunk.metadata or {}

            # Extract struct/interface features from the actual node
            struct_features = self._extract_struct_features(node)
            chunk.metadata.update(struct_features)

            if chunk.name:
                chunk.metadata["is_exported"] = chunk.name[0].isupper()
        return chunk

    def _extract_const_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a const declaration as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.CONSTANT, file_path, content, content_lines, self.language)
        if chunk:
            chunk.metadata = chunk.metadata or {}
            if chunk.name:
                chunk.metadata["is_exported"] = chunk.name[0].isupper()
        return chunk

    def _extract_var_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract a var declaration as a chunk."""
        chunk = self.ast_extractor.create_chunk_from_node(node, ChunkType.VARIABLE, file_path, content, content_lines, self.language)
        if chunk:
            chunk.metadata = chunk.metadata or {}
            if chunk.name:
                chunk.metadata["is_exported"] = chunk.name[0].isupper()
        return chunk

    def _extract_import_chunk(self, node: Node, file_path: str, content: str, content_lines: list[str]) -> CodeChunk | None:
        """Extract an import declaration as a chunk."""
        return self.ast_extractor.create_chunk_from_node(node, ChunkType.IMPORT, file_path, content, content_lines, self.language)

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Determine if a Go node should be included as a chunk."""
        if chunk_type == ChunkType.FUNCTION:
            # Include all function and method declarations
            return True

        elif chunk_type == ChunkType.STRUCT:
            # Include type declarations (structs, interfaces)
            return self._is_go_type_declaration(node)

        elif chunk_type == ChunkType.CONSTANT:
            # Include const declarations
            return True

        elif chunk_type == ChunkType.VARIABLE:
            # Include package-level variable declarations
            return self._is_package_level_var(node)

        elif chunk_type == ChunkType.IMPORT:
            # Include import declarations
            return True

        return True

    def _is_valid_go_chunk(self, chunk: CodeChunk) -> bool:
        """Validate Go-specific chunk requirements."""
        if not self.validate_chunk(chunk):
            return False

        # Go-specific validations
        if chunk.chunk_type == ChunkType.FUNCTION:
            # Function should have a name and body
            return chunk.name and "{" in chunk.content

        elif chunk.chunk_type == ChunkType.STRUCT:
            # Type declaration should have a name
            return chunk.name and chunk.name != "unnamed_struct"

        elif chunk.chunk_type == ChunkType.CONSTANT:
            # Constant should have assignment
            return "=" in chunk.content or "iota" in chunk.content

        return True

    def _is_go_type_declaration(self, node: Node) -> bool:
        """Check if a node is a Go type declaration (struct or interface)."""
        if node.type != "type_declaration":
            return False

        # Check for struct_type or interface_type in children
        node_text = node.text.decode("utf-8") if node.text else ""
        return "struct" in node_text or "interface" in node_text

    def _is_package_level_var(self, node: Node) -> bool:
        """Check if a var declaration is at package level."""
        # In Go, package-level vars are typically outside function bodies
        # Simple heuristic: check parent is not a block
        if node.parent and node.parent.type == "block":
            return False
        return True

    def _extract_receiver(self, node: Node) -> str | None:
        """Extract method receiver from Go method declaration."""
        if node.type == "method_declaration":
            # Find parameter_list (receiver)
            for child in node.children:
                if child.type == "parameter_list":
                    receiver_text = child.text.decode("utf-8") if child.text else ""
                    if receiver_text:
                        return receiver_text
                    break
        return None

    def _extract_struct_features(self, node: Node) -> dict[str, any]:
        """Extract Go struct/interface features."""
        features = {}

        if node.type == "type_declaration":
            node_text = node.text.decode("utf-8") if node.text else ""

            if "struct" in node_text:
                features["type_kind"] = "struct"
                # Count fields (simplified)
                features["field_count"] = node_text.count("\n") - 1

            elif "interface" in node_text:
                features["type_kind"] = "interface"
                # Count methods in interface (simplified)
                features["method_count"] = node_text.count("(")

        return features


class PlainTextChunkingStrategy(BaseChunkingStrategy):
    """
    Chunking strategy for plain text files (.txt).

    This strategy handles plain text files by splitting them into meaningful
    chunks based on paragraphs (empty line separators) or fixed-size sections.
    """

    # Configuration
    MIN_PARAGRAPH_LENGTH = 50  # Minimum characters for a standalone paragraph
    MAX_CHUNK_SIZE = 2000  # Maximum characters per chunk
    PARAGRAPH_SEPARATOR = "\n\n"  # Double newline indicates paragraph break

    def __init__(self, language: str = "plaintext"):
        """Initialize the plain text strategy."""
        super().__init__(language)

    def get_node_mappings(self) -> dict[ChunkType, list[str]]:
        """Return empty mappings as plain text doesn't use AST."""
        return {}

    def extract_chunks(self, root_node: Node, file_path: str, content: str) -> list[CodeChunk]:
        """
        Extract chunks from plain text files based on paragraphs.

        Args:
            root_node: Root node (not used for plain text)
            file_path: Path to the source file
            content: Original file content

        Returns:
            List of extracted chunks based on paragraph structure
        """
        chunks = []

        # Split by paragraphs (double newlines)
        paragraphs = self._split_into_paragraphs(content)

        if not paragraphs:
            # Empty file, return single chunk
            return [self._create_single_chunk(file_path, content)]

        # Track line numbers
        current_line = 1

        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue

            # Find the start line of this paragraph
            start_line = self._find_paragraph_start_line(content, paragraph, current_line)
            paragraph_lines = paragraph.count("\n") + 1
            end_line = start_line + paragraph_lines - 1

            # Calculate byte positions
            start_byte = content.find(paragraph)
            end_byte = start_byte + len(paragraph.encode("utf-8"))

            # Generate content hash
            content_hash = hashlib.md5(paragraph.encode("utf-8")).hexdigest()

            # Create chunk
            chunk_id = f"{file_path}:para{i}:{content_hash[:8]}"

            chunk = CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                content=paragraph,
                chunk_type=ChunkType.SECTION,
                language=self.language,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte if start_byte >= 0 else 0,
                end_byte=end_byte,
                name=self._extract_paragraph_title(paragraph, i),
                signature=None,
                docstring=None,
                content_hash=content_hash,
            )

            chunks.append(chunk)
            current_line = end_line + 1

        # If no meaningful chunks were created, create a single whole-file chunk
        if not chunks:
            return [self._create_single_chunk(file_path, content)]

        return self.post_process_chunks(chunks)

    def should_include_chunk(self, node: Node, chunk_type: ChunkType) -> bool:
        """Include all chunks for plain text."""
        return True

    def _split_into_paragraphs(self, content: str) -> list[str]:
        """
        Split content into paragraphs based on empty lines.

        Merges small paragraphs and splits large ones to maintain reasonable chunk sizes.
        """
        # Split by double newlines (paragraph breaks)
        raw_paragraphs = content.split(self.PARAGRAPH_SEPARATOR)

        # Process paragraphs: merge small ones, split large ones
        processed = []
        current_chunk = []
        current_size = 0

        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If adding this paragraph would exceed max size, save current and start new
            if current_size + para_size > self.MAX_CHUNK_SIZE and current_chunk:
                processed.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            # If single paragraph is too large, split it
            if para_size > self.MAX_CHUNK_SIZE:
                if current_chunk:
                    processed.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences or fixed size
                split_parts = self._split_large_paragraph(para)
                processed.extend(split_parts)
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for separator

        # Add remaining chunk
        if current_chunk:
            processed.append("\n\n".join(current_chunk))

        return processed

    def _split_large_paragraph(self, paragraph: str) -> list[str]:
        """Split a large paragraph into smaller chunks."""
        chunks = []

        # Try to split by sentences first
        sentences = self._split_into_sentences(paragraph)

        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.MAX_CHUNK_SIZE and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_size + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences (simple heuristic)."""
        # Simple sentence splitting by common terminators
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_paragraph_start_line(self, content: str, paragraph: str, min_line: int) -> int:
        """Find the starting line number of a paragraph in the content."""
        lines = content.split("\n")
        paragraph_first_line = paragraph.split("\n")[0].strip()

        for i in range(min_line - 1, len(lines)):
            if lines[i].strip() == paragraph_first_line:
                return i + 1  # 1-indexed

        return min_line

    def _extract_paragraph_title(self, paragraph: str, index: int) -> str:
        """Extract a title or summary for the paragraph."""
        # Use first line as title if it's short enough
        first_line = paragraph.split("\n")[0].strip()

        if len(first_line) <= 60:
            return first_line[:50] + "..." if len(first_line) > 50 else first_line

        # Use first few words
        words = first_line.split()[:5]
        return " ".join(words) + "..."

    def _create_single_chunk(self, file_path: str, content: str) -> CodeChunk:
        """Create a single chunk for the entire file."""
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        content_lines = content.split("\n")

        chunk_id = f"{file_path}:{content_hash[:8]}"

        return CodeChunk(
            chunk_id=chunk_id,
            file_path=file_path,
            content=content,
            chunk_type=ChunkType.WHOLE_FILE,
            language=self.language,
            start_line=1,
            end_line=len(content_lines),
            start_byte=0,
            end_byte=len(content.encode("utf-8")),
            name="plaintext_file",
            signature=None,
            docstring=None,
            content_hash=content_hash,
        )


# Register plain text strategy
plaintext_strategy = PlainTextChunkingStrategy("plaintext")
chunking_strategy_registry.register_strategy("plaintext", plaintext_strategy)
chunking_strategy_registry.register_strategy("txt", plaintext_strategy)
