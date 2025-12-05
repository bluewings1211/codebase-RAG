"""
Language support service for managing Tree-sitter parsers and language-specific configurations.

This service provides a high-level interface for managing programming language support,
including parser management, language detection, and configuration management.
"""

import logging
from pathlib import Path

try:
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")

from models.code_chunk import ChunkType
from utils.tree_sitter_manager import TreeSitterManager


class LanguageSupportService:
    """
    Service for managing programming language support and Tree-sitter parsers.

    This service provides a centralized interface for language detection,
    parser management, and language-specific configuration handling.
    """

    def __init__(self):
        """Initialize the language support service."""
        self.logger = logging.getLogger(__name__)

        # Use TreeSitterManager for robust parser management
        self._tree_sitter_manager = TreeSitterManager()

        # Get initialization summary for logging
        summary = self._tree_sitter_manager.get_initialization_summary()
        self.logger.info(f"Language support initialized: {summary['successful_languages']}/{summary['total_languages']} languages")

        if summary["failed_languages"]:
            self.logger.warning(f"Failed to initialize: {', '.join(summary['failed_languages'])}")

        # Language-specific node type mappings for chunk extraction
        self._node_mappings = self._initialize_node_mappings()

        # Language-specific configuration settings
        self._language_configs = self._initialize_language_configs()

    def _initialize_node_mappings(self) -> dict[str, dict[ChunkType, list[str]]]:
        """Initialize language-specific AST node type mappings."""
        return {
            "python": {
                ChunkType.FUNCTION: ["function_definition"],
                ChunkType.CLASS: ["class_definition"],
                ChunkType.CONSTANT: ["assignment"],  # Filtered by context
                ChunkType.VARIABLE: ["assignment"],
                ChunkType.IMPORT: ["import_statement", "import_from_statement"],
            },
            "javascript": {
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
            },
            "typescript": {
                ChunkType.FUNCTION: [
                    "function_declaration",
                    "arrow_function",
                    "method_definition",
                    "method_signature",
                ],
                ChunkType.ASYNC_FUNCTION: ["async_function_declaration"],
                ChunkType.CLASS: ["class_declaration"],
                ChunkType.INTERFACE: ["interface_declaration"],
                ChunkType.TYPE_ALIAS: ["type_alias_declaration"],
                ChunkType.CONSTANT: ["lexical_declaration"],
                ChunkType.VARIABLE: ["variable_declaration"],
                ChunkType.IMPORT: ["import_statement"],
                ChunkType.EXPORT: ["export_statement"],
            },
            "tsx": {
                ChunkType.FUNCTION: [
                    "function_declaration",
                    "arrow_function",
                    "method_definition",
                    "method_signature",
                ],
                ChunkType.ASYNC_FUNCTION: ["async_function_declaration"],
                ChunkType.CLASS: ["class_declaration"],
                ChunkType.INTERFACE: ["interface_declaration"],
                ChunkType.TYPE_ALIAS: ["type_alias_declaration"],
                ChunkType.CONSTANT: ["lexical_declaration"],
                ChunkType.VARIABLE: ["variable_declaration"],
                ChunkType.IMPORT: ["import_statement"],
                ChunkType.EXPORT: ["export_statement"],
            },
            "go": {
                ChunkType.FUNCTION: ["function_declaration", "method_declaration"],
                ChunkType.STRUCT: ["type_declaration"],  # Go structs and interfaces
                ChunkType.CONSTANT: ["const_declaration"],
                ChunkType.VARIABLE: ["var_declaration"],
                ChunkType.IMPORT: ["import_declaration"],
            },
            "rust": {
                ChunkType.FUNCTION: ["function_item"],
                ChunkType.STRUCT: ["struct_item"],
                ChunkType.ENUM: ["enum_item"],
                ChunkType.IMPL: ["impl_item"],
                ChunkType.CONSTANT: ["const_item"],
                ChunkType.VARIABLE: ["let_declaration"],
                ChunkType.IMPORT: ["use_declaration"],
            },
            "java": {
                ChunkType.FUNCTION: ["method_declaration"],
                ChunkType.CONSTRUCTOR: ["constructor_declaration"],
                ChunkType.CLASS: ["class_declaration"],
                ChunkType.INTERFACE: ["interface_declaration"],
                ChunkType.ENUM: ["enum_declaration"],
                ChunkType.CONSTANT: ["field_declaration"],  # Static final fields
                ChunkType.VARIABLE: ["field_declaration"],
                ChunkType.IMPORT: ["import_declaration"],
            },
            "cpp": {
                ChunkType.FUNCTION: ["function_definition", "function_declarator"],
                ChunkType.CLASS: ["class_specifier"],
                ChunkType.STRUCT: ["struct_specifier"],
                ChunkType.NAMESPACE: ["namespace_definition"],
                ChunkType.CONSTANT: ["declaration"],  # const declarations (filtered by context)
                ChunkType.VARIABLE: ["declaration"],  # variable declarations
                ChunkType.IMPORT: ["preproc_include"],  # #include statements
                ChunkType.CONSTRUCTOR: ["function_definition"],  # constructors (special handling needed)
                ChunkType.DESTRUCTOR: ["function_definition"],  # destructors (special handling needed)
                ChunkType.TEMPLATE: ["template_declaration"],  # template definitions
            },
        }

    def _initialize_language_configs(self) -> dict[str, dict[str, any]]:
        """Initialize language-specific configuration settings."""
        return {
            "python": {
                "supports_async": True,
                "supports_classes": True,
                "supports_interfaces": False,
                "supports_generics": False,
                "docstring_support": True,
                "comment_prefixes": ["#"],
                "string_delimiters": ['"', "'", '"""', "'''"],
            },
            "javascript": {
                "supports_async": True,
                "supports_classes": True,
                "supports_interfaces": False,
                "supports_generics": False,
                "docstring_support": False,
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"', "'", "`"],
            },
            "typescript": {
                "supports_async": True,
                "supports_classes": True,
                "supports_interfaces": True,
                "supports_generics": True,
                "docstring_support": False,
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"', "'", "`"],
            },
            "tsx": {
                "supports_async": True,
                "supports_classes": True,
                "supports_interfaces": True,
                "supports_generics": True,
                "docstring_support": False,
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"', "'", "`"],
            },
            "go": {
                "supports_async": False,
                "supports_classes": False,
                "supports_interfaces": True,
                "supports_generics": True,
                "docstring_support": False,
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"', "`"],
            },
            "rust": {
                "supports_async": True,
                "supports_classes": False,
                "supports_interfaces": True,  # traits
                "supports_generics": True,
                "docstring_support": True,
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"', "'"],
            },
            "java": {
                "supports_async": False,
                "supports_classes": True,
                "supports_interfaces": True,
                "supports_generics": True,
                "docstring_support": True,  # javadoc
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"'],
            },
            "cpp": {
                "supports_async": False,
                "supports_classes": True,
                "supports_interfaces": False,
                "supports_generics": True,  # templates
                "docstring_support": False,
                "comment_prefixes": ["//", "/*"],
                "string_delimiters": ['"', "'"],
            },
            "plaintext": {
                "supports_async": False,
                "supports_classes": False,
                "supports_interfaces": False,
                "supports_generics": False,
                "docstring_support": False,
                "comment_prefixes": [],
                "string_delimiters": [],
            },
        }

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return self._tree_sitter_manager.get_supported_languages()

    def get_failed_languages(self) -> list[str]:
        """Get list of languages that failed to initialize."""
        return self._tree_sitter_manager.get_failed_languages()

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported and available."""
        return self._tree_sitter_manager.is_language_supported(language)

    def detect_language(self, file_path: str) -> str | None:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language name if supported, None otherwise
        """
        # Use TreeSitterManager for primary language detection
        detected = self._tree_sitter_manager.detect_language_from_extension(file_path)

        # Handle special cases not covered by TreeSitterManager
        if detected is None:
            path = Path(file_path)
            extension = path.suffix.lower()

            # Handle structured file types
            if extension in [".json", ".jsonl"]:
                return "json"
            elif extension in [".yaml", ".yml"]:
                return "yaml"
            elif extension in [".md", ".markdown"]:
                return "markdown"
            elif extension == ".txt":
                return "plaintext"

        return detected

    def get_parser(self, language: str) -> Parser | None:
        """Get a parser for the specified language."""
        return self._tree_sitter_manager.get_parser(language)

    def get_language_object(self, language: str) -> Language | None:
        """Get a Language object for the specified language."""
        return self._tree_sitter_manager.get_language(language)

    def get_node_mappings(self, language: str) -> dict[ChunkType, list[str]] | None:
        """
        Get AST node type mappings for a specific language.

        Args:
            language: Programming language name

        Returns:
            Dictionary mapping chunk types to AST node types, or None if not supported
        """
        return self._node_mappings.get(language)

    def get_language_config(self, language: str) -> dict[str, any] | None:
        """
        Get configuration settings for a specific language.

        Args:
            language: Programming language name

        Returns:
            Dictionary with language configuration, or None if not supported
        """
        return self._language_configs.get(language)

    def supports_feature(self, language: str, feature: str) -> bool:
        """
        Check if a language supports a specific feature.

        Args:
            language: Programming language name
            feature: Feature name (e.g., 'async', 'classes', 'interfaces', 'generics')

        Returns:
            True if the language supports the feature, False otherwise
        """
        config = self.get_language_config(language)
        if not config:
            return False

        feature_key = f"supports_{feature}"
        return config.get(feature_key, False)

    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        # Get Tree-sitter supported extensions
        extensions = self._tree_sitter_manager.get_all_extensions()

        # Add structured file extensions
        extensions.update([".json", ".jsonl", ".yaml", ".yml", ".md", ".markdown", ".txt"])

        return extensions

    def get_language_info(self, language: str) -> dict[str, any] | None:
        """
        Get comprehensive information about a language.

        Args:
            language: Language name

        Returns:
            Dictionary with detailed language information or None if not supported
        """
        # Get basic info from TreeSitterManager
        info = self._tree_sitter_manager.get_language_info(language)
        if not info:
            return None

        # Add language-specific configuration
        config = self.get_language_config(language)
        if config:
            info.update(
                {
                    "features": {
                        "async": config.get("supports_async", False),
                        "classes": config.get("supports_classes", False),
                        "interfaces": config.get("supports_interfaces", False),
                        "generics": config.get("supports_generics", False),
                        "docstrings": config.get("docstring_support", False),
                    },
                    "comment_prefixes": config.get("comment_prefixes", []),
                    "string_delimiters": config.get("string_delimiters", []),
                }
            )

        # Add node mapping information
        mappings = self.get_node_mappings(language)
        if mappings:
            info["supported_chunk_types"] = list(mappings.keys())
            info["chunk_type_count"] = len(mappings)

        return info

    def reinitialize_language(self, language: str) -> bool:
        """
        Attempt to reinitialize a specific language parser.

        Args:
            language: Language name to reinitialize

        Returns:
            True if reinitialization succeeded, False otherwise
        """
        return self._tree_sitter_manager.reinitialize_language(language)

    def get_initialization_summary(self) -> dict[str, any]:
        """Get a summary of the language support initialization status."""
        base_summary = self._tree_sitter_manager.get_initialization_summary()

        # Add service-specific information
        base_summary.update(
            {
                "configured_languages": len(self._node_mappings),
                "languages_with_node_mappings": list(self._node_mappings.keys()),
                "languages_with_configs": list(self._language_configs.keys()),
                "structured_file_support": ["json", "yaml", "markdown", "plaintext"],
            }
        )

        return base_summary

    def validate_language_configuration(self, language: str) -> dict[str, any]:
        """
        Validate the configuration for a specific language.

        Args:
            language: Language name to validate

        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "language": language,
            "is_valid": True,
            "warnings": [],
            "errors": [],
        }

        # Check if language is supported by TreeSitter
        if not self.is_language_supported(language):
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Language '{language}' is not supported by TreeSitter")
            return validation_result

        # Check if node mappings exist
        if language not in self._node_mappings:
            validation_result["warnings"].append(f"No node mappings defined for '{language}'")

        # Check if language config exists
        if language not in self._language_configs:
            validation_result["warnings"].append(f"No language configuration defined for '{language}'")

        # Validate node mappings if they exist
        node_mappings = self.get_node_mappings(language)
        if node_mappings:
            if not node_mappings:
                validation_result["warnings"].append(f"Empty node mappings for '{language}'")
            elif len(node_mappings) < 2:
                validation_result["warnings"].append(f"Very few node mappings defined for '{language}' ({len(node_mappings)})")

        return validation_result
