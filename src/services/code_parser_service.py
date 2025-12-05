"""
CodeParser service for intelligent code chunking using Tree-sitter.

This refactored service acts as a coordinator that orchestrates the newly created
specialized services and chunking strategies to provide semantic code parsing capabilities.
"""

import logging
import time
from datetime import datetime

try:
    from tree_sitter import Parser
except ImportError:
    raise ImportError("Tree-sitter dependencies not installed. Run: poetry install")

from models.code_chunk import CodeChunk, CodeSyntaxError, ParseResult
from services.ast_extraction_service import AstExtractionService
from services.chunking_strategies import (
    FallbackChunkingStrategy,
    StructuredFileChunkingStrategy,
    chunking_strategy_registry,
)

# Import the new refactored services
from services.language_support_service import LanguageSupportService
from utils.chunking_metrics_tracker import chunking_metrics_tracker
from utils.fallback_tracker import fallback_tracker
from utils.file_system_utils import get_file_mtime, get_file_size


class CodeParserService:
    """
    Refactored coordinator service for parsing source code into intelligent semantic chunks.

    This service orchestrates specialized services and chunking strategies to provide
    semantic code parsing capabilities with enhanced modularity and maintainability.
    """

    def __init__(self):
        """Initialize the CodeParser coordinator with specialized services."""
        self.logger = logging.getLogger(__name__)

        # Initialize specialized services
        self.language_support = LanguageSupportService()
        self.ast_extractor = AstExtractionService()

        # Get initialization summary for logging
        summary = self.language_support.get_initialization_summary()
        self.logger.info(
            f"CodeParser coordinator initialized: {summary['successful_languages']}/{summary['total_languages']} languages supported"
        )

        if summary["failed_languages"]:
            self.logger.warning(f"Failed language support: {', '.join(summary['failed_languages'])}")

        # Performance metrics
        self._parse_stats = {
            "total_files_processed": 0,
            "total_chunks_extracted": 0,
            "total_processing_time_ms": 0,
            "strategy_usage": {},
            "error_recovery_count": 0,
        }

    # =================== Public API Methods ===================

    def get_supported_languages(self) -> list[str]:
        """Get list of supported programming languages."""
        return self.language_support.get_supported_languages()

    def detect_language(self, file_path: str) -> str | None:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to the source file

        Returns:
            Language name if supported, None otherwise
        """
        return self.language_support.detect_language(file_path)

    def parse_file(self, file_path: str, content: str | None = None) -> ParseResult:
        """
        Parse a source file into intelligent code chunks using the coordinator approach.

        This method orchestrates the language support service, chunking strategies,
        and AST extraction service to provide sophisticated code parsing capabilities.

        Args:
            file_path: Path to the source file
            content: Optional file content (if None, will read from file)

        Returns:
            ParseResult containing extracted chunks and metadata
        """
        start_time = time.time()

        try:
            # Step 1: Language detection
            language = self.detect_language(file_path)
            if not language:
                return self._create_fallback_result(
                    file_path,
                    content,
                    start_time,
                    error_message="Language not supported",
                )

            # Step 2: Content reading
            if content is None:
                content = self._read_file_content(file_path)
                if content is None:
                    return self._create_fallback_result(file_path, "", start_time, error=True)

            # Step 3: Get appropriate chunking strategy
            strategy = self._get_chunking_strategy(language)

            # Step 4: Execute chunking based on strategy type
            if language in ["json", "yaml", "markdown", "plaintext"]:
                # Use structured file chunking (no AST needed)
                chunks = strategy.extract_chunks(None, file_path, content)
                parse_success = True
                error_count = 0
                syntax_errors = []
                error_recovery_used = False
            else:
                # Use Tree-sitter based chunking
                parser = self.language_support.get_parser(language)
                if not parser:
                    return self._create_fallback_result(
                        file_path,
                        content,
                        start_time,
                        error_message=f"No parser available for {language}",
                    )

                # Parse with Tree-sitter
                tree = parser.parse(bytes(content, "utf8"))

                # Extract chunks using strategy
                chunks = strategy.extract_chunks(tree.root_node, file_path, content)

                # Collect error information
                error_count = self.ast_extractor.count_errors(tree.root_node)
                parse_success = error_count == 0
                syntax_errors = self.ast_extractor.traverse_for_errors(tree.root_node, content.split("\n"), language)
                error_recovery_used = error_count > 0 and len(chunks) > 0

                if error_recovery_used:
                    self._parse_stats["error_recovery_count"] += 1

            # Step 5: Post-process and validate chunks
            validated_chunks = self._validate_and_enhance_chunks(chunks, language, strategy)

            # Step 6: Calculate metrics and create result
            processing_time = (time.time() - start_time) * 1000

            parse_result = ParseResult(
                chunks=validated_chunks,
                file_path=file_path,
                language=language,
                parse_success=parse_success,
                error_count=error_count,
                fallback_used=False,
                processing_time_ms=processing_time,
                syntax_errors=syntax_errors,
                error_recovery_used=error_recovery_used,
                valid_sections_count=len(validated_chunks),
            )

            # Step 7: Update statistics and metrics
            self._update_parsing_statistics(parse_result, language, strategy)

            # Step 8: Log results
            self._log_parsing_results(parse_result, file_path)

            return parse_result

        except Exception as e:
            self.logger.error(f"Critical parsing failure for {file_path}: {e}")
            return self._create_fallback_result(
                file_path,
                content or "",
                start_time,
                error=True,
                exception_context=str(e),
            )

    # =================== Coordinator Helper Methods ===================

    def _read_file_content(self, file_path: str) -> str | None:
        """Read file content with error handling."""
        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError) as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def _get_chunking_strategy(self, language: str, fallback_reason: str = "language_not_supported"):
        """Get the appropriate chunking strategy for a language.

        Args:
            language: The programming language
            fallback_reason: Reason for using fallback if needed

        Returns:
            Appropriate chunking strategy
        """
        strategy = chunking_strategy_registry.get_strategy(language)
        if strategy:
            return strategy

        # Fallback to language-specific fallback strategy
        if language in ["json", "yaml", "markdown"]:
            return StructuredFileChunkingStrategy(language)
        elif language == "plaintext":
            # Import PlainTextChunkingStrategy
            from services.chunking_strategies import PlainTextChunkingStrategy

            return PlainTextChunkingStrategy(language)
        else:
            self.logger.warning(f"No Tree-sitter strategy for language '{language}' - using fallback chunking")
            return FallbackChunkingStrategy(language, reason=fallback_reason)

    def _validate_and_enhance_chunks(self, chunks: list[CodeChunk], language: str, strategy) -> list[CodeChunk]:
        """Validate and enhance chunks with additional metadata."""
        validated_chunks = []

        for chunk in chunks:
            # Validate chunk using strategy
            if strategy.validate_chunk(chunk):
                # Add coordinator-level metadata
                self._enhance_chunk_metadata(chunk, language)
                validated_chunks.append(chunk)
            else:
                self.logger.debug(f"Chunk validation failed for {chunk.name} in {chunk.file_path}")

        return validated_chunks

    def _enhance_chunk_metadata(self, chunk: CodeChunk, language: str):
        """Add coordinator-level metadata to chunks."""
        # Add file-level metadata
        chunk.file_size = get_file_size(chunk.file_path)
        chunk.file_mtime = get_file_mtime(chunk.file_path)

        # Add processing timestamp
        chunk.processed_at = datetime.utcnow().isoformat()

        # Add language features
        if hasattr(chunk, "metadata"):
            chunk.metadata = getattr(chunk, "metadata", {})
        else:
            chunk.metadata = {}

        chunk.metadata["language_features"] = self.language_support.get_language_config(language)

    def _update_parsing_statistics(self, parse_result: ParseResult, language: str, strategy):
        """Update internal parsing statistics."""
        self._parse_stats["total_files_processed"] += 1
        self._parse_stats["total_chunks_extracted"] += len(parse_result.chunks)
        self._parse_stats["total_processing_time_ms"] += parse_result.processing_time_ms

        # Track strategy usage
        strategy_name = strategy.__class__.__name__
        if strategy_name not in self._parse_stats["strategy_usage"]:
            self._parse_stats["strategy_usage"][strategy_name] = 0
        self._parse_stats["strategy_usage"][strategy_name] += 1

        # Record metrics with chunking tracker
        if chunking_metrics_tracker:
            chunking_metrics_tracker.record_file_processed(
                language=language,
                chunk_count=len(parse_result.chunks),
                processing_time_ms=parse_result.processing_time_ms,
                parse_success=parse_result.parse_success,
                error_count=parse_result.error_count,
            )

    def _log_parsing_results(self, parse_result: ParseResult, file_path: str):
        """Log parsing results with appropriate detail level."""
        if parse_result.parse_success:
            self.logger.debug(
                f"Successfully parsed {file_path}: {len(parse_result.chunks)} chunks, " f"{parse_result.processing_time_ms:.1f}ms"
            )
        else:
            self.logger.warning(
                f"Parsed {file_path} with {parse_result.error_count} errors: "
                f"{len(parse_result.chunks)} chunks recovered, "
                f"{parse_result.processing_time_ms:.1f}ms"
            )

            # Log detailed error information
            for error in parse_result.syntax_errors[:5]:  # Limit to first 5 errors
                self.logger.debug(f"  Error at line {error.line}: {error.error_text}")

    def get_parsing_statistics(self) -> dict:
        """Get current parsing statistics."""
        return self._parse_stats.copy()

    def reset_parsing_statistics(self):
        """Reset internal parsing statistics."""
        self._parse_stats = {
            "total_files_processed": 0,
            "total_chunks_extracted": 0,
            "total_processing_time_ms": 0,
            "strategy_usage": {},
            "error_recovery_count": 0,
        }
        # Also reset fallback tracker
        fallback_tracker.reset()

    def get_fallback_report(self) -> dict:
        """Get a report of fallback chunking usage.

        Returns:
            Dictionary with fallback usage statistics and recommendations
        """
        return fallback_tracker.get_summary()

    def log_fallback_report(self) -> None:
        """Log a formatted report of fallback usage to help identify languages needing support."""
        fallback_tracker.log_report()

    def get_recommended_tree_sitter_languages(self) -> list[dict]:
        """Get a prioritized list of languages that would benefit from Tree-sitter support.

        Returns:
            List of dictionaries with extension, language name, and priority info
        """
        return fallback_tracker.get_recommended_languages()

    def _create_fallback_result(
        self,
        file_path: str,
        content: str | None,
        start_time: float,
        error: bool = False,
        error_message: str = None,
        exception_context: str | None = None,
    ) -> ParseResult:
        """Create a simplified fallback ParseResult for the coordinator approach."""
        if content is None:
            content = ""

        processing_time = (time.time() - start_time) * 1000
        language = self.detect_language(file_path) or "unknown"

        # Determine fallback reason
        if error_message:
            if "not supported" in error_message.lower():
                fallback_reason = "language_not_supported"
            elif "parser" in error_message.lower():
                fallback_reason = "parser_unavailable"
            else:
                fallback_reason = "parse_error"
        elif exception_context:
            fallback_reason = "parse_error"
        else:
            fallback_reason = "language_not_supported"

        # Create a simple whole-file chunk using fallback strategy
        fallback_strategy = FallbackChunkingStrategy(language, reason=fallback_reason)
        chunks = []

        if content:
            # Create a dummy root node for structured compatibility
            chunks = fallback_strategy.extract_chunks(None, file_path, content)

        # Create simple error information
        syntax_errors = []
        if error and (error_message or exception_context):
            error_details = error_message or f"Exception: {exception_context}"
            syntax_errors.append(
                CodeSyntaxError(
                    start_line=1,
                    start_column=0,
                    end_line=len(content.split("\n")) if content else 1,
                    end_column=0,
                    error_type="parsing_error",
                    context=error_details,
                )
            )

        return ParseResult(
            chunks=chunks,
            file_path=file_path,
            language=language,
            parse_success=not error,
            error_count=1 if error else 0,
            fallback_used=True,
            processing_time_ms=processing_time,
            syntax_errors=syntax_errors,
            error_recovery_used=False,
            valid_sections_count=len(chunks),
        )
