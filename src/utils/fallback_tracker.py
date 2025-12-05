"""
Fallback Tracker utility for tracking Tree-sitter fallback usage.

This module provides centralized tracking and reporting of files that
fall back to whole-file chunking due to lack of Tree-sitter support.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass
class FallbackFileInfo:
    """Information about a file that used fallback chunking."""

    file_path: str
    extension: str
    char_count: int
    line_count: int
    estimated_tokens: int
    reason: str  # "language_not_supported", "parser_unavailable", "parse_error"
    was_truncated: bool = False  # True if chunk exceeded threshold and was split


@dataclass
class ExtensionStats:
    """Statistics for a specific file extension."""

    file_count: int = 0
    total_chars: int = 0
    total_lines: int = 0
    total_estimated_tokens: int = 0
    truncated_count: int = 0
    files: list[str] = field(default_factory=list)

    @property
    def avg_chars(self) -> float:
        return self.total_chars / self.file_count if self.file_count > 0 else 0

    @property
    def avg_tokens(self) -> float:
        return self.total_estimated_tokens / self.file_count if self.file_count > 0 else 0


class FallbackTracker:
    """
    Singleton tracker for monitoring Tree-sitter fallback usage.

    This class tracks files that cannot be parsed with Tree-sitter and
    provides reporting capabilities to identify which languages need support.
    """

    _instance = None
    _lock = Lock()

    # Token estimation: ~4 chars per token for code (conservative estimate)
    CHARS_PER_TOKEN = 4

    # Threshold for considering a chunk "large" (in estimated tokens)
    LARGE_CHUNK_TOKEN_THRESHOLD = 6000

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the tracker."""
        if self._initialized:
            return

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._fallback_files: list[FallbackFileInfo] = []
        self._extension_stats: dict[str, ExtensionStats] = defaultdict(ExtensionStats)
        self._data_lock = Lock()
        self._initialized = True

    def record_fallback(
        self,
        file_path: str,
        content: str,
        reason: str = "language_not_supported",
        was_truncated: bool = False,
    ) -> FallbackFileInfo:
        """
        Record a file that used fallback chunking.

        Args:
            file_path: Path to the file
            content: File content
            reason: Reason for fallback (language_not_supported, parser_unavailable, parse_error)
            was_truncated: Whether the chunk was split due to size

        Returns:
            FallbackFileInfo with recorded details
        """
        extension = Path(file_path).suffix.lower() or "(no extension)"
        char_count = len(content)
        line_count = content.count("\n") + 1
        estimated_tokens = char_count // self.CHARS_PER_TOKEN

        info = FallbackFileInfo(
            file_path=file_path,
            extension=extension,
            char_count=char_count,
            line_count=line_count,
            estimated_tokens=estimated_tokens,
            reason=reason,
            was_truncated=was_truncated,
        )

        with self._data_lock:
            self._fallback_files.append(info)

            # Update extension stats
            stats = self._extension_stats[extension]
            stats.file_count += 1
            stats.total_chars += char_count
            stats.total_lines += line_count
            stats.total_estimated_tokens += estimated_tokens
            if was_truncated:
                stats.truncated_count += 1
            stats.files.append(file_path)

        # Log warning for large files
        if estimated_tokens > self.LARGE_CHUNK_TOKEN_THRESHOLD:
            self.logger.warning(
                f"Large fallback chunk: {char_count:,} chars (~{estimated_tokens:,} tokens) " f"for {file_path} [{extension}]"
            )
        else:
            self.logger.info(f"Tree-sitter fallback used for {file_path} [{extension}]: {reason}")

        return info

    def get_extension_report(self) -> dict[str, ExtensionStats]:
        """Get statistics grouped by file extension."""
        with self._data_lock:
            return dict(self._extension_stats)

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of fallback usage.

        Returns:
            Dictionary with summary statistics
        """
        with self._data_lock:
            total_files = len(self._fallback_files)
            total_chars = sum(f.char_count for f in self._fallback_files)
            total_tokens = sum(f.estimated_tokens for f in self._fallback_files)
            truncated_files = sum(1 for f in self._fallback_files if f.was_truncated)

            # Group by reason
            by_reason: dict[str, int] = defaultdict(int)
            for f in self._fallback_files:
                by_reason[f.reason] += 1

            # Extensions sorted by token count (most impactful first)
            extensions_by_impact = sorted(
                self._extension_stats.items(),
                key=lambda x: x[1].total_estimated_tokens,
                reverse=True,
            )

            return {
                "total_fallback_files": total_files,
                "total_chars": total_chars,
                "total_estimated_tokens": total_tokens,
                "truncated_files": truncated_files,
                "by_reason": dict(by_reason),
                "extensions_by_impact": [
                    {
                        "extension": ext,
                        "file_count": stats.file_count,
                        "avg_chars": round(stats.avg_chars),
                        "avg_tokens": round(stats.avg_tokens),
                        "truncated_count": stats.truncated_count,
                    }
                    for ext, stats in extensions_by_impact
                ],
            }

    def log_report(self) -> None:
        """Log a formatted report of fallback usage."""
        summary = self.get_summary()

        if summary["total_fallback_files"] == 0:
            self.logger.info("No fallback chunking was used - all files parsed with Tree-sitter")
            return

        self.logger.info("=" * 60)
        self.logger.info("FALLBACK CHUNKING REPORT")
        self.logger.info("=" * 60)
        self.logger.info(f"Total files using fallback: {summary['total_fallback_files']}")
        self.logger.info(f"Total characters: {summary['total_chars']:,}")
        self.logger.info(f"Total estimated tokens: {summary['total_estimated_tokens']:,}")

        if summary["truncated_files"] > 0:
            self.logger.warning(
                f"Files requiring smart splitting: {summary['truncated_files']} "
                f"(exceeded {self.LARGE_CHUNK_TOKEN_THRESHOLD} token threshold)"
            )

        # Log by reason
        self.logger.info("-" * 40)
        self.logger.info("Fallback reasons:")
        for reason, count in summary["by_reason"].items():
            self.logger.info(f"  {reason}: {count} files")

        # Log extensions needing support
        self.logger.info("-" * 40)
        self.logger.info("Extensions needing Tree-sitter support (by impact):")

        for ext_info in summary["extensions_by_impact"]:
            truncated_note = ""
            if ext_info["truncated_count"] > 0:
                truncated_note = f" [{ext_info['truncated_count']} truncated]"

            self.logger.info(
                f"  {ext_info['extension']}: {ext_info['file_count']} files, " f"avg ~{ext_info['avg_tokens']} tokens/file{truncated_note}"
            )

        self.logger.info("=" * 60)

    def get_recommended_languages(self) -> list[dict[str, Any]]:
        """
        Get a prioritized list of languages that would benefit from Tree-sitter support.

        Returns:
            List of dictionaries with extension and priority info
        """
        summary = self.get_summary()
        recommendations = []

        # Known language mappings for common extensions
        extension_to_language = {
            ".cs": "C#",
            ".php": "PHP",
            ".rb": "Ruby",
            ".swift": "Swift",
            ".kt": "Kotlin",
            ".kts": "Kotlin Script",
            ".scala": "Scala",
            ".sh": "Shell/Bash",
            ".bash": "Bash",
            ".zsh": "Zsh",
            ".sql": "SQL",
            ".vue": "Vue",
            ".svelte": "Svelte",
            ".html": "HTML",
            ".htm": "HTML",
            ".css": "CSS",
            ".scss": "SCSS",
            ".sass": "Sass",
            ".less": "Less",
            ".lua": "Lua",
            ".r": "R",
            ".R": "R",
            ".pl": "Perl",
            ".pm": "Perl",
            ".ex": "Elixir",
            ".exs": "Elixir",
            ".erl": "Erlang",
            ".hrl": "Erlang",
            ".hs": "Haskell",
            ".ml": "OCaml",
            ".mli": "OCaml",
            ".fs": "F#",
            ".fsx": "F#",
            ".dart": "Dart",
            ".clj": "Clojure",
            ".cljs": "ClojureScript",
            ".groovy": "Groovy",
            ".gradle": "Gradle",
        }

        for ext_info in summary["extensions_by_impact"]:
            ext = ext_info["extension"]
            language = extension_to_language.get(ext, f"Unknown ({ext})")

            # Calculate priority based on file count and token impact
            priority_score = ext_info["file_count"] * ext_info["avg_tokens"]

            recommendations.append(
                {
                    "extension": ext,
                    "language": language,
                    "file_count": ext_info["file_count"],
                    "avg_tokens": ext_info["avg_tokens"],
                    "priority_score": priority_score,
                    "truncated_count": ext_info["truncated_count"],
                }
            )

        # Sort by priority score
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)

        return recommendations

    def reset(self) -> None:
        """Reset all tracked data."""
        with self._data_lock:
            self._fallback_files.clear()
            self._extension_stats.clear()

        self.logger.debug("Fallback tracker reset")

    def has_fallbacks(self) -> bool:
        """Check if any fallbacks were recorded."""
        with self._data_lock:
            return len(self._fallback_files) > 0


# Global singleton instance
fallback_tracker = FallbackTracker()
