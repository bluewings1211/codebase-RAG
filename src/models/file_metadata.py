"""
File metadata model for incremental indexing.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FileMetadata:
    """
    Metadata for tracking file changes in incremental indexing.

    This model stores essential information about indexed files to enable
    efficient change detection and selective reprocessing.
    """

    file_path: str  # Path to the file (relative to project root if available, otherwise absolute)
    mtime: float  # File modification timestamp (Unix timestamp)
    content_hash: str  # SHA256 hash of file content
    file_size: int  # File size in bytes
    indexed_at: datetime  # When this file was last indexed

    # Optional metadata for enhanced functionality
    relative_path: str | None = None  # Path relative to project root
    language: str | None = None  # Detected programming language
    chunk_count: int = 1  # Number of chunks created from this file
    collection_name: str | None = None  # Qdrant collection where this file is stored

    @classmethod
    def from_file_path(cls, file_path: str, project_root: str | None = None) -> "FileMetadata":
        """
        Create FileMetadata instance by reading file system information.

        Args:
            file_path: Path to the file (can be absolute or relative)
            project_root: Optional project root for relative path calculation

        Returns:
            FileMetadata instance with current file state.
            The file_path will be relative to project_root if provided, otherwise absolute.

        Raises:
            FileNotFoundError: If the file doesn't exist
            OSError: If file cannot be read
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file stats
        stat = path.stat()
        mtime = stat.st_mtime
        file_size = stat.st_size

        # Calculate content hash
        content_hash = cls._calculate_content_hash(file_path)

        # Calculate relative path if project root provided
        relative_path = None
        if project_root:
            try:
                relative_path = str(path.relative_to(Path(project_root)))
            except ValueError:
                # File is outside project root
                relative_path = None

        # Use relative path for file_path if available to match indexing behavior
        # This ensures consistency with IndexingService._sanitize_file_path()
        primary_path = relative_path if relative_path else str(path.absolute())

        return cls(
            file_path=primary_path,
            mtime=mtime,
            content_hash=content_hash,
            file_size=file_size,
            indexed_at=datetime.now(),
            relative_path=relative_path,
        )

    @staticmethod
    def _calculate_content_hash(file_path: str) -> str:
        """
        Calculate SHA256 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            Hexadecimal SHA256 hash string
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            # Return a consistent hash for unreadable files
            # This will cause them to be flagged as changed if they become readable
            return f"error_{hash(str(e)):#x}"

        return hash_sha256.hexdigest()

    def has_changed(self, current_mtime: float, current_size: int) -> bool:
        """
        Quick check if file has changed based on mtime and size.

        Args:
            current_mtime: Current file modification time
            current_size: Current file size

        Returns:
            True if file appears to have changed
        """
        return abs(self.mtime - current_mtime) > 0.001 or self.file_size != current_size  # Account for floating point precision

    def verify_content_hash(self, file_path: str | None = None) -> bool:
        """
        Verify if the current file content matches stored hash.

        Args:
            file_path: Optional path to verify, defaults to self.file_path

        Returns:
            True if content hash matches, False otherwise
        """
        try:
            path_to_check = file_path or self.file_path
            current_hash = self._calculate_content_hash(path_to_check)
            return current_hash == self.content_hash
        except Exception:
            return False

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for storage in Qdrant.

        Returns:
            Dictionary representation suitable for vector database storage
        """
        return {
            "file_path": self.file_path,
            "mtime": self.mtime,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "indexed_at": self.indexed_at.isoformat(),
            "relative_path": self.relative_path,
            "language": self.language,
            "chunk_count": self.chunk_count,
            "collection_name": self.collection_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileMetadata":
        """
        Create FileMetadata instance from dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            FileMetadata instance
        """
        indexed_at = datetime.fromisoformat(data["indexed_at"])

        return cls(
            file_path=data["file_path"],
            mtime=data["mtime"],
            content_hash=data["content_hash"],
            file_size=data["file_size"],
            indexed_at=indexed_at,
            relative_path=data.get("relative_path"),
            language=data.get("language"),
            chunk_count=data.get("chunk_count", 1),
            collection_name=data.get("collection_name"),
        )

    def __str__(self) -> str:
        """String representation for debugging."""
        return f"FileMetadata(path={self.relative_path or self.file_path}, size={self.file_size}, hash={self.content_hash[:8]}...)"
