"""
Change detection service for incremental indexing.

This service compares the current state of files with stored metadata
to identify what files have been added, modified, or deleted since
the last indexing operation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from models.file_metadata import FileMetadata

from .file_metadata_service import FileMetadataService


class ChangeType(Enum):
    """Types of changes that can be detected."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    UNCHANGED = "unchanged"


@dataclass
class FileChange:
    """Represents a change to a file."""

    file_path: str
    change_type: ChangeType
    old_path: str | None = None  # For moved files
    metadata: FileMetadata | None = None  # Current file metadata
    old_metadata: FileMetadata | None = None  # Previous metadata


@dataclass
class ChangeDetectionResult:
    """Result of change detection analysis."""

    added_files: list[FileChange]
    modified_files: list[FileChange]
    deleted_files: list[FileChange]
    moved_files: list[FileChange]
    unchanged_files: list[FileChange]

    @property
    def total_changes(self) -> int:
        """Total number of files that changed."""
        return len(self.added_files) + len(self.modified_files) + len(self.deleted_files) + len(self.moved_files)

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return self.total_changes > 0

    def get_files_to_reindex(self) -> list[str]:
        """Get list of file paths that need to be reindexed."""
        files_to_reindex = []

        # Added and modified files need reindexing
        for change in self.added_files + self.modified_files:
            files_to_reindex.append(change.file_path)

        # Moved files need reindexing at new location
        for change in self.moved_files:
            files_to_reindex.append(change.file_path)

        return files_to_reindex

    def get_files_to_remove(self) -> list[str]:
        """Get list of file paths that need to be removed from index."""
        files_to_remove = []

        # Deleted files need removal
        for change in self.deleted_files:
            files_to_remove.append(change.file_path)

        # Moved files need removal from old location
        for change in self.moved_files:
            if change.old_path:
                files_to_remove.append(change.old_path)

        return files_to_remove

    def get_summary(self) -> dict[str, int]:
        """Get a summary of detected changes."""
        return {
            "added": len(self.added_files),
            "modified": len(self.modified_files),
            "deleted": len(self.deleted_files),
            "moved": len(self.moved_files),
            "unchanged": len(self.unchanged_files),
            "total_changes": self.total_changes,
        }


class ChangeDetectorService:
    """
    Service for detecting file changes since last indexing.

    This service compares the current state of files in a directory
    with stored metadata to identify changes for incremental indexing.
    """

    def __init__(self, metadata_service: FileMetadataService | None = None):
        """
        Initialize the change detector service.

        Args:
            metadata_service: Optional FileMetadataService instance
        """
        self.metadata_service = metadata_service or FileMetadataService()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def detect_changes(
        self,
        project_name: str,
        current_files: list[str],
        project_root: str | None = None,
    ) -> ChangeDetectionResult:
        """
        Detect changes between current files and stored metadata.

        Args:
            project_name: Name of the project
            current_files: List of current file paths to check
            project_root: Optional project root for relative path calculation

        Returns:
            ChangeDetectionResult with detailed change information
        """
        self.logger.info(f"Detecting changes for project '{project_name}' with {len(current_files)} current files")

        # Get stored metadata
        stored_metadata = self.metadata_service.get_project_file_metadata(project_name)
        self.logger.debug(f"Found {len(stored_metadata)} stored metadata entries")

        # Convert current files to absolute paths and create metadata
        current_file_metadata = {}
        valid_current_files = []

        for file_path in current_files:
            try:
                # Ensure absolute path
                abs_path = str(Path(file_path).resolve())

                # Skip if file doesn't exist (shouldn't happen, but be safe)
                if not Path(abs_path).exists():
                    self.logger.warning(f"File does not exist: {abs_path}")
                    continue

                # Create metadata for current file
                metadata = FileMetadata.from_file_path(abs_path, project_root)
                # Use the same path format as metadata for consistency
                current_file_metadata[metadata.file_path] = metadata
                valid_current_files.append(abs_path)

            except Exception as e:
                self.logger.warning(f"Failed to process file {file_path}: {e}")
                continue

        self.logger.debug(f"Successfully processed {len(valid_current_files)} current files")

        # Perform change detection
        result = self._analyze_changes(stored_metadata=stored_metadata, current_metadata=current_file_metadata)

        self.logger.info(f"Change detection complete: {result.get_summary()}")
        return result

    def _analyze_changes(
        self,
        stored_metadata: dict[str, FileMetadata],
        current_metadata: dict[str, FileMetadata],
    ) -> ChangeDetectionResult:
        """
        Analyze changes between stored and current metadata.

        Args:
            stored_metadata: Previously stored file metadata
            current_metadata: Current file metadata

        Returns:
            ChangeDetectionResult with categorized changes
        """
        added_files = []
        modified_files = []
        deleted_files = []
        moved_files = []
        unchanged_files = []

        stored_paths = set(stored_metadata.keys())
        current_paths = set(current_metadata.keys())

        # Find added files (in current but not in stored)
        for file_path in current_paths - stored_paths:
            change = FileChange(
                file_path=file_path,
                change_type=ChangeType.ADDED,
                metadata=current_metadata[file_path],
            )
            added_files.append(change)

        # Find deleted files (in stored but not in current)
        for file_path in stored_paths - current_paths:
            change = FileChange(
                file_path=file_path,
                change_type=ChangeType.DELETED,
                old_metadata=stored_metadata[file_path],
            )
            deleted_files.append(change)

        # Check existing files for modifications
        common_paths = stored_paths & current_paths
        for file_path in common_paths:
            stored_meta = stored_metadata[file_path]
            current_meta = current_metadata[file_path]

            # Check if file has changed
            if self._has_file_changed(stored_meta, current_meta):
                change = FileChange(
                    file_path=file_path,
                    change_type=ChangeType.MODIFIED,
                    metadata=current_meta,
                    old_metadata=stored_meta,
                )
                modified_files.append(change)
            else:
                change = FileChange(
                    file_path=file_path,
                    change_type=ChangeType.UNCHANGED,
                    metadata=current_meta,
                    old_metadata=stored_meta,
                )
                unchanged_files.append(change)

        # Detect moved files (same content hash but different path)
        # This is more complex and might be implemented in a future version
        # For now, moved files will be detected as deleted + added

        return ChangeDetectionResult(
            added_files=added_files,
            modified_files=modified_files,
            deleted_files=deleted_files,
            moved_files=moved_files,  # Empty for now
            unchanged_files=unchanged_files,
        )

    def _has_file_changed(self, stored_metadata: FileMetadata, current_metadata: FileMetadata) -> bool:
        """
        Determine if a file has changed based on metadata comparison.

        Args:
            stored_metadata: Previously stored metadata
            current_metadata: Current file metadata

        Returns:
            True if file has changed
        """
        # Primary check: modification time and size
        if stored_metadata.has_changed(current_metadata.mtime, current_metadata.file_size):
            # Secondary verification: content hash
            # Only calculate if quick check suggests change
            return stored_metadata.content_hash != current_metadata.content_hash

        return False

    def detect_moved_files(
        self,
        stored_metadata: dict[str, FileMetadata],
        current_metadata: dict[str, FileMetadata],
        similarity_threshold: float = 0.9,
    ) -> list[tuple[str, str]]:
        """
        Detect files that have been moved (same content, different path).

        This is an advanced feature that can be used to optimize handling
        of file moves by updating metadata instead of reindexing.

        Args:
            stored_metadata: Previously stored file metadata
            current_metadata: Current file metadata
            similarity_threshold: Minimum similarity to consider a move

        Returns:
            List of (old_path, new_path) tuples for detected moves
        """
        moved_files = []

        # Create content hash mappings
        stored_by_hash = {}
        for path, metadata in stored_metadata.items():
            hash_key = metadata.content_hash
            if hash_key not in stored_by_hash:
                stored_by_hash[hash_key] = []
            stored_by_hash[hash_key].append((path, metadata))

        current_by_hash = {}
        for path, metadata in current_metadata.items():
            hash_key = metadata.content_hash
            if hash_key not in current_by_hash:
                current_by_hash[hash_key] = []
            current_by_hash[hash_key].append((path, metadata))

        # Find files with same content hash but different paths
        for content_hash in stored_by_hash:
            if content_hash in current_by_hash:
                stored_files = stored_by_hash[content_hash]
                current_files = current_by_hash[content_hash]

                # Simple case: one-to-one mapping
                if len(stored_files) == 1 and len(current_files) == 1:
                    old_path, old_meta = stored_files[0]
                    new_path, new_meta = current_files[0]

                    if old_path != new_path:
                        # Additional checks for file moves
                        if self._is_likely_move(old_meta, new_meta):
                            moved_files.append((old_path, new_path))

        return moved_files

    def _is_likely_move(self, old_metadata: FileMetadata, new_metadata: FileMetadata) -> bool:
        """
        Determine if two files with same content hash represent a file move.

        Args:
            old_metadata: Metadata from old location
            new_metadata: Metadata from new location

        Returns:
            True if this appears to be a file move
        """
        # Same content hash is primary indicator
        if old_metadata.content_hash != new_metadata.content_hash:
            return False

        # Same file size (should be true if hash matches, but double-check)
        if old_metadata.file_size != new_metadata.file_size:
            return False

        # Similar modification time (within reasonable window)
        mtime_diff = abs(old_metadata.mtime - new_metadata.mtime)
        if mtime_diff > 3600:  # More than 1 hour difference seems unlikely for a move
            return False

        return True

    def get_change_summary_text(self, result: ChangeDetectionResult) -> str:
        """
        Generate a human-readable summary of changes.

        Args:
            result: Change detection result

        Returns:
            Formatted summary text
        """
        if not result.has_changes:
            return "No changes detected - all files are up to date."

        summary_parts = []

        if result.added_files:
            summary_parts.append(f"{len(result.added_files)} files added")

        if result.modified_files:
            summary_parts.append(f"{len(result.modified_files)} files modified")

        if result.deleted_files:
            summary_parts.append(f"{len(result.deleted_files)} files deleted")

        if result.moved_files:
            summary_parts.append(f"{len(result.moved_files)} files moved")

        summary = "Changes detected: " + ", ".join(summary_parts)

        # Add file count info
        total_files = result.total_changes + len(result.unchanged_files)
        summary += f" (out of {total_files} total files)"

        return summary
