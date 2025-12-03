"""
File Discovery Service for efficient file system traversal and analysis.

This service provides file discovery, filtering, and analysis capabilities
for indexing operations with support for .ragignore patterns and project detection.
"""

import logging
import os
from pathlib import Path
from typing import Any

from services.project_analysis_service import ProjectAnalysisService

logger = logging.getLogger(__name__)


class FileDiscoveryService:
    """
    Service for discovering and analyzing files for indexing operations.

    This service handles project detection, file filtering, and provides
    analytics for indexing planning and optimization.
    """

    def __init__(self):
        """Initialize the file discovery service."""
        self.logger = logger
        self.project_analysis = ProjectAnalysisService()

    def discover_project_files(self, directory: str) -> dict[str, Any]:
        """
        Discover and analyze all relevant files in a project directory.

        Args:
            directory: Root directory to analyze

        Returns:
            Dictionary with file discovery results and statistics
        """
        try:
            dir_path = Path(directory).resolve()

            # Get project context
            project_context = self.project_analysis.get_project_context(str(dir_path))

            # Get all relevant files
            relevant_files = self.project_analysis.get_relevant_files(str(dir_path))

            # Get directory structure analysis
            structure_analysis = self.project_analysis.analyze_directory_structure(str(dir_path))

            return {
                "project_context": project_context,
                "relevant_files": relevant_files,
                "file_count": len(relevant_files),
                "structure_analysis": structure_analysis,
                "directory": str(dir_path),
            }

        except Exception as e:
            self.logger.error(f"Failed to discover project files: {e}")
            return {
                "error": str(e),
                "directory": directory,
                "file_count": 0,
                "relevant_files": [],
            }

    def estimate_processing_requirements(self, directory: str, mode: str = "clear_existing") -> dict[str, Any]:
        """
        Estimate processing requirements for indexing operation.

        Args:
            directory: Directory to analyze
            mode: Processing mode ("clear_existing" or "incremental")

        Returns:
            Dictionary with processing estimates and recommendations
        """
        try:
            # Get basic analysis
            analysis_result = self.project_analysis.analyze_directory_structure(directory)

            total_files = analysis_result.get("relevant_files", 0)
            total_size_mb = analysis_result.get("size_analysis", {}).get("total_size_mb", 0)

            # For incremental mode, check for actual changes
            files_to_process = total_files
            change_summary = None

            if mode == "incremental":
                try:
                    from services.change_detector_service import ChangeDetectorService
                    from services.file_metadata_service import FileMetadataService
                    from services.qdrant_service import QdrantService

                    # Get project name
                    project_context = self.project_analysis.get_project_context(directory)
                    project_name = project_context.get("project_name", "unknown")

                    # Initialize services for change detection
                    qdrant_service = QdrantService()
                    metadata_service = FileMetadataService(qdrant_service)
                    change_detector = ChangeDetectorService(metadata_service)

                    # Get relevant files and detect changes
                    relevant_files = self.project_analysis.get_relevant_files(directory)
                    changes = change_detector.detect_changes(
                        project_name=project_name,
                        current_files=relevant_files,
                        project_root=directory,
                    )

                    # Update files_to_process based on actual changes
                    files_to_process = len(changes.get_files_to_reindex())
                    change_summary = changes.get_summary()

                    self.logger.debug(f"Incremental mode: {files_to_process} files need processing")

                except Exception as e:
                    self.logger.warning(f"Could not perform change detection: {e}")
                    # Fall back to total files if change detection fails

            # Calculate estimates based on files to process
            base_rate = 100  # files per minute baseline
            size_factor = max(1.0, total_size_mb / 10)  # Adjust for file size
            estimated_minutes = (files_to_process / base_rate) * size_factor if files_to_process > 0 else 0

            # Determine recommendation
            if estimated_minutes > 30:
                recommendation = "large_operation_confirm"
                use_manual_tool = True
            elif estimated_minutes > 10:
                recommendation = "medium_operation"
                use_manual_tool = False
            else:
                recommendation = "quick_operation"
                use_manual_tool = False

            result = {
                "file_count": files_to_process,
                "total_files": total_files,
                "total_size_mb": total_size_mb,
                "estimated_minutes": round(estimated_minutes, 1),
                "recommendation": recommendation,
                "use_manual_tool_recommended": use_manual_tool,
                "complexity_analysis": analysis_result.get("indexing_complexity", {}),
            }

            if change_summary:
                result["change_summary"] = change_summary

            return result

        except Exception as e:
            self.logger.error(f"Failed to estimate processing requirements: {e}")
            return {
                "error": str(e),
                "file_count": 0,
                "estimated_minutes": 0.0,
                "recommendation": "error",
            }

    def validate_directory(self, directory: str) -> tuple[bool, str]:
        """
        Validate that a directory is suitable for indexing.

        Args:
            directory: Directory path to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            dir_path = Path(directory)

            if not dir_path.exists():
                return False, f"Directory does not exist: {directory}"

            if not dir_path.is_dir():
                return False, f"Path is not a directory: {directory}"

            if not os.access(dir_path, os.R_OK):
                return False, f"Directory is not readable: {directory}"

            # Check if directory has any indexable content
            try:
                relevant_files = self.project_analysis.get_relevant_files(str(dir_path))
                if not relevant_files:
                    return False, f"No indexable files found in directory: {directory}"
            except Exception as e:
                self.logger.warning(f"Could not check file content: {e}")
                # Still consider valid if we can access the directory

            return True, ""

        except Exception as e:
            return False, f"Directory validation failed: {str(e)}"

    def get_file_statistics(self, directory: str) -> dict[str, Any]:
        """
        Get detailed file statistics for a directory.

        Args:
            directory: Directory to analyze

        Returns:
            Dictionary with detailed file statistics
        """
        try:
            # Use existing project analysis capabilities
            analysis = self.project_analysis.analyze_repository(directory)

            # Extract key statistics
            return {
                "total_files": analysis.get("total_files", 0),
                "relevant_files": analysis.get("relevant_files", 0),
                "excluded_files": analysis.get("excluded_files", 0),
                "exclusion_rate": analysis.get("exclusion_rate", 0),
                "languages": analysis.get("languages", {}),
                "size_analysis": analysis.get("size_analysis", {}),
                "directory_breakdown": analysis.get("directory_breakdown", {}),
            }

        except Exception as e:
            self.logger.error(f"Failed to get file statistics: {e}")
            return {
                "error": str(e),
                "total_files": 0,
                "relevant_files": 0,
            }
