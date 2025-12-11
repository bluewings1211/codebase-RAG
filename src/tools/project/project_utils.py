"""Project utility functions for project detection and management.

This module provides utility functions for detecting project boundaries,
managing project configuration, and handling collection naming.
"""

import logging
from pathlib import Path
from typing import Any

# Re-export get_qdrant_client for easier mocking in tests
from tools.database.qdrant_utils import get_qdrant_client

# Configure logging
logger = logging.getLogger(__name__)

# Export for external use
__all__ = [
    "get_qdrant_client",
    "get_current_project",
    "get_collection_name",
    "load_ragignore_patterns",
    "clear_project_collections",
    "list_indexed_projects",
    "get_available_project_names",
    "validate_project_exists",
    "get_project_collections",
    "normalize_project_name",
    "get_project_metadata",
    "delete_file_chunks",
]

# Global variables for caching
_current_project = None

# Configuration
PROJECT_MARKERS = [
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
]


def get_current_project(
    client_directory: str | None = None,
) -> dict[str, str] | None:
    """
    Detect the current project based on common project markers.

    This function traverses up the directory tree looking for project markers
    like .git, pyproject.toml, package.json, etc.

    Args:
        client_directory: Optional directory to start search from (default: current directory)

    Returns:
        Dictionary with project information or None if no project found
    """
    global _current_project

    if client_directory:
        cwd = Path(client_directory).resolve()
    else:
        cwd = Path.cwd()

    # Search for project markers in current directory and parent directories
    for parent in [cwd] + list(cwd.parents):
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                _current_project = {
                    "name": project_name,
                    "root": str(parent),
                    "collection_prefix": f"project_{project_name}",
                }
                logger.debug(f"Detected project: {project_name} at {parent}")
                return _current_project

    # Fallback to directory name if no project markers found
    project_name = cwd.name.replace(" ", "_").replace("-", "_")
    _current_project = {
        "name": project_name,
        "root": str(cwd),
        "collection_prefix": f"dir_{project_name}",
    }
    logger.debug(f"No project markers found, using directory name: {project_name}")
    return _current_project


def get_collection_name(file_path: str, file_type: str = "code") -> str:
    """
    Generate a collection name for a file based on project context and file type.

    Args:
        file_path: Path to the file
        file_type: Type of content (code, config, documentation)

    Returns:
        Collection name string
    """
    path = Path(file_path).resolve()

    # Try to use current project context
    current_project = get_current_project()
    if current_project:
        project_root = Path(current_project["root"])
        try:
            # Check if file is within project root
            path.relative_to(project_root)
            return f"{current_project['collection_prefix']}_{file_type}"
        except ValueError:
            # File is outside project root, continue with fallback logic
            pass

    # Fallback: search for project markers in file's directory tree
    for parent in path.parents:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                return f"project_{project_name}_{file_type}"

    # Final fallback: use global collection
    return f"global_{file_type}"


def load_ragignore_patterns(directory: Path) -> tuple[set[str], set[str]]:
    """
    Load .ragignore patterns for excluding files and directories from indexing.

    Args:
        directory: Directory to start search for .ragignore file

    Returns:
        Tuple of (excluded_directories, excluded_patterns)
    """
    exclude_dirs = set()
    exclude_patterns = set()

    # Default exclusions
    default_exclude_dirs = {
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        ".pytest_cache",
        ".mypy_cache",
        ".coverage",
        "htmlcov",
        ".tox",
        "data",
        "logs",
        "tmp",
        "temp",
        ".idea",
        ".vscode",
        ".vs",
        "qdrant_storage",
        "models",
        ".cache",
    }

    default_exclude_patterns = {
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.class",
        "*.log",
        "*.lock",
        "*.swp",
        "*.swo",
        "*.bak",
        "*.tmp",
        "*.temp",
        "*.old",
        "*.orig",
        "*.rej",
        ".env*",
        "*.sqlite",
        "*.db",
        "*.pid",
    }

    # Search for .ragignore file in directory tree
    ragignore_path = None
    for parent in [directory] + list(directory.parents):
        potential_path = parent / ".ragignore"
        if potential_path.exists():
            ragignore_path = potential_path
            break

    if not ragignore_path:
        logger.debug("No .ragignore file found, using default exclusions")
        return default_exclude_dirs, default_exclude_patterns

    try:
        logger.debug(f"Loading .ragignore patterns from {ragignore_path}")
        with open(ragignore_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.endswith("/"):
                    exclude_dirs.add(line.rstrip("/"))
                else:
                    exclude_patterns.add(line)

        logger.debug(f"Loaded {len(exclude_dirs)} directory patterns and {len(exclude_patterns)} file patterns")

    except Exception as e:
        logger.warning(f"Error reading .ragignore: {e}, using defaults")
        return default_exclude_dirs, default_exclude_patterns

    # If no patterns loaded, use defaults
    if not exclude_dirs and not exclude_patterns:
        return default_exclude_dirs, default_exclude_patterns

    return exclude_dirs, exclude_patterns


def clear_project_collections() -> dict[str, Any]:
    """
    Clear all collections for the current project.

    Returns:
        Dictionary with results of clearing operations
    """
    current_project = get_current_project()
    if not current_project:
        return {"error": "No project context found", "cleared": []}

    client = get_qdrant_client()
    cleared = []
    errors = []

    try:
        existing_collections = [c.name for c in client.get_collections().collections]

        for collection_type in ["code", "config", "documentation", "file_metadata"]:
            collection_name = f"{current_project['collection_prefix']}_{collection_type}"
            if collection_name in existing_collections:
                try:
                    client.delete_collection(collection_name)
                    cleared.append(collection_name)
                    logger.info(f"Cleared collection: {collection_name}")
                except Exception as e:
                    error_msg = f"Failed to clear {collection_name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        return {
            "project": current_project["name"],
            "cleared_collections": cleared,
            "errors": errors if errors else None,
        }

    except Exception as e:
        error_msg = f"Failed to access collections: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "project": current_project.get("name", "unknown")}


def list_indexed_projects() -> dict[str, Any]:
    """List all projects that have indexed data.

    Returns:
        Dictionary with information about all indexed projects
    """
    try:
        client = get_qdrant_client()
        all_collections = [c.name for c in client.get_collections().collections]

        project_names = get_available_project_names(all_collections)

        project_info = []
        for project_name in project_names:
            # Get collections for this project
            project_collections = [
                c
                for c in all_collections
                if (c.startswith(f"project_{project_name}_") or c.startswith(f"dir_{project_name}_")) and not c.endswith("_file_metadata")
            ]

            # Count total points across all collections for this project
            total_points = 0
            collection_details = []

            for collection_name in project_collections:
                try:
                    collection_info = client.get_collection(collection_name)
                    points_count = collection_info.points_count
                    total_points += points_count

                    # Determine collection type
                    if collection_name.endswith("_code"):
                        collection_type = "code"
                    elif collection_name.endswith("_config"):
                        collection_type = "config"
                    elif collection_name.endswith("_documentation"):
                        collection_type = "documentation"
                    else:
                        collection_type = "unknown"

                    collection_details.append(
                        {
                            "name": collection_name,
                            "type": collection_type,
                            "points_count": points_count,
                        }
                    )

                except Exception as e:
                    logger.warning(f"Could not get info for collection {collection_name}: {e}")

            if project_collections:  # Only include projects with actual collections
                project_info.append(
                    {
                        "name": project_name,
                        "collections": project_collections,
                        "total_points": total_points,
                        "collection_details": collection_details,
                        "collection_types": list({detail["type"] for detail in collection_details}),
                    }
                )

        return {
            "total_projects": len(project_info),
            "projects": project_info,
            "timestamp": logger.info.__module__,  # This is a simple way to get current time without importing datetime
        }

    except Exception as e:
        error_msg = f"Failed to list indexed projects: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def get_available_project_names(collections: list[str]) -> list[str]:
    """Extract available project names from collection names.

    Args:
        collections: List of collection names to analyze

    Returns:
        List of available project names
    """
    # Known collection type suffixes
    collection_type_suffixes = {"code", "config", "documentation", "file_metadata"}

    project_names = set()
    for collection in collections:
        # Skip metadata collections entirely - they use format "project_{name}_file_metadata"
        # which would incorrectly extract "{name}_file" as the project name
        if collection.endswith("_file_metadata"):
            continue

        if collection.startswith(("project_", "dir_")):
            # Extract project name from "project_{name}_{type}" or "dir_{name}_{type}"
            # Handle multi-part names like "project_PocketFlow_Template_Python_code"
            parts = collection.split("_")
            if len(parts) >= 3 and parts[-1] in collection_type_suffixes:
                # Everything between the prefix and the last '_type' is the project name
                project_name = "_".join(parts[1:-1])
                project_names.add(project_name)

    return sorted(project_names)


def validate_project_exists(project_name: str) -> dict[str, Any]:
    """Validate that a project exists and has indexed data.

    Args:
        project_name: Name of the project to validate

    Returns:
        Dictionary with validation results
    """
    try:
        client = get_qdrant_client()
        all_collections = [c.name for c in client.get_collections().collections]

        # Normalize project name
        normalized_name = project_name.replace(" ", "_").replace("-", "_").lower()

        # Find collections for this project
        project_collections = [
            c
            for c in all_collections
            if (c.startswith(f"project_{normalized_name}_") or c.startswith(f"dir_{normalized_name}_")) and not c.endswith("_file_metadata")
        ]

        if not project_collections:
            available_projects = get_available_project_names(all_collections)
            return {
                "exists": False,
                "project_name": project_name,
                "normalized_name": normalized_name,
                "available_projects": available_projects,
                "message": f"Project '{project_name}' not found or has no indexed data",
            }

        # Count total points
        total_points = 0
        for collection_name in project_collections:
            try:
                collection_info = client.get_collection(collection_name)
                total_points += collection_info.points_count
            except Exception as e:
                logger.warning(f"Could not get points count for {collection_name}: {e}")

        return {
            "exists": True,
            "project_name": project_name,
            "normalized_name": normalized_name,
            "collections": project_collections,
            "total_points": total_points,
            "message": f"Project '{project_name}' found with {len(project_collections)} collections and {total_points} indexed items",
        }

    except Exception as e:
        error_msg = f"Failed to validate project '{project_name}': {str(e)}"
        logger.error(error_msg)
        return {"exists": False, "error": error_msg, "project_name": project_name}


def get_project_collections(project_name: str) -> dict[str, Any]:
    """Get all collections for a specific project.

    Args:
        project_name: Name of the project

    Returns:
        Dictionary with project collections information
    """
    try:
        client = get_qdrant_client()
        all_collections = [c.name for c in client.get_collections().collections]

        # Normalize project name
        normalized_name = project_name.replace(" ", "_").replace("-", "_").lower()

        # Find all collections for this project (including metadata)
        project_collections = [
            c for c in all_collections if c.startswith(f"project_{normalized_name}_") or c.startswith(f"dir_{normalized_name}_")
        ]

        # Categorize collections
        collections_by_type = {
            "code": [],
            "config": [],
            "documentation": [],
            "file_metadata": [],
            "other": [],
        }

        for collection_name in project_collections:
            if collection_name.endswith("_code"):
                collections_by_type["code"].append(collection_name)
            elif collection_name.endswith("_config"):
                collections_by_type["config"].append(collection_name)
            elif collection_name.endswith("_documentation"):
                collections_by_type["documentation"].append(collection_name)
            elif collection_name.endswith("_file_metadata"):
                collections_by_type["file_metadata"].append(collection_name)
            else:
                collections_by_type["other"].append(collection_name)

        # Get detailed info for each collection
        collection_details = []
        for collection_name in project_collections:
            try:
                collection_info = client.get_collection(collection_name)
                collection_details.append(
                    {
                        "name": collection_name,
                        "points_count": collection_info.points_count,
                        "vector_size": collection_info.config.params.vectors.size,
                        "distance": collection_info.config.params.vectors.distance.value,
                    }
                )
            except Exception as e:
                logger.warning(f"Could not get detailed info for {collection_name}: {e}")
                collection_details.append({"name": collection_name, "error": str(e)})

        return {
            "project_name": project_name,
            "normalized_name": normalized_name,
            "collections": project_collections,
            "collections_by_type": collections_by_type,
            "collection_details": collection_details,
            "total_collections": len(project_collections),
        }

    except Exception as e:
        error_msg = f"Failed to get collections for project '{project_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "project_name": project_name}


def normalize_project_name(project_name: str) -> str:
    """Normalize project name for consistent collection naming.

    Args:
        project_name: Original project name

    Returns:
        Normalized project name
    """
    return project_name.replace(" ", "_").replace("-", "_").lower()


def get_project_metadata(project_name: str) -> dict[str, Any]:
    """Extract project metadata including name, path, and collection info.

    Args:
        project_name: Name of the project

    Returns:
        Dictionary with project metadata
    """
    try:
        # Get project collections info
        collections_info = get_project_collections(project_name)

        if "error" in collections_info:
            return collections_info

        # Try to determine project path from collection metadata
        project_path = None
        project_root = None

        # Look for metadata in file_metadata collections
        file_metadata_collections = collections_info["collections_by_type"]["file_metadata"]

        if file_metadata_collections:
            try:
                client = get_qdrant_client()

                # Query a few points from metadata collection to extract path info
                sample_results = client.search(
                    collection_name=file_metadata_collections[0],
                    query_vector=[0.0] * 768,  # Dummy vector for sampling
                    limit=5,
                    score_threshold=0.0,
                )

                if sample_results:
                    for result in sample_results:
                        if isinstance(result.payload, dict):
                            file_path = result.payload.get("file_path", "")
                            if file_path:
                                # Try to infer project root from file paths
                                path_obj = Path(file_path)
                                for parent in path_obj.parents:
                                    if any((parent / marker).exists() for marker in PROJECT_MARKERS):
                                        project_root = str(parent)
                                        project_path = str(parent)
                                        break
                                if project_root:
                                    break

            except Exception as e:
                logger.debug(f"Could not extract path info from metadata: {e}")

        # Calculate total points across all content collections
        total_points = sum(
            detail.get("points_count", 0)
            for detail in collections_info["collection_details"]
            if not detail["name"].endswith("_file_metadata") and "error" not in detail
        )

        return {
            "project_name": project_name,
            "normalized_name": collections_info["normalized_name"],
            "project_path": project_path,
            "project_root": project_root,
            "collections": collections_info["collections"],
            "collections_by_type": collections_info["collections_by_type"],
            "total_collections": collections_info["total_collections"],
            "total_points": total_points,
            "content_types": [
                content_type
                for content_type, collections in collections_info["collections_by_type"].items()
                if collections and content_type != "file_metadata"
            ],
        }

    except Exception as e:
        error_msg = f"Failed to get metadata for project '{project_name}': {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "project_name": project_name}


def delete_file_chunks(file_path: str, collection_name: str | None = None) -> dict[str, Any]:
    """
    Delete all chunks for a specific file from the vector database.

    Args:
        file_path: Path to the file whose chunks should be deleted
        collection_name: Optional specific collection name to delete from

    Returns:
        Dictionary with deletion results
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue

    logger.info(f"Deleting chunks for file: {file_path}")

    try:
        if not file_path or not isinstance(file_path, str):
            return {"error": "Invalid file path"}

        abs_path = Path(file_path).resolve()
        qdrant_client = get_qdrant_client()

        # Determine collection name if not provided
        if collection_name is None:
            suffix = abs_path.suffix.lower()
            if suffix in [".json", ".yaml", ".yml", ".xml", ".toml", ".ini", ".env"]:
                file_type = "config"
            elif suffix in [".md", ".markdown", ".rst", ".txt", ".mdx"]:
                file_type = "documentation"
            else:
                file_type = "code"
            collection_name = get_collection_name(str(abs_path), file_type)

        # Check if collection exists
        try:
            collections = [c.name for c in qdrant_client.get_collections().collections]
            if collection_name not in collections:
                return {"error": f"Collection '{collection_name}' does not exist"}
        except Exception:
            return {"error": f"Could not access collection '{collection_name}'"}

        # Count existing points for the file
        filter_condition = Filter(must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))])

        count_response = qdrant_client.count(collection_name=collection_name, count_filter=filter_condition, exact=True)
        points_before = count_response.count

        # Delete the points
        qdrant_client.delete(collection_name=collection_name, points_selector=filter_condition)

        logger.info(f"Deleted {points_before} points for {file_path} from {collection_name}")

        return {
            "file_path": str(abs_path),
            "collection": collection_name,
            "deleted_points": points_before,
        }

    except Exception as e:
        error_msg = f"Failed to delete chunks for {file_path}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "file_path": file_path}
