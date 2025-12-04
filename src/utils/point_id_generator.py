"""Point ID generator for deterministic UUID generation."""

import uuid
from functools import lru_cache

# Namespace UUID for deterministic generation
# Using a fixed namespace ensures consistent UUIDs across runs
POINT_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # Standard namespace UUID


@lru_cache(maxsize=10000)  # Cache for frequently used paths
def generate_deterministic_uuid(file_path: str) -> str:
    """
    Generate a deterministic UUID based on file path.

    Uses uuid5 with a fixed namespace to ensure the same file path
    always generates the same UUID, even across different runs.
    Includes LRU cache for performance optimization.

    Args:
        file_path: The file path to generate UUID for

    Returns:
        A string representation of the UUID (e.g., "550e8400-e29b-41d4-a716-446655440000")
    """
    # Generate deterministic UUID using uuid5
    # uuid5 uses SHA-1 hashing, making it deterministic for the same input
    deterministic_uuid = uuid.uuid5(POINT_ID_NAMESPACE, file_path)

    return str(deterministic_uuid)


def is_valid_uuid_format(uuid_string: str) -> bool:
    """
    Check if a string is a valid UUID format.

    Args:
        uuid_string: The string to validate

    Returns:
        True if valid UUID format, False otherwise
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False
