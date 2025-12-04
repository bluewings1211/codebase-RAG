"""Point ID validation utilities for Qdrant compatibility."""

import re
import uuid

# UUID v4 regex pattern (standard UUID format)
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Hex string pattern (for detecting old invalid format)
HEX_STRING_PATTERN = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)


class InvalidPointIDError(ValueError):
    """Raised when a Point ID is invalid for Qdrant."""

    pass


def is_valid_point_id(point_id: str) -> bool:
    """
    Check if a Point ID is valid for Qdrant.

    Qdrant accepts:
    - Valid UUID strings (e.g., "550e8400-e29b-41d4-a716-446655440000")
    - Unsigned integers (as strings, e.g., "12345")

    Args:
        point_id: The Point ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not point_id or not isinstance(point_id, str):
        return False

    # Check if it's a valid UUID
    try:
        uuid.UUID(point_id)
        return True
    except ValueError:
        pass

    # Check if it's an unsigned integer
    try:
        int_val = int(point_id)
        return int_val >= 0
    except ValueError:
        return False


def validate_point_id(point_id: str, context: str | None = None) -> None:
    """
    Validate a Point ID and raise an exception if invalid.

    Args:
        point_id: The Point ID to validate
        context: Optional context for better error messages

    Raises:
        InvalidPointIDError: If the Point ID is invalid
    """
    if not point_id:
        raise InvalidPointIDError("Point ID cannot be empty")

    if not isinstance(point_id, str):
        raise InvalidPointIDError(f"Point ID must be a string, got {type(point_id).__name__}")

    # Check for the old hex string format
    if len(point_id) == 16 and HEX_STRING_PATTERN.match(point_id):
        error_msg = (
            f"Invalid Point ID format: '{point_id}' appears to be a 16-character hex string. "
            "Qdrant requires valid UUIDs or unsigned integers. "
        )
        if context:
            error_msg += f"Context: {context}"
        raise InvalidPointIDError(error_msg)

    if not is_valid_point_id(point_id):
        error_msg = (
            f"Invalid Point ID: '{point_id}'. "
            "Qdrant requires valid UUIDs (e.g., '550e8400-e29b-41d4-a716-446655440000') "
            "or unsigned integers (e.g., '12345'). "
        )
        if context:
            error_msg += f"Context: {context}"
        raise InvalidPointIDError(error_msg)


def detect_invalid_hex_format(point_id: str) -> bool:
    """
    Detect if a Point ID is using the old invalid hex format.

    Args:
        point_id: The Point ID to check

    Returns:
        True if it matches the old hex format pattern
    """
    return (
        isinstance(point_id, str)
        and len(point_id) == 16
        and HEX_STRING_PATTERN.match(point_id) is not None
        and not is_valid_point_id(point_id)
    )
