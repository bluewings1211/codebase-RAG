# Contributing to Codebase RAG MCP

Thank you for your interest in contributing to the Codebase RAG MCP Server! This document provides guidelines and instructions for setting up your development environment and contributing to the project.

## Development Environment Setup

We provide a helper script to quickly set up your environment.

### Prerequisites

*   **Python 3.10+**
*   **uv**: A fast Python package manager. [Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)
*   **Docker**: Recommended for running Qdrant.
*   **Ollama**: For local embedding generation.

### Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd codebase-rag-mcp
    ```

2.  **Run the setup script**:
    ```bash
    ./setup.sh
    ```
    This script will:
    *   Install dependencies using `uv sync`.
    *   Create a `.env` file from the example.
    *   Check for Docker and Ollama availability.

3.  **Install development dependencies**:
    ```bash
    # Install with dev dependencies (testing, linting, RAGAS)
    uv sync --dev

    # Or install production dependencies only
    uv sync
    ```

    The `--dev` flag installs additional packages for development:
    *   Testing: `pytest`, `pytest-cov`, `pytest-xdist`
    *   RAG Evaluation: `ragas`, `datasets`
    *   Linting: `ruff`, `black`, `pre-commit`

4.  **Activate Virtual Environment**:
    `uv` manages the virtual environment for you. You can run commands using `uv run <command>` or activate it manually:
    ```bash
    source .venv/bin/activate
    ```

## Coding Standards

This project maintains high code quality standards using `ruff` for linting and formatting, and `black` for code formatting.

### Formatting and Linting

Before submitting a Pull Request, please ensure your code adheres to the project's standards.

1.  **Format Code**:
    We use `black` for code formatting.
    ```bash
    uv run black .
    ```

2.  **Lint Code**:
    We use `ruff` for linting.
    ```bash
    uv run ruff check . --fix
    ```

### Configuration

The configuration for `ruff` and `black` is located in `pyproject.toml`.
*   **Line Length**: 140 characters.
*   **Target Python Version**: 3.10.

## Running Tests

We use `pytest` for testing with marker-based test categorization.

### Quick Reference

```bash
# Run all fast tests (recommended before commits)
uv run pytest src/tests/ -m "not slow and not integration and not requires_qdrant"

# Run only unit tests (no external dependencies)
uv run pytest src/tests/ -m "unit"

# Run all tests
uv run pytest src/tests/

# Run with coverage report
uv run pytest src/tests/ --cov=src --cov-report=html

# Run tests in parallel (faster)
uv run pytest src/tests/ -n auto
```

### Test Markers

Tests are categorized using pytest markers:

| Marker | Description | When to Run |
|--------|-------------|-------------|
| `unit` | Fast tests with no external dependencies | Always |
| `slow` | Tests that take > 10 seconds | Before major commits |
| `integration` | Requires external services (Qdrant, Ollama) | Before PR |
| `requires_qdrant` | Requires Qdrant database | With Qdrant running |
| `requires_ollama` | Requires Ollama service | With Ollama running |
| `rag_quality` | RAG quality evaluation tests | Periodically |

### Adding Markers to Tests

When writing new tests, add appropriate markers:

```python
import pytest

@pytest.mark.unit
def test_fast_function():
    """Test that runs quickly without external services."""
    pass

@pytest.mark.requires_qdrant
@pytest.mark.integration
def test_with_database():
    """Test that requires Qdrant to be running."""
    pass
```

### RAG Quality Tests

For RAG-specific quality testing, see `src/tests/test_rag_quality.py` and the documentation in `docs/RAG-testing-framework-and-ci-cd.md`.

```bash
# Run RAG quality tests (basic, no Ollama required)
uv run pytest src/tests/test_rag_quality.py -m "rag_quality and not requires_ollama"

# Run full RAG quality tests (requires Ollama)
uv run pytest src/tests/test_rag_quality.py -m "rag_quality"
```

Please ensure all relevant tests pass before submitting your changes. If you are adding new features, please include appropriate tests with proper markers.

## Architecture and Design

For a high-level overview of the system architecture, please refer to [ARCHITECTURE.md](ARCHITECTURE.md).
For a deep dive into the implementation details, see [docs/ARCHITECTURE_DEEP_DIVE.md](docs/ARCHITECTURE_DEEP_DIVE.md).

## Pull Request Process

1.  Fork the repository and create your branch from `main`.
2.  If you've added code that should be tested, add tests.
3.  Ensure the test suite passes.
4.  Make sure your code lints and formats correctly.
5.  Issue that pull request!

## MCP Tools Documentation

If you are modifying or adding MCP tools, please update [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md) to reflect the changes.
