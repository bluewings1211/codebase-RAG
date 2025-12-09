# User Manual

This manual provides detailed instructions on how to use the Codebase RAG MCP Server effectively.

## Table of Contents
1.  [Core Features](#core-features)
2.  [Tutorials](#tutorials)
    *   [How to Index Effectively](#how-to-index-effectively)
    *   [How to Search Effectively](#how-to-search-effectively)
3.  [Intelligent Code Chunking](#intelligent-code-chunking)
4.  [Advanced Prompts](#advanced-prompts)
5.  [Troubleshooting](#troubleshooting)

## Core Features

### Natural Language Querying
Ask questions about your codebase in plain English. The system converts your question into vectors and finds the most semantically relevant code chunks.

### Cross-Language Support
Supported languages include:
*   Python, JavaScript, TypeScript (including JSX/TSX)
*   Go, Rust, Java, C++
*   JSON, YAML, Markdown

### Intelligent Parsing
Unlike simple text search, this tool understands code structure (functions, classes, methods) using Tree-sitter. This means it returns complete logical units of code, not just random lines.

## Tutorials

### How to Index Effectively

Indexing is the process of reading your code and storing it in the vector database.

**Option 1: Manual Indexing (Recommended for first run)**
Use the command-line tool for the initial setup or large codebases.
```bash
# Full fresh index
uv run python manual_indexing.py -d /path/to/repo -m clear_existing

# Incremental update (only changed files)
uv run python manual_indexing.py -d /path/to/repo -m incremental
```

**Option 2: Chat-based Indexing**
You can ask Claude to index for you:
> "Please index the current directory."

*Note: For very large repositories, the manual tool is faster and more reliable.*

### How to Search Effectively

The `search` tool is your main interface. You can be specific or broad.

*   **Specific**: "Find the `CodeParser` class definition."
*   **Conceptual**: "How does the system handle retry logic for network requests?"
*   **Architectural**: "Show me the relationship between the Service layer and the Data layer."

## Intelligent Code Chunking

We use **syntax-aware code chunking**. Instead of cutting files at fixed character limits, we split them by logical blocks:

*   **`function`**: Complete function bodies.
*   **`class`**: Class definitions with properties.
*   **`method`**: Individual methods within classes.
*   **`interface`**: Type definitions.

This ensures that when you retrieve code, you get the full context needed to understand it.

## Advanced Prompts

The server includes specialized prompts to help you navigate complex projects.

### `explore_project`
Use this when you are new to a codebase. It analyzes the structure and gives you a guided tour.

**Usage:**
> "Run the explore_project prompt on this directory."

**What it does:**
1.  Analyzes directory structure.
2.  Identifies entry points and core modules.
3.  Suggests a navigation strategy.

### `advance_search`
Use this for deep dives or cross-project searches.

**Usage:**
> "Use advance_search to find how authentication is implemented across all projects."

**Features:**
*   **Targeted Search**: Search specific projects.
*   **Cross-Project**: Search everything you've indexed.
*   **Search Modes**: Hybrid (Keyword + Semantic) for best results.

## MCP Tools Reference

For a complete detailed reference of all available tools and their arguments, please refer to:
**[docs/MCP_TOOLS.md](docs/MCP_TOOLS.md)**

## Troubleshooting

**Q: The search returns no results.**
A: Ensure you have indexed the directory. Run `uv run python manual_indexing.py -d . -m incremental` to check.

**Q: Indexing is slow.**
A: If you are on a Mac with Apple Silicon, ensure you are using the MLX provider (see [Configuration Guide](docs/CONFIGURATION.md#high-speed-indexing-with-mlx-server)) or adjust the `INDEXING_BATCH_SIZE` in `.env`.

**Q: "Port already in use" error.**
A: Check if Qdrant is already running in Docker.
