# System Architecture

This document provides a high-level overview of the Codebase RAG MCP Server architecture. For a comprehensive deep dive into the internal mechanics, please refer to [docs/ARCHITECTURE_DEEP_DIVE.md](docs/ARCHITECTURE_DEEP_DIVE.md).

## Overview

The Codebase RAG MCP Server is designed to bridge the gap between Large Language Models (LLMs) and local codebases. It implements a Retrieval-Augmented Generation (RAG) system exposed via the Model Context Protocol (MCP).

### Key Components

1.  **MCP Server Layer**: Handles communication with MCP clients (like Claude Code/Desktop).
2.  **Service Layer**: Core business logic for indexing, searching, and analyzing code.
3.  **Data Layer**:
    *   **Qdrant**: Vector database for storing code embeddings.
    *   **Ollama/MLX**: Embedding providers for generating vector representations of code.
4.  **Parsing Layer**: Uses **Tree-sitter** for syntax-aware code chunking.

## Server Architecture

The server is structured around a modular service architecture:

*   **`src/main.py`**: The entry point that initializes the MCP server.
*   **`src/services/`**: Contains the core logic.
    *   `indexing_service.py`: Orchestrates the file processing and indexing pipeline.
    *   `code_parser_service.py`: Intelligent code chunking using Tree-sitter.
    *   `embedding_service.py`: Interface for embedding providers (Ollama, MLX).
    *   `reranker_service.py`: Two-stage RAG with cross-encoder reranking (Qwen3-Reranker).
    *   `qdrant_service.py`: Manages interactions with the vector database.
    *   `project_analysis_service.py`: Analyzes project structure and dependencies.
*   **`src/tools/`**: Implementations of the MCP tools exposed to the LLM.
*   **`src/utils/`**: Shared utilities.
    *   `logging_config.py`: Centralized logging with file rotation support.
    *   `tree_sitter_manager.py`: Tree-sitter parser management and caching.

## Code Encapsulation

The system uses a clean separation of concerns:

### 1. Intelligent Chunking Strategy
Instead of simple text splitting, the system understands code structure.
*   **Source**: `src/services/code_parser_service.py`
*   **Logic**: Files are parsed into Abstract Syntax Trees (AST). Nodes (functions, classes) are extracted as self-contained chunks with their context (docstrings, decorators).

### 2. Vector Storage Schema
Data is stored in Qdrant with specific collections for different content types:
*   `project_{name}_code`: Semantic code chunks.
*   `project_{name}_config`: Configuration files.
*   `project_{name}_documentation`: Markdown and docs.

### 3. Abstraction Layers
*   **`FastMCP`**: The server uses the `fastmcp` library to abstract the MCP protocol details.
*   **`BasePromptImplementation`**: Prompts are implemented as classes inheriting from a base class to ensure consistent registration and error handling.

## Data Flow

1.  **Indexing**:
    `File` -> `Tree-sitter Parser` -> `CodeChunks` -> `Embedding Model` -> `Vectors` -> `Qdrant`

2.  **Searching (Two-Stage RAG)**:
    ```
    Query -> Embedding Model -> Vector ->
    Stage 1: Qdrant (ANN Search) -> Top-K Candidates (default: 50) ->
    Stage 2: Cross-Encoder Reranking (Qwen3-Reranker) ->
    Top-N Reranked Results -> LLM
    ```

    The two-stage retrieval improves search accuracy by 22-31% compared to single-stage vector search:
    - **Stage 1**: Fast approximate nearest neighbor (ANN) search retrieves candidate results
    - **Stage 2**: Cross-encoder evaluates query-document pairs together for precise relevance scoring

## Further Reading

*   **[Deep Dive Architecture](docs/ARCHITECTURE_DEEP_DIVE.md)**: detailed analysis of the system's internals.
*   **[Import Hierarchy](docs/IMPORT_HIERARCHY.md)**: Visualizing module dependencies.
*   **[Intelligent Chunking Guide](docs/INTELLIGENT_CHUNKING_GUIDE.md)**: How code is parsed and split.
