# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Setup and Installation
./setup.sh                          # Run setup script

# Run MCP Server
uv run python src/run_mcp.py        # Start MCP server

# Manual Indexing
uv run python manual_indexing.py -d "." -m clear_existing    # Full reindex
uv run python manual_indexing.py -d "." -m incremental       # Incremental update

# Testing
uv run pytest src/tests/            # Run tests
```

## Architecture Overview

This is a **Codebase RAG (Retrieval-Augmented Generation) MCP Server** that enables AI agents to understand and query codebases using natural language with **function-level precision** through intelligent syntax-aware code chunking.

### Project Structure

```
src/
├── main.py                    # MCP server entry point
├── run_mcp.py                 # Server startup script
├── models/                    # Data models and structures
│   ├── code_chunk.py         # Intelligent chunk representations
│   └── file_metadata.py      # File tracking and metadata
├── services/                  # Core business logic
│   ├── code_parser_service.py    # AST parsing and chunking
│   ├── indexing_service.py       # Orchestration and processing
│   ├── embedding_service.py      # Ollama integration
│   ├── reranker_service.py       # Cross-encoder reranking (Two-Stage RAG)
│   ├── qdrant_service.py         # Vector database operations
│   └── project_analysis_service.py # Repository analysis
├── tools/                     # MCP tool implementations
│   ├── core/                 # Error handling and utilities
│   ├── indexing/             # Parsing and chunking tools
│   └── project/              # Project management tools
├── utils/                     # Shared utilities
│   ├── logging_config.py        # Centralized logging with file rotation
│   ├── language_registry.py     # Language support definitions
│   ├── tree_sitter_manager.py   # Parser management
│   └── performance_monitor.py   # Progress tracking
└── prompts/                   # Advanced query prompts (future)

Root Files:
├── manual_indexing.py         # Standalone indexing tool
├── pyproject.toml            # uv/Python configuration
└── docs/                     # Documentation (referenced)
```

### Configuration

Key environment variables (`.env` file). See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for full reference.

```bash
# Embedding Provider
EMBEDDING_PROVIDER=ollama              # ollama or mlx_server
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text

# Two-Stage RAG / Reranker
RERANKER_ENABLED=true                  # Enable cross-encoder reranking
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
RERANK_TOP_K=50                        # Stage 1 candidates count

# Qdrant Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Logging
LOG_LEVEL=INFO
LOG_FILE_ENABLED=false                 # Enable file logging for debugging
LOG_FILE_PATH=logs/codebase-rag.log
```

### MCP Tools Available

#### Core Tools
- `health_check_tool()`: Check system health (Qdrant, Ollama, Reranker)

#### Indexing Tools
- `index_directory(directory, patterns, recursive, clear_existing, incremental, project_name)`: Index a codebase
- `check_index_status(directory)`: Check indexing status and recommendations
- `analyze_repository_tool(directory)`: Analyze repository structure
- `get_file_filtering_stats_tool(directory)`: Get file filtering statistics

#### Search Tools
- `search(query, n_results, cross_project, search_mode, include_context, context_chunks, target_projects, collection_types, minimal_output, enable_reranking, rerank_top_k)`: Semantic search with two-stage RAG

**Key Search Parameters:**
- `enable_reranking`: Enable two-stage retrieval with cross-encoder reranking (default: true via env)
- `rerank_top_k`: Number of candidates for Stage 1 before reranking (default: 50)
- `collection_types`: Filter by type - ["code"], ["config"], ["documentation"]

#### Chunking & Parser Tools
- `get_chunking_metrics_tool(language, export_path)`: Get chunking performance metrics
- `reset_chunking_metrics_tool()`: Reset session metrics
- `diagnose_parser_health_tool(comprehensive, language)`: Diagnose Tree-sitter parser health
- `get_indexing_progress_tool()`: Get real-time indexing progress

#### Project Management Tools
- `get_project_info_tool(directory)`: Get project information
- `list_indexed_projects_tool()`: List all indexed projects
- `clear_project_data_tool(project_name, directory)`: Clear project data

#### File Management Tools
- `get_file_metadata_tool(file_path)`: Get file metadata from vector database
- `clear_file_metadata_tool(file_path, collection_name)`: Clear file chunks
- `reindex_file_tool(file_path)`: Reindex a specific file

## Intelligent Code Chunking System

### Overview
The system uses **Tree-sitter** parsers to perform syntax-aware code analysis, breaking down source code into semantically meaningful chunks (functions, classes, methods) instead of processing entire files as single units.

### Supported Languages and Chunk Types

**Fully Implemented Languages:**
- **Python (.py, .pyw, .pyi)**: Functions, classes, methods, constants, docstrings, decorators
- **JavaScript (.js, .jsx, .mjs, .cjs)**: Functions, classes, modules, arrow functions
- **TypeScript (.ts)**: Interfaces, types, classes, functions, generics, annotations
- **TypeScript JSX (.tsx)**: React components, interfaces, types, functions
- **Go (.go)**: Functions, structs, interfaces, methods, packages
- **Rust (.rs)**: Functions, structs, impl blocks, traits, modules, macros
- **Java (.java)**: Classes, methods, interfaces, annotations, generics
- **C++ (.cpp, .cxx, .cc, .c, .hpp, .hxx, .hh, .h)**: Functions, classes, structs, namespaces, templates

**Structured Files:**
- **JSON/YAML**: Object-level chunking (e.g., `scripts`, `dependencies` as separate chunks)
- **Markdown**: Header-based hierarchical chunking with section organization
- **Configuration Files**: Section-based parsing with semantic grouping

### Incremental Indexing Workflow

1. **Initial Indexing**: Full codebase processing with metadata storage
2. **Change Detection**: Compare file modification times and content hashes
3. **Selective Processing**: Only reprocess files with detected changes
4. **Metadata Updates**: Update file metadata after successful processing
5. **Collection Management**: Automatic cleanup of stale entries

### Collection Architecture

**Content Collections** (store intelligent chunks with embeddings):
- `project_{name}_code`: **Intelligent code chunks** - functions, classes, methods from (.py, .js, .java, etc.)
- `project_{name}_config`: **Structured config chunks** - JSON/YAML objects, configuration sections
- `project_{name}_documentation`: **Document chunks** - Markdown headers, documentation sections

**Metadata Collection** (tracks file states):
- `project_{name}_file_metadata`: File change tracking for incremental indexing
  - Stores: file_path, mtime, content_hash, file_size, indexed_at, syntax_error_count
  - Used for: change detection, incremental processing, progress tracking, error monitoring
