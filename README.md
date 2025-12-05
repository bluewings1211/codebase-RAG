# Codebase RAG MCP Server

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server designed to assist AI agents and junior developers in understanding and navigating codebases. It leverages Qdrant as its vector database for storing codebase embeddings and integrates with Ollama for generating these embeddings. The server provides API endpoints for indexing local directories or GitHub repositories and for querying the indexed codebases using natural language.

## Features

### Core Functionality
- **Codebase Indexing**: Index local project directories or remote GitHub repositories
- **Automatic Project Analysis**: Intelligently identifies project types and relevant source code files, respecting `.gitignore` rules
- **Flexible Embedding Models**: Supports using various embedding models available through Ollama
- **Metal Performance Shaders (MPS) Acceleration**: Automatically utilizes MPS on macOS for faster embedding generation
- **Natural Language Querying**: Ask questions about your codebase in natural language and retrieve relevant code snippets

## Intelligent Code Chunking

This system implements state-of-the-art **syntax-aware code chunking** using Tree-sitter parsers to break down source code into meaningful, semantically coherent chunks instead of processing entire files as single units.

### Key Benefits

- **🎯 Precise Retrieval**: Find specific functions, classes, or methods instead of entire files
- **📈 Better Embeddings**: Each chunk represents a complete semantic unit, improving vector quality
- **🔍 Function-Level Search**: Query for specific functionality and get exact matches
- **📚 Rich Metadata**: Each chunk includes detailed information like function signatures, docstrings, and context

### Supported Languages

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
- **JSON/YAML**: Object-level chunking (e.g., separate chunks for `scripts`, `dependencies`)
- **Markdown**: Header-based hierarchical chunking
- **Configuration files**: Section-based parsing

### How It Works

1. **AST Parsing**: Uses Tree-sitter to generate syntax trees for each source file
2. **Semantic Extraction**: Identifies functions, classes, methods, and other semantic units
3. **Context Enhancement**: Preserves surrounding context (imports, docstrings, inline comments)
4. **Metadata Enrichment**: Extracts signatures, parameter types, access modifiers, and relationships
5. **Error Handling**: Gracefully handles syntax errors with smart fallback to whole-file processing

### Chunk Types

- `function` - Standalone functions with their complete implementation
- `class` - Class definitions including properties and documentation
- `method` - Individual methods within classes
- `interface` - TypeScript/Java interfaces and type definitions
- `constant` - Important constants and configuration objects
- `config_object` - Structured configuration sections

### Advanced Features
- **🎯 Intelligent Code Chunking**: Function-level and class-level intelligent chunking using Tree-sitter for precise code understanding and retrieval
- **🚀 Incremental Indexing**: Only process changed files for dramatically faster re-indexing (80%+ time savings)
- **🔧 Manual Indexing Tool**: Standalone script for heavy indexing operations that don't block conversational workflows
- **⚡ Large Codebase Optimization**: Parallel processing, batch operations, and streaming for handling 10,000+ file repositories
- **🧠 Intelligent Recommendations**: Automatic detection of existing indexes with smart time estimates and workflow suggestions
- **📊 Progress Tracking**: Real-time progress monitoring with ETA estimation and memory usage tracking
- **🗂️ Smart Collection Management**: Automatic categorization into code, config, documentation, and metadata collections
- **🌐 Multi-Language Support**: Support for 10+ programming languages with syntax-aware parsing
- **🛡️ Error Tolerance**: Graceful handling of syntax errors with smart fallback mechanisms

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: The project is developed with Python.
- **uv**: Modern Python package manager for fast dependency management. Install from [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).
- **Docker (Recommended for Qdrant)**: Qdrant is best run as a Docker container.
- **Ollama**: For running local language models and embedding models. Download from [ollama.com](https://ollama.com/).
- **Tree-sitter Language Parsers**: Automatically installed with the project dependencies for intelligent code chunking support.
- **MLX Embedding Server (Optional, Recommended for Apple Silicon)**: For 48-161x faster embedding generation on MacBooks. See [High-Speed Indexing with MLX Server](#high-speed-indexing-with-mlx-server).

## Configuration

This project uses environment variables for configuration, loaded from a `.env` file in the project root.

Example `.env` file:

```bash
# Embedding Provider Selection
EMBEDDING_PROVIDER=ollama                 # ollama (default) or mlx_server

# MLX Server Configuration (48-161x faster on Apple Silicon)
MLX_SERVER_URL=http://localhost:8000      # MLX server URL
MLX_BATCH_SIZE=64                         # Batch size (64-128 recommended)
MLX_MODEL_SIZE=small                      # Model: small, medium, large
MLX_FALLBACK_TO_OLLAMA=true               # Auto-fallback if MLX unavailable

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Performance Tuning
INDEXING_CONCURRENCY=4                    # Parallel file processing workers
INDEXING_BATCH_SIZE=20                    # Files processed per batch
EMBEDDING_BATCH_SIZE=10                   # Texts per embedding API call
QDRANT_BATCH_SIZE=500                     # Points per database insertion
MEMORY_WARNING_THRESHOLD_MB=1000          # Memory usage warning threshold
MAX_FILE_SIZE_MB=5                       # Skip files larger than this
MAX_DIRECTORY_DEPTH=20                   # Maximum directory traversal depth

# Logging
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
```

### Configuration Options

#### Embedding Provider Settings
- `EMBEDDING_PROVIDER`: Embedding provider selection - `ollama` (default) or `mlx_server`
- `MLX_SERVER_URL`: MLX server URL (default: `http://localhost:8000`)
- `MLX_BATCH_SIZE`: MLX batch size for embedding requests (default: `64`, recommended: 64-128)
- `MLX_MODEL_SIZE`: MLX model size - `small`, `medium`, `large` (default: `small`)
- `MLX_TIMEOUT`: MLX request timeout in seconds (default: `120`)
- `MLX_FALLBACK_TO_OLLAMA`: Auto-fallback to Ollama if MLX unavailable (default: `true`)

#### Basic Settings
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_DEFAULT_EMBEDDING_MODEL`: Default embedding model (default: `nomic-embed-text`)
- `QDRANT_HOST`: Qdrant host (default: `localhost`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)

#### Performance Settings
- `INDEXING_CONCURRENCY`: Number of parallel workers for file processing (default: `4`)
- `INDEXING_BATCH_SIZE`: Files processed in each batch (default: `20`)
- `EMBEDDING_BATCH_SIZE`: Texts sent to embedding API per call (default: `10`)
- `QDRANT_BATCH_SIZE`: Points inserted to database per batch (default: `500`)
- `MEMORY_WARNING_THRESHOLD_MB`: Memory usage threshold for warnings (default: `1000`)

#### File Processing Settings
- `MAX_FILE_SIZE_MB`: Skip files larger than this size (default: `5`)
- `MAX_DIRECTORY_DEPTH`: Maximum directory traversal depth (default: `20`)
- `FOLLOW_SYMLINKS`: Whether to follow symbolic links (default: `false`)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd codebase-rag-mcp # Or your project directory name
    ```

2.  **Install dependencies and create virtual environment**:
    ```bash
    uv sync
    ```
    *This automatically creates a virtual environment and installs all dependencies from the lock file.*

3.  **Copy environment configuration**:
    ```bash
    cp .env.example .env
    # Edit .env file to customize settings as needed
    ```

## Running Qdrant

It's recommended to run Qdrant using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```
This command will start Qdrant and map its default ports (6333 for gRPC, 6334 for HTTP) to your host. Data will be persisted in a `qdrant_data` directory in your current working directory.

## Running Ollama

1.  **Start the Ollama server**: Follow the instructions on [ollama.com](https://ollama.com/) to download and run Ollama.

2.  **Pull an embedding model**: This project uses `nomic-embed-text` by default for examples. You can pull it using:
    ```bash
    ollama pull nomic-embed-text
    ```
    You can use any other embedding model available in Ollama, just ensure you specify its name correctly when making API calls.

## High-Speed Indexing with MLX Server

For dramatically faster embedding generation on Apple Silicon Macs, you can use the MLX Embedding Server instead of Ollama. This provides **48-161x faster** indexing performance.

### Performance Comparison

| Method | Speed (emb/s) | vs Ollama |
|--------|--------------|-----------|
| **MLX Server (batch=128)** | **1707** | **48x faster** |
| MLX Server (batch=64) | 1646 | 47x faster |
| Ollama (nomic-embed-text) | 35 | baseline |

### Expected Indexing Times

| Codebase Size | Ollama | MLX Server |
|--------------|--------|------------|
| 10K chunks | ~5 min | ~6 sec |
| 50K chunks | ~24 min | ~30 sec |
| 100K chunks | ~48 min | ~1 min |

### Setup MLX Server

1. **Clone the MLX embedding server**:
   ```bash
   git clone https://github.com/jakedahn/qwen3-embeddings-mlx.git
   cd qwen3-embeddings-mlx
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**:
   ```bash
   python server.py
   # Server runs at http://localhost:8000
   ```

4. **Configure the RAG server** to use MLX:
   ```bash
   # In your .env file
   EMBEDDING_PROVIDER=mlx_server
   MLX_SERVER_URL=http://localhost:8000
   MLX_BATCH_SIZE=64
   ```

### MLX Server Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `ollama` | Set to `mlx_server` to enable MLX |
| `MLX_SERVER_URL` | `http://localhost:8000` | MLX server URL |
| `MLX_BATCH_SIZE` | `64` | Batch size (64-128 recommended) |
| `MLX_MODEL_SIZE` | `small` | Model size: `small`, `medium`, `large` |
| `MLX_TIMEOUT` | `120` | Request timeout in seconds |
| `MLX_FALLBACK_TO_OLLAMA` | `true` | Auto-fallback if MLX unavailable |

### Important Notes

- **Vector Dimensions**: MLX uses Qwen3-Embedding (1024 dimensions) vs Ollama's nomic-embed-text (768 dimensions). Collections are created with the correct dimension automatically.
- **Requires Re-indexing**: If switching providers, you need to re-index your codebase due to different embedding dimensions.
- **Apple Silicon Only**: MLX Server requires Apple Silicon (M1/M2/M3) Mac.
- **Auto-Fallback**: If MLX server is unavailable, the system automatically falls back to Ollama (configurable).

## IDE Integration

This MCP server integrates seamlessly with various AI development environments:

### Claude Code Integration

Register the server with Claude Code for use in conversations:

```bash
claude mcp add codebase-rag-mcp \
  --command "uv" \
  --args "run" \
  --args "python" \
  --args "src/run_mcp.py"
```

Once registered, you can use natural language queries like:
- "Find functions that handle file uploads"
- "Show me React components that use hooks"
- "Locate error handling patterns in this codebase"

### Gemini CLI Integration

*Integration instructions and screenshots coming soon*

### VS Code Integration

*Extension and configuration instructions coming soon*

## MCP Tools

The server provides powerful MCP tools for intelligent codebase search and analysis:

### Primary Tool: `search` - Semantic Code Search

Search indexed codebases using natural language queries with function-level precision.

**Example Queries:**
- "Find functions that handle file uploads"
- "Show me React components that use useState hook"
- "Locate error handling patterns in Python"
- "Find database connection initialization code"

**Key Features:**
- **🔍 Function-Level Precision**: Returns specific functions, classes, and methods instead of entire files
- **📝 Natural Language**: Use conversational queries to find code
- **🌐 Cross-Project Search**: Search across all projects or target specific projects with `target_projects`
- **📚 Rich Context**: Include surrounding code for better understanding
- **⚡ Multiple Search Modes**: Semantic, keyword, or hybrid search strategies

### Additional Tools

For comprehensive functionality, additional tools are available:
- **`index_directory`**: Index a codebase for intelligent searching
- **`health_check`**: Verify server connectivity and status
- **`analyze_repository_tool`**: Get repository statistics and analysis

For complete tool documentation with parameters and examples, see **[docs/MCP_TOOLS.md](docs/MCP_TOOLS.md)**.

## Manual Indexing Tool

For large codebases or operations that might take several minutes, use the standalone manual indexing tool:

```bash
# Full indexing (clear existing data)
uv run python manual_indexing.py -d /path/to/large/repo -m clear_existing

# Incremental indexing (only changed files)
uv run python manual_indexing.py -d /path/to/repo -m incremental

# With verbose output
uv run python manual_indexing.py -d /path/to/repo -m incremental --verbose

# Skip confirmation prompts
uv run python manual_indexing.py -d /path/to/repo -m clear_existing --no-confirm
```

### Manual Tool Features
- **Pre-indexing Analysis**: Shows file count, size, and time estimates
- **Change Detection**: In incremental mode, shows exactly which files changed
- **Progress Reporting**: Real-time progress with ETA and memory usage
- **Error Handling**: Graceful handling of individual file failures
- **Dependency Validation**: Checks Qdrant and embedding service availability

### When to Use Manual Tool
- Large repositories (1000+ files)
- Operations estimated to take >5 minutes
- Heavy indexing that shouldn't block interactive workflows
- Batch processing scenarios

## Incremental Indexing

Incremental indexing dramatically reduces processing time by only handling changed files:

### How It Works
1. **File Metadata Tracking**: Stores modification times and content hashes
2. **Change Detection**: Compares current file state with stored metadata
3. **Selective Processing**: Only re-indexes files that have changed
4. **Smart Updates**: Handles file additions, modifications, and deletions

### Performance Benefits
- **80%+ Time Savings**: For typical development scenarios (5-50 changed files)
- **Memory Efficient**: Only processes changed content
- **Network Friendly**: Fewer embedding API calls


## Performance and Best Practices

### For Large Codebases (1000+ files)
1. **Use Manual Tool**: Avoid blocking interactive workflows
2. **Tune Batch Sizes**: Adjust `INDEXING_BATCH_SIZE` and `EMBEDDING_BATCH_SIZE`
3. **Monitor Memory**: Set appropriate `MEMORY_WARNING_THRESHOLD_MB`
4. **Use Incremental Mode**: After initial indexing, always use incremental updates

### Memory Optimization
- The system automatically manages memory with garbage collection between batches
- Monitor memory warnings in logs and adjust batch sizes if needed
- Large files (>5MB) are automatically skipped

### Network Optimization
- Batch embedding generation reduces API calls
- Incremental indexing minimizes redundant processing
- Failed operations have automatic retry with exponential backoff

### File Filtering
Create a `.ragignore` file in your project root to exclude directories:
```
node_modules/
.git/
dist/
build/
__pycache__/
.venv/
*.pyc
*.log
```

### Intelligent Chunking Error Handling

The system includes sophisticated error handling for syntax errors and malformed code:

- **Syntax Error Tolerance**: Files with syntax errors are processed using smart error recovery
- **Graceful Fallback**: When intelligent chunking fails, the system automatically falls back to whole-file processing
- **Error Reporting**: Detailed error statistics are provided in manual indexing tool output
- **Partial Processing**: Correct code sections are preserved even when parts of the file have errors

Example error handling in action:
```bash
# Manual indexing with verbose error reporting
python manual_indexing.py -d /path/to/project --verbose
# Output includes syntax error statistics and affected files
```

## Running Tests

To run the unit and integration tests, use `pytest`:

```bash
uv run pytest tests/
```

## Integrating with AI Assistants

This MCP server can be integrated with various AI assistants and development tools.

### Claude Code Integration

The easiest way to register this server with Claude Code:

1. **Automatic Registration** (recommended):
   ```bash
   ./register_mcp.sh
   ```
   This script will create the necessary configuration and provide instructions.

2. **Manual Registration**:
   ```bash
   # Recommended: Use the wrapper script
   claude mcp add codebase-rag-mcp "$(pwd)/mcp_server"

   # Alternative: Direct uv execution
   claude mcp add codebase-rag-mcp uv run python "$(pwd)/src/run_mcp.py"
   ```

3. **Usage in Claude Code**:
   Once registered, you can use the tools in your prompts:
   - `@codebase-rag-mcp:index_directory` - Index a directory (with intelligent recommendations)
   - `@codebase-rag-mcp:search` - Search indexed content
   - `@codebase-rag-mcp:health_check` - Check server status

   **Smart Indexing Workflow**:
   - First call to `index_directory` will detect existing data and provide recommendations
   - Choose from: use existing data, incremental update, full reindex, or manual tool
   - For large repos, the system will automatically recommend the manual indexing tool

### Other MCP Clients

For other MCP-compatible clients, configure them to run:
```bash
uv run python /path/to/your/project/src/run_mcp.py
```

Refer to your specific MCP client documentation for configuration details.

## Project Structure

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
│   ├── qdrant_service.py         # Vector database operations
│   └── project_analysis_service.py # Repository analysis
├── tools/                     # MCP tool implementations
│   ├── core/                 # Error handling and utilities
│   ├── indexing/             # Parsing and chunking tools
│   └── project/              # Project management tools
├── utils/                     # Shared utilities
│   ├── language_registry.py     # Language support definitions
│   ├── tree_sitter_manager.py   # Parser management
│   └── performance_monitor.py   # Progress tracking
└── prompts/                   # Advanced query prompts

Root Files:
├── manual_indexing.py         # Standalone indexing tool
├── pyproject.toml            # uv/Python configuration
├── uv.lock                   # Dependency lock file
└── docs/                     # Documentation
    ├── MCP_TOOLS.md          # Comprehensive tool reference
    └── BEST_PRACTICES.md     # Optimization guides
```

### Key Components

#### Services Layer
- **IndexingService**: Orchestrates the entire indexing process with parallel processing
- **CodeParserService**: Implements intelligent code chunking using Tree-sitter parsers for syntax-aware code analysis
- **QdrantService**: Manages vector database operations with batch optimization
- **EmbeddingService**: Handles Ollama integration with batch embedding generation
- **ProjectAnalysisService**: Analyzes project structure and filters relevant files
- **FileMetadataService**: Tracks file states for incremental indexing
- **ChangeDetectorService**: Detects file changes for selective reprocessing

#### Models
- **FileMetadata**: Tracks file modification times, content hashes, and indexing state
- **CodeChunk**: Represents intelligent code chunks with rich metadata (function signatures, docstrings, syntax tree information)

#### Utilities
- **PerformanceMonitor**: Progress tracking, ETA estimation, and memory monitoring
- **StageLogger**: Detailed timing and performance logging for each processing stage

#### Collection Organization
The system creates the following Qdrant collections:
- `project_{name}_code`: Source code files
- `project_{name}_config`: Configuration files (JSON, YAML, etc.)
- `project_{name}_documentation`: Documentation files (Markdown, etc.)
- `project_{name}_file_metadata`: File change tracking for incremental indexing

## Real-World Usage Examples

This section demonstrates real-world usage of the Agentic-RAG system with actual exploration and analysis examples.

### 📋 Project Architecture Exploration Example

The [ARCHITECTURE_DEEP_DIVE.md](./docs/ARCHITECTURE_DEEP_DIVE.md) file was generated using this RAG system as a live demonstration. It showcases:

- **Comprehensive codebase analysis** using function-level search precision
- **Component relationship mapping** through intelligent chunking
- **Architecture documentation** generated from actual code exploration
- **Performance insights** derived from real system metrics

This document serves as both:
1. **Usage Example**: Shows how the system explores and understands complex codebases
2. **Architecture Reference**: Complete technical documentation of system components

### 🔍 How the Example Was Generated

The architecture documentation was created by:

1. **Initial Exploration**: Using `codebaseRAG:search` tools to understand entry points
2. **Component Discovery**: Function-level searches to map service relationships
3. **Data Flow Analysis**: Tracing execution paths through intelligent chunking
4. **Performance Analysis**: Real metrics from the current 11,363 indexed chunks
5. **Best Practices**: Derived from actual system behavior and optimization

### 📊 Key Insights from Real Usage

**Search Precision Results**:
- 🎯 **Function-level accuracy**: Returns specific functions like `src/services/code_parser_service.py:30`
- 🌳 **AST parsing insights**: Tree-sitter integration details from live code
- ⚡ **Performance data**: < 100ms parsing times from actual benchmarks
- 📈 **Scalability metrics**: 11,363 chunks indexed in ~1.1 minutes

**Real System Stats** (Current Project):
```
📊 Indexed Collections:
├── Code: 8,524 intelligent chunks (functions, classes, methods)
├── Config: 280 configuration chunks (JSON/YAML objects)
├── Documentation: 2,559 documentation chunks (Markdown sections)
└── Total: 11,363 semantic chunks ready for search
```

## Documentation

For comprehensive guides and references:
- **[docs/MCP_TOOLS.md](docs/MCP_TOOLS.md)**: Complete MCP tools reference with parameters and examples
- **[docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)**: Best practices for search optimization and cross-project workflows

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
