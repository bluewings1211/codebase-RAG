# 🏗️ Agentic-RAG Project Architecture Deep Dive

## Overview

**Agentic-RAG** is a sophisticated **Codebase RAG (Retrieval-Augmented Generation) MCP Server** that enables AI agents to understand and query codebases with **function-level precision** through intelligent syntax-aware code chunking.

### Core Innovation: Intelligent Code Chunking
Unlike traditional RAG systems that process entire files, this system uses **Tree-sitter AST parsing** to break code into semantically meaningful chunks (functions, classes, methods) with rich metadata.

## System Architecture & Data Flow

### Entry Points
- **`src/main.py:7`**: FastMCP server initialization and tool registration
- **`manual_indexing.py`**: Standalone heavy indexing operations
- **`demo_mcp_usage.py`**: Usage demonstrations

### Complete Data Flow

**Indexing Flow:**
```
📁 Source Code
    ↓
🔍 File Discovery (project_analysis_service.py)
    ↓
🌳 AST Parsing (code_parser_service.py + Tree-sitter)
    ↓
⚡ Intelligent Chunking (Function/Class-level granularity)
    ↓
🧠 Batch Embedding Generation (embedding_service.py + Ollama/MLX)
    ↓
💾 Vector Storage (qdrant_service.py + Qdrant)
    ↓
📊 Metadata Tracking (file_metadata_service.py)
```

**Query Flow (Two-Stage RAG):**
```
📝 Natural Language Query
    ↓
🧠 Query Embedding (embedding_service.py)
    ↓
⚡ Stage 1: Vector Search (qdrant_service.py)
    ├─ Fast ANN search using HNSW algorithm
    ├─ Retrieves top-K candidates (default: 50)
    └─ Bi-encoder semantic similarity scoring
    ↓
🎯 Stage 2: Cross-Encoder Reranking (reranker_service.py)
    ├─ Qwen3-Reranker evaluates query-document pairs
    ├─ Deep semantic relevance scoring
    └─ 22-31% accuracy improvement over single-stage
    ↓
📋 Context Enhancement (if enabled)
    ├─ Adds surrounding code chunks
    └─ Builds breadcrumb navigation
    ↓
✅ Function-Level Results with Precision Ranking
```

## Key Architectural Components

### 🎯 Intelligent Chunking System (`src/services/code_parser_service.py:30`)
- **Tree-sitter Integration**: Supports 8+ programming languages (Python, JS/TS, Go, Rust, Java, C/C++)
- **AST-based Parsing**: Function-level precision instead of whole-file processing
- **Rich Metadata**: Extracts signatures, docstrings, breadcrumbs, access modifiers
- **Error Recovery**: Graceful handling of syntax errors with partial content preservation

**Supported Languages & Chunk Types:**

| Language | Supported Chunk Types |
|----------|----------------------|
| Python | Functions, Classes, Constants, Variables, Imports |
| JavaScript/TypeScript | Functions, Classes, Interfaces, Types, Imports/Exports |
| Go | Functions, Structs, Constants, Variables |
| Rust | Functions, Structs, Enums, Traits |
| Java | Classes, Methods, Interfaces |
| C/C++ | Functions, Structs, Classes |

### 🛠️ MCP Server & Tools (`src/tools/registry.py:12`)

**Core Tools Available:**
- `index_directory()`: Smart indexing with time estimation and incremental updates
- `search()`: Natural language semantic search with two-stage RAG and function-level results
- `health_check_tool()`: System health monitoring (Qdrant, Ollama, Reranker)
- `analyze_repository_tool()`: Repository structure analysis
- `check_index_status()`: Indexing status and recommendations
- `get_chunking_metrics_tool()`: Performance monitoring
- `get_project_info_tool()`: Project information and statistics
- `list_indexed_projects_tool()`: List all indexed projects
- `clear_project_data_tool()`: Clear project indexed data
- `get_file_metadata_tool()`: File-level metadata inspection
- `reindex_file_tool()`: Reindex specific files

### ⚙️ Services Layer (`src/services/`)

| Service | Purpose | Key Features |
|---------|---------|--------------|
| `indexing_service.py` | Orchestrates processing | Parallel processing, batch optimization |
| `code_parser_service.py` | AST parsing | Tree-sitter integration, intelligent chunking |
| `qdrant_service.py` | Vector database | Streaming operations, retry logic |
| `embedding_service.py` | Embeddings | Ollama/MLX integration, batch processing |
| `reranker_service.py` | Two-stage RAG | Cross-encoder reranking, 22-31% accuracy boost |
| `project_analysis_service.py` | Repository analysis | File filtering, structure analysis |
| `file_metadata_service.py` | Change tracking | Incremental indexing, metadata storage |

### 🔧 Utilities Layer (`src/utils/`)

| Utility | Purpose | Key Features |
|---------|---------|--------------|
| `logging_config.py` | Centralized logging | File rotation, third-party suppression |
| `tree_sitter_manager.py` | Parser management | Parser caching, language detection |
| `performance_monitor.py` | Progress tracking | ETA calculation, memory monitoring |
| `chunking_metrics_tracker.py` | Metrics collection | Per-language statistics, error tracking |

## Navigation Strategy & Development Workflow

### 🚀 Most Efficient Exploration Path

1. **Start Here**: `src/main.py:7` - FastMCP app initialization
2. **Core Tools**: `src/tools/registry.py:12` - Understand all available MCP tools
3. **Intelligent Chunking**: `src/services/code_parser_service.py:30` - Tree-sitter AST parsing
4. **Data Flow**: `src/services/indexing_service.py` - Parallel processing orchestration
5. **Vector Operations**: `src/services/qdrant_service.py` - Database interactions

### 🔄 Request/Data Flow
```
MCP Client → FastMCP Tools → Services Layer → Tree-sitter Parser →
Embedding Service → Qdrant Storage → Search Results
```

### Development Commands Quick Reference

```bash
# Setup
.venv/bin/poetry install
.venv/bin/python src/run_mcp.py

# Testing
.venv/bin/pytest tests/
python test_full_functionality.py

# Manual Indexing
python manual_indexing.py -d /path/to/repo -m clear_existing
python manual_indexing.py -d /path/to/repo -m incremental --verbose
```

## Key Design Insights

### Why This Architecture?

1. **Function-Level Precision**: Instead of searching entire files, you get specific functions/classes
2. **Language-Agnostic**: Tree-sitter supports multiple programming languages consistently
3. **Scalable**: Parallel processing, batch operations, and incremental indexing
4. **Error-Tolerant**: Graceful fallback for syntax errors
5. **Memory-Efficient**: Streaming operations with garbage collection

### Performance Characteristics

- **AST Parsing**: < 100ms per file (99th percentile)
- **Memory Usage**: < 50MB additional per 1000 files
- **Total Processing Impact**: < 20% increase over whole-file approach
- **Indexing Speed**: ~1.1 minutes for 129 files (typical small project)

### Collection Architecture

**Content Collections** (store intelligent chunks with embeddings):
- `project_{name}_code`: Intelligent code chunks - functions, classes, methods
- `project_{name}_config`: Structured config chunks - JSON/YAML objects
- `project_{name}_documentation`: Document chunks - Markdown headers, docs

**Metadata Collection** (tracks file states):
- `project_{name}_file_metadata`: File change tracking for incremental indexing

**Example Collection Stats** (Current Project):
- Code: 8,524 intelligent chunks
- Config: 280 configuration chunks
- Documentation: 2,559 documentation chunks
- **Total**: 11,363 indexed chunks

## Advanced Features

### Intelligent Error Handling

**Syntax Error Tolerance:**
- **ERROR Node Detection**: Identifies Tree-sitter ERROR nodes in AST
- **Partial Processing**: Preserves correct code sections even with syntax errors
- **Smart Recovery**: Includes surrounding context for better understanding
- **Graceful Fallback**: Falls back to whole-file processing when AST parsing fails

**Error Classification:**
- **Minor Errors**: Small syntax issues that don't affect overall structure
- **Major Errors**: Significant parsing failures requiring fallback
- **Recoverable Errors**: Errors where partial content can still be extracted

### Performance Optimization Features

- **Parallel Processing**: Multi-threaded file reading and AST parsing
- **Parser Caching**: Cached Tree-sitter parsers for improved performance
- **Batch Operations**: Grouped embedding generation and database insertions
- **Streaming Architecture**: Memory-efficient processing for large codebases
- **Progress Monitoring**: Real-time progress tracking with ETA calculation
- **Memory Management**: Automatic cleanup and garbage collection
- **Adaptive Batching**: Dynamic batch size adjustment based on memory usage
- **Retry Logic**: Exponential backoff for failed operations

### Configuration & Environment

**Environment Variables** (`.env` file):

```bash
# Basic Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Two-Stage RAG / Reranker Configuration
RERANKER_ENABLED=true                    # Enable cross-encoder reranking
RERANKER_PROVIDER=transformers           # transformers | ollama | mlx
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B  # Reranker model
RERANKER_MAX_LENGTH=512                  # Max input length
RERANKER_BATCH_SIZE=8                    # Batch size for reranking
RERANK_TOP_K=50                          # Stage 1 candidates count

# Performance Tuning
INDEXING_CONCURRENCY=4
INDEXING_BATCH_SIZE=20
EMBEDDING_BATCH_SIZE=10
QDRANT_BATCH_SIZE=500
MEMORY_WARNING_THRESHOLD_MB=1000
MAX_FILE_SIZE_MB=5
MAX_DIRECTORY_DEPTH=20

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE_ENABLED=false                   # Enable file logging
LOG_FILE_PATH=logs/codebase-rag.log      # Log file path
LOG_FILE_MAX_SIZE=10                     # Max size in MB before rotation
LOG_FILE_BACKUP_COUNT=5                  # Number of backup files

# Development Settings
FOLLOW_SYMLINKS=false
```

## Development Workflow

### How Components Work Together

1. **Project Discovery**: `ProjectAnalysisService` scans directories respecting `.ragignore`
2. **AST Processing**: `CodeParserService` uses Tree-sitter to extract semantic chunks
3. **Parallel Processing**: `IndexingService` coordinates batch processing across multiple threads
4. **Vector Generation**: `EmbeddingService` creates embeddings via Ollama
5. **Storage**: `QdrantService` streams data to vector database with retry logic
6. **Search**: Natural language queries return function-level matches with context

### Incremental Indexing Workflow

1. **Initial Indexing**: Full codebase processing with metadata storage
2. **Change Detection**: Compare file modification times and content hashes
3. **Selective Processing**: Only reprocess files with detected changes
4. **Metadata Updates**: Update file metadata after successful processing
5. **Collection Management**: Automatic cleanup of stale entries

## Next Steps for Deep Understanding

### Immediate Actions
1. **Explore AST Parsing**: Read `src/services/code_parser_service.py` to understand Tree-sitter integration
2. **Test Intelligent Search**: Use the RAG search tools to see function-level precision in action
3. **Review Language Support**: Check `src/utils/tree_sitter_manager.py` for supported languages
4. **Study Performance**: Examine `src/utils/performance_monitor.py` for optimization strategies

### Advanced Exploration
1. **Custom Language Support**: Extend Tree-sitter parsers for new languages
2. **Chunk Optimization**: Tune chunk types and metadata extraction
3. **Search Enhancement**: Improve semantic search algorithms
4. **Scaling**: Optimize for larger codebases

### Testing & Validation

```bash
# Core functionality tests
.venv/bin/pytest tests/test_code_parser_service.py
.venv/bin/pytest tests/test_intelligent_chunking.py

# Performance benchmarks
.venv/bin/pytest tests/test_performance_benchmarks.py

# Full integration test
python test_full_functionality.py
```

## Key Takeaways

This is a **production-ready, intelligent codebase RAG system** with:

- ✅ **Function-level search precision**
- ✅ **Two-Stage RAG** with 22-31% accuracy improvement via cross-encoder reranking
- ✅ **Multi-language support** via Tree-sitter
- ✅ **Scalable parallel processing**
- ✅ **Error-tolerant AST parsing**
- ✅ **Incremental indexing capabilities**
- ✅ **Rich MCP tool ecosystem**
- ✅ **Comprehensive performance monitoring**
- ✅ **Memory-efficient streaming operations**
- ✅ **Centralized logging** with file rotation support

The system is designed for immediate productivity with advanced semantic search operations. You can start using the RAG search tools to explore code relationships and understand implementation patterns with unprecedented precision.

## External Dependencies

- **Qdrant**: Vector database for embeddings storage
- **Ollama**: Local embedding model service
- **Tree-sitter**: Syntax-aware parsing for multiple languages
- **FastMCP**: Model Context Protocol server framework
- **PyTorch**: Deep learning framework for reranker inference
- **Transformers (HuggingFace)**: Cross-encoder model loading and inference

## File Organization Patterns

- **Smart Filtering**: Uses `.ragignore` files for excluding directories/files
- **Automatic Categorization**: Files categorized by type (code/config/documentation)
- **Collection Naming**: Deterministic collection names: `project_{name}_{type}`
- **Binary Detection**: Automatically skips binary files and large files
- **Language Detection**: Identifies programming languages for better categorization
- **Gitignore Integration**: Respects `.gitignore` patterns for relevant file detection
