# Configuration Guide

This document provides a comprehensive reference for all configuration options available in the Codebase RAG MCP Server.

## Table of Contents

1.  [Quick Start](#quick-start)
2.  [Environment Variables Reference](#environment-variables-reference)
    *   [Embedding Provider](#embedding-provider)
    *   [MLX Server (Apple Silicon)](#mlx-server-apple-silicon)
    *   [Ollama](#ollama)
    *   [Reranker (Two-Stage RAG)](#reranker-two-stage-rag)
    *   [Qdrant Database](#qdrant-database)
    *   [File Processing](#file-processing)
    *   [Performance Tuning](#performance-tuning)
    *   [Memory Management](#memory-management)
    *   [Logging](#logging)
3.  [High-Speed Indexing with MLX Server](#high-speed-indexing-with-mlx-server)
4.  [Performance Recommendations](#performance-recommendations)

---

## Quick Start

1.  Copy the example configuration file:
    ```bash
    cp .env.example .env
    ```

2.  Edit `.env` to customize your settings. The defaults work for most setups.

3.  Ensure your services are running:
    *   **Qdrant**: `docker run -d -p 6333:6333 qdrant/qdrant`
    *   **Ollama**: `ollama serve` and `ollama pull nomic-embed-text`

---

## Environment Variables Reference

### Embedding Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `ollama` | Embedding provider selection: `ollama` or `mlx_server` |

Choose your embedding provider based on your hardware:
*   **`ollama`**: Works on all platforms. Good for general use.
*   **`mlx_server`**: **48-161x faster** on Apple Silicon (M1/M2/M3/M4). Recommended for large codebases.

---

### MLX Server (Apple Silicon)

For dramatically faster embedding generation on Apple Silicon Macs.

| Variable | Default | Description |
|----------|---------|-------------|
| `MLX_SERVER_URL` | `http://localhost:8000` | MLX embedding server URL |
| `MLX_BATCH_SIZE` | `64` | Batch size for embedding requests (64-128 recommended) |
| `MLX_MODEL_SIZE` | `small` | Model size: `small`, `medium`, `large` |
| `MLX_TIMEOUT` | `120` | Request timeout in seconds |
| `MLX_FALLBACK_TO_OLLAMA` | `true` | Auto-fallback to Ollama if MLX is unavailable |

#### Performance Comparison

| Method | Speed (emb/s) | vs Ollama |
|--------|--------------|-----------|
| **MLX Server (batch=128)** | **1707** | **48x faster** |
| MLX Server (batch=64) | 1646 | 47x faster |
| Ollama (nomic-embed-text) | 35 | baseline |

#### Expected Indexing Times

| Codebase Size | Ollama | MLX Server |
|--------------|--------|------------|
| 10K chunks | ~5 min | ~6 sec |
| 50K chunks | ~24 min | ~30 sec |
| 100K chunks | ~48 min | ~1 min |

---

### Ollama

Default embedding provider configuration.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_DEFAULT_EMBEDDING_MODEL` | `nomic-embed-text` | Default embedding model |

#### Supported Embedding Models

*   `nomic-embed-text` (recommended, 768 dimensions)
*   `mxbai-embed-large` (1024 dimensions)
*   Any other Ollama-compatible embedding model

---

### Reranker (Two-Stage RAG)

Configure the cross-encoder reranker for improved search accuracy. The two-stage RAG system improves search accuracy by **22-31%** over single-stage vector search.

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking for two-stage RAG |
| `RERANKER_PROVIDER` | `transformers` | Reranker provider: `transformers`, `ollama`, or `mlx` |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Cross-encoder model name |
| `RERANKER_MAX_LENGTH` | `512` | Maximum input length for query + document |
| `RERANKER_BATCH_SIZE` | `8` | Batch size for reranking operations |
| `RERANK_TOP_K` | `50` | Number of candidates retrieved in Stage 1 for reranking |

#### How Two-Stage RAG Works

1. **Stage 1 - Fast Vector Search**: Retrieves top-K candidates (default: 50) using approximate nearest neighbor (ANN) search
2. **Stage 2 - Cross-Encoder Reranking**: Evaluates query-document pairs together using Qwen3-Reranker for precise relevance scoring

#### Performance Characteristics

| Hardware | Reranking Latency (50 candidates) |
|----------|-----------------------------------|
| Apple Silicon (MPS) | ~100ms |
| CUDA GPU | ~80-150ms |
| CPU | ~400ms |

#### Reranker Model Options

*   `Qwen/Qwen3-Reranker-0.6B` (default, smallest, fastest)
*   `Qwen/Qwen3-Reranker-4B` (medium, better quality)
*   `Qwen/Qwen3-Reranker-8B` (largest, best quality)

#### Disabling Reranking

For speed-critical applications, disable reranking:

```bash
# In your .env file
RERANKER_ENABLED=false
```

Or disable per-query using the search tool's `enable_reranking=false` parameter.

---

### Qdrant Database

Vector database configuration.

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_HOST` | `localhost` | Qdrant server hostname |
| `QDRANT_PORT` | `6333` | Qdrant gRPC port |

#### Running Qdrant

```bash
# Basic (data not persisted)
docker run -p 6333:6333 qdrant/qdrant

# With persistence (recommended)
docker run -d -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_data:/qdrant/storage" \
  qdrant/qdrant
```

---

### File Processing

Control how files are discovered and processed during indexing.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE_MB` | `5` | Skip files larger than this size (MB) |
| `MAX_DIRECTORY_DEPTH` | `20` | Maximum directory traversal depth |
| `FOLLOW_SYMLINKS` | `false` | Whether to follow symbolic links |
| `DETECT_BINARY_FILES` | `true` | Auto-detect and skip binary files |
| `LOG_SKIPPED_FILES` | `true` | Log when files are skipped |

---

### Performance Tuning

Fine-tune indexing performance for your hardware.

| Variable | Default | Description |
|----------|---------|-------------|
| `INDEXING_CONCURRENCY` | `4` | Number of parallel file processing workers |
| `INDEXING_BATCH_SIZE` | `20` | Files processed per batch |
| `EMBEDDING_BATCH_SIZE` | `10` | Texts sent to embedding API per call |
| `QDRANT_BATCH_SIZE` | `500` | Points inserted to database per batch |

#### Recommended Settings by Hardware

**High-end workstation (32GB+ RAM, 8+ cores)**:
```bash
INDEXING_CONCURRENCY=8
INDEXING_BATCH_SIZE=50
EMBEDDING_BATCH_SIZE=20
QDRANT_BATCH_SIZE=1000
```

**Standard laptop (16GB RAM, 4 cores)**:
```bash
INDEXING_CONCURRENCY=4
INDEXING_BATCH_SIZE=20
EMBEDDING_BATCH_SIZE=10
QDRANT_BATCH_SIZE=500
```

**Resource-constrained (8GB RAM)**:
```bash
INDEXING_CONCURRENCY=2
INDEXING_BATCH_SIZE=10
EMBEDDING_BATCH_SIZE=5
QDRANT_BATCH_SIZE=200
```

---

### Memory Management

Control memory usage during large indexing operations.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_WARNING_THRESHOLD_MB` | `1000` | Warn when memory usage exceeds this |
| `MEMORY_CLEANUP_INTERVAL` | `5` | Batches between memory cleanup checks |
| `FORCE_CLEANUP_THRESHOLD_MB` | `1500` | Force garbage collection above this |

---

### Database Operations

Configure database operation behavior and reliability.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_OPERATION_TIMEOUT` | `30` | Operation timeout in seconds |
| `DB_RETRY_ATTEMPTS` | `3` | Number of retry attempts on failure |
| `DB_RETRY_DELAY` | `1.0` | Delay between retries (seconds) |
| `DB_HEALTH_CHECK_INTERVAL` | `50` | Operations between health checks |

---

### MCP Response

Control MCP tool response behavior.

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILES_IN_RESPONSE` | `50` | Maximum files returned in a single response |

---

### Logging

Configure logging verbosity and optional file output.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FILE_ENABLED` | `false` | Enable file logging for debugging |
| `LOG_FILE_PATH` | `logs/codebase-rag.log` | Path to log file (relative or absolute) |
| `LOG_FILE_MAX_SIZE` | `10` | Maximum log file size in MB before rotation |
| `LOG_FILE_BACKUP_COUNT` | `5` | Number of backup log files to keep |

**Debugging with File Logging:**

To enable file logging for debugging issues:

```bash
# In your .env file
LOG_FILE_ENABLED=true
LOG_LEVEL=DEBUG
LOG_FILE_PATH=logs/debug.log
```

Log files are automatically rotated when they reach the maximum size. The `logs/` directory will be created automatically if it doesn't exist.

---

## High-Speed Indexing with MLX Server

For Apple Silicon users who need maximum indexing performance.

### Setup

1.  **Clone the MLX embedding server**:
    ```bash
    git clone https://github.com/jakedahn/qwen3-embeddings-mlx.git
    cd qwen3-embeddings-mlx
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the server**:
    ```bash
    python server.py
    # Server runs at http://localhost:8000
    ```

4.  **Configure the RAG server**:
    ```bash
    # In your .env file
    EMBEDDING_PROVIDER=mlx_server
    MLX_SERVER_URL=http://localhost:8000
    MLX_BATCH_SIZE=64
    ```

### Important Notes

*   **Vector Dimensions**: MLX uses Qwen3-Embedding (1024 dimensions) vs Ollama's nomic-embed-text (768 dimensions). Collections are created with the correct dimension automatically.
*   **Requires Re-indexing**: If switching providers, you need to re-index your codebase due to different embedding dimensions.
*   **Apple Silicon Only**: MLX Server requires Apple Silicon (M1/M2/M3/M4) Mac.
*   **Auto-Fallback**: If MLX server is unavailable, the system automatically falls back to Ollama (configurable).

---

## Performance Recommendations

### For Large Codebases (10,000+ files)

1.  Use MLX Server if on Apple Silicon
2.  Increase batch sizes:
    ```bash
    INDEXING_BATCH_SIZE=50
    EMBEDDING_BATCH_SIZE=20
    ```
3.  Use manual indexing tool for initial index:
    ```bash
    uv run python manual_indexing.py -d /path/to/repo -m clear_existing
    ```

### For CI/CD Environments

1.  Use incremental indexing to minimize processing time
2.  Consider pre-built Qdrant snapshots for faster startup
3.  Set appropriate timeouts:
    ```bash
    DB_OPERATION_TIMEOUT=60
    MLX_TIMEOUT=180
    ```

### For Memory-Constrained Systems

1.  Reduce batch sizes
2.  Lower memory thresholds:
    ```bash
    MEMORY_WARNING_THRESHOLD_MB=500
    FORCE_CLEANUP_THRESHOLD_MB=800
    ```
3.  Limit concurrent workers:
    ```bash
    INDEXING_CONCURRENCY=2
    ```
