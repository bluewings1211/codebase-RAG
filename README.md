# Codebase RAG MCP Server

A Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server designed to assist AI agents and developers in understanding and navigating codebases.

## Quick Links

*   **[Getting Started](GETTING_STARTED.md)**: Installation and Basic Usage.
*   **[User Manual](MANUAL.md)**: Tutorials, Features, and Advanced Usage.
*   **[Configuration](docs/CONFIGURATION.md)**: Environment variables and performance tuning.
*   **[Architecture](ARCHITECTURE.md)**: System design and high-level overview.
*   **[Contributing](CONTRIBUTING.md)**: Developer setup and guidelines.

## Overview

This tool allows you to "chat" with your code. It indexes your local directories or GitHub repositories into a vector database (Qdrant), allowing Large Language Models to perform semantic searches and retrieve accurate code snippets.

### Key Features

*   **🔍 Semantic Search**: Find code by meaning, not just keywords.
*   **🧠 Intelligent Chunking**: Parses code into functions and classes using Tree-sitter for better context.
*   **⚡ High Performance**: Supports incremental indexing and MPS acceleration on macOS.
*   **🌐 Multi-Language**: Supports Python, JavaScript, TypeScript, Go, Rust, Java, C++, and more.

## Quick Start

1.  **Setup**:
    ```bash
    ./setup.sh
    ```

2.  **Index Code**:
    ```bash
    uv run python manual_indexing.py -d "." -m clear_existing
    ```

3.  **Connect**:
    Follow the instructions from `setup.sh` to add the server to your MCP client (e.g., Claude Desktop).

For detailed instructions, please visit **[Getting Started](GETTING_STARTED.md)**.

## Documentation

Detailed documentation is available in the `docs/` directory:
*   [MCP Tools Reference](docs/MCP_TOOLS.md)
*   [Architecture Deep Dive](docs/ARCHITECTURE_DEEP_DIVE.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
