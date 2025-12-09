# Getting Started

This guide will help you set up the Codebase RAG MCP Server and start using it with your projects.

## Installation

We provide a helper script to automate the setup process.

### 1. Prerequisites
Ensure you have the following installed:
*   **Git**
*   **Python 3.10+**
*   **uv** (Python package manager): [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
*   **Docker** (Recommended for Qdrant database)
*   **Ollama** (For embedding models)

### 2. Setup Script

Run the following commands in your terminal:

```bash
git clone <repository_url>
cd codebase-rag-mcp
chmod +x setup.sh
./setup.sh
```

The script will:
*   Install Python dependencies.
*   Configure your environment (`.env`).
*   Verify your Qdrant and Ollama setup.
*   Provide the exact command to register the server with Claude.

### 3. Add to Claude

After running the setup script, copy the output command and run it to register the MCP server with Claude Desktop or Claude Code. It will look something like this:

```bash
claude mcp add codebase-rag-mcp --command "uv" --args "run" --args "python" --args "src/run_mcp.py"
```

## Basic Usage

Once installed, you need to index your code so the AI can understand it.

### Indexing a Project

The most reliable way to start is using the manual indexing tool. This ensures your project is fully processed before you start asking questions.

To index the current directory (for example, this repository itself):

```bash
uv run python manual_indexing.py -d "." -m clear_existing
```

*   `-d "."`: Specifies the current directory as the target.
*   `-m clear_existing`: Clears any previous data for this project to ensure a fresh index.

### Searching Your Codebase

Once indexed, you can ask Claude questions about your code.

**Example Queries:**
*   "How does the indexing service work?"
*   "Find all functions related to vector database connections."
*   "Explain the architecture of this project."

## MCP Tools Quick Start

The server exposes several tools to the AI. Here are the most common ones:

*   **`search`**: The primary tool. Finds relevant code based on natural language.
    *   *Usage*: "Search for authentication logic."
*   **`index_directory`**: triggers indexing from within the chat (best for small updates).
    *   *Usage*: "Index the 'src/utils' folder."
*   **`health_check`**: Verifies everything is running smoothly.
    *   *Usage*: "Check if the RAG server is healthy."

For a complete list of tools and their parameters, see [docs/MCP_TOOLS.md](docs/MCP_TOOLS.md).
