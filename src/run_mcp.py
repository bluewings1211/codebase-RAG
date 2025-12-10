#!/usr/bin/env python3
"""
MCP Server runner for codebase RAG.
This script runs the MCP server in stdio mode for integration with Claude Code and other MCP clients.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import app
from utils.logging_config import setup_logging

# Configure logging to stderr so it doesn't interfere with JSON-RPC communication
setup_logging(stream=sys.stderr)


async def main():
    """Run the MCP server in stdio mode."""
    logger = logging.getLogger(__name__)
    logger.info("Starting Codebase RAG MCP Server...")
    logger.info("Server name: codebase-rag-mcp")
    logger.info("Listening for JSON-RPC requests on stdin...")

    try:
        await app.run_stdio_async()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
