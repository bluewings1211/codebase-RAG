#!/bin/bash

# Setup script for Codebase RAG MCP Server

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

printf "${GREEN}Starting setup for Codebase RAG MCP Server...${NC}\n"

# 1. Check for uv
if ! command -v uv &> /dev/null; then
    printf "${RED}Error: 'uv' is not installed.${NC}\n"
    printf "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/\n"
    exit 1
fi

# 2. Install dependencies
printf "${YELLOW}Installing dependencies with uv sync...${NC}\n"
uv sync

# 3. Setup .env
if [ ! -f .env ]; then
    printf "${YELLOW}Creating .env file from .env.example...${NC}\n"
    cp .env.example .env
    printf "${GREEN}.env created. Please edit it to customize your configuration.${NC}\n"
else
    printf "${GREEN}.env file already exists.${NC}\n"
fi

# 4. Check Docker (for Qdrant)
if command -v docker &> /dev/null; then
    printf "${GREEN}Docker is installed.${NC}\n"
    # Check if Qdrant container is running
    if docker ps | grep -q qdrant; then
         printf "${GREEN}Qdrant container is running.${NC}\n"
    else
         printf "${YELLOW}Qdrant container is NOT running.${NC}\n"
         printf "Recommended command to start Qdrant:\n"
         printf "docker run -d -p 6333:6333 -p 6334:6334 -v \$(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant\n"
    fi
else
    printf "${YELLOW}Warning: Docker is not installed. You will need to run Qdrant manually.${NC}\n"
fi

# 5. Check Ollama
if command -v ollama &> /dev/null; then
    printf "${GREEN}Ollama is installed.${NC}\n"
    # Check if nomic-embed-text is pulled
    if ollama list | grep -q nomic-embed-text; then
        printf "${GREEN}Embedding model 'nomic-embed-text' is available.${NC}\n"
    else
        printf "${YELLOW}Embedding model 'nomic-embed-text' not found.${NC}\n"
        printf "Please run: ollama pull nomic-embed-text\n"
    fi
else
    printf "${YELLOW}Warning: Ollama is not installed. You will need a running Ollama instance or another embedding provider.${NC}\n"
fi

printf "\n${GREEN}Setup complete!${NC}\n"
printf "----------------------------------------------------------------\n"
printf "To add this MCP server to Claude Code, run the following command:\n"
printf "\n"
printf "claude mcp add codebase-rag-mcp --command \"uv\" --args \"run\" --args \"python\" --args \"src/run_mcp.py\"\n"
printf "\n"
printf "----------------------------------------------------------------\n"
