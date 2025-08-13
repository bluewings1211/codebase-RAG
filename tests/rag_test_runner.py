"""
RAG Test Runner

This script runs a series of tests against the RAG pipeline to evaluate its
performance and accuracy. It is configured via environment variables.
"""

import asyncio
import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Add src directory to path to allow importing project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Import the function to be used
from src.tools.project.project_tools import list_indexed_projects_sync

def get_git_commit_hash() -> str:
    """Gets the current git commit hash of the repository."""
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            text=True,
            stderr=subprocess.PIPE
        ).strip()
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not get git commit hash: {e}")
        return "unknown"

# Import the function to be tested
from src.tools.indexing.search_tools import search_async_cached

async def run_test_for_project(project_name: str, query: str, max_tool_uses: int, commit_hash: str):
    """
    Runs a single test for a given project and generates a report.
    """
    print(f"--> Running test for project: {project_name}")

    # Ensure the report directory exists
    report_dir = Path(__file__).parent.parent / ".test"
    report_dir.mkdir(exist_ok=True)

    tool_calls = []
    total_start_time = time.time()

    # As per refined understanding, we only make one tool call: search.
    # The framework can be extended later if more complex agent simulation is needed.
    if max_tool_uses > 0:
        # Define the parameters for the search tool call
        search_params = {
            "query": query,
            "n_results": 10,
            "cross_project": False,
            "search_mode": "hybrid",
            "include_context": True,
            "context_chunks": 1,
            "target_projects": [project_name],
            "enable_multi_modal": True, # Use the advanced features
        }

        tool_start_time = time.time()
        try:
            # Execute the search tool
            search_result = await search_async_cached(**search_params)
            tool_execution_time = time.time() - tool_start_time

            # Create a summary of the output for quick review
            output_summary = {
                "results_found": search_result.get("total", 0),
                "top_3_files": [r.get("file_path") for r in search_result.get("results", [])[:3]],
                "error": search_result.get("error")
            }

            tool_calls.append({
                "call_number": 1,
                "tool_name": "search",
                "parameters": search_params,
                "execution_time_seconds": round(tool_execution_time, 4),
                "output_summary": output_summary,
                "full_output": search_result
            })

        except Exception as e:
            tool_execution_time = time.time() - tool_start_time
            print(f"    ERROR during search for project {project_name}: {e}")
            tool_calls.append({
                "call_number": 1,
                "tool_name": "search",
                "parameters": search_params,
                "execution_time_seconds": round(tool_execution_time, 4),
                "output_summary": {"error": str(e)},
                "full_output": None
            })

    total_execution_time = time.time() - total_start_time

    # Generate the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"report_{project_name}_{timestamp}.json"
    report_filepath = report_dir / report_filename

    report = {
        "test_run_id": f"{project_name}_{timestamp}_{commit_hash[:7]}",
        "project_name": project_name,
        "git_commit_hash": commit_hash,
        "initial_search_string": query,
        "max_tool_uses_configured": max_tool_uses,
        "total_execution_time_seconds": round(total_execution_time, 4),
        "tool_calls": tool_calls
    }

    # Write report to file
    try:
        with open(report_filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"    SUCCESS: Report saved to {report_filepath}")
    except Exception as e:
        print(f"    ERROR: Failed to write report for {project_name}: {e}")

    print(f"--> Finished test for project: {project_name}")


async def main():
    """Main function to run the RAG tests."""
    print("RAG Test Runner starting...")
    load_dotenv()

    # Get configuration from environment variables
    test_projects_str = os.getenv("TEST_PROJECTS")
    if not test_projects_str:
        print("Error: TEST_PROJECTS environment variable not set or empty. Please set it in your .env file.")
        return

    initial_query = os.getenv("TEST_INITIAL_SEARCH_STRING")
    if not initial_query:
        print("Error: TEST_INITIAL_SEARCH_STRING environment variable not set or empty.")
        return

    max_tool_uses_str = os.getenv("TEST_MAX_TOOL_USES")
    if not max_tool_uses_str or not max_tool_uses_str.isdigit():
        print("Warning: TEST_MAX_TOOL_USES not set or invalid. Defaulting to 1.")
        max_tool_uses = 1
    else:
        max_tool_uses = int(max_tool_uses_str)

    commit_hash = get_git_commit_hash()
    print(f"Testing on git commit: {commit_hash[:7]}")

    # Get available indexed projects
    print("\nFetching list of indexed projects...")
    try:
        indexed_projects_data = list_indexed_projects_sync()
        if "error" in indexed_projects_data:
            print(f"Error fetching indexed projects: {indexed_projects_data['error']}")
            return

        available_projects = {p["name"] for p in indexed_projects_data.get("projects", [])}
        if not available_projects:
            print("Error: No indexed projects found in Qdrant.")
            return
        print(f"Found {len(available_projects)} indexed projects: {', '.join(available_projects)}")
    except Exception as e:
        print(f"An exception occurred while fetching indexed projects: {e}")
        print("Please ensure Qdrant service is running and accessible.")
        return

    # Validate requested test projects
    requested_projects = {p.strip() for p in test_projects_str.split(',') if p.strip()}
    valid_projects = requested_projects.intersection(available_projects)
    invalid_projects = requested_projects.difference(available_projects)

    if invalid_projects:
        print(f"\nWarning: The following requested projects are not indexed and will be skipped: {', '.join(invalid_projects)}")

    if not valid_projects:
        print("\nError: No valid projects to test. Please check your TEST_PROJECTS in .env.")
        return

    print(f"\nWill run tests on: {', '.join(valid_projects)}")
    print("-" * 30)

    # Run tests for each valid project
    for project_name in valid_projects:
        await run_test_for_project(project_name, initial_query, max_tool_uses, commit_hash)

    print("-" * 30)
    print("RAG Test Runner finished.")

if __name__ == "__main__":
    asyncio.run(main())
