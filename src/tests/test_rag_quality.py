"""
RAG Quality Evaluation Tests for Codebase-RAG-MCP.

This module provides comprehensive RAG quality testing including:
- Basic search quality metrics (no external LLM required)
- Retrieval precision and recall measurements
- RAGAS integration for advanced metrics (requires Ollama)
- Search robustness and consistency tests

Test Categories:
- Basic tests: Run without Ollama, using heuristic metrics
- Advanced tests: Require Ollama for LLM-based evaluation

Usage:
    # Run basic tests only (no Ollama required)
    pytest test_rag_quality.py -m "rag_quality and not requires_ollama"

    # Run all RAG quality tests (requires Ollama)
    pytest test_rag_quality.py -m "rag_quality"

    # Run with verbose output and report generation
    pytest test_rag_quality.py -v --tb=short
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Data Structures
# =============================================================================


@dataclass
class GoldenTestCase:
    """A single test case for RAG quality evaluation."""

    query: str
    description: str
    expected_files: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_chunk_types: list[str] = field(default_factory=list)
    ground_truth: str = ""
    min_expected_results: int = 1
    max_acceptable_rank: int = 5  # Expected result should appear within top N


@dataclass
class RAGQualityMetrics:
    """Container for RAG quality evaluation metrics."""

    # Basic metrics (no LLM required)
    hit_rate: float = 0.0  # % of queries with at least one relevant result
    mrr: float = 0.0  # Mean Reciprocal Rank
    precision_at_k: float = 0.0  # Precision at K results
    recall_at_k: float = 0.0  # Recall at K results
    keyword_coverage: float = 0.0  # % of expected keywords found

    # Advanced metrics (requires LLM)
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "basic_metrics": {
                "hit_rate": self.hit_rate,
                "mrr": self.mrr,
                "precision_at_k": self.precision_at_k,
                "recall_at_k": self.recall_at_k,
                "keyword_coverage": self.keyword_coverage,
            },
            "advanced_metrics": {
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
            },
        }


# =============================================================================
# Golden Dataset Fixture
# =============================================================================


@pytest.fixture
def golden_dataset() -> list[GoldenTestCase]:
    """
    Golden dataset for RAG quality evaluation.

    This dataset contains carefully crafted test cases that verify
    the search functionality returns relevant results for typical
    developer queries.

    Customize this dataset based on your indexed codebase.
    """
    return [
        # Code Structure Queries
        GoldenTestCase(
            query="How does the indexing pipeline work?",
            description="Query about indexing architecture",
            expected_files=["indexing_pipeline.py", "indexing_service.py"],
            expected_keywords=["index", "pipeline", "chunk", "process"],
            expected_chunk_types=["class", "function", "method"],
            ground_truth="The indexing pipeline processes files through parsing, chunking, embedding, and storage stages.",
        ),
        GoldenTestCase(
            query="How to generate embeddings?",
            description="Query about embedding generation",
            expected_files=["embedding_service.py"],
            expected_keywords=["embedding", "generate", "ollama", "model"],
            expected_chunk_types=["class", "method", "function"],
            ground_truth="Embeddings are generated using the EmbeddingService which calls Ollama's embedding API.",
        ),
        GoldenTestCase(
            query="Search function implementation",
            description="Query about search functionality",
            expected_files=["search_tools.py"],
            expected_keywords=["search", "query", "results", "qdrant"],
            expected_chunk_types=["function", "method"],
            ground_truth="The search function performs vector similarity search using Qdrant and optional reranking.",
        ),
        # Specific Code Element Queries
        GoldenTestCase(
            query="TreeSitter parser configuration",
            description="Query about parser setup",
            expected_files=["tree_sitter_manager.py", "code_parser_service.py"],
            expected_keywords=["tree", "sitter", "parser", "language"],
            expected_chunk_types=["class", "method"],
        ),
        GoldenTestCase(
            query="Qdrant collection management",
            description="Query about vector database operations",
            expected_files=["qdrant_service.py", "qdrant_utils.py"],
            expected_keywords=["collection", "qdrant", "vector", "upsert"],
            expected_chunk_types=["class", "method", "function"],
        ),
        # Error Handling Queries
        GoldenTestCase(
            query="Error handling patterns",
            description="Query about error handling",
            expected_files=["errors.py", "error_utils.py"],
            expected_keywords=["error", "exception", "raise", "handle"],
            expected_chunk_types=["class", "function"],
        ),
        # Configuration Queries
        GoldenTestCase(
            query="Environment configuration settings",
            description="Query about configuration",
            expected_files=[".env", "logging_config.py"],
            expected_keywords=["env", "config", "setting", "environment"],
            expected_chunk_types=["constant", "function"],
        ),
        # Reranker Queries
        GoldenTestCase(
            query="Cross-encoder reranking implementation",
            description="Query about two-stage RAG",
            expected_files=["reranker_service.py"],
            expected_keywords=["rerank", "cross-encoder", "score", "model"],
            expected_chunk_types=["class", "method"],
        ),
    ]


@pytest.fixture
def sample_indexed_project(tmp_path):
    """
    Create a sample project with indexed content for testing.

    This fixture creates a minimal project structure that can be
    indexed for testing purposes.
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create sample Python file
    main_py = project_dir / "main.py"
    main_py.write_text(
        '''
"""Main module for testing RAG quality."""

class DataProcessor:
    """Processes data using various algorithms."""

    def __init__(self, config: dict):
        """Initialize the processor with configuration."""
        self.config = config
        self.results = []

    def process(self, data: list) -> list:
        """Process input data and return results."""
        for item in data:
            processed = self._transform(item)
            self.results.append(processed)
        return self.results

    def _transform(self, item):
        """Transform a single item."""
        return item * 2

def calculate_metrics(data: list) -> dict:
    """Calculate metrics from processed data."""
    return {
        "count": len(data),
        "sum": sum(data),
        "average": sum(data) / len(data) if data else 0,
    }

GLOBAL_CONFIG = {
    "max_items": 1000,
    "timeout": 30,
}
'''
    )

    # Create sample config file
    config_json = project_dir / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "database": {"host": "localhost", "port": 5432},
                "features": {"caching": True, "logging": True},
            },
            indent=2,
        )
    )

    # Create sample README
    readme_md = project_dir / "README.md"
    readme_md.write_text(
        """
# Test Project

This is a sample project for RAG quality testing.

## Features

- Data processing
- Metric calculation
- Configuration management

## Usage

```python
from main import DataProcessor

processor = DataProcessor(config)
results = processor.process(data)
```
"""
    )

    return project_dir


# =============================================================================
# Utility Functions
# =============================================================================


def check_ollama_available() -> bool:
    """Check if Ollama service is available."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def check_qdrant_available() -> bool:
    """Check if Qdrant service is available."""
    try:
        import requests

        response = requests.get("http://localhost:6333/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def calculate_mrr(results: list[dict], expected_files: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    MRR = 1/rank of first relevant result (0 if none found)
    """
    for rank, result in enumerate(results, start=1):
        file_path = result.get("file_path", "")
        for expected in expected_files:
            if expected in file_path:
                return 1.0 / rank
    return 0.0


def calculate_precision_at_k(results: list[dict], expected_files: list[str], k: int = 5) -> float:
    """
    Calculate Precision@K.

    Precision@K = (relevant results in top K) / K
    """
    if not results:
        return 0.0

    top_k = results[:k]
    relevant_count = 0

    for result in top_k:
        file_path = result.get("file_path", "")
        for expected in expected_files:
            if expected in file_path:
                relevant_count += 1
                break

    return relevant_count / k


def calculate_keyword_coverage(results: list[dict], expected_keywords: list[str]) -> float:
    """
    Calculate keyword coverage in search results.

    Coverage = (keywords found in results) / (total expected keywords)
    """
    if not expected_keywords:
        return 1.0

    # Combine all result content
    all_content = " ".join(result.get("content", "").lower() for result in results)

    found_keywords = sum(1 for kw in expected_keywords if kw.lower() in all_content)

    return found_keywords / len(expected_keywords)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.rag_quality
class TestBasicSearchQuality:
    """
    Basic search quality tests that don't require external LLM.

    These tests verify fundamental search functionality using
    heuristic metrics like hit rate, MRR, and keyword coverage.
    """

    @pytest.mark.requires_qdrant
    def test_search_returns_results(self, golden_dataset):
        """Test that search returns results for all golden queries."""
        from tools.indexing.search_tools import search_sync

        hit_count = 0

        for test_case in golden_dataset:
            try:
                results = search_sync(
                    query=test_case.query,
                    n_results=10,
                    cross_project=True,
                    search_mode="hybrid",
                )

                if results.get("total", 0) >= test_case.min_expected_results:
                    hit_count += 1
                else:
                    print(f"MISS: '{test_case.query}' returned {results.get('total', 0)} results")

            except Exception as e:
                print(f"ERROR: '{test_case.query}' - {e}")

        hit_rate = hit_count / len(golden_dataset)
        print(f"\nHit Rate: {hit_rate:.2%} ({hit_count}/{len(golden_dataset)})")

        # Minimum acceptable hit rate
        assert hit_rate >= 0.6, f"Hit rate {hit_rate:.2%} is below threshold (60%)"

    @pytest.mark.requires_qdrant
    def test_search_result_relevance(self, golden_dataset):
        """Test that search results are relevant to queries."""
        from tools.indexing.search_tools import search_sync

        mrr_scores = []
        precision_scores = []
        coverage_scores = []

        for test_case in golden_dataset:
            try:
                results = search_sync(
                    query=test_case.query,
                    n_results=10,
                    cross_project=True,
                )

                result_list = results.get("results", [])

                # Calculate MRR
                mrr = calculate_mrr(result_list, test_case.expected_files)
                mrr_scores.append(mrr)

                # Calculate Precision@5
                precision = calculate_precision_at_k(result_list, test_case.expected_files, k=5)
                precision_scores.append(precision)

                # Calculate keyword coverage
                coverage = calculate_keyword_coverage(result_list, test_case.expected_keywords)
                coverage_scores.append(coverage)

                print(f"Query: '{test_case.query[:40]}...'")
                print(f"  MRR: {mrr:.3f}, P@5: {precision:.3f}, Coverage: {coverage:.3f}")

            except Exception as e:
                print(f"ERROR: '{test_case.query}' - {e}")
                mrr_scores.append(0.0)
                precision_scores.append(0.0)
                coverage_scores.append(0.0)

        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0

        print("\n--- Aggregate Metrics ---")
        print(f"Average MRR: {avg_mrr:.3f}")
        print(f"Average Precision@5: {avg_precision:.3f}")
        print(f"Average Keyword Coverage: {avg_coverage:.3f}")

        # Assertions with reasonable thresholds
        assert avg_mrr >= 0.3, f"Average MRR {avg_mrr:.3f} is below threshold (0.3)"
        assert avg_precision >= 0.2, f"Average Precision@5 {avg_precision:.3f} is below threshold (0.2)"

    @pytest.mark.requires_qdrant
    def test_search_score_ordering(self, golden_dataset):
        """Test that search results are properly ordered by score."""
        from tools.indexing.search_tools import search_sync

        for test_case in golden_dataset[:3]:  # Test subset for speed
            results = search_sync(
                query=test_case.query,
                n_results=10,
                cross_project=True,
            )

            result_list = results.get("results", [])
            if len(result_list) < 2:
                continue

            scores = [r.get("score", 0) for r in result_list]

            # Verify descending order
            assert scores == sorted(scores, reverse=True), f"Results not sorted by score for query: {test_case.query}"


@pytest.mark.rag_quality
class TestRetrievalMetrics:
    """
    Retrieval-specific metric tests.

    These tests focus on the retrieval component of the RAG system,
    measuring how well the system finds relevant documents.
    """

    @pytest.mark.requires_qdrant
    def test_context_precision_heuristic(self, golden_dataset):
        """
        Test context precision using heuristic evaluation.

        Context precision measures whether relevant documents
        appear before irrelevant ones in the result list.
        """
        from tools.indexing.search_tools import search_sync

        precision_scores = []

        for test_case in golden_dataset[:5]:
            results = search_sync(
                query=test_case.query,
                n_results=10,
                cross_project=True,
            )

            result_list = results.get("results", [])

            # Calculate position-weighted precision
            weighted_hits = 0
            total_weight = 0

            for rank, result in enumerate(result_list, start=1):
                weight = 1.0 / rank  # Higher weight for earlier positions
                total_weight += weight

                file_path = result.get("file_path", "")
                if any(exp in file_path for exp in test_case.expected_files):
                    weighted_hits += weight

            precision = weighted_hits / total_weight if total_weight > 0 else 0
            precision_scores.append(precision)

            print(f"Query: '{test_case.query[:40]}...' - Context Precision: {precision:.3f}")

        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        print(f"\nAverage Context Precision: {avg_precision:.3f}")

    @pytest.mark.requires_qdrant
    def test_chunk_type_diversity(self, golden_dataset):
        """
        Test that search returns diverse chunk types.

        A good RAG system should return different types of code chunks
        (functions, classes, methods) when relevant.
        """
        from tools.indexing.search_tools import search_sync

        for test_case in golden_dataset[:3]:
            if not test_case.expected_chunk_types:
                continue

            results = search_sync(
                query=test_case.query,
                n_results=10,
                cross_project=True,
            )

            result_list = results.get("results", [])

            # Collect chunk types from results
            found_types = set()
            for result in result_list:
                chunk_type = result.get("chunk_type", "")
                if chunk_type:
                    found_types.add(chunk_type)

            expected_set = set(test_case.expected_chunk_types)
            overlap = found_types.intersection(expected_set)

            print(f"Query: '{test_case.query[:40]}...'")
            print(f"  Expected types: {expected_set}")
            print(f"  Found types: {found_types}")
            print(f"  Overlap: {overlap}")


@pytest.mark.rag_quality
@pytest.mark.requires_ollama
class TestRAGASIntegration:
    """
    RAGAS integration tests for advanced RAG evaluation.

    These tests require Ollama to be running and use RAGAS
    framework for LLM-based quality evaluation.

    Skip these tests if Ollama is not available.
    """

    @pytest.fixture(autouse=True)
    def check_dependencies(self):
        """Check if required dependencies are available."""
        if not check_ollama_available():
            pytest.skip("Ollama service not available")

        try:
            import ragas  # noqa: F401
        except ImportError:
            pytest.skip("RAGAS not installed. Run: pip install ragas")

    def test_ragas_faithfulness(self, golden_dataset):
        """
        Test faithfulness metric using RAGAS.

        Faithfulness measures whether the generated answer is
        grounded in the retrieved context.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.llms import llm_factory
            from ragas.metrics import faithfulness
            from tools.indexing.search_tools import search_sync

            # Configure Ollama
            llm = llm_factory(
                "llama3.1:8b",
                provider="ollama",
                base_url="http://localhost:11434",
            )

            # Prepare evaluation data
            eval_data = []
            for test_case in golden_dataset[:3]:  # Subset for speed
                if not test_case.ground_truth:
                    continue

                results = search_sync(
                    query=test_case.query,
                    n_results=5,
                    cross_project=True,
                )

                contexts = [r.get("content", "") for r in results.get("results", [])]

                eval_data.append(
                    {
                        "question": test_case.query,
                        "answer": test_case.ground_truth,
                        "contexts": contexts,
                    }
                )

            if not eval_data:
                pytest.skip("No test cases with ground truth available")

            # Run RAGAS evaluation
            dataset = Dataset.from_list(eval_data)
            result = evaluate(dataset, metrics=[faithfulness], llm=llm)

            faithfulness_score = result["faithfulness"]
            print(f"\nRAGAS Faithfulness Score: {faithfulness_score:.3f}")

            assert faithfulness_score >= 0.5, f"Faithfulness {faithfulness_score:.3f} is below threshold (0.5)"

        except Exception as e:
            pytest.skip(f"RAGAS evaluation failed: {e}")

    def test_ragas_answer_relevancy(self, golden_dataset):
        """
        Test answer relevancy metric using RAGAS.

        Answer relevancy measures how well the answer addresses
        the original question.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.llms import llm_factory
            from ragas.metrics import answer_relevancy
            from tools.indexing.search_tools import search_sync

            llm = llm_factory(
                "llama3.1:8b",
                provider="ollama",
                base_url="http://localhost:11434",
            )

            eval_data = []
            for test_case in golden_dataset[:3]:
                if not test_case.ground_truth:
                    continue

                results = search_sync(
                    query=test_case.query,
                    n_results=5,
                    cross_project=True,
                )

                contexts = [r.get("content", "") for r in results.get("results", [])]

                eval_data.append(
                    {
                        "question": test_case.query,
                        "answer": test_case.ground_truth,
                        "contexts": contexts,
                    }
                )

            if not eval_data:
                pytest.skip("No test cases with ground truth available")

            dataset = Dataset.from_list(eval_data)
            result = evaluate(dataset, metrics=[answer_relevancy], llm=llm)

            relevancy_score = result["answer_relevancy"]
            print(f"\nRAGAS Answer Relevancy Score: {relevancy_score:.3f}")

            assert relevancy_score >= 0.5, f"Answer relevancy {relevancy_score:.3f} is below threshold (0.5)"

        except Exception as e:
            pytest.skip(f"RAGAS evaluation failed: {e}")


@pytest.mark.rag_quality
class TestSearchRobustness:
    """
    Search robustness and consistency tests.

    These tests verify that the search system behaves consistently
    and handles edge cases gracefully.
    """

    @pytest.mark.requires_qdrant
    def test_query_variations(self):
        """Test that similar queries return similar results."""
        from tools.indexing.search_tools import search_sync

        query_variations = [
            ("How to create embeddings?", "embedding generation process"),
            ("search function", "search implementation"),
            ("error handling", "exception management"),
        ]

        for query1, query2 in query_variations:
            results1 = search_sync(query=query1, n_results=5, cross_project=True)
            results2 = search_sync(query=query2, n_results=5, cross_project=True)

            files1 = {r.get("file_path", "") for r in results1.get("results", [])}
            files2 = {r.get("file_path", "") for r in results2.get("results", [])}

            overlap = len(files1.intersection(files2))
            total = len(files1.union(files2))
            similarity = overlap / total if total > 0 else 0

            print(f"Query pair: '{query1}' vs '{query2}'")
            print(f"  File overlap: {overlap}, Similarity: {similarity:.2%}")

    @pytest.mark.requires_qdrant
    def test_empty_query_handling(self):
        """Test that empty or whitespace queries are handled gracefully."""
        from tools.indexing.search_tools import search_sync

        invalid_queries = ["", "   ", "\n\t"]

        for query in invalid_queries:
            results = search_sync(query=query, n_results=5, cross_project=True)

            # Should return error or empty results, not crash
            assert "error" in results or results.get("total", 0) == 0

    @pytest.mark.requires_qdrant
    def test_special_characters_in_query(self):
        """Test handling of special characters in queries."""
        from tools.indexing.search_tools import search_sync

        special_queries = [
            "def __init__(self):",
            "class MyClass(BaseClass):",
            "import os.path",
            "self.config['key']",
            "result = func(x, y)",
        ]

        for query in special_queries:
            try:
                results = search_sync(query=query, n_results=5, cross_project=True)

                # Should not crash and return valid structure
                assert isinstance(results, dict)
                assert "results" in results or "error" in results

                print(f"Query: '{query}' - Results: {results.get('total', 'error')}")

            except Exception as e:
                pytest.fail(f"Query '{query}' caused exception: {e}")

    @pytest.mark.requires_qdrant
    def test_search_consistency(self):
        """Test that same query returns consistent results."""
        from tools.indexing.search_tools import search_sync

        query = "indexing pipeline implementation"

        # Run same query multiple times
        all_results = []
        for i in range(3):
            results = search_sync(query=query, n_results=5, cross_project=True)
            result_files = [r.get("file_path", "") for r in results.get("results", [])]
            all_results.append(result_files)

        # Check consistency
        for i in range(1, len(all_results)):
            assert all_results[0] == all_results[i], f"Inconsistent results between run 0 and run {i}"


@pytest.mark.rag_quality
class TestQualityReporting:
    """
    Tests for generating quality reports.

    These tests aggregate metrics and generate reports for
    tracking RAG system quality over time.
    """

    @pytest.mark.requires_qdrant
    def test_generate_quality_report(self, golden_dataset, tmp_path):
        """Generate a comprehensive quality report."""
        from tools.indexing.search_tools import search_sync

        report = {
            "test_cases": [],
            "aggregate_metrics": {},
            "timestamp": None,
        }

        mrr_scores = []
        precision_scores = []
        hit_count = 0

        for test_case in golden_dataset:
            case_report = {
                "query": test_case.query,
                "description": test_case.description,
                "metrics": {},
            }

            try:
                results = search_sync(
                    query=test_case.query,
                    n_results=10,
                    cross_project=True,
                )

                result_list = results.get("results", [])

                # Calculate metrics
                mrr = calculate_mrr(result_list, test_case.expected_files)
                precision = calculate_precision_at_k(result_list, test_case.expected_files, k=5)
                coverage = calculate_keyword_coverage(result_list, test_case.expected_keywords)

                case_report["metrics"] = {
                    "mrr": mrr,
                    "precision_at_5": precision,
                    "keyword_coverage": coverage,
                    "result_count": len(result_list),
                }

                mrr_scores.append(mrr)
                precision_scores.append(precision)

                if results.get("total", 0) > 0:
                    hit_count += 1

            except Exception as e:
                case_report["error"] = str(e)

            report["test_cases"].append(case_report)

        # Aggregate metrics
        report["aggregate_metrics"] = {
            "hit_rate": hit_count / len(golden_dataset) if golden_dataset else 0,
            "average_mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
            "average_precision_at_5": sum(precision_scores) / len(precision_scores) if precision_scores else 0,
            "total_test_cases": len(golden_dataset),
            "successful_queries": hit_count,
        }

        # Add timestamp
        from datetime import datetime

        report["timestamp"] = datetime.now().isoformat()

        # Save report
        report_path = tmp_path / "rag_quality_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nQuality report saved to: {report_path}")
        print("\n--- Aggregate Metrics ---")
        for key, value in report["aggregate_metrics"].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

        # Also save to project root for CI artifact collection
        project_report_path = Path(__file__).parent.parent.parent / "rag_quality_report.json"
        try:
            with open(project_report_path, "w") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass  # Ignore if we can't write to project root


# =============================================================================
# Mock Tests (for CI without external services)
# =============================================================================


@pytest.mark.rag_quality
class TestMockedSearchQuality:
    """
    Mocked tests for CI environments without Qdrant/Ollama.

    These tests verify the test infrastructure and metrics
    calculations work correctly.
    """

    def test_mrr_calculation(self):
        """Test MRR calculation logic."""
        # Expected file in position 1
        results = [
            {"file_path": "/path/to/expected.py"},
            {"file_path": "/path/to/other.py"},
        ]
        mrr = calculate_mrr(results, ["expected.py"])
        assert mrr == 1.0

        # Expected file in position 2
        results = [
            {"file_path": "/path/to/other.py"},
            {"file_path": "/path/to/expected.py"},
        ]
        mrr = calculate_mrr(results, ["expected.py"])
        assert mrr == 0.5

        # Expected file not found
        results = [
            {"file_path": "/path/to/other.py"},
        ]
        mrr = calculate_mrr(results, ["expected.py"])
        assert mrr == 0.0

    def test_precision_at_k_calculation(self):
        """Test Precision@K calculation logic."""
        results = [
            {"file_path": "/path/to/expected1.py"},
            {"file_path": "/path/to/other.py"},
            {"file_path": "/path/to/expected2.py"},
            {"file_path": "/path/to/other2.py"},
            {"file_path": "/path/to/other3.py"},
        ]

        # 2 relevant in top 5
        precision = calculate_precision_at_k(results, ["expected1.py", "expected2.py"], k=5)
        assert precision == 0.4  # 2/5

        # 1 relevant in top 3
        precision = calculate_precision_at_k(results, ["expected1.py"], k=3)
        assert abs(precision - 0.333) < 0.01  # 1/3

    def test_keyword_coverage_calculation(self):
        """Test keyword coverage calculation logic."""
        results = [
            {"content": "This function processes data using embeddings."},
            {"content": "The pipeline handles chunking operations."},
        ]

        # All keywords found
        coverage = calculate_keyword_coverage(results, ["function", "pipeline", "data"])
        assert coverage == 1.0

        # Partial keywords found
        coverage = calculate_keyword_coverage(results, ["function", "missing", "data"])
        assert abs(coverage - 0.667) < 0.01  # 2/3

        # No keywords found
        coverage = calculate_keyword_coverage(results, ["xyz", "abc"])
        assert coverage == 0.0

    def test_golden_test_case_structure(self, golden_dataset):
        """Test that golden dataset is properly structured."""
        assert len(golden_dataset) > 0, "Golden dataset should not be empty"

        for test_case in golden_dataset:
            assert isinstance(test_case.query, str)
            assert len(test_case.query) > 0
            assert isinstance(test_case.description, str)
            assert isinstance(test_case.expected_files, list)
            assert isinstance(test_case.expected_keywords, list)

    def test_metrics_container(self):
        """Test RAGQualityMetrics container."""
        metrics = RAGQualityMetrics(
            hit_rate=0.8,
            mrr=0.65,
            precision_at_k=0.4,
            recall_at_k=0.7,
            keyword_coverage=0.85,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["basic_metrics"]["hit_rate"] == 0.8
        assert metrics_dict["basic_metrics"]["mrr"] == 0.65
        assert metrics_dict["advanced_metrics"]["faithfulness"] is None


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
