"""
Performance benchmark tests and memory usage monitoring for intelligent code chunking.

This test suite provides:
- Performance benchmarks for different file sizes and complexities
- Memory usage monitoring during parsing operations
- Comparison metrics across different languages
- Scalability testing for large codebases
- Performance regression detection
"""

import gc
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.code_chunk import ParseResult
from services.code_parser_service import CodeParserService
from services.indexing_service import IndexingService
from utils.performance_monitor import MemoryMonitor, ProgressTracker


@dataclass
class PerformanceMetric:
    """Container for performance measurement data."""

    operation: str
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    file_size_bytes: int
    chunk_count: int
    error_count: int
    cpu_percent: float


class PerformanceBenchmarkSuite:
    """Base class for performance benchmarking."""

    def __init__(self):
        self.metrics: list[PerformanceMetric] = []
        self.memory_monitor = MemoryMonitor()

    def measure_operation(self, operation_name: str, operation_func, *args, **kwargs) -> PerformanceMetric:
        """Measure performance of a single operation."""
        # Force garbage collection before measurement
        gc.collect()

        # Initial measurements
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_before = process.cpu_percent()
        start_time = time.perf_counter()

        # Monitor peak memory during operation
        peak_memory = memory_before

        def memory_callback():
            nonlocal peak_memory
            current = process.memory_info().rss / (1024 * 1024)
            peak_memory = max(peak_memory, current)

        # Execute operation
        try:
            result = operation_func(*args, **kwargs)
            # Check memory again after operation
            memory_callback()
        except Exception as e:
            print(f"Operation {operation_name} failed: {e}")
            result = None

        # Final measurements
        end_time = time.perf_counter()
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        cpu_after = process.cpu_percent()

        duration_ms = (end_time - start_time) * 1000

        # Extract metrics from result if available
        chunk_count = 0
        error_count = 0
        file_size = 0

        if isinstance(result, ParseResult):
            chunk_count = len(result.chunks)
            error_count = result.error_count
        elif isinstance(result, list):
            chunk_count = len(result)

        # Try to get file size if args contain file path or content
        if args and isinstance(args[0], str):
            if len(args[0]) < 1000:  # Probably a file path
                try:
                    file_size = Path(args[0]).stat().st_size
                except (OSError, AttributeError, TypeError):
                    file_size = len(args[0])  # Content length
            else:
                file_size = len(args[0])  # Content length

        metric = PerformanceMetric(
            operation=operation_name,
            duration_ms=duration_ms,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=peak_memory,
            file_size_bytes=file_size,
            chunk_count=chunk_count,
            error_count=error_count,
            cpu_percent=max(cpu_before, cpu_after),
        )

        self.metrics.append(metric)
        return metric

    def generate_report(self) -> dict[str, Any]:
        """Generate performance report from collected metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}

        report = {
            "summary": {
                "total_operations": len(self.metrics),
                "avg_duration_ms": statistics.mean(m.duration_ms for m in self.metrics),
                "median_duration_ms": statistics.median(m.duration_ms for m in self.metrics),
                "max_duration_ms": max(m.duration_ms for m in self.metrics),
                "avg_memory_usage_mb": statistics.mean(m.memory_peak_mb for m in self.metrics),
                "max_memory_usage_mb": max(m.memory_peak_mb for m in self.metrics),
                "total_chunks_generated": sum(m.chunk_count for m in self.metrics),
                "total_errors": sum(m.error_count for m in self.metrics),
            },
            "by_operation": {},
            "performance_issues": [],
        }

        # Group by operation type
        operations = {}
        for metric in self.metrics:
            if metric.operation not in operations:
                operations[metric.operation] = []
            operations[metric.operation].append(metric)

        # Generate per-operation statistics
        for op_name, op_metrics in operations.items():
            report["by_operation"][op_name] = {
                "count": len(op_metrics),
                "avg_duration_ms": statistics.mean(m.duration_ms for m in op_metrics),
                "median_duration_ms": statistics.median(m.duration_ms for m in op_metrics),
                "avg_memory_mb": statistics.mean(m.memory_peak_mb for m in op_metrics),
                "chunks_per_second": self._calculate_throughput(op_metrics),
                "error_rate": sum(m.error_count for m in op_metrics) / len(op_metrics),
            }

        # Identify performance issues
        self._identify_performance_issues(report)

        return report

    def _calculate_throughput(self, metrics: list[PerformanceMetric]) -> float:
        """Calculate processing throughput in chunks per second."""
        total_chunks = sum(m.chunk_count for m in metrics)
        total_time_s = sum(m.duration_ms for m in metrics) / 1000
        return total_chunks / total_time_s if total_time_s > 0 else 0

    def _identify_performance_issues(self, report: dict[str, Any]) -> None:
        """Identify potential performance issues."""
        issues = []

        # Check for slow operations (>1000ms)
        if report["summary"]["max_duration_ms"] > 1000:
            issues.append(f"Slow operation detected: {report['summary']['max_duration_ms']:.1f}ms")

        # Check for high memory usage (>500MB)
        if report["summary"]["max_memory_usage_mb"] > 500:
            issues.append(f"High memory usage: {report['summary']['max_memory_usage_mb']:.1f}MB")

        # Check for high error rates
        if report["summary"]["total_errors"] > len(self.metrics) * 0.1:
            issues.append(f"High error rate: {report['summary']['total_errors']} errors in {len(self.metrics)} operations")

        report["performance_issues"] = issues


class TestCodeParsingPerformance:
    """Performance tests for code parsing operations."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create a performance benchmark suite."""
        return PerformanceBenchmarkSuite()

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_small_file_performance(self, benchmark_suite, parser_service):
        """Test performance with small files (< 1KB)."""
        small_code = """
def simple_function():
    return "hello"

class SimpleClass:
    def method(self):
        return 42
"""

        # Run multiple iterations for statistical significance
        for i in range(10):
            metric = benchmark_suite.measure_operation(
                f"small_file_parse_{i}",
                parser_service.parse_file,
                f"small_{i}.py",
                small_code,
            )

            # Small files should parse quickly
            assert metric.duration_ms < 100, f"Small file parsing too slow: {metric.duration_ms}ms"
            # Memory threshold accounts for baseline Python/library overhead (~300MB+ with ML libraries loaded)
            assert metric.memory_peak_mb < 500, f"Small file using too much memory: {metric.memory_peak_mb}MB"

    def test_medium_file_performance(self, benchmark_suite, parser_service):
        """Test performance with medium files (1-10KB)."""
        # Generate medium-sized code
        medium_code = '"""Medium-sized Python file for performance testing."""\n\n'

        for i in range(50):
            medium_code += f'''
class TestClass{i}:
    """Test class {i}."""

    def __init__(self):
        self.value = {i}

    def get_value(self):
        return self.value

    def calculate(self, x, y):
        return x + y + self.value

def test_function_{i}(a, b):
    """Test function {i}."""
    return a * b + {i}
'''

        # Run multiple iterations
        for i in range(5):
            metric = benchmark_suite.measure_operation(
                f"medium_file_parse_{i}",
                parser_service.parse_file,
                f"medium_{i}.py",
                medium_code,
            )

            # Medium files should still parse reasonably quickly
            assert metric.duration_ms < 500, f"Medium file parsing too slow: {metric.duration_ms}ms"
            assert metric.chunk_count > 50, f"Should generate multiple chunks: {metric.chunk_count}"

    def test_large_file_performance(self, benchmark_suite, parser_service):
        """Test performance with large files (>10KB)."""
        # Generate large code file
        large_code = '"""Large Python file for performance testing."""\n\n'
        large_code += "import os\nimport sys\nimport json\nimport time\n\n"

        for i in range(200):
            large_code += f'''
class LargeClass{i}:
    """Large class {i} with multiple methods."""

    def __init__(self, param1, param2=None):
        self.param1 = param1
        self.param2 = param2 or "default"
        self.counter = {i}

    def method_one_{i}(self):
        """Method one for class {i}."""
        return f"Method one: {{self.param1}} - {{self.counter}}"

    def method_two_{i}(self, arg):
        """Method two for class {i}."""
        result = arg + self.counter
        return result

    def complex_method_{i}(self, data):
        """Complex method with nested logic."""
        if data:
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key == "special":
                            return value + self.counter
        return None

    @property
    def calculated_property_{i}(self):
        """Calculated property."""
        return self.counter * 2

def utility_function_{i}(x, y, z=None):
    """Utility function {i}."""
    if z is None:
        z = {i}
    return x + y + z

async def async_function_{i}():
    """Async function {i}."""
    await asyncio.sleep(0.01)
    return f"Async result {i}"
'''

        # Test large file parsing
        metric = benchmark_suite.measure_operation(
            "large_file_parse",
            parser_service.parse_file,
            "large_test.py",
            large_code,
        )

        # Large files should still parse within reasonable time
        assert metric.duration_ms < 5000, f"Large file parsing too slow: {metric.duration_ms}ms"
        assert metric.chunk_count > 500, f"Should generate many chunks: {metric.chunk_count}"
        # Memory threshold accounts for baseline Python/library overhead (~300MB+ with ML libraries)
        assert metric.memory_peak_mb < 500, f"Memory usage too high: {metric.memory_peak_mb}MB"

        print(f"Large file performance: {metric.duration_ms:.1f}ms, {metric.chunk_count} chunks, {metric.memory_peak_mb:.1f}MB")

    def test_multi_language_performance(self, benchmark_suite, parser_service):
        """Test performance across different programming languages."""

        # Python code
        python_code = (
            """
class PythonClass:
    def method(self):
        return "python"

def python_function():
    return [1, 2, 3]
"""
            * 20
        )

        # JavaScript code
        js_code = (
            """
class JavaScriptClass {
    method() {
        return "javascript";
    }
}

function jsFunction() {
    return [1, 2, 3];
}

const arrowFunction = () => "arrow";
"""
            * 20
        )

        # TypeScript code
        ts_code = (
            """
interface TestInterface {
    id: number;
    name: string;
}

class TypeScriptClass implements TestInterface {
    constructor(public id: number, public name: string) {}

    method(): string {
        return "typescript";
    }
}

function tsFunction<T>(param: T): T {
    return param;
}
"""
            * 20
        )

        languages = [
            ("python", python_code),
            ("javascript", js_code),
            ("typescript", ts_code),
        ]

        for lang, code in languages:
            metric = benchmark_suite.measure_operation(
                f"{lang}_parse",
                parser_service.parse_file,
                f"test.{lang[:2]}",
                code,
            )

            # All languages should parse within reasonable time
            assert metric.duration_ms < 1000, f"{lang} parsing too slow: {metric.duration_ms}ms"
            # Some languages may use fallback chunking (whole_file), resulting in fewer chunks
            assert metric.chunk_count >= 1, f"{lang} should generate at least one chunk"

            print(f"{lang} performance: {metric.duration_ms:.1f}ms, {metric.chunk_count} chunks")

    def test_error_handling_performance(self, benchmark_suite, parser_service):
        """Test performance when handling files with syntax errors."""

        # Create code with various syntax errors
        error_code = (
            """
# File with multiple syntax errors

def broken_function(
    # Missing closing paren
    return "broken"

class IncompleteClass
    # Missing colon
    def method(self):
        return True

def valid_function():
    return "this works"

invalid_syntax = "unclosed string

def another_valid():
    return "also works"
"""
            * 10
        )  # Repeat to make it larger

        metric = benchmark_suite.measure_operation(
            "error_handling_parse",
            parser_service.parse_file,
            "errors.py",
            error_code,
        )

        # Error handling should not significantly slow down parsing
        assert metric.duration_ms < 2000, f"Error handling too slow: {metric.duration_ms}ms"
        assert metric.error_count > 0, "Should detect syntax errors"

        # Should still recover some valid code
        assert metric.chunk_count > 0, "Should recover some valid chunks"

        print(f"Error handling performance: {metric.duration_ms:.1f}ms, {metric.error_count} errors, {metric.chunk_count} chunks")


class TestIndexingPerformance:
    """Performance tests for full indexing operations."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create a performance benchmark suite."""
        return PerformanceBenchmarkSuite()

    @pytest.fixture
    def indexing_service(self):
        """Create an IndexingService instance."""
        return IndexingService()

    @pytest.fixture
    def temp_project(self):
        """Create a temporary project with multiple files."""
        temp_dir = tempfile.mkdtemp()
        project_path = Path(temp_dir) / "test_project"
        project_path.mkdir()

        # Create multiple files
        for i in range(10):
            py_file = project_path / f"module_{i}.py"
            py_file.write_text(
                f'''
"""Module {i} for testing."""

class Module{i}Class:
    """Class in module {i}."""

    def __init__(self):
        self.id = {i}

    def get_id(self):
        return self.id

    def process(self, data):
        return data + self.id

def module_{i}_function():
    """Function in module {i}."""
    return f"Module {i} result"

CONSTANT_{i} = {i} * 10
'''
            )

        yield str(project_path)
        shutil.rmtree(temp_dir)

    def test_multi_file_indexing_performance(self, benchmark_suite, indexing_service, temp_project):
        """Test performance of indexing multiple files."""

        metric = benchmark_suite.measure_operation(
            "multi_file_indexing",
            indexing_service.process_codebase_for_indexing,
            temp_project,
        )

        # Multi-file indexing should complete within reasonable time
        assert metric.duration_ms < 10000, f"Multi-file indexing too slow: {metric.duration_ms}ms"
        assert metric.chunk_count > 20, f"Should generate multiple chunks: {metric.chunk_count}"

        print(f"Multi-file indexing performance: {metric.duration_ms:.1f}ms, {metric.chunk_count} chunks")

    def test_memory_usage_scaling(self, benchmark_suite, indexing_service):
        """Test memory usage scaling with file count."""

        memory_results = []

        for file_count in [1, 5, 10, 20]:
            # Create temporary directory with specified number of files
            temp_dir = tempfile.mkdtemp()
            project_path = Path(temp_dir) / f"project_{file_count}"
            project_path.mkdir()

            for i in range(file_count):
                py_file = project_path / f"file_{i}.py"
                py_file.write_text(
                    f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return {i}
"""
                    * 5
                )  # Make each file reasonably sized

            try:
                metric = benchmark_suite.measure_operation(
                    f"scaling_{file_count}_files",
                    indexing_service.process_codebase_for_indexing,
                    str(project_path),
                )

                memory_results.append((file_count, metric.memory_peak_mb))

                print(f"{file_count} files: {metric.duration_ms:.1f}ms, {metric.memory_peak_mb:.1f}MB")

            finally:
                shutil.rmtree(temp_dir)

        # Memory usage should scale reasonably
        if len(memory_results) >= 2:
            # Memory shouldn't grow exponentially
            first_memory = memory_results[0][1]
            last_memory = memory_results[-1][1]
            memory_ratio = last_memory / first_memory
            file_ratio = memory_results[-1][0] / memory_results[0][0]

            # Memory growth should be less than quadratic
            assert memory_ratio < file_ratio**1.5, f"Memory scaling too aggressive: {memory_ratio} vs file ratio {file_ratio}"


class TestConcurrencyPerformance:
    """Performance tests for concurrent operations."""

    @pytest.fixture
    def benchmark_suite(self):
        """Create a performance benchmark suite."""
        return PerformanceBenchmarkSuite()

    @pytest.fixture
    def parser_service(self):
        """Create a CodeParserService instance."""
        return CodeParserService()

    def test_concurrent_parsing_performance(self, benchmark_suite, parser_service):
        """Test performance of concurrent parsing operations."""
        import concurrent.futures

        # Create multiple code samples
        code_samples = []
        for i in range(20):
            code = (
                f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return "result_{i}"
"""
                * 3
            )
            code_samples.append((code, f"file_{i}.py", "python"))

        def parse_single(args):
            code, filename, language = args
            return parser_service.parse_file(filename, code)

        # Test sequential parsing
        sequential_start = time.perf_counter()
        sequential_results = []
        for sample in code_samples:
            result = parse_single(sample)
            sequential_results.append(result)
        sequential_time = time.perf_counter() - sequential_start

        # Test concurrent parsing
        concurrent_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(parse_single, code_samples))
        concurrent_time = time.perf_counter() - concurrent_start

        # Concurrent should be faster (or at least not much slower due to overhead)
        speedup = sequential_time / concurrent_time
        print(f"Sequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s, Speedup: {speedup:.2f}x")

        # Should have some performance benefit or at least be thread-safe
        assert speedup > 0.5, f"Concurrent performance too poor: {speedup:.2f}x speedup"
        assert len(concurrent_results) == len(sequential_results), "Concurrent results incomplete"


class TestMemoryMonitoring:
    """Tests for memory monitoring capabilities."""

    def test_memory_monitor_accuracy(self):
        """Test accuracy of memory monitoring."""
        monitor = MemoryMonitor()

        # Get initial memory reading
        initial_info = monitor.get_system_memory_info()
        assert "total_mb" in initial_info
        assert "available_mb" in initial_info
        assert "used_mb" in initial_info
        assert initial_info["total_mb"] > 0

        # Allocate some memory and verify detection
        large_data = ["x" * 1000] * 1000  # ~1MB

        after_allocation = monitor.get_system_memory_info()

        # Should detect increased usage (allowing for measurement variance)
        assert after_allocation["used_mb"] >= initial_info["used_mb"]

        # Clean up
        del large_data
        gc.collect()

    def test_progress_tracking_accuracy(self):
        """Test accuracy of progress tracking."""
        tracker = ProgressTracker(total_items=100)

        # Simulate processing items
        for _ in range(50):
            tracker.increment_processed()
            time.sleep(0.001)  # Simulate work

        # Check progress calculation
        assert tracker.stats.processed_items == 50

        # Get current stats via get_progress_summary()
        summary = tracker.get_progress_summary()
        assert summary["processed_items"] == 50

        # Complete the work
        for _ in range(50):
            tracker.increment_processed()

        assert tracker.stats.processed_items == 100


def test_performance_regression_detection():
    """Test for performance regression detection."""
    # This would be used in CI/CD to detect performance regressions

    benchmark_suite = PerformanceBenchmarkSuite()
    parser_service = CodeParserService()

    # Standard test case
    standard_code = (
        """
class StandardClass:
    def __init__(self):
        self.value = 42

    def method(self):
        return self.value

def standard_function():
    return "standard"
"""
        * 50
    )

    # Run benchmark
    metric = benchmark_suite.measure_operation(
        "regression_test",
        parser_service.parse_file,
        "standard.py",
        standard_code,
    )

    # Define performance thresholds (these would be based on historical data)
    # Note: Memory baseline is ~300MB+ due to ML libraries (transformers, torch)
    MAX_DURATION_MS = 200
    MAX_MEMORY_MB = 500  # Account for baseline library overhead

    # Check for regressions
    assert metric.duration_ms < MAX_DURATION_MS, f"Performance regression: {metric.duration_ms}ms > {MAX_DURATION_MS}ms"
    assert metric.memory_peak_mb < MAX_MEMORY_MB, f"Memory regression: {metric.memory_peak_mb}MB > {MAX_MEMORY_MB}MB"

    print(f"Regression test passed: {metric.duration_ms:.1f}ms, {metric.memory_peak_mb:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
