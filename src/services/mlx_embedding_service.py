"""
MLX Server Embedding Service for high-performance Apple Silicon inference.

This service integrates with the qwen3-embeddings-mlx server to provide
significantly faster embedding generation on MacBooks with Apple Silicon.

Performance benchmarks show MLX Server is 48-161x faster than Ollama:
- MLX Server (batch=128): ~1700 emb/s
- MLX Server (batch=64): ~1646 emb/s
- Ollama (nomic-embed-text): ~35 emb/s

Server: https://github.com/jakedahn/qwen3-embeddings-mlx
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import requests
import torch


@dataclass
class MLXBatchMetrics:
    """Metrics for MLX batch processing."""

    batch_size: int
    total_chars: int
    duration_seconds: float
    embeddings_per_second: float
    successful: int
    failed: int


class MLXServerEmbeddingService:
    """
    MLX Server embedding service for high-performance Apple Silicon inference.

    This service connects to a qwen3-embeddings-mlx server running locally,
    providing batch embedding generation that is 48-161x faster than Ollama.
    """

    # Qwen3-Embedding models output 1024 dimensions
    EMBEDDING_DIMENSION = 1024

    def __init__(
        self,
        server_url: str | None = None,
        batch_size: int | None = None,
        timeout: int | None = None,
        model_size: str | None = None,
    ):
        """
        Initialize the MLX Server embedding service.

        Args:
            server_url: MLX server URL (default: from env or http://localhost:8000)
            batch_size: Batch size for embedding requests (default: from env or 64)
            timeout: Request timeout in seconds (default: from env or 120)
            model_size: Model size to use: 'small', 'medium', 'large' (default: from env or 'small')
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

        # Configuration with environment variable fallbacks
        self.server_url = server_url or os.getenv("MLX_SERVER_URL", "http://localhost:8000")
        self.batch_size = batch_size or int(os.getenv("MLX_BATCH_SIZE", "64"))
        self.timeout = timeout or int(os.getenv("MLX_TIMEOUT", "120"))
        self.model_size = model_size or os.getenv("MLX_MODEL_SIZE", "small")

        # Session for connection pooling
        self._session: requests.Session | None = None

        # Metrics tracking
        self._total_embeddings = 0
        self._total_time = 0.0
        self._total_batches = 0

        self.logger.info(
            f"MLX Server embedding service initialized - "
            f"URL: {self.server_url}, batch_size: {self.batch_size}, "
            f"model: {self.model_size}"
        )

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if not self.logger.handlers:
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.logger.propagate = False

    def _get_session(self) -> requests.Session:
        """Get or create a requests session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()

            # Configure retry strategy with exponential backoff
            from urllib3.util.retry import Retry

            retry_strategy = Retry(
                total=5,  # Total retry attempts
                backoff_factor=1.0,  # Wait 1, 2, 4, 8, 16 seconds between retries
                status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
                allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],  # Allow POST retries
                raise_on_status=False,  # Don't raise on status, let us handle it
            )

            # Configure connection pool with retry
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=retry_strategy,
            )
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)

            # Set keep-alive headers
            self._session.headers.update(
                {
                    "Connection": "keep-alive",
                    "Keep-Alive": "timeout=60, max=1000",
                }
            )
        return self._session

    def health_check(self) -> dict[str, Any]:
        """
        Check if MLX server is running and healthy.

        Returns:
            Dictionary with health status and server info
        """
        try:
            response = self._get_session().get(
                f"{self.server_url}/health",
                timeout=5,
            )

            if response.status_code == 200:
                health_data = response.json()
                return {
                    "healthy": True,
                    "server_url": self.server_url,
                    "server_response": health_data,
                    "provider": "mlx_server",
                }
            else:
                return {
                    "healthy": False,
                    "server_url": self.server_url,
                    "error": f"Server returned status {response.status_code}",
                    "provider": "mlx_server",
                }

        except requests.exceptions.ConnectionError:
            return {
                "healthy": False,
                "server_url": self.server_url,
                "error": f"Cannot connect to MLX server at {self.server_url}. Is it running?",
                "provider": "mlx_server",
            }
        except Exception as e:
            return {
                "healthy": False,
                "server_url": self.server_url,
                "error": str(e),
                "provider": "mlx_server",
            }

    def generate_single_embedding(self, text: str) -> torch.Tensor | None:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding tensor or None on error
        """
        if not text or not text.strip():
            self.logger.warning("Empty or whitespace-only text provided for embedding")
            return None

        try:
            response = self._get_session().post(
                f"{self.server_url}/embed",
                json={"text": text, "model": self.model_size},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                self.logger.error(f"Single embedding failed: {response.text}")
                return None

            result = response.json()
            embedding = result.get("embedding")

            if not embedding:
                self.logger.warning("Received empty embedding from MLX server")
                return None

            return torch.tensor(embedding, dtype=torch.float32)

        except Exception as e:
            self.logger.error(f"Error generating single embedding: {e}")
            return None

    def generate_embeddings_batch(
        self,
        texts: list[str],
        chunk_metadata: list[dict] | None = None,
    ) -> list[torch.Tensor | None]:
        """
        Generate embeddings for a batch of texts using MLX server.

        This is the primary method for high-performance embedding generation.
        Texts are automatically split into optimal batch sizes.

        Args:
            texts: List of texts to embed
            chunk_metadata: Optional metadata for error tracking

        Returns:
            List of embedding tensors (or None for failed items)
        """
        if not texts:
            return []

        self.logger.info(f"MLX Server: Generating embeddings for {len(texts)} texts")
        start_time = time.time()

        all_embeddings: list[torch.Tensor | None] = []
        successful = 0
        failed = 0

        # Process in batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

            try:
                batch_embeddings = self._process_batch(batch_texts, batch_start, chunk_metadata)

                for i, emb in enumerate(batch_embeddings):
                    if emb is not None:
                        successful += 1
                    else:
                        failed += 1
                        self._log_failed_embedding(batch_start + i, chunk_metadata)

                all_embeddings.extend(batch_embeddings)

                # Progress logging at intervals
                if total_batches > 5 and batch_num % 5 == 0:
                    self.logger.info(
                        f"  MLX progress: {batch_num}/{total_batches} batches "
                        f"({successful}/{batch_start + len(batch_texts)} successful)"
                    )

            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed: {e}")
                # Add None for all items in failed batch
                for i in range(len(batch_texts)):
                    all_embeddings.append(None)
                    failed += 1
                    self._log_failed_embedding(batch_start + i, chunk_metadata)

        # Update metrics
        duration = time.time() - start_time
        self._total_embeddings += successful
        self._total_time += duration
        self._total_batches += (len(texts) + self.batch_size - 1) // self.batch_size

        # Log summary
        rate = len(texts) / duration if duration > 0 else 0
        self.logger.info(f"MLX Server: Completed {successful}/{len(texts)} embeddings " f"in {duration:.2f}s ({rate:.1f} emb/s)")

        if failed > 0:
            self.logger.warning(f"MLX Server: {failed} embeddings failed")

        return all_embeddings

    def _process_batch(
        self,
        texts: list[str],
        batch_start_index: int,
        chunk_metadata: list[dict] | None,
    ) -> list[torch.Tensor | None]:
        """
        Process a single batch of texts with automatic retry on failure.

        Args:
            texts: Batch of texts to process
            batch_start_index: Starting index for error tracking
            chunk_metadata: Optional metadata for error tracking

        Returns:
            List of embedding tensors
        """
        # Filter empty texts but preserve positions
        valid_indices = []
        valid_texts = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text)

        # If no valid texts, return all None
        if not valid_texts:
            return [None] * len(texts)

        # Try batch API first
        try:
            embeddings_list = self._call_batch_api(valid_texts)

            if len(embeddings_list) != len(valid_texts):
                raise RuntimeError(f"Embedding count mismatch: got {len(embeddings_list)}, " f"expected {len(valid_texts)}")

            # Reconstruct full results with None for invalid texts
            full_results: list[torch.Tensor | None] = [None] * len(texts)

            for idx, embedding in zip(valid_indices, embeddings_list, strict=True):
                if embedding:
                    full_results[idx] = torch.tensor(embedding, dtype=torch.float32)

            return full_results

        except Exception as batch_error:
            self.logger.warning(f"Batch API failed ({batch_error}), falling back to individual processing")
            return self._process_individually(texts, batch_start_index, chunk_metadata)

    def _call_batch_api(self, texts: list[str], retry_count: int = 3) -> list:
        """Call MLX server batch API with retries."""
        last_error = None

        for attempt in range(retry_count):
            try:
                response = self._get_session().post(
                    f"{self.server_url}/embed_batch",
                    json={"texts": texts, "model": self.model_size},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("embeddings", [])

                # Server error, may be retryable
                if response.status_code >= 500:
                    last_error = RuntimeError(f"Server error {response.status_code}: {response.text}")
                    if attempt < retry_count - 1:
                        wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                        self.logger.warning(f"MLX batch attempt {attempt + 1} failed, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                raise RuntimeError(f"MLX batch request failed: {response.text}")

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 2
                    self.logger.warning(f"MLX connection error on attempt {attempt + 1}: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)

        raise last_error or RuntimeError("Batch API failed after retries")

    def _process_individually(
        self,
        texts: list[str],
        batch_start_index: int,
        chunk_metadata: list[dict] | None,
    ) -> list[torch.Tensor | None]:
        """Process texts individually as fallback when batch fails."""
        results: list[torch.Tensor | None] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results.append(None)
                continue

            # Try individual embedding with retry
            embedding = self._generate_single_with_retry(text)
            if embedding is None:
                self._log_failed_embedding(batch_start_index + i, chunk_metadata)
            results.append(embedding)

        return results

    def _generate_single_with_retry(self, text: str, retry_count: int = 3) -> torch.Tensor | None:
        """Generate single embedding with retry logic."""
        for attempt in range(retry_count):
            try:
                response = self._get_session().post(
                    f"{self.server_url}/embed",
                    json={"text": text, "model": self.model_size},
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    embedding = result.get("embedding")
                    if embedding:
                        return torch.tensor(embedding, dtype=torch.float32)

                if attempt < retry_count - 1:
                    time.sleep((attempt + 1) * 1)

            except requests.exceptions.RequestException as e:
                self.logger.debug(f"Single embedding attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep((attempt + 1) * 1)

        return None

    def _log_failed_embedding(
        self,
        index: int,
        chunk_metadata: list[dict] | None,
    ) -> None:
        """Log information about a failed embedding."""
        if chunk_metadata and index < len(chunk_metadata):
            meta = chunk_metadata[index]
            file_path = meta.get("file_path", "unknown")
            chunk_name = meta.get("name", "unknown")
            chunk_type = meta.get("chunk_type", "unknown")

            self.logger.warning(f"Failed embedding at index {index}: " f"{file_path} - {chunk_type}:{chunk_name}")

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for Qwen3 models."""
        return self.EMBEDDING_DIMENSION

    def get_metrics(self) -> dict[str, Any]:
        """Get cumulative metrics for MLX embedding service."""
        avg_rate = self._total_embeddings / self._total_time if self._total_time > 0 else 0

        return {
            "provider": "mlx_server",
            "total_embeddings": self._total_embeddings,
            "total_time_seconds": self._total_time,
            "total_batches": self._total_batches,
            "average_rate_per_second": avg_rate,
            "server_url": self.server_url,
            "batch_size": self.batch_size,
            "model_size": self.model_size,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
        }

    def reset_metrics(self) -> None:
        """Reset cumulative metrics."""
        self._total_embeddings = 0
        self._total_time = 0.0
        self._total_batches = 0
        self.logger.info("MLX embedding metrics reset")

    def close(self) -> None:
        """Close the session and release resources."""
        if self._session:
            self._session.close()
            self._session = None
