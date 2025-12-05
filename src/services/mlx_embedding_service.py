"""
MLX Server Embedding Service for high-performance Apple Silicon inference.

This service integrates with the qwen3-embeddings-mlx server to provide
significantly faster embedding generation on MacBooks with Apple Silicon.

Performance benchmarks show MLX Server is ~48x faster than Ollama:
- MLX Server (batch=128): ~1700 emb/s
- MLX Server (batch=64): ~1646 emb/s
- Ollama (nomic-embed-text): ~35 emb/s

Server: https://github.com/jakedahn/qwen3-embeddings-mlx
"""

import logging
import os
import time
from typing import Any

import requests
import torch


class MLXServerEmbeddingService:
    """
    MLX Server embedding service for high-performance Apple Silicon inference.

    This service connects to a qwen3-embeddings-mlx server running locally,
    providing batch embedding generation that is ~48x faster than Ollama.
    """

    # Qwen3-Embedding models output 1024 dimensions
    EMBEDDING_DIMENSION = 1024

    def __init__(
        self,
        server_url: str | None = None,
        batch_size: int | None = None,
        timeout: int | None = None,
        model_size: str | None = None,
        max_batch_chars: int | None = None,
    ):
        """
        Initialize the MLX Server embedding service.

        Args:
            server_url: MLX server URL (default: from env or http://localhost:8000)
            batch_size: Max texts per batch (default: from env or 64)
            timeout: Request timeout in seconds (default: from env or 120)
            model_size: Model size to use: 'small', 'medium', 'large' (default: from env or 'small')
            max_batch_chars: Max characters per batch for optimal performance (default: from env or 15000)
                           Lower values = faster per-batch response, higher throughput
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

        # Configuration with environment variable fallbacks
        self.server_url = server_url or os.getenv("MLX_SERVER_URL", "http://localhost:8000")
        self.batch_size = batch_size or int(os.getenv("MLX_BATCH_SIZE", "64"))
        self.timeout = timeout or int(os.getenv("MLX_TIMEOUT", "120"))
        self.model_size = model_size or os.getenv("MLX_MODEL_SIZE", "small")
        # Character-based batching for consistent performance
        # 15000 chars ≈ 30 emb/s, 50000 chars ≈ 10 emb/s
        self.max_batch_chars = max_batch_chars or int(os.getenv("MLX_MAX_BATCH_CHARS", "15000"))

        # Session for connection pooling
        self._session: requests.Session | None = None

        # Metrics tracking
        self._total_embeddings = 0
        self._total_time = 0.0
        self._total_batches = 0

        self.logger.info(
            f"MLX Server embedding service initialized - "
            f"URL: {self.server_url}, batch_size: {self.batch_size}, "
            f"max_batch_chars: {self.max_batch_chars}, model: {self.model_size}"
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
        Texts are automatically split into optimal batch sizes based on
        character count for consistent performance.

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

        # Create character-based batches for optimal performance
        batches = self._create_char_based_batches(texts)
        total_batches = len(batches)

        self.logger.info(
            f"MLX Server: Created {total_batches} batches " f"(max {self.max_batch_chars} chars/batch, max {self.batch_size} texts/batch)"
        )

        # Process each batch
        for batch_num, (batch_texts, batch_start_idx) in enumerate(batches, 1):
            batch_chars = sum(len(t) for t in batch_texts)

            try:
                batch_embeddings = self._process_batch(batch_texts, batch_start_idx, chunk_metadata)

                for i, emb in enumerate(batch_embeddings):
                    if emb is not None:
                        successful += 1
                    else:
                        failed += 1
                        self._log_failed_embedding(batch_start_idx + i, chunk_metadata)

                all_embeddings.extend(batch_embeddings)

                # Progress logging at intervals
                if total_batches > 5 and batch_num % 5 == 0:
                    elapsed = time.time() - start_time
                    rate = successful / elapsed if elapsed > 0 else 0
                    self.logger.info(f"  MLX progress: {batch_num}/{total_batches} batches, " f"{successful} embeddings ({rate:.1f} emb/s)")

            except Exception as e:
                self.logger.error(f"Batch {batch_num} failed ({len(batch_texts)} texts, {batch_chars} chars): {e}")
                # Add None for all items in failed batch
                for i in range(len(batch_texts)):
                    all_embeddings.append(None)
                    failed += 1
                    self._log_failed_embedding(batch_start_idx + i, chunk_metadata)

        # Update metrics
        duration = time.time() - start_time
        self._total_embeddings += successful
        self._total_time += duration
        self._total_batches += total_batches

        # Log summary
        rate = successful / duration if duration > 0 else 0
        self.logger.info(f"MLX Server: Completed {successful}/{len(texts)} embeddings in {duration:.2f}s ({rate:.1f} emb/s)")

        if failed > 0:
            self.logger.warning(f"MLX Server: {failed} embeddings failed")

        return all_embeddings

    def _create_char_based_batches(self, texts: list[str]) -> list[tuple[list[str], int]]:
        """
        Create batches based on character count for consistent performance.

        Each batch will have at most max_batch_chars total characters and
        at most batch_size texts.

        Args:
            texts: List of texts to batch

        Returns:
            List of (batch_texts, start_index) tuples
        """
        batches = []
        current_batch = []
        current_batch_chars = 0
        current_batch_start_idx = 0

        for i, text in enumerate(texts):
            text_len = len(text) if text else 0

            # Check if adding this text would exceed limits
            would_exceed_chars = current_batch_chars + text_len > self.max_batch_chars
            would_exceed_count = len(current_batch) >= self.batch_size

            # If current batch would be exceeded, start a new batch
            if current_batch and (would_exceed_chars or would_exceed_count):
                batches.append((current_batch, current_batch_start_idx))
                current_batch = []
                current_batch_chars = 0
                current_batch_start_idx = i

            # Add text to current batch
            current_batch.append(text)
            current_batch_chars += text_len

            # Handle single texts that exceed max_batch_chars
            if text_len > self.max_batch_chars and len(current_batch) == 1:
                # Process oversized text in its own batch
                batches.append((current_batch, current_batch_start_idx))
                current_batch = []
                current_batch_chars = 0
                current_batch_start_idx = i + 1

        # Add remaining texts
        if current_batch:
            batches.append((current_batch, current_batch_start_idx))

        return batches

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

    def _call_batch_api(self, texts: list[str]) -> list:
        """Call MLX server batch API."""
        response = self._get_session().post(
            f"{self.server_url}/embed_batch",
            json={"texts": texts, "model": self.model_size},
            timeout=self.timeout,
        )

        # The session's retry mechanism handles transient errors.
        # We just need to check the final status and raise if it's not successful.
        response.raise_for_status()

        result = response.json()
        return result.get("embeddings", [])

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

    def _generate_single_with_retry(self, text: str) -> torch.Tensor | None:
        """Generate single embedding."""
        try:
            response = self._get_session().post(
                f"{self.server_url}/embed",
                json={"text": text, "model": self.model_size},
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            embedding = result.get("embedding")
            if embedding:
                return torch.tensor(embedding, dtype=torch.float32)

            return None
        except requests.exceptions.RequestException as e:
            self.logger.debug(f"Single embedding request failed after retries: {e}")
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
            "max_batch_chars": self.max_batch_chars,
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
