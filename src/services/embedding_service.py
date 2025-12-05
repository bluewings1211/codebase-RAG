import functools
import logging
import os
import platform
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ollama
import torch

if TYPE_CHECKING:
    from services.mlx_embedding_service import MLXServerEmbeddingService


@dataclass
class FailedEmbeddingInfo:
    """Information about a failed embedding for debugging."""

    index: int
    error_message: str
    file_path: str = ""
    chunk_name: str = ""
    chunk_type: str = ""
    start_line: int = 0
    end_line: int = 0
    text_preview: str = ""


@dataclass
class BatchMetrics:
    """Metrics for tracking batch processing performance."""

    batch_id: str = ""
    batch_size: int = 0
    total_chars: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    api_calls: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    retry_attempts: int = 0
    subdivisions: int = 0
    failed_items: list[FailedEmbeddingInfo] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def embeddings_per_second(self) -> float:
        """Calculate embeddings per second rate."""
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return self.successful_embeddings / duration

    @property
    def chars_per_second(self) -> float:
        """Calculate characters per second rate."""
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return self.total_chars / duration

    @property
    def api_efficiency(self) -> float:
        """Calculate API efficiency (successful embeddings per API call)."""
        if self.api_calls <= 0:
            return 0.0
        return self.successful_embeddings / self.api_calls


@dataclass
class CumulativeMetrics:
    """Cumulative metrics for entire indexing operation."""

    total_batches: int = 0
    total_embeddings: int = 0
    total_successful: int = 0
    total_failed: int = 0
    total_chars: int = 0
    total_api_calls: int = 0
    total_retry_attempts: int = 0
    total_subdivisions: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    all_failed_items: list[FailedEmbeddingInfo] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def overall_success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_embeddings <= 0:
            return 0.0
        return self.total_successful / self.total_embeddings

    @property
    def overall_embeddings_per_second(self) -> float:
        """Calculate overall embeddings per second."""
        duration = self.duration_seconds
        if duration <= 0:
            return 0.0
        return self.total_successful / duration


class EmbeddingService:
    """
    Embedding service with provider selection support.

    Supports two embedding providers:
    - ollama: Default provider using Ollama API (sequential, ~35 emb/s)
    - mlx_server: High-performance MLX server for Apple Silicon (~1700 emb/s)

    Set EMBEDDING_PROVIDER environment variable to select provider.
    MLX server provides 48-161x faster embedding generation on Apple Silicon.
    """

    # Provider constants
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_MLX_SERVER = "mlx_server"

    # Default embedding dimensions by provider
    OLLAMA_DEFAULT_DIMENSION = 768  # nomic-embed-text
    MLX_DEFAULT_DIMENSION = 1024  # Qwen3-Embedding

    def __init__(self):
        # Initialize logger first since other methods depend on it
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

        # Provider selection
        self.provider = os.getenv("EMBEDDING_PROVIDER", self.PROVIDER_OLLAMA).lower()
        self._mlx_service: MLXServerEmbeddingService | None = None
        self._fallback_enabled = os.getenv("MLX_FALLBACK_TO_OLLAMA", "true").lower() == "true"

        # Initialize provider-specific settings
        if self.provider == self.PROVIDER_MLX_SERVER:
            self._init_mlx_provider()
        else:
            self.provider = self.PROVIDER_OLLAMA  # Normalize to ollama

        # Now initialize other components that may use the logger
        self.device = self._get_device()
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Default Ollama host
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        self.max_batch_chars = int(os.getenv("MAX_BATCH_CHARS", "50000"))  # Max chars per batch

        # Retry configuration
        self.max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
        self.base_delay = float(os.getenv("EMBEDDING_RETRY_BASE_DELAY", "1.0"))  # Base delay in seconds
        self.max_delay = float(os.getenv("EMBEDDING_RETRY_MAX_DELAY", "60.0"))  # Max delay in seconds
        self.backoff_multiplier = float(os.getenv("EMBEDDING_BACKOFF_MULTIPLIER", "2.0"))

        # Metrics tracking
        self.cumulative_metrics = CumulativeMetrics()
        self.current_batch_metrics: BatchMetrics | None = None
        self._metrics_enabled = os.getenv("EMBEDDING_METRICS_ENABLED", "true").lower() == "true"

        # Log verbosity control
        self._log_successful_batches = os.getenv("LOG_SUCCESSFUL_BATCHES", "false").lower() == "true"
        self._log_batch_interval = int(os.getenv("LOG_BATCH_INTERVAL", "100"))  # Log progress every N batches

        # Current batch metadata for error tracking
        self._current_batch_metadata: list[dict] | None = None

        self.logger.info(f"Embedding service initialized with provider: {self.provider}")

    def _init_mlx_provider(self) -> None:
        """Initialize MLX server provider."""
        try:
            from services.mlx_embedding_service import MLXServerEmbeddingService

            self._mlx_service = MLXServerEmbeddingService()

            # Verify MLX server is accessible
            health = self._mlx_service.health_check()
            if health.get("healthy"):
                self.logger.info(f"MLX server provider initialized successfully - " f"URL: {self._mlx_service.server_url}")
            else:
                error_msg = health.get("error", "Unknown error")
                self.logger.warning(
                    f"MLX server not available: {error_msg}. " f"Will {'fallback to Ollama' if self._fallback_enabled else 'fail'}."
                )
                if self._fallback_enabled:
                    self.provider = self.PROVIDER_OLLAMA
                    self._mlx_service = None

        except (ImportError, Exception) as e:
            self.logger.warning(f"MLX provider initialization failed: {e}")
            if self._fallback_enabled:
                self.provider = self.PROVIDER_OLLAMA
                self._mlx_service = None
            else:
                raise

    def _get_device(self):
        if platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                self.logger.info("MPS is available. Using Metal for acceleration.")
                return torch.device("mps")
            else:
                self.logger.info("MPS not available, using CPU.")
        return torch.device("cpu")

    def _retry_with_exponential_backoff(self, func: Callable) -> Callable:
        """Decorator for implementing exponential backoff retry logic."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(self.max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Don't retry on final attempt
                    if attempt == self.max_retries:
                        break

                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.base_delay * (self.backoff_multiplier**attempt),
                        self.max_delay,
                    )

                    # Add jitter (random variation) to prevent thundering herd
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter

                    self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. " f"Retrying in {total_delay:.2f}s...")

                    time.sleep(total_delay)

            # If we get here, all retries failed
            self.logger.error(f"All {self.max_retries + 1} attempts failed. Final error: {last_exception}")
            raise last_exception

        return wrapper

    def _should_retry_exception(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Define retryable exceptions (connection errors, timeouts, rate limits)
        retryable_errors = [
            "connection",
            "timeout",
            "rate limit",
            "server error",
            "503",
            "502",
            "504",
            "429",  # Too Many Requests
        ]

        error_str = str(exception).lower()
        return any(error_type in error_str for error_type in retryable_errors)

    def _split_oversized_batch(self, texts: list[str]) -> list[list[str]]:
        """Split a batch that's too large for processing."""
        if len(texts) <= 1:
            return [texts]  # Can't split further

        mid = len(texts) // 2
        left_batch = texts[:mid]
        right_batch = texts[mid:]

        self.logger.info(f"Splitting batch of {len(texts)} into batches of {len(left_batch)} and {len(right_batch)}")

        return [left_batch, right_batch]

    def generate_embeddings(
        self,
        model: str,
        text: str | list[str],
        chunk_metadata: list[dict] | None = None,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """Generate embeddings for single text or batch of texts.

        Automatically routes to the configured provider (Ollama or MLX Server).
        MLX Server provides 48-161x faster embedding generation on Apple Silicon.

        Args:
            model: The embedding model to use (ignored for MLX server, uses configured model)
            text: Single text string or list of text strings
            chunk_metadata: Optional list of metadata dicts for each text (for error tracking).
                           Each dict should contain: file_path, name, chunk_type, start_line, end_line

        Returns:
            Single tensor for single text, list of tensors for batch, or None on error
        """
        # Route to MLX server if configured
        if self.provider == self.PROVIDER_MLX_SERVER and self._mlx_service:
            return self._generate_mlx_embeddings(text, chunk_metadata)

        # Handle both single text and batch processing with Ollama
        if isinstance(text, str):
            return self._generate_single_embedding(model, text)
        elif isinstance(text, list):
            return self._generate_batch_embeddings(model, text, chunk_metadata)
        else:
            self.logger.error(f"Invalid input type for text: {type(text)}")
            return None

    def _generate_mlx_embeddings(
        self,
        text: str | list[str],
        chunk_metadata: list[dict] | None = None,
    ) -> torch.Tensor | list[torch.Tensor] | None:
        """Generate embeddings using MLX server with automatic fallback.

        Args:
            text: Single text string or list of text strings
            chunk_metadata: Optional metadata for error tracking

        Returns:
            Embeddings or None on error
        """
        if not self._mlx_service:
            self.logger.error("MLX service not initialized")
            return None

        try:
            if isinstance(text, str):
                return self._mlx_service.generate_single_embedding(text)
            elif isinstance(text, list):
                result = self._mlx_service.generate_embeddings_batch(text, chunk_metadata)
                # Update our metrics with MLX results
                mlx_metrics = self._mlx_service.get_metrics()
                if self._metrics_enabled:
                    self.cumulative_metrics.total_successful += mlx_metrics.get("total_embeddings", 0)
                return result
            else:
                self.logger.error(f"Invalid input type for text: {type(text)}")
                return None

        except Exception as e:
            self.logger.error(f"MLX embedding generation failed: {e}")

            # Fallback to Ollama if enabled
            if self._fallback_enabled:
                self.logger.warning("Falling back to Ollama provider")
                model = os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

                if isinstance(text, str):
                    return self._generate_single_embedding(model, text)
                elif isinstance(text, list):
                    return self._generate_batch_embeddings(model, text, chunk_metadata)

            return None

    def _generate_single_embedding(self, model: str, text: str) -> torch.Tensor | None:
        """Generate embedding for a single text (backward compatibility)."""
        try:
            # Handle empty or whitespace-only text
            if not text or not text.strip():
                self.logger.warning("Empty or whitespace-only text provided for embedding")
                return None

            # Note: The Ollama library itself doesn't directly expose device selection.
            # The torch device is set for potential future use with local models that might run via PyTorch.
            # For now, this primarily serves the requirement of detecting and acknowledging MPS support.

            # Create Ollama client with host configuration
            client = ollama.Client(host=self.ollama_host)
            response = client.embeddings(model=model, prompt=text)

            # Check if embedding is empty
            if not response.get("embedding") or len(response["embedding"]) == 0:
                self.logger.warning(f"Received empty embedding for text: {text[:50]}...")
                return None

            # Convert to torch tensor for consistency
            import numpy as np

            embedding_array = np.array(response["embedding"])
            return torch.tensor(embedding_array, dtype=torch.float32)

        except Exception as e:
            self.logger.error(f"An error occurred while generating single embedding: {e}")
            return None

    def _generate_batch_embeddings(
        self,
        model: str,
        texts: list[str],
        chunk_metadata: list[dict] | None = None,
    ) -> list[torch.Tensor] | None:
        """Generate embeddings for multiple texts using intelligent batching with metrics tracking."""
        if not texts:
            self.logger.warning("Empty text list provided for batch embedding")
            return []

        # Store metadata for error tracking
        self._current_batch_metadata = chunk_metadata

        # Initialize cumulative metrics if this is the first call
        if self.cumulative_metrics.start_time == 0:
            self.cumulative_metrics.start_time = time.time()

        self.logger.info(f"Generating embeddings for batch of {len(texts)} texts using intelligent batching")

        try:
            # Split texts into optimal batches (each batch includes start_index)
            batches = self._create_intelligent_batches(texts)
            self.logger.info(f"Created {len(batches)} intelligent batches for processing")

            all_embeddings = []

            for batch_idx, (batch_texts, batch_start_index) in enumerate(batches):
                batch_start_time = time.time()

                # Create batch metrics
                batch_id = f"batch_{batch_idx + 1}_{len(batch_texts)}texts"
                batch_chars = sum(len(text) for text in batch_texts if text)

                if self._metrics_enabled:
                    self.current_batch_metrics = BatchMetrics(
                        batch_id=batch_id,
                        batch_size=len(batch_texts),
                        total_chars=batch_chars,
                        start_time=batch_start_time,
                    )

                # Only log batch start info in verbose mode or at intervals
                if self._log_successful_batches or batch_idx % self._log_batch_interval == 0:
                    self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch_texts)} texts ({batch_chars} chars)")

                batch_embeddings = self._process_single_batch(model, batch_texts, batch_start_index)

                # Update metrics
                if self._metrics_enabled and self.current_batch_metrics:
                    self.current_batch_metrics.end_time = time.time()
                    successful_in_batch = sum(1 for emb in batch_embeddings if emb is not None) if batch_embeddings else 0
                    failed_in_batch = len(batch_texts) - successful_in_batch

                    self.current_batch_metrics.successful_embeddings = successful_in_batch
                    self.current_batch_metrics.failed_embeddings = failed_in_batch

                    # Log batch metrics
                    self._log_batch_metrics(self.current_batch_metrics)

                    # Update cumulative metrics
                    self._update_cumulative_metrics(self.current_batch_metrics)

                if batch_embeddings is None:
                    self.logger.error(f"Failed to process batch {batch_idx + 1}")
                    # Add None placeholders for failed batch
                    all_embeddings.extend([None] * len(batch_texts))
                else:
                    all_embeddings.extend(batch_embeddings)

            successful_count = sum(1 for emb in all_embeddings if emb is not None)
            self.logger.info(f"Successfully generated {successful_count}/{len(texts)} embeddings")

            # Log cumulative metrics if enabled
            if self._metrics_enabled:
                self._log_cumulative_metrics()

            return all_embeddings

        except Exception as e:
            self.logger.error(f"An error occurred while generating batch embeddings: {e}")
            return None

    def _create_intelligent_batches(self, texts: list[str]) -> list[tuple[list[str], int]]:
        """Create intelligent batches based on content size and batch limits.

        Returns:
            List of tuples: (batch_texts, start_index) where start_index is the
            index of the first text in the original texts list.
        """
        if not texts:
            return []

        batches = []
        current_batch = []
        current_batch_start_idx = 0
        current_batch_chars = 0

        for idx, text in enumerate(texts):
            text_length = len(text) if text else 0

            # Check if adding this text would exceed limits
            would_exceed_chars = current_batch_chars + text_length > self.max_batch_chars
            would_exceed_count = len(current_batch) >= self.embedding_batch_size

            # If current batch would be exceeded, start a new batch
            if current_batch and (would_exceed_chars or would_exceed_count):
                batches.append((current_batch, current_batch_start_idx))
                current_batch = []
                current_batch_start_idx = idx
                current_batch_chars = 0

            # Add text to current batch
            current_batch.append(text)
            current_batch_chars += text_length

            # Handle oversized single texts by putting them in their own batch
            if text_length > self.max_batch_chars:
                self.logger.warning(f"Text exceeds max batch size ({text_length} > {self.max_batch_chars} chars), processing individually")
                if len(current_batch) > 1:
                    # Move the oversized text to its own batch
                    oversized_text = current_batch.pop()
                    current_batch_chars -= text_length

                    # Save current batch and create new batch for oversized text
                    batches.append((current_batch, current_batch_start_idx))
                    batches.append(([oversized_text], idx))
                    current_batch = []
                    current_batch_start_idx = idx + 1
                    current_batch_chars = 0

        # Add remaining texts
        if current_batch:
            batches.append((current_batch, current_batch_start_idx))

        return batches

    def _process_single_batch(self, model: str, texts: list[str], batch_start_index: int = 0) -> list[torch.Tensor] | None:
        """Process a single batch of texts with retry logic and subdivision on failure."""
        return self._process_batch_with_retry(model, texts, batch_start_index=batch_start_index)

    def _process_batch_with_retry(
        self,
        model: str,
        texts: list[str],
        attempt_subdivision: bool = True,
        batch_start_index: int = 0,
    ) -> list[torch.Tensor] | None:
        """Process batch with retry logic and optional subdivision on failure."""
        try:
            # Try processing the batch normally first
            return self._process_batch_core(model, texts, batch_start_index)

        except Exception as e:
            # Check if we should retry this exception
            if not self._should_retry_exception(e):
                self.logger.error(f"Non-retryable error processing batch: {e}")
                return None

            # If batch subdivision is enabled and batch has more than 1 item, try splitting
            if attempt_subdivision and len(texts) > 1:
                self.logger.warning(f"Batch processing failed: {e}. Attempting batch subdivision...")

                # Track subdivision attempt
                if self._metrics_enabled and self.current_batch_metrics:
                    self.current_batch_metrics.subdivisions += 1

                try:
                    # Split the batch into smaller batches
                    sub_batches = self._split_oversized_batch(texts)
                    all_embeddings = []

                    current_offset = 0
                    for sub_batch in sub_batches:
                        # Process each sub-batch without further subdivision to avoid infinite recursion
                        sub_start_index = batch_start_index + current_offset
                        sub_embeddings = self._process_batch_with_retry(
                            model, sub_batch, attempt_subdivision=False, batch_start_index=sub_start_index
                        )

                        if sub_embeddings is None:
                            # If sub-batch fails, add None placeholders
                            all_embeddings.extend([None] * len(sub_batch))
                        else:
                            all_embeddings.extend(sub_embeddings)

                        current_offset += len(sub_batch)

                    return all_embeddings

                except Exception as subdivision_error:
                    self.logger.error(f"Batch subdivision also failed: {subdivision_error}")
                    return None
            else:
                # Cannot subdivide further or subdivision disabled
                self.logger.error(f"Batch processing failed and cannot subdivide further: {e}")
                return None

    def _process_batch_core(self, model: str, texts: list[str], batch_start_index: int = 0) -> list[torch.Tensor] | None:
        """Core batch processing logic with retry decorator applied."""

        # Apply retry decorator to the core processing logic
        @self._retry_with_exponential_backoff
        def _core_processing():
            client = ollama.Client(host=self.ollama_host)
            embeddings = []
            api_call_start = time.time()

            for i, text in enumerate(texts):
                # Calculate the original index in the full texts list
                original_index = batch_start_index + i

                if not text or not text.strip():
                    self.logger.warning(f"Skipping empty text at index {original_index}")
                    self._record_failed_embedding(original_index, "Empty or whitespace-only text", text)
                    embeddings.append(None)
                    continue

                # Individual text processing with its own error handling
                try:
                    individual_start = time.time()
                    response = client.embeddings(model=model, prompt=text)
                    individual_duration = time.time() - individual_start

                    # Track API call metrics
                    if self._metrics_enabled and self.current_batch_metrics:
                        self.current_batch_metrics.api_calls += 1

                    # Log individual API response time for debugging
                    if individual_duration > 5.0:  # Log slow API calls
                        self.logger.warning(f"Slow API response for index {original_index}: {individual_duration:.2f}s")

                    if not response.get("embedding") or len(response["embedding"]) == 0:
                        self._log_embedding_error(original_index, "Received empty embedding", text)
                        self._record_failed_embedding(original_index, "Empty embedding returned", text)
                        embeddings.append(None)
                        continue

                    # Convert to torch tensor
                    import numpy as np

                    embedding_array = np.array(response["embedding"])
                    tensor = torch.tensor(embedding_array, dtype=torch.float32)
                    embeddings.append(tensor)

                except Exception as text_error:
                    # Track failed API call
                    if self._metrics_enabled and self.current_batch_metrics:
                        self.current_batch_metrics.api_calls += 1

                    # Log detailed error with chunk info
                    self._log_embedding_error(original_index, str(text_error), text)
                    self._record_failed_embedding(original_index, str(text_error), text)
                    embeddings.append(None)

            total_api_duration = time.time() - api_call_start
            if self._metrics_enabled and len(texts) > 1:
                avg_api_time = total_api_duration / len(texts)
                self.logger.debug(f"Average API response time for batch: {avg_api_time:.3f}s per text")

            return embeddings

        # Track retry attempts

        def retry_with_metrics_tracking(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                attempt = 0
                last_exception = None

                for attempt in range(self.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        # Track retry attempt
                        if self._metrics_enabled and self.current_batch_metrics and attempt > 0:
                            self.current_batch_metrics.retry_attempts += 1

                        if attempt == self.max_retries:
                            break

                        delay = min(
                            self.base_delay * (self.backoff_multiplier**attempt),
                            self.max_delay,
                        )
                        jitter = random.uniform(0.1, 0.3) * delay
                        total_delay = delay + jitter

                        self.logger.warning(
                            f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. " f"Retrying in {total_delay:.2f}s..."
                        )

                        time.sleep(total_delay)

                self.logger.error(f"All {self.max_retries + 1} attempts failed. Final error: {last_exception}")
                raise last_exception

            return wrapper

        # Apply metrics-aware retry decorator
        decorated_func = retry_with_metrics_tracking(_core_processing)

        try:
            return decorated_func()
        except Exception as e:
            self.logger.error(f"Core batch processing failed after all retries: {e}")
            return None

    def _log_embedding_error(self, original_index: int, error_message: str, text: str) -> None:
        """Log detailed embedding error with chunk metadata if available."""
        # Try to get metadata for this index
        chunk_info = ""
        if self._current_batch_metadata and original_index < len(self._current_batch_metadata):
            meta = self._current_batch_metadata[original_index]
            file_path = meta.get("file_path", "unknown")
            chunk_name = meta.get("name", "unknown")
            chunk_type = meta.get("chunk_type", "unknown")
            # Support both naming conventions: line_start/line_end and start_line/end_line
            start_line = meta.get("line_start") or meta.get("start_line", "?")
            end_line = meta.get("line_end") or meta.get("end_line", "?")

            chunk_info = (
                f"\n    📁 File: {file_path}" f"\n    🔖 Chunk: {chunk_type}:{chunk_name}" f"\n    📍 Lines: {start_line}-{end_line}"
            )

        text_preview = text[:150].replace("\n", " ") if text else "(empty)"
        if len(text) > 150:
            text_preview += "..."

        self.logger.error(
            f"Embedding failed at index {original_index}: {error_message}" f"{chunk_info}" f"\n    📝 Preview: {text_preview}"
        )

    def _record_failed_embedding(self, original_index: int, error_message: str, text: str) -> None:
        """Record a failed embedding for later reporting."""
        if not self._metrics_enabled:
            return

        # Get metadata if available
        file_path = ""
        chunk_name = ""
        chunk_type = ""
        start_line = 0
        end_line = 0

        if self._current_batch_metadata and original_index < len(self._current_batch_metadata):
            meta = self._current_batch_metadata[original_index]
            file_path = meta.get("file_path", "")
            chunk_name = meta.get("name", "")
            chunk_type = meta.get("chunk_type", "")
            # Support both naming conventions: line_start/line_end and start_line/end_line
            start_line = meta.get("line_start") or meta.get("start_line", 0)
            end_line = meta.get("line_end") or meta.get("end_line", 0)

        failed_info = FailedEmbeddingInfo(
            index=original_index,
            error_message=error_message,
            file_path=file_path,
            chunk_name=chunk_name,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
            text_preview=text[:100] if text else "",
        )

        # Add to current batch metrics
        if self.current_batch_metrics:
            self.current_batch_metrics.failed_items.append(failed_info)

        # Also add to cumulative metrics
        self.cumulative_metrics.all_failed_items.append(failed_info)

    def _log_batch_metrics(self, metrics: BatchMetrics) -> None:
        """Log metrics for a single batch - only log if there are issues or at intervals."""
        if not self._metrics_enabled:
            return

        has_failures = metrics.failed_embeddings > 0
        has_retries = metrics.retry_attempts > 0
        has_subdivisions = metrics.subdivisions > 0
        is_interval = self.cumulative_metrics.total_batches % self._log_batch_interval == 0

        # Always log batches with issues
        if has_failures or has_retries or has_subdivisions:
            self.logger.warning(
                f"⚠️  Batch {metrics.batch_id} - "
                f"Success: {metrics.successful_embeddings}/{metrics.batch_size}, "
                f"Failed: {metrics.failed_embeddings}, "
                f"Retries: {metrics.retry_attempts}, "
                f"Subdivisions: {metrics.subdivisions}"
            )
        # Log progress at intervals or if verbose mode enabled
        elif self._log_successful_batches:
            self.logger.info(
                f"Batch {metrics.batch_id} completed - "
                f"Duration: {metrics.duration_seconds:.2f}s, "
                f"Success: {metrics.successful_embeddings}/{metrics.batch_size}, "
                f"Rate: {metrics.embeddings_per_second:.2f} emb/s"
            )
        elif is_interval:
            # Compact progress log at intervals
            self.logger.info(
                f"📊 Progress: {self.cumulative_metrics.total_batches} batches, "
                f"{self.cumulative_metrics.total_successful}/{self.cumulative_metrics.total_embeddings} embeddings "
                f"({self.cumulative_metrics.overall_success_rate:.1%} success)"
            )

    def _update_cumulative_metrics(self, batch_metrics: BatchMetrics) -> None:
        """Update cumulative metrics with batch results."""
        if not self._metrics_enabled:
            return

        self.cumulative_metrics.total_batches += 1
        self.cumulative_metrics.total_embeddings += batch_metrics.batch_size
        self.cumulative_metrics.total_successful += batch_metrics.successful_embeddings
        self.cumulative_metrics.total_failed += batch_metrics.failed_embeddings
        self.cumulative_metrics.total_chars += batch_metrics.total_chars
        self.cumulative_metrics.total_api_calls += batch_metrics.api_calls
        self.cumulative_metrics.total_retry_attempts += batch_metrics.retry_attempts
        self.cumulative_metrics.total_subdivisions += batch_metrics.subdivisions

    def _log_cumulative_metrics(self) -> None:
        """Log cumulative metrics for the entire operation."""
        if not self._metrics_enabled:
            return

        self.cumulative_metrics.end_time = time.time()

        self.logger.info("=== Cumulative Embedding Metrics ===")
        self.logger.info(f"Total duration: {self.cumulative_metrics.duration_seconds:.2f}s")
        self.logger.info(f"Total batches processed: {self.cumulative_metrics.total_batches}")
        self.logger.info(f"Total embeddings: {self.cumulative_metrics.total_embeddings}")
        self.logger.info(f"Successful: {self.cumulative_metrics.total_successful}")
        self.logger.info(f"Failed: {self.cumulative_metrics.total_failed}")
        self.logger.info(f"Success rate: {self.cumulative_metrics.overall_success_rate:.1%}")
        self.logger.info(f"Overall rate: {self.cumulative_metrics.overall_embeddings_per_second:.2f} emb/s")
        self.logger.info(f"Total characters processed: {self.cumulative_metrics.total_chars:,}")
        self.logger.info(f"Total API calls: {self.cumulative_metrics.total_api_calls}")

        if self.cumulative_metrics.total_retry_attempts > 0:
            self.logger.info(f"Total retry attempts: {self.cumulative_metrics.total_retry_attempts}")

        if self.cumulative_metrics.total_subdivisions > 0:
            self.logger.info(f"Total batch subdivisions: {self.cumulative_metrics.total_subdivisions}")

        overall_efficiency = (
            self.cumulative_metrics.total_successful / self.cumulative_metrics.total_api_calls
            if self.cumulative_metrics.total_api_calls > 0
            else 0
        )
        self.logger.info(f"Overall API efficiency: {overall_efficiency:.2f} emb/call")

        # Log failed items summary if any
        if self.cumulative_metrics.all_failed_items:
            self.logger.warning(f"=== Failed Embeddings Summary ({len(self.cumulative_metrics.all_failed_items)} items) ===")

            # Group failures by file
            failures_by_file: dict[str, list[FailedEmbeddingInfo]] = {}
            for item in self.cumulative_metrics.all_failed_items:
                file_key = item.file_path or "unknown"
                if file_key not in failures_by_file:
                    failures_by_file[file_key] = []
                failures_by_file[file_key].append(item)

            for file_path, items in failures_by_file.items():
                self.logger.warning(f"  📁 {file_path}: {len(items)} failures")
                for item in items[:3]:  # Show first 3 failures per file
                    self.logger.warning(
                        f"      - {item.chunk_type}:{item.chunk_name} (lines {item.start_line}-{item.end_line}): {item.error_message[:50]}"
                    )
                if len(items) > 3:
                    self.logger.warning(f"      ... and {len(items) - 3} more")

        self.logger.info("=====================================")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of current metrics for external reporting."""
        if not self._metrics_enabled:
            return {"metrics_enabled": False}

        # Convert failed items to dicts
        failed_items_list = [
            {
                "index": item.index,
                "error_message": item.error_message,
                "file_path": item.file_path,
                "chunk_name": item.chunk_name,
                "chunk_type": item.chunk_type,
                "start_line": item.start_line,
                "end_line": item.end_line,
            }
            for item in self.cumulative_metrics.all_failed_items
        ]

        return {
            "metrics_enabled": True,
            "cumulative": {
                "total_batches": self.cumulative_metrics.total_batches,
                "total_embeddings": self.cumulative_metrics.total_embeddings,
                "total_successful": self.cumulative_metrics.total_successful,
                "total_failed": self.cumulative_metrics.total_failed,
                "success_rate": self.cumulative_metrics.overall_success_rate,
                "duration_seconds": self.cumulative_metrics.duration_seconds,
                "embeddings_per_second": self.cumulative_metrics.overall_embeddings_per_second,
                "total_chars": self.cumulative_metrics.total_chars,
                "total_api_calls": self.cumulative_metrics.total_api_calls,
                "total_retry_attempts": self.cumulative_metrics.total_retry_attempts,
                "total_subdivisions": self.cumulative_metrics.total_subdivisions,
            },
            "failed_items": failed_items_list,
        }

    def get_failed_items(self) -> list[FailedEmbeddingInfo]:
        """Get the list of all failed embedding items."""
        return self.cumulative_metrics.all_failed_items

    def reset_metrics(self) -> None:
        """Reset all metrics for a new operation."""
        if self._metrics_enabled:
            self.cumulative_metrics = CumulativeMetrics()
            self.current_batch_metrics = None
            self.logger.info("Embedding metrics reset for new operation")

    def _setup_logging(self) -> None:
        """Setup logging configuration for embedding service."""
        if not self.logger.handlers:
            # Set level from environment or default to INFO
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))

            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # Prevent duplicate logging
            self.logger.propagate = False

    # Backward compatibility method (deprecated)
    def generate_embedding(self, model: str, text: str) -> torch.Tensor | None:
        """Deprecated: Use generate_embeddings instead."""
        self.logger.warning("generate_embedding is deprecated, use generate_embeddings instead")
        return self._generate_single_embedding(model, text)

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current provider.

        Returns:
            int: Embedding dimension (1024 for MLX/Qwen3, 768 for Ollama/nomic-embed-text)
        """
        if self.provider == self.PROVIDER_MLX_SERVER and self._mlx_service:
            return self._mlx_service.get_embedding_dimension()
        return self.OLLAMA_DEFAULT_DIMENSION

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about the current embedding provider.

        Returns:
            Dictionary with provider details
        """
        info = {
            "provider": self.provider,
            "fallback_enabled": self._fallback_enabled,
        }

        if self.provider == self.PROVIDER_MLX_SERVER and self._mlx_service:
            info.update(
                {
                    "server_url": self._mlx_service.server_url,
                    "batch_size": self._mlx_service.batch_size,
                    "model_size": self._mlx_service.model_size,
                    "embedding_dimension": self._mlx_service.get_embedding_dimension(),
                }
            )
        else:
            info.update(
                {
                    "host": self.ollama_host,
                    "batch_size": self.embedding_batch_size,
                    "embedding_dimension": self.OLLAMA_DEFAULT_DIMENSION,
                }
            )

        return info

    def check_provider_health(self) -> dict[str, Any]:
        """Check the health of the current embedding provider.

        Returns:
            Dictionary with health status and details
        """
        if self.provider == self.PROVIDER_MLX_SERVER and self._mlx_service:
            return self._mlx_service.health_check()

        # Check Ollama health
        try:
            client = ollama.Client(host=self.ollama_host)
            # Try listing models to verify connection
            client.list()
            return {
                "healthy": True,
                "provider": "ollama",
                "host": self.ollama_host,
            }
        except Exception as e:
            return {
                "healthy": False,
                "provider": "ollama",
                "host": self.ollama_host,
                "error": str(e),
            }
