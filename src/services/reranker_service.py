"""
Reranker service for two-stage RAG retrieval.

Implements cross-encoder reranking using Qwen3-Reranker for improved
search accuracy. The reranker evaluates query-document pairs together,
providing more nuanced relevance scoring than bi-encoder embeddings alone.

Two-stage retrieval flow:
1. Stage 1: Fast vector search retrieves top-K candidates (e.g., 50-100)
2. Stage 2: Cross-encoder reranks candidates for precise top-N results

Performance characteristics:
- Reranking 50 candidates: ~400ms (CPU) / ~100ms (MPS)
- Improves search accuracy by 22-31% over embedding-only approaches
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RerankedResult:
    """Reranked search result with updated relevance score."""

    content: str
    file_path: str
    original_score: float
    rerank_score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankerMetrics:
    """Metrics for reranking operations."""

    total_reranked: int = 0
    total_duration_seconds: float = 0.0
    average_latency_ms: float = 0.0
    batch_count: int = 0

    def update(self, count: int, duration: float) -> None:
        """Update metrics with new reranking operation."""
        self.total_reranked += count
        self.total_duration_seconds += duration
        self.batch_count += 1
        if self.total_reranked > 0:
            self.average_latency_ms = (self.total_duration_seconds / self.total_reranked) * 1000


class RerankerService:
    """
    Cross-encoder reranking service using Qwen3-Reranker.

    Provides two-stage retrieval capability by reranking initial search
    results using a cross-encoder model that evaluates query-document
    pairs together for more accurate relevance scoring.

    Supports multiple deployment options:
    - transformers: Direct HuggingFace model loading
    - ollama: Ollama-based deployment (future)
    - mlx: MLX-optimized for Apple Silicon (future)
    """

    PROVIDER_TRANSFORMERS = "transformers"
    PROVIDER_OLLAMA = "ollama"
    PROVIDER_MLX = "mlx"

    # Default reranking instruction for code search
    DEFAULT_CODE_INSTRUCTION = (
        "Given a code search query, determine if the code snippet is relevant "
        "for understanding or implementing the requested functionality."
    )

    def __init__(self):
        """Initialize the reranker service."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

        # Configuration from environment
        self.enabled = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
        self.provider = os.getenv("RERANKER_PROVIDER", self.PROVIDER_TRANSFORMERS).lower()
        self.model_name = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
        try:
            self.max_length = int(os.getenv("RERANKER_MAX_LENGTH", "512"))
            self.batch_size = int(os.getenv("RERANKER_BATCH_SIZE", "8"))
            self.default_top_k = int(os.getenv("RERANK_TOP_K", "50"))
        except ValueError as e:
            self.logger.error(f"Invalid reranker configuration. Please check your .env file: {e}")
            raise

        # Model state (lazy loading)
        self._model = None
        self._tokenizer = None
        self._token_true_id = None
        self._token_false_id = None
        self._device = None

        # Metrics tracking
        self.metrics = RerankerMetrics()

        if self.enabled:
            self.logger.info(
                f"RerankerService initialized - provider: {self.provider}, " f"model: {self.model_name}, max_length: {self.max_length}"
            )
        else:
            self.logger.info("RerankerService initialized but disabled")

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

    def _get_device(self) -> torch.device:
        """Determine the best available device for inference."""
        if self._device is not None:
            return self._device

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
            self.logger.info("Using MPS (Apple Silicon) for reranker")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
            self.logger.info("Using CUDA for reranker")
        else:
            self._device = torch.device("cpu")
            self.logger.info("Using CPU for reranker")

        return self._device

    def _load_model(self) -> None:
        """Lazy load the reranker model."""
        if self._model is not None:
            return

        if not self.enabled:
            raise RuntimeError("RerankerService is disabled")

        self.logger.info(f"Loading reranker model: {self.model_name}")
        start_time = time.time()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer with left padding for causal LM
            # Note: trust_remote_code=True is required for Qwen3-Reranker models
            # as they use custom tokenization code. This is a known trusted model
            # from Qwen/Alibaba. For untrusted models, review the code before enabling.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="left",
                trust_remote_code=True,
            )

            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model
            # Note: trust_remote_code=True is required for Qwen3-Reranker models.
            # See tokenizer comment above for security considerations.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use FP16 for efficiency
                trust_remote_code=True,
            ).eval()

            # Get yes/no token IDs for relevance scoring
            self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
            self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")

            # Handle case where tokens might not be in vocabulary
            if self._token_true_id == self._tokenizer.unk_token_id:
                # Fallback to alternative tokens
                self._token_true_id = self._tokenizer.convert_tokens_to_ids("Yes")
            if self._token_false_id == self._tokenizer.unk_token_id:
                self._token_false_id = self._tokenizer.convert_tokens_to_ids("No")

            # Move model to appropriate device
            device = self._get_device()
            self._model = self._model.to(device)

            load_time = time.time() - start_time
            self.logger.info(f"Reranker model loaded successfully in {load_time:.2f}s " f"(device: {device}, dtype: float16)")

        except ImportError as e:
            raise ImportError("transformers library required for reranker. " "Install with: pip install transformers>=4.51.0") from e
        except Exception as e:
            self.logger.error(f"Failed to load reranker model: {e}")
            raise

    def compute_relevance_score(
        self,
        query: str,
        document: str,
        instruction: str | None = None,
    ) -> float:
        """
        Compute relevance score for a query-document pair.

        Uses the cross-encoder to evaluate how relevant the document
        is to the given query, returning a probability score.

        Args:
            query: Search query
            document: Document content to score
            instruction: Optional task instruction for instruction-aware reranking

        Returns:
            Relevance score between 0 and 1
        """
        self._load_model()

        # Build input prompt
        if instruction:
            prompt = f"<instruct>{instruction}</instruct>\n" f"<query>{query}</query>\n" f"<doc>{document}</doc>"
        else:
            prompt = f"<query>{query}</query>\n<doc>{document}</doc>"

        try:
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,
            )

            # Move to device
            device = self._get_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get logits
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[:, -1, :]

            # Calculate relevance score from yes/no logits
            yes_logit = logits[:, self._token_true_id]
            no_logit = logits[:, self._token_false_id]

            # Softmax to get probability
            scores = torch.softmax(torch.stack([no_logit, yes_logit], dim=-1), dim=-1)
            relevance_score = scores[:, 1].item()  # Probability of "yes"

            return relevance_score

        except Exception as e:
            self.logger.error(f"Error computing relevance score: {e}", exc_info=True)
            return 0.0

    def _compute_batch_scores(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[float]:
        """
        Compute relevance scores for multiple documents in a single batch.

        This is significantly faster than computing scores one by one because:
        1. Reduces GPU kernel launch overhead
        2. Better memory access patterns
        3. Enables parallel computation on GPU

        Args:
            query: Search query
            documents: List of document contents to score
            instruction: Optional task instruction

        Returns:
            List of relevance scores (0-1) for each document
        """
        if not documents:
            return []

        self._load_model()
        device = self._get_device()

        # Build prompts for all documents
        prompts = []
        for doc in documents:
            if instruction:
                prompt = f"<instruct>{instruction}</instruct>\n" f"<query>{query}</query>\n" f"<doc>{doc}</doc>"
            else:
                prompt = f"<query>{query}</query>\n<doc>{doc}</doc>"
            prompts.append(prompt)

        try:
            # Batch tokenize all prompts
            inputs = self._tokenizer(
                prompts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True,  # Pad to same length for batching
            )

            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Single forward pass for entire batch
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Get logits for last token of each sequence
                logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Calculate relevance scores from yes/no logits for all items
            yes_logits = logits[:, self._token_true_id]  # [batch_size]
            no_logits = logits[:, self._token_false_id]  # [batch_size]

            # Softmax to get probabilities
            stacked = torch.stack([no_logits, yes_logits], dim=-1)  # [batch_size, 2]
            probs = torch.softmax(stacked, dim=-1)
            scores = probs[:, 1].tolist()  # Probability of "yes" for each item

            # Handle NaN values (can occur with very short or problematic inputs)
            scores = [0.0 if (s != s) else s for s in scores]  # NaN check: x != x is True for NaN

            return scores

        except Exception as e:
            self.logger.warning(f"Batch scoring failed: {e}, falling back to individual scoring")
            # Fallback to individual scoring
            return [self.compute_relevance_score(query, doc, instruction) for doc in documents]

    def rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
        instruction: str | None = None,
        min_score_threshold: float = 0.0,
    ) -> list[RerankedResult]:
        """
        Rerank search results using cross-encoder with batch processing.

        Takes initial search results from Stage 1 (vector search) and
        reranks them using the cross-encoder for more accurate ordering.
        Uses batch processing for significantly improved performance.

        Args:
            query: Original search query
            results: List of search results from Stage 1 vector search
            top_k: Number of top results to return (default: all)
            instruction: Optional task instruction for domain-specific reranking
            min_score_threshold: Minimum rerank score to include in results

        Returns:
            List of reranked results sorted by relevance (descending)
        """
        if not results:
            return []

        if not self.enabled:
            self.logger.debug("Reranking disabled, returning original results")
            return self._convert_to_reranked_results(results)

        self._load_model()

        # Use default instruction for code if none provided
        if instruction is None:
            instruction = self.DEFAULT_CODE_INSTRUCTION

        start_time = time.time()

        self.logger.info(f"Reranking {len(results)} candidates for query: '{query[:50]}...' " f"(batch_size={self.batch_size})")

        # Prepare documents and filter empty content
        valid_results = []
        documents = []
        for result in results:
            content = result.get("content", "")
            if not content or not content.strip():
                continue

            # Truncate very long content to fit model context
            if len(content) > 2000:
                content = content[:2000] + "..."

            valid_results.append(result)
            documents.append(content)

        if not documents:
            return []

        # Sort documents by length for more efficient batching
        # This reduces padding waste when documents have varying lengths
        indexed_docs = [(i, doc, len(doc)) for i, doc in enumerate(documents)]
        indexed_docs.sort(key=lambda x: x[2])  # Sort by length

        # Process in batches for memory efficiency
        all_scores = [0.0] * len(documents)  # Pre-allocate with original indices
        num_batches = (len(indexed_docs) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(indexed_docs))
            batch_items = indexed_docs[batch_start:batch_end]

            batch_docs = [item[1] for item in batch_items]
            batch_indices = [item[0] for item in batch_items]

            batch_scores = self._compute_batch_scores(
                query=query,
                documents=batch_docs,
                instruction=instruction,
            )

            # Place scores back at original indices
            for orig_idx, score in zip(batch_indices, batch_scores, strict=True):
                all_scores[orig_idx] = score

        # Build reranked results
        reranked = []
        for result, score in zip(valid_results, all_scores, strict=True):
            # Skip results below threshold
            if score < min_score_threshold:
                continue

            reranked.append(
                RerankedResult(
                    content=result.get("content", ""),
                    file_path=result.get("file_path", ""),
                    original_score=result.get("score", 0.0),
                    rerank_score=score,
                    metadata={
                        "chunk_type": result.get("chunk_type", ""),
                        "name": result.get("name", ""),
                        "line_start": result.get("line_start", 0),
                        "line_end": result.get("line_end", 0),
                        "language": result.get("language", ""),
                        "breadcrumb": result.get("breadcrumb", ""),
                        "collection": result.get("collection", ""),
                        "parent_name": result.get("parent_name", ""),
                        "signature": result.get("signature", ""),
                        "docstring": result.get("docstring", ""),
                    },
                )
            )

        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        # Limit to top_k if specified
        if top_k and len(reranked) > top_k:
            reranked = reranked[:top_k]

        # Update metrics
        duration = time.time() - start_time
        self.metrics.update(len(results), duration)

        self.logger.info(
            f"Reranking complete: {len(reranked)} results in {duration:.3f}s "
            f"({num_batches} batches, avg: {self.metrics.average_latency_ms:.1f}ms/result)"
        )

        # Log score distribution
        if reranked:
            scores = [r.rerank_score for r in reranked]
            self.logger.debug(
                f"Score distribution - max: {max(scores):.3f}, " f"min: {min(scores):.3f}, avg: {sum(scores)/len(scores):.3f}"
            )

        return reranked

    def _convert_to_reranked_results(self, results: list[dict[str, Any]]) -> list[RerankedResult]:
        """Convert raw results to RerankedResult format without reranking."""
        return [
            RerankedResult(
                content=r.get("content", ""),
                file_path=r.get("file_path", ""),
                original_score=r.get("score", 0.0),
                rerank_score=r.get("score", 0.0),  # Use original score
                metadata={
                    "chunk_type": r.get("chunk_type", ""),
                    "name": r.get("name", ""),
                    "line_start": r.get("line_start", 0),
                    "line_end": r.get("line_end", 0),
                    "language": r.get("language", ""),
                    "breadcrumb": r.get("breadcrumb", ""),
                    "collection": r.get("collection", ""),
                },
            )
            for r in results
        ]

    def get_metrics(self) -> dict[str, Any]:
        """Get reranker performance metrics."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model_name,
            "model_loaded": self._model is not None,
            "device": str(self._device) if self._device else "not_loaded",
            "metrics": {
                "total_reranked": self.metrics.total_reranked,
                "total_duration_seconds": round(self.metrics.total_duration_seconds, 3),
                "average_latency_ms": round(self.metrics.average_latency_ms, 2),
                "batch_count": self.metrics.batch_count,
            },
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = RerankerMetrics()
        self.logger.info("Reranker metrics reset")

    def is_available(self) -> bool:
        """Check if reranker is available and enabled."""
        if not self.enabled:
            return False

        try:
            # Check if transformers is available
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def health_check(self) -> dict[str, Any]:
        """Perform health check on reranker service."""
        result = {
            "healthy": False,
            "enabled": self.enabled,
            "provider": self.provider,
            "model": self.model_name,
        }

        if not self.enabled:
            result["status"] = "disabled"
            result["healthy"] = True  # Disabled is a valid state
            return result

        try:
            # Check if model can be loaded
            self._load_model()
            result["healthy"] = True
            result["status"] = "ready"
            result["device"] = str(self._device)

            # Test with a simple query
            test_score = self.compute_relevance_score(
                query="test query",
                document="test document",
            )
            result["test_score"] = round(test_score, 4)

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result


# Global singleton instance
_reranker_instance: RerankerService | None = None


def get_reranker_instance() -> RerankerService:
    """Get or create the global reranker service instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance


def reset_reranker_instance() -> None:
    """Reset the global reranker instance (for testing)."""
    global _reranker_instance
    _reranker_instance = None
