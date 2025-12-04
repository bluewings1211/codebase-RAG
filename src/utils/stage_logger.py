"""
Stage-specific logging utilities for detailed performance tracking.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageMetrics:
    """Metrics for a processing stage."""

    stage_name: str
    start_time: float
    end_time: float | None = None
    item_count: int = 0
    processed_count: int = 0
    failed_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def processing_rate(self) -> float:
        """Get processing rate (items per second)."""
        duration = self.duration
        if duration == 0:
            return 0.0
        return self.processed_count / duration

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total_attempted = self.processed_count + self.failed_count
        if total_attempted == 0:
            return 100.0
        return (self.processed_count / total_attempted) * 100.0


class StageLogger:
    """Enhanced logger with stage-specific tracking and structured logging."""

    def __init__(self, name: str, parent_logger: logging.Logger | None = None):
        self.name = name
        self.logger = parent_logger or logging.getLogger(name)
        self._lock = threading.Lock()
        self._stages: dict[str, StageMetrics] = {}
        self._current_stage: str | None = None

        # Set up structured formatter if not already configured
        if not self.logger.handlers:
            self._setup_structured_logging()

    def _setup_structured_logging(self) -> None:
        """Setup structured logging with consistent format."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    @contextmanager
    def stage(self, stage_name: str, item_count: int = 0, **details):
        """Context manager for tracking a processing stage."""
        self.start_stage(stage_name, item_count, **details)
        try:
            yield self._stages[stage_name]
        finally:
            self.end_stage(stage_name)

    def start_stage(self, stage_name: str, item_count: int = 0, **details) -> None:
        """Start tracking a processing stage."""
        with self._lock:
            self._current_stage = stage_name
            self._stages[stage_name] = StageMetrics(
                stage_name=stage_name,
                start_time=time.time(),
                item_count=item_count,
                details=details,
            )

        detail_str = f" ({', '.join(f'{k}={v}' for k, v in details.items())})" if details else ""
        self.logger.info(f"STAGE_START: {stage_name} - {item_count} items{detail_str}")

    def end_stage(self, stage_name: str) -> StageMetrics:
        """End tracking a processing stage and log summary."""
        with self._lock:
            if stage_name not in self._stages:
                self.logger.warning(f"Attempted to end unknown stage: {stage_name}")
                return None

            stage = self._stages[stage_name]
            stage.end_time = time.time()

            if self._current_stage == stage_name:
                self._current_stage = None

        # Log comprehensive stage summary
        self.logger.info(
            f"STAGE_END: {stage_name} - "
            f"Duration: {stage.duration:.2f}s - "
            f"Items: {stage.processed_count}/{stage.item_count} - "
            f"Rate: {stage.processing_rate:.2f}/s - "
            f"Success: {stage.success_rate:.1f}% - "
            f"Failures: {stage.failed_count}"
        )

        return stage

    def log_item_processed(self, stage_name: str = None, **details) -> None:
        """Log a successfully processed item."""
        stage_name = stage_name or self._current_stage
        if stage_name and stage_name in self._stages:
            with self._lock:
                self._stages[stage_name].processed_count += 1

            if details:
                detail_str = f" ({', '.join(f'{k}={v}' for k, v in details.items())})"
                self.logger.debug(f"ITEM_PROCESSED: {stage_name}{detail_str}")

    def log_item_failed(self, stage_name: str = None, error: str = None, **details) -> None:
        """Log a failed item."""
        stage_name = stage_name or self._current_stage
        if stage_name and stage_name in self._stages:
            with self._lock:
                self._stages[stage_name].failed_count += 1

            detail_str = f" ({', '.join(f'{k}={v}' for k, v in details.items())})" if details else ""
            error_str = f" - Error: {error}" if error else ""
            self.logger.warning(f"ITEM_FAILED: {stage_name}{detail_str}{error_str}")

    def log_progress(self, stage_name: str = None, force: bool = False) -> None:
        """Log current progress for a stage."""
        stage_name = stage_name or self._current_stage
        if stage_name and stage_name in self._stages:
            stage = self._stages[stage_name]

            # Only log progress if we have significant progress or forced
            if force or stage.processed_count % 10 == 0:  # Log every 10 items
                completion = 0.0
                if stage.item_count > 0:
                    completion = (stage.processed_count / stage.item_count) * 100

                self.logger.info(
                    f"PROGRESS: {stage_name} - "
                    f"{stage.processed_count}/{stage.item_count} ({completion:.1f}%) - "
                    f"Rate: {stage.processing_rate:.2f}/s - "
                    f"Elapsed: {stage.duration:.1f}s"
                )

    def get_stage_summary(self, stage_name: str) -> dict[str, Any] | None:
        """Get comprehensive summary for a stage."""
        if stage_name not in self._stages:
            return None

        stage = self._stages[stage_name]
        return {
            "stage_name": stage.stage_name,
            "duration": round(stage.duration, 2),
            "item_count": stage.item_count,
            "processed_count": stage.processed_count,
            "failed_count": stage.failed_count,
            "processing_rate": round(stage.processing_rate, 2),
            "success_rate": round(stage.success_rate, 1),
            "details": stage.details,
            "is_complete": stage.end_time is not None,
        }

    def get_all_stage_summaries(self) -> dict[str, Any]:
        """Get summaries for all stages."""
        return {name: self.get_stage_summary(name) for name in self._stages.keys()}

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional structured data."""
        self._log_with_context("info", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional structured data."""
        self._log_with_context("warning", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional structured data."""
        self._log_with_context("error", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional structured data."""
        self._log_with_context("debug", message, **kwargs)

    def _log_with_context(self, level: str, message: str, **kwargs) -> None:
        """Log message with current stage context."""
        if kwargs:
            context_str = f" ({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
            message = f"{message}{context_str}"

        if self._current_stage:
            message = f"[{self._current_stage}] {message}"

        getattr(self.logger, level)(message)


def create_stage_logger(name: str, parent_logger: logging.Logger | None = None) -> StageLogger:
    """Factory function to create a stage logger."""
    return StageLogger(name, parent_logger)


# Pre-configured stage loggers for common operations
def get_file_discovery_logger() -> StageLogger:
    """Get logger for file discovery stage."""
    return create_stage_logger("indexing.file_discovery")


def get_file_reading_logger() -> StageLogger:
    """Get logger for file reading stage."""
    return create_stage_logger("indexing.file_reading")


def get_embedding_logger() -> StageLogger:
    """Get logger for embedding generation stage."""
    return create_stage_logger("indexing.embedding")


def get_database_logger() -> StageLogger:
    """Get logger for database operations stage."""
    return create_stage_logger("indexing.database")


# Utility functions for common logging patterns
def log_timing(logger: StageLogger, operation: str, duration: float, **details) -> None:
    """Log timing information for an operation."""
    detail_str = f" ({', '.join(f'{k}={v}' for k, v in details.items())})" if details else ""
    logger.info(f"TIMING: {operation} took {duration:.3f}s{detail_str}")


def log_batch_summary(
    logger: StageLogger,
    batch_num: int,
    batch_size: int,
    processed: int,
    failed: int,
    duration: float,
    log_interval: int = 50,
    force: bool = False,
) -> None:
    """Log summary for a batch operation.

    Only logs in the following cases to reduce noise:
    - There are failures (always log)
    - It's a log interval batch (every N batches)
    - force=True is passed

    Args:
        logger: The stage logger to use
        batch_num: Current batch number
        batch_size: Size of the batch
        processed: Number of items processed
        failed: Number of items that failed
        duration: Duration of the batch in seconds
        log_interval: Log every N batches (default: 50)
        force: Force logging regardless of interval
    """
    rate = processed / duration if duration > 0 else 0

    # Always log if there are failures
    if failed > 0:
        logger.warning(
            f"⚠️  BATCH: {batch_num} - Size: {batch_size} - "
            f"Processed: {processed} - Failed: {failed} - "
            f"Duration: {duration:.2f}s - Rate: {rate:.2f}/s"
        )
    # Log at intervals or when forced
    elif force or batch_num % log_interval == 0:
        logger.info(f"📊 Progress: batch {batch_num} - " f"Processed: {processed}/{batch_size} - " f"Rate: {rate:.2f}/s")
