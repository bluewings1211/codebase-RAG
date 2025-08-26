"""
Query History Analyzer for Wave 4.0 Query Analysis and Routing System

This service analyzes historical query patterns to optimize routing decisions,
learn from past performance, and provide intelligent recommendations.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from ..models.query_features import QueryComplexity, QueryFeatures, QueryType
from ..models.routing_decision import RoutingDecision, RoutingHistory

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """Represents a pattern discovered in query history."""

    pattern_id: str
    pattern_type: str  # 'keyword', 'intent', 'complexity', 'temporal'
    pattern_description: str

    # Pattern characteristics
    frequency: int = 0
    success_rate: float = 0.0
    average_performance: dict[str, float] = field(default_factory=dict)

    # Associated queries
    example_queries: list[str] = field(default_factory=list)
    query_hashes: set[str] = field(default_factory=set)

    # Routing preferences
    preferred_modes: dict[str, float] = field(default_factory=dict)
    mode_success_rates: dict[str, float] = field(default_factory=dict)

    # Temporal information
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    peak_usage_hours: list[int] = field(default_factory=list)

    # Quality metrics
    pattern_strength: float = 0.0  # How well-defined this pattern is
    prediction_accuracy: float = 0.0  # How well it predicts routing success


@dataclass
class HistoryInsights:
    """Insights derived from query history analysis."""

    # Overall statistics
    total_queries_analyzed: int = 0
    analysis_time_range: tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))

    # Performance insights
    overall_success_rate: float = 0.0
    mode_performance: dict[str, dict[str, float]] = field(default_factory=dict)
    trend_direction: str = "stable"  # improving, degrading, stable

    # Pattern insights
    discovered_patterns: list[QueryPattern] = field(default_factory=list)
    pattern_coverage: float = 0.0  # Percentage of queries matching patterns

    # Routing insights
    routing_effectiveness: dict[str, float] = field(default_factory=dict)
    misrouting_rate: float = 0.0
    routing_recommendations: list[str] = field(default_factory=list)

    # Temporal insights
    usage_patterns: dict[str, Any] = field(default_factory=dict)
    seasonal_trends: dict[str, float] = field(default_factory=dict)

    # Quality and confidence
    analysis_confidence: float = 0.0
    data_completeness: float = 0.0


class QueryHistoryAnalyzer:
    """
    Analyzes query history to identify patterns, optimize routing decisions,
    and provide intelligent recommendations for future queries.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # History storage (in production, this would use persistent storage)
        self.query_histories: dict[str, RoutingHistory] = {}
        self.analysis_cache: dict[str, tuple[HistoryInsights, datetime]] = {}

        # Pattern discovery configuration
        self.pattern_config = {
            "min_pattern_frequency": 3,
            "min_success_rate_for_recommendation": 0.7,
            "pattern_cache_ttl_hours": 24,
            "analysis_window_days": 30,
            "similarity_threshold": 0.8,
        }

        # Learning parameters
        self.learning_rates = {"routing_weight_adjustment": 0.1, "confidence_boost_factor": 0.05, "pattern_strength_decay": 0.95}

        # Analysis statistics
        self.analysis_stats = {
            "total_analyses": 0,
            "patterns_discovered": 0,
            "recommendations_generated": 0,
            "average_analysis_time_ms": 0.0,
        }

    async def analyze_query_history(self, time_window_days: int = 30, force_refresh: bool = False) -> HistoryInsights:
        """Perform comprehensive analysis of query history."""
        start_time = time.time()

        try:
            self.logger.debug(f"Analyzing query history for {time_window_days} days")

            # Check cache first
            cache_key = f"analysis_{time_window_days}d"
            if not force_refresh and cache_key in self.analysis_cache:
                insights, timestamp = self.analysis_cache[cache_key]
                if datetime.now() - timestamp < timedelta(hours=self.pattern_config["pattern_cache_ttl_hours"]):
                    self.logger.debug("Using cached history analysis")
                    return insights

            # Gather historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_window_days)

            relevant_histories = self._filter_histories_by_date(start_date, end_date)

            if not relevant_histories:
                self.logger.warning("No historical data available for analysis")
                return HistoryInsights(analysis_time_range=(start_date, end_date), analysis_confidence=0.0, data_completeness=0.0)

            # Perform comprehensive analysis
            insights = await self._perform_comprehensive_analysis(relevant_histories, start_date, end_date)

            # Cache the results
            self.analysis_cache[cache_key] = (insights, datetime.now())

            # Update statistics
            analysis_time_ms = (time.time() - start_time) * 1000
            self._update_analysis_stats(insights, analysis_time_ms)

            self.logger.debug(
                f"History analysis complete: {insights.total_queries_analyzed} queries, "
                f"{len(insights.discovered_patterns)} patterns, "
                f"confidence={insights.analysis_confidence:.2f}, "
                f"time={analysis_time_ms:.2f}ms"
            )

            return insights

        except Exception as e:
            self.logger.error(f"Error in query history analysis: {e}")
            return HistoryInsights(analysis_confidence=0.0, data_completeness=0.0)

    def _filter_histories_by_date(self, start_date: datetime, end_date: datetime) -> list[RoutingHistory]:
        """Filter query histories by date range."""
        relevant_histories = []

        for history in self.query_histories.values():
            # Check if history has data in the time range
            last_updated = datetime.fromisoformat(history.last_updated)
            if start_date <= last_updated <= end_date:
                relevant_histories.append(history)

        return relevant_histories

    async def _perform_comprehensive_analysis(
        self, histories: list[RoutingHistory], start_date: datetime, end_date: datetime
    ) -> HistoryInsights:
        """Perform comprehensive analysis of filtered histories."""
        insights = HistoryInsights(
            analysis_time_range=(start_date, end_date), total_queries_analyzed=sum(len(h.routing_decisions) for h in histories)
        )

        # 1. Overall performance analysis
        await self._analyze_overall_performance(histories, insights)

        # 2. Pattern discovery
        await self._discover_query_patterns(histories, insights)

        # 3. Routing effectiveness analysis
        await self._analyze_routing_effectiveness(histories, insights)

        # 4. Temporal pattern analysis
        await self._analyze_temporal_patterns(histories, insights)

        # 5. Generate recommendations
        await self._generate_routing_recommendations(insights)

        # 6. Calculate analysis quality metrics
        self._calculate_analysis_quality(insights)

        return insights

    async def _analyze_overall_performance(self, histories: list[RoutingHistory], insights: HistoryInsights) -> None:
        """Analyze overall system performance from history."""
        all_success_rates = []
        mode_performance = defaultdict(lambda: {"successes": 0, "total": 0, "latencies": []})

        for history in histories:
            all_success_rates.append(history.success_rate)

            # Analyze per-mode performance
            for perf in history.performance_history:
                mode = perf.get("decision_mode")
                actual_perf = perf.get("actual_performance", {})

                if mode:
                    mode_performance[mode]["total"] += 1
                    if actual_perf.get("success", False):
                        mode_performance[mode]["successes"] += 1

                    if "latency_ms" in actual_perf:
                        mode_performance[mode]["latencies"].append(actual_perf["latency_ms"])

        # Calculate overall success rate
        if all_success_rates:
            insights.overall_success_rate = statistics.mean(all_success_rates)

        # Calculate mode-specific performance
        for mode, stats in mode_performance.items():
            if stats["total"] > 0:
                success_rate = stats["successes"] / stats["total"]
                avg_latency = statistics.mean(stats["latencies"]) if stats["latencies"] else 0

                insights.mode_performance[mode] = {
                    "success_rate": success_rate,
                    "average_latency_ms": avg_latency,
                    "total_queries": stats["total"],
                }

        # Determine trend direction
        insights.trend_direction = self._calculate_trend_direction(histories)

    def _calculate_trend_direction(self, histories: list[RoutingHistory]) -> str:
        """Calculate whether performance is improving, degrading, or stable."""
        # Simple trend analysis based on recent vs older performance
        recent_performances = []
        older_performances = []

        cutoff_date = datetime.now() - timedelta(days=7)  # Last week vs before

        for history in histories:
            last_updated = datetime.fromisoformat(history.last_updated)

            if last_updated >= cutoff_date:
                recent_performances.append(history.success_rate)
            else:
                older_performances.append(history.success_rate)

        if not recent_performances or not older_performances:
            return "stable"

        recent_avg = statistics.mean(recent_performances)
        older_avg = statistics.mean(older_performances)

        difference = recent_avg - older_avg

        if difference > 0.05:
            return "improving"
        elif difference < -0.05:
            return "degrading"
        else:
            return "stable"

    # Factory function
    _history_analyzer_instance = None

    def get_query_history_analyzer() -> "QueryHistoryAnalyzer":
        """Get or create a QueryHistoryAnalyzer instance."""
        global _history_analyzer_instance
        if _history_analyzer_instance is None:
            _history_analyzer_instance = QueryHistoryAnalyzer()
        return _history_analyzer_instance
