from abc import abstractmethod
from typing import List, Tuple, Dict, Optional
from src.workers.sql_worker import SQLWorker
from src.evaluation.metrics.metric import Metric
from src.typing.query import DBQuery
from src.typing.result import ExecutionResult
import json
from pathlib import Path
from utils.loggers.metrics_logger import MetricsLogger, log_with_emoji
import logging

# Get logger instance
metrics_logger = MetricsLogger.get_instance().get_metrics_logger(
    log_file='logs/metrics_evaluation.log',
    level=logging.INFO
)

class ExecutionBasedMetric(Metric):
    """Base class for metrics that require query execution (no caching)."""

    def __init__(
        self,
        sql_worker: SQLWorker = None,
        runs_per_query: int = 1,
        **kwargs
    ):
        """
        Args:
            sql_worker: Optional pre-configured SQLWorker
            runs_per_query: How many times to run each query (for timing)
        """
        if sql_worker is not None:
            self.sql_worker = sql_worker
        else:
            self.sql_worker = SQLWorker(runs_per_query=runs_per_query, **kwargs)

        super().__init__()

    def _execute_targets(self, target_queries: List[DBQuery]) -> Dict[str, ExecutionResult]:
        """Execute target queries without caching."""
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Executing {len(target_queries)} target queries...",
            "hourglass"
        )
        results_list = self.sql_worker.execute_parallel(target_queries)

        successful_count = sum(1 for r in results_list if r.success)
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Target execution complete: {successful_count}/{len(results_list)} successful",
            "bar_chart"
        )

        # Convert to dict
        return {str(r.query_id): r for r in results_list}

    def _execute_predictions(self, prediction_queries: List[DBQuery]) -> List[ExecutionResult]:
        """Execute prediction queries."""
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Executing {len(prediction_queries)} prediction queries...",
            "rocket"
        )
        
        results = self.sql_worker.execute_parallel(prediction_queries)
        
        successful = sum(1 for r in results if r.success)
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Prediction execution complete: {successful}/{len(results)} successful",
            "bar_chart"
        )
        
        return results

    def _compute_score(self, target: ExecutionResult, prediction: ExecutionResult) -> float:
        """Compute metric score for a single target-prediction pair.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("_compute_score must be implemented by subclasses.")

    def compute_many(
        self,
        target_results: Dict[str, ExecutionResult],
        prediction_results: List[ExecutionResult]
    ) -> Tuple[List[float], List[str]]:
        """
        Compute metric for multiple queries.

        Args:
            target_results: Dict mapping query_id to ExecutionResult (already executed)
            prediction_results: List of ExecutionResult (already executed)

        Returns:
            Tuple of (scores, skipped_ids)
            - scores: List of metric values (only for valid targets)
            - skipped_ids: List of query IDs that were skipped due to invalid targets
        """
        scores = []
        skipped = []

        for pred_res in prediction_results:
            query_id = str(pred_res.query_id)
            target_res = target_results.get(query_id)

            # Skip if target doesn't exist or failed
            if target_res is None:
                log_with_emoji(
                    metrics_logger,
                    logging.WARNING,
                    f"Query {query_id}: Target result not found",
                    "warning"
                )
                skipped.append(query_id)
                continue

            if not target_res.success:
                log_with_emoji(
                    metrics_logger,
                    logging.WARNING,
                    f"Query {query_id}: Invalid target result (error: {target_res.error})",
                    "x"
                )
                skipped.append(query_id)
                continue

            # Compute score using subclass logic
            score = self._compute_score(target_res, pred_res)
            scores.append(score)

        # Summary
        log_with_emoji(
            metrics_logger,
            logging.INFO,
            f"Evaluation complete: {len(scores)} valid, {len(skipped)} skipped",
            "trophy"
        )

        if skipped:
            metrics_logger.info(f"Skipped query IDs: {', '.join(skipped[:10])}{'...' if len(skipped) > 10 else ''}")

        return scores, skipped

    def compute(self, target: DBQuery, prediction: DBQuery) -> float:
        """Compute metric for a single query pair."""
        query_id = str(target.query_id)

        # Execute both queries
        target_res = self.sql_worker.execute_single(
            target.db_path, target.query_id, target.query
        )

        pred_res = self.sql_worker.execute_single(
            prediction.db_path, prediction.query_id, prediction.query
        )

        # Skip if target invalid
        if not target_res.success:
            log_with_emoji(
                metrics_logger,
                logging.WARNING,
                f"Query {query_id}: Invalid target, skipping",
                "warning"
            )
            return None

        score = self._compute_score(target_res, pred_res)

        log_with_emoji(
            metrics_logger,
            logging.DEBUG,
            f"Query {query_id}: Score = {score:.4f}",
            "dart"
        )

        return score
    
    def compute_metric(self, metric_name: str, target, prediction) -> float:
        """Compute a specific metric given target and prediction.

        Args:
            metric_name (str): The name of the metric to compute.
            target (Any): The ground truth data.
            prediction (Any): The predicted data.
        Returns:
            float | int: The computed metric value.
        """
        raise NotImplementedError("Use compute or compute_many methods for ExecutionBasedMetric.")

class CachedExecutionMetricWrapper:
    """Wrapper that handles ALL caching (runtime + file-based) for ExecutionBasedMetric instances.

    Can be used as both a wrapper and a class decorator:

    As a wrapper:
        metric = CachedExecutionMetricWrapper(ExecAccuracy())

    As a decorator:
        @CachedExecutionMetricWrapper
        class ExecAccuracy(ExecutionBasedMetric):
            ...
    """

    _shared_runtime_cache = {}  # Shared runtime cache for all instances

    def __init__(self, metric_or_class, cache_file_path: str = None):
        """
        Args:
            metric_or_class: Either an ExecutionBasedMetric instance or a class (when used as decorator)
            cache_file_path: Optional path to file-based cache for target results
        """
        # Check if used as a decorator (metric_or_class is a class, not an instance)
        if isinstance(metric_or_class, type) and issubclass(metric_or_class, ExecutionBasedMetric):
            # Decorator mode: store the class for later instantiation
            self._metric_class = metric_or_class
            self._is_decorator = True
            self.metric = None  # Will be instantiated on first call
        else:
            # Wrapper mode: validate instance
            if not isinstance(metric_or_class, ExecutionBasedMetric):
                raise TypeError(f"Expected ExecutionBasedMetric instance or class, got {type(metric_or_class)}")
            self.metric = metric_or_class
            self._is_decorator = False
            self._metric_class = None

        self.cache_file_path = cache_file_path

        # Load file-based cache on initialization
        if cache_file_path:
            cached_data = self._load_file_cache(cache_file_path)
            if cached_data:
                # Store directly in runtime cache by query_id
                self._shared_runtime_cache.update(cached_data)

    def __call__(self, *args, **kwargs):
        """Support decorator usage by returning a wrapped instance when class is instantiated."""
        if self._is_decorator:
            # Create instance of the decorated class
            metric_instance = self._metric_class(*args, **kwargs)
            # Return a new wrapper around the instance
            return CachedExecutionMetricWrapper(metric_instance, self.cache_file_path)
        else:
            raise TypeError("CachedExecutionMetricWrapper instance is not callable. Use compute_many() method.")

    @staticmethod
    def _load_file_cache(cache_path: str) -> Optional[Dict[str, ExecutionResult]]:
        """Load cached execution results from file."""
        if not Path(cache_path).is_file():
            metrics_logger.warning(f"Cache file not found: {cache_path}")
            return None

        try:
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)

            results = {}
            for _id, v in cached_data.items():
                results[str(_id)] = ExecutionResult(
                    query_id=str(_id),
                    results=[tuple(row) for row in v['results']],
                    exec_time_ms=v['exec_time_ms'],
                    success=v['success'],
                    error=v.get('error', '')
                )

            log_with_emoji(
                metrics_logger,
                logging.INFO,
                f"Loaded {len(results)} cached target results from {cache_path}",
                "check_mark"
            )
            return results

        except Exception as e:
            metrics_logger.error(f"Error loading cache from {cache_path}: {e}")
            return None

    @staticmethod
    def _save_file_cache(cache_path: str, results: Dict[str, ExecutionResult]) -> None:
        """Save execution results to file cache."""
        try:
            cache_dict = {}
            for query_id, res in results.items():
                cache_dict[int(query_id)] = {
                    'results': [list(row) for row in res.results],
                    'exec_time_ms': res.exec_time_ms,
                    'success': res.success,
                    'error': res.error
                }

            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(cache_dict, f, indent=4)

            log_with_emoji(
                metrics_logger,
                logging.INFO,
                f"Saved {len(results)} target results to cache: {cache_path}",
                "floppy_disk"
            )

        except Exception as e:
            metrics_logger.error(f"Error saving cache to {cache_path}: {e}")

    def compute_many(self, target_queries: List[DBQuery], predicted_queries: List[DBQuery]) -> Tuple[List[float], List[str]]:
        """Compute metric with full caching support."""

        # Find which queries are missing from cache
        missing_queries = []
        cached_results = {}

        for query in target_queries:
            query_id = str(query.query_id)
            if query_id in self._shared_runtime_cache:
                # Already cached
                cached_results[query_id] = self._shared_runtime_cache[query_id]
            else:
                # Need to execute
                missing_queries.append(query)

        if missing_queries:
            # Execute only missing queries
            log_with_emoji(
                metrics_logger,
                logging.INFO,
                f"Executing {len(missing_queries)} missing target queries (cached: {len(cached_results)})",
                "hourglass"
            )
            new_results = self.metric._execute_targets(missing_queries)

            # Store in shared cache by query_id
            self._shared_runtime_cache.update(new_results)

            # Save to file if configured
            if self.cache_file_path:
                self._save_file_cache(self.cache_file_path, self._shared_runtime_cache)

            # Merge for this computation
            target_results = {**cached_results, **new_results}
        else:
            # All cached
            target_results = cached_results
            log_with_emoji(
                metrics_logger,
                logging.INFO,
                f"Full cache hit - using {len(target_results)} cached target results",
                "high_voltage"
            )

        # Execute predictions (never cached)
        prediction_results = self.metric._execute_predictions(predicted_queries)

        # Delegate to metric for scoring
        return self.metric.compute_many(target_results, prediction_results)