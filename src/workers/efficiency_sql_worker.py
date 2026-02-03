"""
Extended SQL Worker for efficiency metrics.

Provides specialized execution methods for:
- Memory tracking during query execution
- Query plan analysis (EXPLAIN QUERY PLAN)

All methods follow the same timeout/retry patterns as the base SQLWorker.
"""

import time
import tracemalloc
import re
from dataclasses import dataclass
from typing import List, Tuple
from sqlalchemy import create_engine, text, pool
import sqlalchemy

from src.workers.sql_worker import SQLWorker
from src.workers.sql_logger import SQLLogger
from src.typing.query import DBQuery

execution_logger = SQLLogger.get_instance().get_execution_logger()


@dataclass
class MemoryExecutionResult:
    """Execution result with memory tracking information."""
    query_id: str
    results: List[Tuple]
    exec_time_ms: float
    peak_memory_bytes: int
    success: bool
    error: str = ""


@dataclass
class QueryPlanResult:
    """Result of EXPLAIN QUERY PLAN analysis."""
    query_id: str
    has_full_scan: bool
    has_index_scan: bool
    has_covering_index: bool
    estimated_rows: int
    operation_count: int
    plan_text: str
    success: bool
    error: str = ""


class EfficiencySQLWorker(SQLWorker):
    """
    Extended SQL Worker with efficiency measurement capabilities.

    Adds methods for:
    - Memory-tracked query execution
    - Query plan analysis

    Inherits all base SQLWorker functionality (timeouts, retries).
    """

    def __init__(
        self,
        num_workers: int = 4,
        runs_per_query: int = 1,
        timeout: int = 6,
        max_try_timeout: int = 5
    ):
        super().__init__(
            num_workers=num_workers,
            runs_per_query=runs_per_query,
            timeout=timeout,
            max_try_timeout=max_try_timeout
        )

    def execute_with_memory(self, db_url: str, query_id: str, sql: str) -> MemoryExecutionResult:
        """
        Execute a single query with memory tracking.

        Uses tracemalloc to measure peak memory during execution.
        Follows same timeout pattern as base SQLWorker.
        """
        _db_url = db_url if db_url.startswith('sqlite:///') else f'sqlite:///{db_url}'

        try:
            engine = create_engine(
                _db_url,
                poolclass=pool.NullPool,
                connect_args={'check_same_thread': False} if 'sqlite' in db_url else {}
            )
        except Exception as e:
            return MemoryExecutionResult(query_id, [], 0, 0, False, f"Connection error: {e}")

        final_results = []
        exec_time_ms = 0
        peak_memory = 0

        try:
            with engine.connect() as connection:
                raw_conn = connection.connection.dbapi_connection

                start_time = time.time()
                timed_out = False

                def progress_check():
                    nonlocal timed_out
                    if time.time() - start_time > self.timeout:
                        timed_out = True
                        return 1
                    return 0

                raw_conn.set_progress_handler(progress_check, 1000)

                try:
                    # Start memory tracking
                    tracemalloc.start()

                    t1 = time.perf_counter()
                    result = connection.execute(text(sql))
                    rows = result.fetchall()
                    t2 = time.perf_counter()

                    # Get memory stats before stopping
                    current, peak_memory = tracemalloc.get_traced_memory()
                    tracemalloc.stop()

                    raw_conn.set_progress_handler(None, 0)

                    if timed_out:
                        raise TimeoutError("Query exceeded time limit")

                    exec_time_ms = (t2 - t1) * 1000
                    final_results = rows

                except (TimeoutError, sqlalchemy.exc.OperationalError) as e:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    raw_conn.set_progress_handler(None, 0)

                    execution_logger.info(f"Timeout/OperationalError for {query_id}")

                    if timed_out or "interrupt" in str(e).lower():
                        try:
                            connection.rollback()
                        except:
                            pass
                        return MemoryExecutionResult(query_id, [], 0, 0, False, "TimeoutError")
                    else:
                        return MemoryExecutionResult(query_id, [], 0, 0, False, str(e))

                except Exception as e:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    raw_conn.set_progress_handler(None, 0)
                    try:
                        connection.rollback()
                    except:
                        pass
                    return MemoryExecutionResult(query_id, [], 0, 0, False, str(e))

        except Exception as e:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            return MemoryExecutionResult(query_id, [], 0, 0, False, str(e))
        finally:
            engine.dispose()

        return MemoryExecutionResult(
            query_id=query_id,
            results=final_results,
            exec_time_ms=exec_time_ms,
            peak_memory_bytes=peak_memory,
            success=True
        )

    def get_query_plan(self, db_url: str, query_id: str, sql: str) -> QueryPlanResult:
        """
        Execute EXPLAIN QUERY PLAN and parse results.

        Uses SQLAlchemy connection with timeout handling.
        """
        _db_url = db_url if db_url.startswith('sqlite:///') else f'sqlite:///{db_url}'

        try:
            engine = create_engine(
                _db_url,
                poolclass=pool.NullPool,
                connect_args={'check_same_thread': False} if 'sqlite' in db_url else {}
            )
        except Exception as e:
            return QueryPlanResult(
                query_id=query_id,
                has_full_scan=False,
                has_index_scan=False,
                has_covering_index=False,
                estimated_rows=0,
                operation_count=0,
                plan_text="",
                success=False,
                error=f"Connection error: {e}"
            )

        try:
            with engine.connect() as connection:
                raw_conn = connection.connection.dbapi_connection

                start_time = time.time()
                timed_out = False

                def progress_check():
                    nonlocal timed_out
                    if time.time() - start_time > self.timeout:
                        timed_out = True
                        return 1
                    return 0

                raw_conn.set_progress_handler(progress_check, 1000)

                try:
                    # Execute EXPLAIN QUERY PLAN
                    result = connection.execute(text(f"EXPLAIN QUERY PLAN {sql}"))
                    plan_rows = result.fetchall()

                    raw_conn.set_progress_handler(None, 0)

                    if timed_out:
                        raise TimeoutError("Query plan analysis exceeded time limit")

                    # Parse plan output
                    plan_text = "\n".join([str(row) for row in plan_rows])
                    plan_text_upper = plan_text.upper()

                    # Detect full table scans
                    has_full_scan = "SCAN TABLE" in plan_text_upper or "SCAN " in plan_text_upper

                    # Detect index usage
                    has_index_scan = "USING INDEX" in plan_text_upper or "SEARCH" in plan_text_upper

                    # Detect covering index (index-only scan)
                    has_covering_index = "COVERING INDEX" in plan_text_upper

                    # Count operations
                    operation_count = len(plan_rows)

                    # Estimate rows (simplified)
                    estimated_rows = 0
                    for row in plan_rows:
                        row_str = str(row)
                        match = re.search(r'\(~?(\d+)\s*rows?\)', row_str, re.IGNORECASE)
                        if match:
                            estimated_rows += int(match.group(1))

                    return QueryPlanResult(
                        query_id=query_id,
                        has_full_scan=has_full_scan,
                        has_index_scan=has_index_scan,
                        has_covering_index=has_covering_index,
                        estimated_rows=estimated_rows,
                        operation_count=operation_count,
                        plan_text=plan_text,
                        success=True
                    )

                except (TimeoutError, sqlalchemy.exc.OperationalError) as e:
                    raw_conn.set_progress_handler(None, 0)
                    return QueryPlanResult(
                        query_id=query_id,
                        has_full_scan=False,
                        has_index_scan=False,
                        has_covering_index=False,
                        estimated_rows=0,
                        operation_count=0,
                        plan_text="",
                        success=False,
                        error=str(e)
                    )

                except Exception as e:
                    raw_conn.set_progress_handler(None, 0)
                    return QueryPlanResult(
                        query_id=query_id,
                        has_full_scan=False,
                        has_index_scan=False,
                        has_covering_index=False,
                        estimated_rows=0,
                        operation_count=0,
                        plan_text="",
                        success=False,
                        error=str(e)
                    )

        except Exception as e:
            return QueryPlanResult(
                query_id=query_id,
                has_full_scan=False,
                has_index_scan=False,
                has_covering_index=False,
                estimated_rows=0,
                operation_count=0,
                plan_text="",
                success=False,
                error=str(e)
            )
        finally:
            engine.dispose()
