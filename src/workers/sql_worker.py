from dataclasses import dataclass
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional
from sqlalchemy import create_engine, text, pool
from tqdm import tqdm
import sqlalchemy
import numpy as np
from src.workers.sql_logger import SQLLogger
from src.typing.result import ExecutionResult
from src.typing.query import DBQuery

execution_logger = SQLLogger.get_instance().get_execution_logger()

def _remove_outliers(array: list[float]) -> list[float]:
    """Remove outliers using 3-sigma rule."""
    if len(array) < 3:
        return array
    mean, std = np.mean(array), np.std(array)
    lower, upper = mean - 3 * std, mean + 3 * std
    return [x for x in array if lower <= x <= upper]

def _execute_single_query_worker(
    db_url: str,
    query_id: str,
    sql: str,
    runs_per_query: int,
    timeout: int,
    max_try_timeout: int
) -> ExecutionResult:
    """
    Standalone worker function for parallel execution.
    Executes one query multiple times and returns averaged result.
    """
    _db_url = db_url if db_url.startswith('sqlite:///') else f'sqlite:///{db_url}'
    
    try:
        engine = create_engine(
            _db_url,
            poolclass=pool.NullPool,
            connect_args={'check_same_thread': False} if 'sqlite' in db_url else {}
        )
    except Exception as e:
        return ExecutionResult(query_id, [], 0, False, f"Connection error: {e}")

    times_ms = []
    final_results = []
    timeout_count = 0

    try:
        with engine.connect() as connection:
            raw_conn = connection.connection.dbapi_connection

            for run in range(runs_per_query):
                start_time = time.time()
                timed_out = False

                def progress_check():
                    nonlocal timed_out
                    if time.time() - start_time > timeout:
                        timed_out = True
                        return 1
                    return 0

                raw_conn.set_progress_handler(progress_check, 1000)

                try:
                    t1 = time.perf_counter()
                    result = connection.execute(text(sql))
                    rows = result.fetchall()
                    t2 = time.perf_counter()
                    
                    raw_conn.set_progress_handler(None, 0)
                    
                    if timed_out:
                        raise TimeoutError("Query exceeded time limit")

                    times_ms.append((t2 - t1) * 1000)
                    final_results = rows

                except (TimeoutError, sqlalchemy.exc.OperationalError) as e:
                    raw_conn.set_progress_handler(None, 0)

                    execution_logger.info(f"Timeout/OperationalError for {query_id} on run attempt ({timeout_count + 1}/{max_try_timeout})")
                    
                    if timed_out or "interrupt" in str(e).lower():
                        try:
                            connection.rollback()
                        except:
                            pass
                        timeout_count += 1
                        if timeout_count >= max_try_timeout:
                            execution_logger.warning(f"Max timeouts for {query_id}")
                            return ExecutionResult(query_id, [], 0, False, "TimeoutError")
                        continue
                    else:
                        execution_logger.warning(f"OperationalError for {query_id}: {e}")
                        return ExecutionResult(query_id, [], 0, False, str(e))

                except sqlalchemy.exc.ResourceClosedError:
                    raw_conn.set_progress_handler(None, 0)
                    execution_logger.warning(f"ResourceClosedError for {query_id}")
                    return ExecutionResult(query_id, [], 0, False, "ResourceClosedError")

                except Exception as e:
                    raw_conn.set_progress_handler(None, 0)
                    try:
                        connection.rollback()
                    except:
                        pass
                    execution_logger.warning(f"Exception for {query_id}: {e}")
                    return ExecutionResult(query_id, [], 0, False, str(e))

    except Exception as e:
        return ExecutionResult(query_id, [], 0, False, str(e))
    finally:
        engine.dispose()

    # Success case
    avg_time = np.mean(_remove_outliers(times_ms)) if times_ms else 0
    execution_logger.info(f"Query {query_id} executed in {avg_time:.2f} ms")
    return ExecutionResult(query_id, final_results, avg_time, True)


class SQLWorker:
    """Execute SQL queries with query-level parallelization."""

    def __init__(
        self,
        num_workers: int = 4,
        runs_per_query: int = 100,
        timeout: int = 6,
        max_try_timeout: int = 5
    ):
        self.num_workers = num_workers
        self.runs_per_query = runs_per_query
        self.timeout = timeout
        self.max_try_timeout = max_try_timeout

    def execute_single(self, db_url: str, query_id: str, sql: str) -> ExecutionResult:
        """Execute a single query (no parallelization)."""
        return _execute_single_query_worker(
            db_url, query_id, sql,
            self.runs_per_query, self.timeout, self.max_try_timeout
        )

    def execute_batch(self, db_url: str, queries: List[DBQuery]) -> List[ExecutionResult]:
        """Execute multiple queries sequentially on one database."""
        return [self.execute_single(db_url, query.query_id, query.query) for query in queries]
    
    def execute_db_parallel(self, db_url: str, queries: List[DBQuery]) -> List[ExecutionResult]:
        """
        Execute multiple queries in parallel on one database.
        
        Args:
            db_url: Database connection URL
            queries: List of (query_id, sql) tuples
        results = []
        """
        results = []

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    _execute_single_query_worker,
                    db_url, query.query_id, query.query,
                    self.runs_per_query, self.timeout, self.max_try_timeout
                ): query.query_id
                for query in queries
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Executing"):
                results.append(future.result())

        return results

    def execute_parallel(self, queries: List[DBQuery]) -> List[ExecutionResult]:
        """
        Execute queries in parallel, preserving input order (handles duplicate query_ids).
        """
        results = [None] * len(queries)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    _execute_single_query_worker,
                    query.db_path, query.query_id, query.query,
                    self.runs_per_query, self.timeout, self.max_try_timeout
                ): idx
                for idx, query in enumerate(queries)
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Executing"):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Process-level failure - create failed result
                    query = queries[idx]
                    results[idx] = ExecutionResult(
                        query_id=str(query.query_id),
                        results=[],
                        exec_time_ms=0,
                        success=False,
                        error=f"Process error: {str(e)}"
                    )

        # Safety check: replace any remaining None with failed result
        for idx, res in enumerate(results):
            if res is None:
                query = queries[idx]
                results[idx] = ExecutionResult(
                    query_id=str(query.query_id),
                    results=[],
                    exec_time_ms=0,
                    success=False,
                    error="Unknown execution failure"
                )
        return results