import time
from statistics import median
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sqlalchemy import create_engine, text, pool
from tqdm import tqdm
import sqlalchemy
import numpy as np
import logging
from pathlib import Path
import signal

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============= SQLAlchemy Logging (Existing) =============
_sqlalchemy_handler = logging.FileHandler(LOG_DIR / "sqlalchemy.log", mode="w", encoding="utf-8")
_sqlalchemy_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
)

for name in ("sqlalchemy.engine", "sqlalchemy.pool"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(_sqlalchemy_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to root (no console output)


# ============= Your Custom Logging =============
execution_logger = logging.getLogger("query_execution")  # Custom logger
execution_logger.handlers.clear()  # Remove any existing handlers

# File handler only (no console)
_execution_handler = logging.FileHandler(LOG_DIR / "execution.log", mode="w", encoding="utf-8")
_execution_handler.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)

execution_logger.addHandler(_execution_handler)
execution_logger.setLevel(logging.INFO)  # Change to WARNING/ERROR if you want less
execution_logger.propagate = False  # ← KEY: Prevents console output

def _remove_outliers(array: list[float]) -> list[float]:
    mean, std = np.mean(array), np.std(array)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    return [x for x in array if lower_bound <= x <= upper_bound]


@dataclass
class ExecutionResult:
    query_id: str
    results: List[Tuple]
    exec_time_ms: float
    success: bool
    error: str = ""


def batch_sql_query_worker(
        db_url: str,
        queries: List[Tuple[str, str]],  # List of (query_id, sql)
        runs: int = 3,
        timeout: int = 6,
        max_try_timeout: int = 5
    ) -> List[ExecutionResult]:

    _db_url = 'sqlite:///' + db_url if not db_url.startswith('sqlite:///') else db_url

    try:
        engine = create_engine(
            _db_url,
            poolclass=pool.NullPool,
            connect_args={'check_same_thread': False} if 'sqlite' in db_url else {}
        )
        
        results = []

        with engine.connect() as connection:
            raw_conn = connection.connection.dbapi_connection
        
            for query_id, sql in queries:
                times_ms = []
                final_results = None
                _try_timeout = 0

                for _ in range(runs):
                    start_time = time.time()
                    timed_out = False

                    def progress_check():
                        nonlocal timed_out
                        if time.time() - start_time > timeout:
                            timed_out = True
                            return 1  # Non-zero cancels query
                        return 0

                    raw_conn.set_progress_handler(progress_check, 1000)

                    try:
                        t1 = time.perf_counter()
                        result = connection.execute(text(sql))
                        rows = result.fetchall()
                        t2 = time.perf_counter()

                        raw_conn.set_progress_handler(None, 0)

                        if timed_out:
                            raise TimeoutError("Query execution exceeded the time limit.")

                        times_ms.append((t2 - t1) * 1000)
                        final_results = rows

                    except (TimeoutError, sqlalchemy.exc.OperationalError) as e:
                        raw_conn.set_progress_handler(None, 0)
                        
                        if timed_out or "interrupt" in str(e).lower():
                            try:
                                connection.rollback()
                            except:
                                pass
                            _try_timeout += 1

                            if _try_timeout >= max_try_timeout:
                                results.append(ExecutionResult(query_id, [], 0, False, "TimeoutError"))
                                execution_logger.warning(f"Max timeouts ({max_try_timeout}) reached for query ID: {query_id}")
                                break
                            else:
                                execution_logger.warning(f"TimeoutError for query ID: {query_id} (attempt {_try_timeout}/{max_try_timeout})")
                                continue
                        else:
                            results.append(ExecutionResult(query_id, [], 0, False, str(e)))
                            execution_logger.warning(f"Exception for query ID: {query_id}: {e}")
                            break

                    except sqlalchemy.exc.ResourceClosedError:
                        raw_conn.set_progress_handler(None, 0)
                        results.append(ExecutionResult(query_id, [], 0, False, "ResourceClosedError"))
                        execution_logger.warning(f"ResourceClosedError for query ID: {query_id}")
                        break
                        
                    except Exception as e:
                        raw_conn.set_progress_handler(None, 0)
                        try:
                            connection.rollback()
                        except:
                            pass
                        results.append(ExecutionResult(query_id, [], 0, False, str(e)))
                        execution_logger.warning(f"Exception for query ID: {query_id}: {e}")
                        break
                
                else:
                    execution_logger.info(f"Query ID {query_id} executed successfully in {np.mean(_remove_outliers(times_ms)):.2f} ms")
                    results.append(ExecutionResult(
                        query_id=query_id,
                        results=final_results,
                        exec_time_ms=np.mean(_remove_outliers(times_ms)),
                        success=True
                    ))

        engine.dispose()
        return results
    
    except Exception as e:
        return [ExecutionResult(query_id, [], 0, False, str(e)) for query_id, _ in queries]


def execute_queries_parallel(
    queries: List[Dict],  
    max_workers: int = 4,
    runs_per_query: int = 100,
    timeout: int = 6,
    max_try_timeout: int = 5
) -> List[ExecutionResult]:
    """
    Execute SQL queries in parallel using SQLAlchemy Core.
    
    Args:
        queries: 
        max_workers: CPU cores
        runs_per_query: Runs for timing
    """
    results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                batch_sql_query_worker,
                db_url=db_tuple[0][1], 
                queries=db_tuple[1],
                runs=runs_per_query,
                timeout=timeout,
                max_try_timeout=max_try_timeout
            ) : db_tuple for db_tuple in queries.items()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Executing queries"):
            results.extend(future.result())
    
    return results
