"""
COMPLETE SOLUTION: SQL Execution Checker with Timeout and Pre-filtering

This combines:
1. Query risk assessment (filters out bad queries)
2. Timeout mechanism (kills hanging queries)
3. Your original logic (exact comparison)
"""

import sqlite3
import time
from typing import Tuple, List, Dict
from multiprocessing import Process, Queue, Manager
import json


# ============================================================================
# PART 1: Risk Assessment (Pre-filter dangerous queries)
# ============================================================================

def is_query_safe(sql: str, max_risk_score: int = 50) -> Tuple[bool, str, int]:
    """
    Check if query is safe to execute.
    
    Returns:
        (is_safe, reason, risk_score)
    """
    if not sql:
        return False, "Empty query", 0
    
    sql_upper = sql.upper()
    risk_score = 0
    issues = []
    
    # Cartesian product check
    if ',' in sql and 'FROM' in sql_upper:
        import re
        from_match = re.search(r'FROM\s+(.*?)(?:WHERE|GROUP|ORDER|LIMIT|$)', 
                               sql_upper, re.DOTALL)
        if from_match:
            table_count = from_match.group(1).count(',') + 1
            if table_count > 1:
                issues.append('Cartesian product')
                risk_score += 40
    
    # JOIN without ON
    import re
    if re.search(r'JOIN\s+\w+\s+(?!ON)', sql_upper):
        issues.append('JOIN without ON')
        risk_score += 50
    
    # Multiple joins without LIMIT
    join_count = sql_upper.count('JOIN')
    if join_count >= 2 and 'LIMIT' not in sql_upper:
        issues.append('Multiple JOINs without LIMIT')
        risk_score += 10
    
    if risk_score > max_risk_score:
        return False, f"Unsafe query: {', '.join(issues)}", risk_score
    
    return True, "", risk_score


# ============================================================================
# PART 2: Execution with Timeout
# ============================================================================

def execute_with_timeout_worker(db_path: str, query: str, result_queue: Queue, 
                                 max_rows: int):
    """Worker to execute query in separate process"""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if len(rows) > max_rows:
            result_queue.put(('error', f"Too many rows: {len(rows)} > {max_rows}"))
        else:
            result_queue.put(('success', rows))
        
        conn.close()
    except Exception as e:
        result_queue.put(('error', str(e)))


def execute_with_timeout(db_path: str, query: str, timeout_sec: int, 
                         max_rows: int) -> Tuple[bool, any, str]:
    """Execute query with timeout using multiprocessing"""
    manager = Manager()
    result_queue = manager.Queue()
    
    process = Process(target=execute_with_timeout_worker,
                     args=(db_path, query, result_queue, max_rows))
    process.start()
    process.join(timeout=timeout_sec)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return False, None, f"Timeout after {timeout_sec}s"
    
    if not result_queue.empty():
        status, result = result_queue.get()
        if status == 'success':
            return True, result, ""
        return False, None, result
    
    return False, None, "No result"


# ============================================================================
# PART 3: Simple Version (In-memory DB, no true timeout but has safety check)
# ============================================================================

def check_execution_accuracy_safe(
    predicted_sql: str,
    ground_truth_sql: str,
    db_connection: sqlite3.Connection,
    skip_unsafe: bool = True,
    max_risk_score: int = 50
) -> Tuple[bool, str]:
    """
    Safe version that pre-filters dangerous queries.
    
    Args:
        predicted_sql: Predicted SQL
        ground_truth_sql: Ground truth SQL  
        db_connection: SQLite connection
        skip_unsafe: If True, skip queries with high risk score
        max_risk_score: Maximum acceptable risk score
        
    Returns:
        (is_correct, error_message)
        - error_message starting with "SKIPPED:" means query was too risky
    """
    # Pre-check: Is predicted SQL safe?
    if skip_unsafe:
        is_safe, reason, risk = is_query_safe(predicted_sql, max_risk_score)
        if not is_safe:
            return False, f"SKIPPED: {reason} (risk={risk})"
    
    try:
        cursor = db_connection.cursor()
        
        # Execute ground truth
        try:
            cursor.execute(ground_truth_sql)
            gt_result = cursor.fetchall()
        except Exception as e:
            return False, f"Ground truth error: {str(e)}"
        
        # Execute predicted
        try:
            cursor.execute(predicted_sql)
            pred_result = cursor.fetchall()
        except Exception as e:
            return False, f"Predicted SQL error: {str(e)}"
        
        # Quick comparison
        if pred_result == gt_result:
            return True, ""
        
        if len(pred_result) == 0 and len(gt_result) == 0:
            return True, ""
        
        if len(pred_result) != len(gt_result):
            return False, ""
        
        if pred_result and gt_result:
            if len(pred_result[0]) != len(gt_result[0]):
                return False, ""
        
        # Normalized comparison
        pred_set = set(tuple(row) for row in pred_result)
        gt_set = set(tuple(row) for row in gt_result)
        
        if pred_set == gt_set:
            return True, ""
        
        return False, ""
        
    except Exception as e:
        return False, f"Error: {str(e)}"


# ============================================================================
# PART 4: Full Version with File-Based DB and True Timeout
# ============================================================================

def check_execution_accuracy_full(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: str,
    timeout_seconds: int = 30,
    max_rows: int = 100000,
    skip_unsafe: bool = True,
    max_risk_score: int = 50
) -> Tuple[bool, str]:
    """
    Full version with timeout for file-based databases.
    
    Args:
        predicted_sql: Predicted SQL
        ground_truth_sql: Ground truth SQL
        db_path: Path to SQLite database file
        timeout_seconds: Timeout per query
        max_rows: Max rows to fetch
        skip_unsafe: Skip high-risk queries
        max_risk_score: Max acceptable risk
        
    Returns:
        (is_correct, error_message)
    """
    # Pre-check
    if skip_unsafe:
        is_safe, reason, risk = is_query_safe(predicted_sql, max_risk_score)
        if not is_safe:
            return False, f"SKIPPED: {reason} (risk={risk})"
    
    # Execute ground truth
    success, gt_result, error = execute_with_timeout(
        db_path, ground_truth_sql, timeout_seconds, max_rows
    )
    if not success:
        return False, f"Ground truth failed: {error}"
    
    # Execute predicted
    success, pred_result, error = execute_with_timeout(
        db_path, predicted_sql, timeout_seconds, max_rows
    )
    if not success:
        return False, f"Predicted failed: {error}"
    
    # Compare
    if pred_result == gt_result:
        return True, ""
    
    if len(pred_result) == 0 and len(gt_result) == 0:
        return True, ""
    
    if len(pred_result) != len(gt_result):
        return False, ""
    
    if pred_result and gt_result:
        if len(pred_result[0]) != len(gt_result[0]):
            return False, ""
    
    # Normalized
    pred_set = set(tuple(row) for row in pred_result)
    gt_set = set(tuple(row) for row in gt_result)
    
    return (pred_set == gt_set), ""
