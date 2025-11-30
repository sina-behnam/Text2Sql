from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ExecutionResult:
    query_id: str
    results: List[Tuple]
    exec_time_ms: float
    success: bool
    error: str = ""