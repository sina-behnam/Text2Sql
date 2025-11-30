from dataclasses import dataclass
from typing import List, Tuple

@dataclass 
class DBQuery:
    db_name: str
    db_path: str
    query_id: int
    query: str

@dataclass
class TargetPredictedDBQuery:
    target: DBQuery
    predicted: DBQuery