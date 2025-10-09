"""
Standardized instance processor for Text2SQL datasets.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

@dataclass
class StandardizedInstance:
    """Standardized instance format for all datasets"""
    id: int
    dataset: str
    question: str
    sql: str
    database: Dict[str, Any]
    schemas: List[Dict]
    difficulty: str
    original_instance_id: Optional[str] = None
    evidence: Optional[str] = None
    question_analysis: Optional[Dict] = None
    sql_analysis: Optional[Dict] = None
    schema_analysis: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)