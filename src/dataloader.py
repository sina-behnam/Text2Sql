import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DatasetInstance:
    """Represents a single Text2SQL instance from the dataset"""
    id: int
    question: str
    sql: str
    database: Dict
    schemas: List[Dict]
    difficulty: str
    dataset: str
    original_instance_id: Optional[str] = None
    evidence: Optional[str] = None
    question_analysis: Optional[Dict] = None
    sql_analysis: Optional[Dict] = None
    inference_results: Optional[Dict] = None
    
    @classmethod
    def from_json(cls, data: Dict) -> 'DatasetInstance':
        """Create instance from JSON data"""
        return cls(
            id=data['id'],
            question=data['question'],
            sql=data['sql'],
            database=data['database'],
            schemas=data['schemas'],
            difficulty=data.get('difficulty', 'unknown'),
            dataset=data.get('dataset', 'unknown'),
            original_instance_id=data.get('original_instance_id'),
            evidence=data.get('evidence'),
            question_analysis=data.get('question_analysis'),
            sql_analysis=data.get('sql_analysis'),
            inference_results=data.get('inference_results')
        )
    
    def to_dict(self) -> Dict:
        """Convert instance to dictionary"""
        return asdict(self)

class DatasetLoader:
    """Handles loading and processing of the Text2SQL datasets"""
    
    def __init__(self, data_path: str = "/app/data"):
        self.data_path = Path(data_path)
        self.instances: List[DatasetInstance] = []
    
    def load_instances(self, pattern: str = "instance_*.json") -> List[Tuple[DatasetInstance, str]]:
        """Load all instances matching the pattern"""
        logger.info(f"Loading instances from {self.data_path} with pattern {pattern}")
        
        instance_files = list(self.data_path.glob(pattern))
        logger.info(f"Found {len(instance_files)} instance files")
        
        instances_with_paths = []
        for file_path in instance_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    instance = DatasetInstance.from_json(data)
                    instances_with_paths.append((instance, str(file_path)))
                    logger.debug(f"Loaded instance {instance.id} from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        self.instances = [inst for inst, _ in instances_with_paths]
        logger.info(f"Successfully loaded {len(instances_with_paths)} instances")
        return instances_with_paths
    
    def get_instance_by_id(self, instance_id: int) -> Optional[DatasetInstance]:
        """Get a specific instance by ID"""
        for instance in self.instances:
            if instance.id == instance_id:
                return instance
        return None
    
    def filter_by_difficulty(self, difficulty: str) -> List[DatasetInstance]:
        """Filter instances by difficulty level"""
        return [inst for inst in self.instances if inst.difficulty == difficulty]
    
    def filter_by_dataset(self, dataset: str) -> List[DatasetInstance]:
        """Filter instances by dataset type"""
        return [inst for inst in self.instances if inst.dataset == dataset]
