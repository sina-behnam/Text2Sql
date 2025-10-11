import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


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
    def from_dict(cls, data: Dict) -> 'DatasetInstance':
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
    
    def to_dataframe(self,flatten_nested: bool = True, 
                         include_schemas: bool = False,
                         include_analysis: bool = True,
                         include_inference: bool = True
                         ) -> pd.DataFrame:
        """
        Convert DatasetInstance(s) to pandas DataFrame.

        Args:
            instance: Single DatasetInstance
            flatten_nested: Whether to flatten nested dictionaries (database info, etc.)
            include_schemas: Whether to include schema information (can make DataFrame very wide)
            include_analysis: Whether to include question_analysis and sql_analysis

        Returns:
            pandas DataFrame with instance data
        """
    
    
        # Start with basic fields
        row = {
            'id': self.id,
            'question': self.question,
            'sql': self.sql,
            'difficulty': self.difficulty,
            'dataset': self.dataset,
            'original_instance_id': self.original_instance_id,
            'evidence': self.evidence
        }
        
        # Handle database information
        if flatten_nested and self.database:
            row['database_name'] = self.database.get('name')
            row['database_type'] = self.database.get('type')
            # Convert path list to string
            if 'path' in self.database:
                row['database_path'] = json.dumps(self.database['path']) if isinstance(self.database['path'], list) else self.database['path']
        else:
            row['database'] = json.dumps(self.database) if self.database else None

        # Handle schemas
        if include_schemas and self.schemas:
            row['schemas'] = [self.schemas]
            row['num_schemas'] = len(self.schemas)
        else:
            row['num_schemas'] = len(self.schemas) if self.schemas else 0
    
        # Handle analysis data
        if include_analysis:
            # Question analysis
            if self.question_analysis:
                qa = self.question_analysis
                row['question_char_length'] = qa.get('char_length')
                row['question_word_length'] = qa.get('word_length')
                row['question_has_entities'] = qa.get('has_entities', False)
                row['question_has_numbers'] = qa.get('has_numbers', False)
                row['question_has_negation'] = qa.get('has_negation', False)
                row['question_has_superlatives'] = qa.get('has_superlatives', False)
                row['question_entity_types'] = json.dumps(qa.get('entity_types', []))
                row['question_numbers'] = json.dumps(qa.get('numbers', []))

            # SQL analysis
            if self.sql_analysis:
                sa = self.sql_analysis
                row['sql_char_length'] = sa.get('char_length')
                row['sql_tables_count'] = sa.get('tables_count')
                row['sql_join_count'] = sa.get('join_count')
                row['sql_where_conditions'] = sa.get('where_conditions')
                row['sql_subquery_count'] = sa.get('subquery_count')
                row['sql_aggregation_function_count'] = sa.get('aggregation_function_count')
                row['sql_tables'] = json.dumps(sa.get('tables', []))
                row['sql_aggregation_functions'] = json.dumps(sa.get('aggregation_functions', []))

        # Handle inference results
        if include_inference and self.inference_results:
            ir = self.inference_results
            row['has_prediction'] = ir.get('has_prediction', False)

            if 'model' in ir:
                model_info = ir['model']
                row['model_name'] = model_info.get('model_name')
                row['model_type'] = model_info.get('model_type')
                row['model_timestamp'] = model_info.get('timestamp')

            if 'predicted_output' in ir:
                po = ir['predicted_output']
                row['generated_sql'] = po.get('generated_sql')
                row['execution_correct'] = po.get('execution_correct')
                row['exact_match'] = po.get('exact_match')
                row['semantic_equivalent'] = po.get('semantic_equivalent')
                row['execution_error'] = po.get('execution_error')
                row['semantic_explanation'] = po.get('semantic_explanation')
    
    
        return pd.DataFrame(row, index=[0])

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
                    instance = DatasetInstance.from_dict(data)
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
