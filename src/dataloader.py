import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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


class Text2SQLDataset(Dataset):
    """PyTorch Dataset for Text2SQL instances using Arctic template"""

    def __init__(self, data_path: str, pattern: str = "instance_*.json",
                 template=None, dialect: str = "SQLite"):
        """
        Initialize the PyTorch Dataset.

        Args:
            data_path: Path to the data directory
            pattern: Glob pattern to match instance files
            template: Prompt template instance (e.g., ArcticText2SQLTemplate)
            dialect: SQL dialect (e.g., "SQLite", "PostgreSQL")
        """
        self.data_path = Path(data_path)
        self.dialect = dialect
        self.template = template

        # Load instances with their file paths
        self.instances_with_paths = self._load_instances(pattern)

        logger.info(f"Initialized Text2SQLDataset with {len(self.instances_with_paths)} instances")

    def _load_instances(self, pattern: str) -> List[Tuple[DatasetInstance, str]]:
        """Load all instances matching the pattern"""
        logger.info(f"Loading instances from {self.data_path} with pattern {pattern}")

        instance_files = list(self.data_path.glob(pattern))
        logger.info(f"Found {len(instance_files)} instance files")

        instances_with_paths = []
        for file_path in sorted(instance_files):  # Sort for deterministic ordering
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    instance = DatasetInstance.from_dict(data)
                    instances_with_paths.append((instance, str(file_path)))
                    logger.debug(f"Loaded instance {instance.id} from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(f"Successfully loaded {len(instances_with_paths)} instances")
        return instances_with_paths

    def _format_schema(self, schemas: List[Dict]) -> str:
        """Format schema information into a string"""
        schema_parts = []
        for i,schema in enumerate(schemas):
            table_name = schema.get('table_name')
            ddl = schema.get('DDL')
            description = schema.get('description', '')

            shcema_part = f"\nTable {i+1} Name : {table_name}\nDDL:\n```{self.dialect}\n{ddl}```\nDescription: {description}"
            # sep = "\n" + "-"*40 + "\n"
            # if i > 0:
            #     schema_parts.append(sep)
            schema_parts.append(shcema_part)

        return "\n\n".join(schema_parts)

    def __len__(self) -> int:
        """Return the number of instances in the dataset"""
        return len(self.instances_with_paths)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single item from the dataset.

        Returns:
            Dictionary containing:
                - system_message: System prompt
                - user_message: User prompt with schema and question
                - assistant_prefix: Assistant's prefix to start generation
                - instance_path: Path to the instance file
                - instance_id: ID of the instance
                - question: Original question
                - ground_truth_sql: Ground truth SQL query
                - evidence: Additional evidence (if available)
                - database_name: Name of the database
        """
        instance, instance_path = self.instances_with_paths[idx]

        # Format schema
        schema_str = self._format_schema(instance.schemas)

        # Create prompt using template if provided
        if self.template:
            system_message, user_message, assistant_prefix = self.template.create_prompt(
                question=instance.question,
                schema=schema_str,
                dialect=self.dialect,
                evidence=instance.evidence
            )
        else:
            # Fallback if no template provided
            system_message = ""
            user_message = f"Schema:\n{schema_str}\n\nQuestion: {instance.question}"
            assistant_prefix = ""

        return {
            'system_message': system_message,
            'user_message': user_message,
            'assistant_prefix': assistant_prefix,
            'instance_path': instance_path,
            'instance_id': instance.id,
            'question': instance.question,
            'ground_truth_sql': instance.sql,
            'evidence': instance.evidence or "",
            'database_name': instance.database.get('name', 'unknown'),
            'difficulty': instance.difficulty,
            'dataset': instance.dataset
        }


def create_text2sql_dataloader(data_path: str,
                               pattern: str = "instance_*.json",
                               template=None,
                               dialect: str = "SQLite",
                               batch_size: int = 1,
                               shuffle: bool = False,
                               num_workers: int = 0,
                               collate_fn=None) -> DataLoader:
    """
    Create a PyTorch DataLoader for Text2SQL dataset.

    Args:
        data_path: Path to the data directory
        pattern: Glob pattern to match instance files
        template: Prompt template instance (e.g., ArcticText2SQLTemplate)
        dialect: SQL dialect
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        collate_fn: Custom collate function for batching

    Returns:
        PyTorch DataLoader instance
    """
    dataset = Text2SQLDataset(
        data_path=data_path,
        pattern=pattern,
        template=template,
        dialect=dialect
    )

    # Default collate function that handles batching of dictionaries
    if collate_fn is None:
        def default_collate(batch):
            """Default collate function for batching"""
            return {
                key: [item[key] for item in batch]
                for key in batch[0].keys()
            }
        collate_fn = default_collate

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader
