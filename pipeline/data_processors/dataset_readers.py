"""
Unified Dataset Reader and Processor for Text2SQL Datasets
Supports Spider, BIRD, and Spider2 datasets with a common interface
"""

import json
import glob
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from instance import StandardizedInstance
from base_reader import BaseDatasetReader

class SpiderDatasetReader(BaseDatasetReader):
    """Reader for Spider dataset"""
    
    def __init__(self, dataset_path: str, split: str = 'dev'):
        super().__init__(dataset_path, split)
        self.instances_file = self.dataset_path / f"{split}.json"
        self.db_dir = self.dataset_path / "database"
        self.schema_dir = self.dataset_path / f"{split}_schemas"
        
    def process_schemas(self, with_description: bool = False):
        """Process database schemas and save them to CSV files"""
        if not self.schema_dir.exists():
            self.schema_dir.mkdir(parents=True)
        
        db_files = glob.glob(str(self.db_dir / "**/*.sqlite"), recursive=True)
        
        for db_file in db_files:
            db_name = Path(db_file).stem
            output_dir = self.schema_dir / db_name
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{db_name}.csv"
            
            ddl_statements = self.generate_ddl_from_sqlite(db_file)
            self.save_schemas_to_csv(ddl_statements, str(output_path))
            logger.info(f"Processed {db_name} and saved to {output_path}")
    
    def get_database_info(self, instance: Dict) -> Dict[str, Any]:
        """Extract database information from Spider instance"""
        db_name = instance.get('db_id')
        db_path = glob.glob(str(self.db_dir / db_name / "*.sqlite"))
        
        if not db_path:
            raise FileNotFoundError(f"No SQLite files found for database {db_name}")
        
        return {
            'name': db_name,
            'path': db_path,
            'type': 'sqlite'
        }
    
    def get_schema_info(self, instance: Dict) -> List[Dict[str, Any]]:
        """Extract schema information from Spider instance"""
        db_name = instance.get('db_id')
        schema_files = glob.glob(str(self.schema_dir / db_name / "*.csv"))
        
        if not schema_files:
            raise FileNotFoundError(f"No schema files found for database {db_name}")
        
        return [{
            'name': db_name,
            'path': schema_files,
            'type': 'csv'
        }]
    
    def load_instances(self, limit: Optional[int] = None) -> List[StandardizedInstance]:
        """Load and standardize Spider instances"""
        # Process schemas if needed
        if not self.schema_dir.exists():
            logger.info(f"Processing schemas for {self.split} split...")
            self.process_schemas()
        
        # Load instances
        with open(self.instances_file, 'r') as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
        
        standardized_instances = []
        
        for idx, instance in enumerate(data):
            try:
                std_instance = StandardizedInstance(
                    id=idx,
                    dataset='spider',
                    question=instance['question'],
                    sql=instance['query'],
                    database=self.get_database_info(instance),
                    schemas=self.get_schema_info(instance),
                    difficulty=self.calculate_difficulty(instance['query']),
                    evidence=''
                )
                standardized_instances.append(std_instance)
            except Exception as e:
                logger.warning(f"Failed to process instance {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(standardized_instances)} Spider instances")
        return standardized_instances


class BIRDDatasetReader(BaseDatasetReader):
    """Reader for BIRD dataset"""
    
    def __init__(self, dataset_path: str, split: str = 'dev'):
        super().__init__(dataset_path, split)
        self.instances_file = self.dataset_path / f"{split}.json"
        self.db_dir = self.dataset_path / f"{split}_databases"
        self.schema_dir = self.dataset_path / f"{split}_schemas"
    
    def process_schemas(self, with_description: bool = False):
        """Process database schemas and save them to CSV files"""
        if not self.schema_dir.exists():
            self.schema_dir.mkdir(parents=True)
        
        db_files = glob.glob(str(self.db_dir / "**/*.sqlite"), recursive=True)
        
        for db_file in db_files:
            db_name = Path(db_file).stem
            output_dir = self.schema_dir / db_name
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{db_name}.csv"
            
            ddl_statements = self.generate_ddl_from_sqlite(db_file)
            self.save_schemas_to_csv(ddl_statements, str(output_path))
            logger.info(f"Processed {db_name} and saved to {output_path}")
    
    def get_database_info(self, instance: Dict) -> Dict[str, Any]:
        """Extract database information from BIRD instance"""
        db_name = instance.get('db_id')
        db_path = glob.glob(str(self.db_dir / db_name / "*.sqlite"))
        csv_files = glob.glob(str(self.db_dir / db_name / "database_description" / "*.csv"))
        
        if not db_path:
            raise FileNotFoundError(f"No SQLite files found for database {db_name}")
        
        return {
            'name': db_name,
            'path': db_path,
            'csv_files': csv_files,
            'type': 'sqlite'
        }
    
    def get_schema_info(self, instance: Dict) -> List[Dict[str, Any]]:
        """Extract schema information from BIRD instance"""
        db_name = instance.get('db_id')
        schema_files = glob.glob(str(self.schema_dir / db_name / "*.csv"))
        
        if not schema_files:
            raise FileNotFoundError(f"No schema files found for database {db_name}")
        
        return [{
            'name': db_name,
            'path': schema_files,
            'type': 'csv'
        }]
    
    def load_instances(self, limit: Optional[int] = None) -> List[StandardizedInstance]:
        """Load and standardize BIRD instances"""
        # Process schemas if needed
        if not self.schema_dir.exists():
            logger.info(f"Processing schemas for {self.split} split...")
            self.process_schemas()
        
        # Load instances
        with open(self.instances_file, 'r') as f:
            data = json.load(f)
        
        if limit:
            data = data[:limit]
        
        standardized_instances = []
        
        for idx, instance in enumerate(data):
            try:
                std_instance = StandardizedInstance(
                    id=idx,
                    original_instance_id=str(instance.get('question_id', idx)),
                    dataset='bird',
                    question=instance['question'],
                    sql=instance['SQL'],
                    database=self.get_database_info(instance),
                    schemas=self.get_schema_info(instance),
                    difficulty=instance.get('difficulty', 'moderate'),
                    evidence=instance.get('evidence', '')
                )
                standardized_instances.append(std_instance)
            except Exception as e:
                logger.warning(f"Failed to process instance {idx}: {e}")
                continue
        
        logger.info(f"Loaded {len(standardized_instances)} BIRD instances")
        return standardized_instances


class Spider2DatasetReader(BaseDatasetReader):
    """Reader for Spider2-lite dataset"""
    
    def __init__(self, dataset_path: str, dataset_type: str = 'lite'):
        super().__init__(dataset_path, 'dev')
        self.dataset_type = dataset_type
        self.base_path = self.dataset_path / f"spider2-{dataset_type}"
        self.instances_file = self.base_path / f"spider2-{dataset_type}.jsonl"
        self.queries_dir = self.base_path / "evaluation_suite" / "gold" / "sql"
        self.external_knowledge_dir = self.base_path / "resource" / "documents"
        self.sqlite_dir = self.base_path / "resource" / "databases" / "spider2-localdb"
        self.db_schemas = self._load_schema_paths()
    
    def _load_schema_paths(self) -> pd.DataFrame:
        """Load schema paths for different database types"""
        schema_paths = {}
        db_base = self.base_path / "resource" / "databases"
        
        for db_type in ['snowflake', 'sqlite', 'bigquery']:
            db_type_dir = db_base / db_type
            if db_type_dir.exists():
                schema_paths[db_type] = self._get_schemas_for_type(db_type_dir)
        
        return pd.DataFrame.from_dict(schema_paths, orient='index').T
    
    def _get_schemas_for_type(self, db_type_dir: Path) -> Dict[str, List[str]]:
        """Get schema files for a specific database type"""
        schemas = {}
        csv_files = glob.glob(str(db_type_dir / "**/*.csv"), recursive=True)
        
        for csv_file in csv_files:
            parts = Path(csv_file).relative_to(db_type_dir).parts
            if parts:
                db_name = parts[0]
                if db_name not in schemas:
                    schemas[db_name] = []
                schemas[db_name].append(csv_file)
        
        return schemas
    
    def _read_jsonl(self, file_path: Path) -> List[Dict]:
        """Read JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _get_sql_query(self, instance_id: str) -> Optional[str]:
        """Get SQL query for an instance from separate file"""
        sql_file = self.queries_dir / f"{instance_id}.sql"
        if sql_file.exists():
            with open(sql_file, 'r') as f:
                return f.read().strip()
        return None
    
    def _get_external_knowledge(self, filename: str) -> Optional[str]:
        """Get external knowledge from markdown file"""
        if filename:
            knowledge_file = self.external_knowledge_dir / filename
            if knowledge_file.exists():
                with open(knowledge_file, 'r') as f:
                    return f.read().strip()
        return None
    
    def get_database_info(self, instance: Dict) -> Optional[Dict[str, Any]]:
        """Extract database information from Spider2 instance"""
        db_name = instance.get('db')
        if not db_name:
            return None
        
        # Check if database exists in schemas
        if db_name not in self.db_schemas.index:
            logger.warning(f"Database {db_name} not found in schemas")
            return None
        
        db_info = self.db_schemas.loc[db_name].dropna().to_dict()
        if not db_info:
            return None
        
        db_type = list(db_info.keys())[0]
        
        if db_type == 'sqlite':
            sqlite_file = self.sqlite_dir / f"{db_name}.sqlite"
            if sqlite_file.exists():
                return {
                    'name': db_name,
                    'path': str(sqlite_file),
                    'type': 'sqlite'
                }
        else:
            return {
                'name': db_name,
                'path': f'Call the {db_type} API to get the database',
                'type': db_type
            }
        
        return None
    
    def get_schema_info(self, instance: Dict) -> List[Dict[str, Any]]:
        """Extract schema information from Spider2 instance"""
        db_name = instance.get('db')
        if not db_name or db_name not in self.db_schemas.index:
            return []
        
        db_info = self.db_schemas.loc[db_name].dropna().to_dict()
        if not db_info:
            return []
        
        schemas = []
        for db_type, schema_files in db_info.items():
            for schema_file in schema_files:
                schema_name = Path(schema_file).parts[-2]
                schemas.append({
                    'name': schema_name,
                    'path': schema_file
                })
        
        return schemas
    
    def load_instances(self, limit: Optional[int] = None) -> List[StandardizedInstance]:
        """Load and standardize Spider2 instances"""
        # Load instances
        data = self._read_jsonl(self.instances_file)
        
        if limit:
            data = data[:limit]
        
        standardized_instances = []
        
        for idx, instance in enumerate(data):
            instance_id = instance.get('instance_id')
            if not instance_id:
                continue
            
            # Get SQL query
            sql_query = self._get_sql_query(instance_id)
            if not sql_query:
                logger.debug(f"No SQL query found for {instance_id}")
                continue
            
            # Get database info
            db_info = self.get_database_info(instance)
            if not db_info:
                continue
            
            # Normalize question key
            question = instance.get('question') or instance.get('instruction') or instance.get('query')
            if not question:
                continue
            
            # Get external knowledge if available
            evidence = None
            external_knowledge = instance.get('external_knowledge')
            if external_knowledge:
                evidence = self._get_external_knowledge(external_knowledge)
            
            try:
                std_instance = StandardizedInstance(
                    id=idx,
                    original_instance_id=instance_id,
                    dataset=f'spider2-{self.dataset_type}',
                    question=question,
                    sql=sql_query,
                    database=db_info,
                    schemas=self.get_schema_info(instance),
                    difficulty=self.calculate_difficulty(sql_query),
                    evidence=evidence
                )
                standardized_instances.append(std_instance)
            except Exception as e:
                logger.warning(f"Failed to process instance {instance_id}: {e}")
                continue
        
        logger.info(f"Loaded {len(standardized_instances)} Spider2 instances")
        return standardized_instances

