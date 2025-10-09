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
from dataclasses import dataclass, asdict

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
        """Process database schemas"""
        if not self.schema_dir.exists():
            self.schema_dir.mkdir(parents=True)
        
        db_files = glob.glob(str(self.db_dir / "**/*.sqlite"), recursive=True)
        schemas = {}
        for db_file in db_files:
            db_name = Path(db_file).stem
            
            ddl_statements = self.generate_ddl_from_sqlite(db_file)
            schemas[db_name] = self.shape_schema(ddl_statements, descriptions={}) # ! No descriptions in Spider dataset

        return schemas
 
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

    def get_schemas(self, schemas, instance):
        db_name = instance.get('db_id')
        if db_name in schemas:
            return schemas[db_name]
        else:
            logger.warning(f"For Spider, No schema found for database {db_name}")
            return []
    
    def load_instances(self, limit: Optional[int] = None) -> List[StandardizedInstance]:
        """Load and standardize Spider instances"""
        # Process schemas if needed
        # if not self.schema_dir.exists():
        logger.info(f"Processing schemas for {self.split} split...")
        schemas = self.process_schemas()
        
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
                    schemas=self.get_schemas(schemas,instance),
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

    @staticmethod
    def generate_table_description(df: pd.DataFrame, table_name) -> str:
        """Generate a comprehensive table description of BIRD dataset from the DataFrame of database_secription folder

        Note: Not all columns are described if they are self-explanatory or not useful.

        Args:
            df (pd.DataFrame): DataFrame containing the database description
            table_name (str): Name of the table/database
        Returns:
            str: Formatted table description
        """
        table_description = f'The description of columns of the `{table_name}` that require more information are as follows:\n\n'

        for index, row in df.iterrows():
            original_column_name = row['original_column_name']
            column_name = row['column_name']
            column_description = row['column_description']
            data_format = row['data_format']
            value_description = row['value_description']

            # checking the type of each field to be string to avoid errors
            original_column_name = str(original_column_name) if pd.notna(original_column_name) else ''
            column_name = str(column_name) if pd.notna(column_name) else ''
            column_description = str(column_description) if pd.notna(column_description) else ''
            data_format = str(data_format) if pd.notna(data_format) else ''
            value_description = str(value_description) if pd.notna(value_description) else ''

            desc = ''
            if column_name != '' and (column_name.lower() != original_column_name.lower()) and (column_name.lower() != column_description.lower()):
                desc = f"- {original_column_name} (also known as {column_name}) {': ' + column_description}"
            elif column_description != '' and (column_description.lower() != original_column_name.lower()):
                desc = f"- {original_column_name} : {column_description}"

            if value_description != '' and value_description != 'not useful' and value_description.lower() != column_description.lower():
                if desc == '':
                    desc = f"- {original_column_name}"
                desc += f": Where it means that the value is about : [{value_description}]"

            if desc != '':
                table_description += desc + '\n'

        return table_description
    
    def process_schemas(self, with_description: bool = True):
        """Process database schemas and save them to CSV files"""
        if not self.schema_dir.exists():
            self.schema_dir.mkdir(parents=True)
        
        db_files = glob.glob(str(self.db_dir / "**/*.sqlite"), recursive=True)
        schemas = {}
        for db_file in db_files:
            db_name = Path(db_file).stem

            descriptions = {}
            if with_description:
                # Process table descriptions
                db_secription_path = self.db_dir / db_name / "database_description"  
                csv_files = glob.glob(str(db_secription_path / "*.csv"))
                for csv_file in csv_files:
                    table_name = Path(csv_file).stem
                    try:
                        df = pd.read_csv(csv_file)
                        table_desc = BIRDDatasetReader.generate_table_description(df, table_name)
                        descriptions[table_name] = table_desc
                    except UnicodeDecodeError as e:
                        df = pd.read_csv(csv_file, encoding='latin1')
                        table_desc = BIRDDatasetReader.generate_table_description(df, table_name)
                        descriptions[table_name] = table_desc
                    except Exception as e:
                        logger.error(f"Failed to process description for table {table_name} in database {db_name}: {e}")
                        descriptions[table_name] = ""
                        continue;
                    
            ddl_statements = self.generate_ddl_from_sqlite(db_file)
            schemas[db_name] = self.shape_schema(ddl_statements, descriptions)

        return schemas
    
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

    def get_schemas(self, schemas, instance):
        db_name = instance.get('db_id')
        if db_name in schemas:
            return schemas[db_name]
        else:
            logger.warning(f"For Bird, No schema found for database {db_name}")
            return []
    
    def load_instances(self, limit: Optional[int] = None) -> List[StandardizedInstance]:
        """Load and standardize BIRD instances"""
        # Process schemas if needed
        # if not self.schema_dir.exists():
        logger.info(f"Processing schemas for {self.split} split...")
        schemas = self.process_schemas()
        
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
                    schemas=self.get_schemas(schemas,instance),
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

    @dataclass
    class Spider2SchemaStruct:
        db_name : str
        schema_name : str
        table_name : str
        DDL : str
        type : str
        description : str = ''

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
        
        db_base = self.base_path / "resource" / "databases"
        
        for db_type in ['snowflake', 'sqlite', 'bigquery']:
            db_type_dir = db_base / db_type
            if db_type_dir.exists():
                csv_files = glob.glob(str(db_type_dir / "**/*.csv"), recursive=True)
                json_files = glob.glob(str(db_type_dir / "**/*.json"), recursive=True)
                
                schema_files = {}
                for csv_file in csv_files:
                    if csv_file not in schema_files:
                        schema_files[csv_file] = []
                    for json_file in json_files:
                        if Path(csv_file).parent == Path(json_file).parent:
                            schema_files[csv_file].append(json_file)

                for csv_file, json_list in schema_files.items():
                    db_name = Path(csv_file).relative_to(db_type_dir).parts[0]
                    ddl_statements = pd.read_csv(csv_file).to_string(index=False)
                    for json_file in json_list:
                        table_name = Path(json_file).stem

                        schema_struct = Spider2DatasetReader.Spider2SchemaStruct(
                            db_name=db_name,
                            schema_name=Path(csv_file).parts[-2],
                            table_name=table_name,
                            DDL=ddl_statements,
                            type=db_type,
                            description=''
                        )


        
        
    
    def _get_schemas_for_type(self, db_type_dir: Path) -> Dict[str, List[str]]:
        """Get schema files for a specific database type"""
        schemas = {}
        files = glob.glob(str(db_type_dir / "**/*.csv"), recursive=True)
        json_files = glob.glob(str(db_type_dir / "**/*.json"), recursive=True)
        files.extend(json_files)
        
        for f in files:
            parts = Path(f).relative_to(db_type_dir).parts
            if parts:
                db_name = parts[0]
                if db_name not in schemas:
                    schemas[db_name] = []
                schemas[db_name].append(f)
        
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
    
    def get_schemas(self, schemas : Dict,instance: Dict) -> List[Dict]:
        """Extract schema information from Spider2 instance"""
        db_name = instance.get('db')
        if not db_name or db_name not in self.db_schemas.index:
            return []
        
        db_info = self.db_schemas.loc[db_name].dropna().to_dict()
        if not db_info:
            return []
        
        schemas = {}
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

