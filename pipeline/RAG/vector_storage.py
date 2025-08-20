import os
import numpy as np
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import certifi
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer
from pymongo.operations import SearchIndexModel
import time
import json
import glob

class Text2SQLVectorDB:
    """Store Text2SQL instances as vector embeddings in MongoDB for semantic search"""
    
    def __init__(
        self, 
        mongodb_uri: str = "mongodb://localhost:27017/",
        db_name: str = "text2sql_vectordb",
        collection_name: str = "instances",
        embedding_model: str = "all-MiniLM-L6-v2",
        root_data_dir: str = "/path/to/data"
    ):
        """
        Initialize the Text2SQL Vector Database
        
        Args:
            mongodb_uri: URI for MongoDB connection
            db_name: Name of the database to use
            collection_name: Name of the collection to store embeddings
            embedding_model: SentenceTransformer model to use for embeddings
        """
        # Set up MongoDB connection
        self.client = MongoClient(mongodb_uri , server_api=ServerApi('1'), tlsCAFile=certifi.where())
        self.db = self.client[db_name]
        self.collection_name = collection_name
        self.collection = self.db[collection_name]
        self.root_data_dir = root_data_dir
        
        # Initialize embedding model
        self.model = SentenceTransformer(embedding_model)
        self.dimensions = self.model.get_sentence_embedding_dimension()


    def create_collection(self):
        # Create collection if it doesn't exist
        if self.collection not in self.db.list_collection_names():
            self.db.create_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' created.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")
        # Create vector index if it doesn't exist
        if "vector_index" not in [idx.get("name") for idx in self.collection.list_search_indexes()]:
            self._create_vector_index()
        else:
            print("Vector index already exists.")

    def _create_vector_index(self):
        # Set up MongoDB collection with vector index
        search_index_model = SearchIndexModel(
            definition={
                "fields" : [{
                    "type" : "vector",
                    "path" : "embedding",
                    "numDimensions" : self.dimensions,
                    "similarity" : "euclidean",
                }]
            },
            name="vector_index",
            type="vectorSearch"
        )
        
        result = self.collection.create_search_index(model=search_index_model)
        print("New search index named " + result + " is building.")

        # Poll until the index is ready for queries
        print("Polling to check if the index is ready. This may take up to 60 seconds.")
        while True:
            # list_search_indexes returns index definitions; filter by name
            infos = [idx for idx in self.collection.list_search_indexes() if idx.get("name") == result]
            if infos and infos[0].get("queryable"):
                break
            time.sleep(5)
        print(f"Index '{result}' is ready for querying.")

    def _create_instance_text(self, instance: Dict) -> str:
        """
        Create a single text representation of the instance for embedding
        
        Args:
            instance: Text2SQL instance data
            
        Returns:
            Combined text representation of question, schema, and evidence
        """
        # Extract question, schema, and evidence
        question = instance.get('question', '')
        evidence = instance.get('evidence', '')
        
        # Convert schema to text if available
        schema_info = instance.get('schemas', {})
        dataset_name = instance.get('dataset', 'unknown')
        schema_text = Text2SQLVectorDB.process_schema(self.root_data_dir, schema_info, dataset_name)
            
        # Combine all elements
        combined_text = f"Question: {question}\nEvidence: {evidence}\n{schema_text}"
        return combined_text
    
    @staticmethod
    def _match_data_path_dataset(root_data_dir,dataset_name: str, requested_path) -> str:
        """
        Match the requested path to the dataset type directory.
        Currently supports 'bird' and 'spider' datasets.
        This function is mainly use for loading schema information and database files.
        Args:
            root_data_dir: Root directory for the dataset
            dataset_name: Name of the dataset (e.g., 'bird', 'spider')
            requested_path: Path to the data file
        Returns:
            Full path to the requested data file
        """

        if dataset_name == 'bird':
            return os.path.join(root_data_dir, 'bird_subset', requested_path)
        elif dataset_name == 'spider':
            return os.path.join(root_data_dir, 'spider_subset', requested_path)
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}. Supported datasets are 'bird' and 'spider'.")    

    @staticmethod
    def process_schema(root_data_dir : str,schema_info: Dict, dataset_name : str) -> str:
        """
        Process schema information into a text representation for embedding.
        
        Args:
            root_data_dir: Root directory for the dataset
            schema_info: Schema information from the database
            dataset_name: Name of the dataset (e.g., 'bird', 'spider')
            
        Returns:
            String representation of the schema
        """
        if not schema_info:
            return ""
            
        schema_text = ""
        
        # Process schema information if it's available as a path
        schemas_path = schema_info.get('path', [])
        for schema_path in schemas_path:
            # Try to load schema from the path
            schema_path = Text2SQLVectorDB._match_data_path_dataset(root_data_dir,dataset_name, schema_path)

            try:
                if os.path.exists(schema_path): 
                    schema_df = pd.read_csv(schema_path)
                    
                    # Extract relevant columns: table_name, description, DDL
                    for _, row in schema_df.iterrows():
                        table_name = row.get('table_name', '')
                        description = row.get('description', '')
                        ddl = row.get('DDL', '')
                        
                        if table_name:
                            schema_text += f"Table: {table_name} "
                        if description:
                            schema_text += f"Description: {description} "
                        if ddl:
                            schema_text += f"DDL: {ddl} "
                else:
                    print(f"Schema file not found: {schema_path}")
                    continue
            except Exception as e:
                print(f"Error loading schema from {schema_path}: {e}")
                continue
            
        return schema_text
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding vector for the provided text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def store_instance(self, instance: Dict) -> str:
        """
        Create and store embedding for a Text2SQL instance
        
        Args:
            instance: Text2SQL instance data
            
        Returns:
            ID of the inserted document
        """
        # Create text representation
        instance_text = self._create_instance_text(instance)
        
        # Generate embedding
        embedding = self.create_embedding(instance_text)
        
        # Prepare document for MongoDB
        document = {
            "_id": instance.get('id'),
            "embedding": embedding,
            "original_instance": instance
        }
        
        self.collection.replace_one(
            {"_id": instance.get('id')},
            document,
            upsert=True
        )
        return str(instance.get('id'))
    
    def find_similar_instances(self, query_text: str, limit: int = 5) -> List[Dict]:
        """
        Find instances similar to the provided query text
        
        Args:
            query_text: Text to find similar instances for
            limit: Maximum number of similar instances to return
            
        Returns:
            List of similar instances with similarity scores
        """
        # Create embedding for query
        query_embedding = self.create_embedding(query_text)
        
        # Use MongoDB's vector search capability
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "instance_id": 1,
                    "original_instance": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return results

