"""
This module defines the configuration and supported models, datasets, worker engines, and etc.

The idea is that the models are mainly two types: local models and remote models.

If any new model wanted to be added, You should also update the following Enums accordingly. Otherwise,
the database storage may fail to find the correct model name or type. !! 
"""

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel, Field

PROJECT_PATH = '/home/sina/Projects/Thesis/Text2Sql/'
BIRD_DATA_PATH = PROJECT_PATH + '/Data/v3_claude/bird_set_stratified'
SPIDER_DATA_PATH = PROJECT_PATH + '/Data/v3_claude/spider_set_stratified'

class SupportedModels(Enum):
    META_LLAMA3_INS_8B = 'Llama-3.1-8B-Instruct'
    OMNISQL_7B = 'seeklhy-OmniSQL-7B-v2'
    ARCTIC_R1_7B = 'Arctic-Text2SQL-R1-7B'

class SupportedRefrenceModels(Enum):
    QWEN_CODER_480B = 'Qwen3-Coder-480B-A35B-Instruct'

class ModelConfigParametersSet(Enum):
    TEMPERATURES = [0.0, 0.5, 1.0]
    FP_PP = [
        (0.0, 0.0),
        (0.0, 0.3),
        (0.2, 0.1),
        (0.5, 0.0)
    ]

def _load_default_mongo_uri() -> str:
    with open(PROJECT_PATH + 'Data/Auth/mongo.remote.uri/uri.key', 'r') as f:
        return f.read().strip()

class MongoDBConfig(BaseModel):
    db_name: str = Field('model_inference_results')
    collection_name: str = Field('inference_results')
    db_uri: str = Field(default_factory=_load_default_mongo_uri)

    def get_db_uri(self) -> str:
        return self.db_uri

    
class WorkerEngines(Enum):
    '''
    This is only for local model inference engines !
    '''
    VLLM = 'vllm'
    TRANSFORMERS = 'transformers' # some features like logprobs and penalties may not be supported ! 

class SupportedAPILibs(Enum):
    # TOGETHER_AI = 'together_ai'
    OPENAI = 'openai'

class SupportedRemoteOperators(Enum):
    TOGETHER_AI = 'togetherai'
    OPENROUTE = 'openroute'

class Datasets(Enum):
    SPIDER = 'spider'
    BIRD = 'bird'

class Dialects(Enum):
    SQLITE = 'sqlite'
    SNOWFLAKE = 'snowflake'

# --------
JUDGE_SYSTEM_MESSAGE = (
            "You are a SQL expert tasked with determining if two SQL queries are semantically equivalent. "
            "This means they may have syntactic differences but would return the same results when executed "
            "on the same database. Common acceptable differences include: "
            "- Different column ordering in SELECT statements "
            "- Presence or absence of column aliases (AS) "
            "- Different formatting, spacing, or capitalization "
            "- Use of quotes around identifiers "
            "- Simple reordering of conditions that doesn't change the logic "
            "\n\nYour response must be in JSON format with two fields: "
            "'equivalent' (true/false) and 'explanation' (a brief explanation of your judgment)."
        )