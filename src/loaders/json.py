from typing import List, Dict, Any
import json
from src.loaders.base import BaseLoader

class JSONFileDataLoader(BaseLoader):
    
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _conversion_format(self, data: Dict[str, Any]) -> Dict[int, Any]:
        converted_data = {}
        for key, value in data.items():
            converted_data[int(key)] = value
        return converted_data

    def load_data(self) -> Dict[str, Any]:
        """Load data from a JSON file."""
        with open(self.file_path, 'r') as f:
            data = json.load(f)
        return self._conversion_format(data)