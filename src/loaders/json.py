from typing import List, Dict, Any
import json
from src.loaders.base import BaseLoader

class JSONFileDataLoader(BaseLoader):
    
    def __init__(self, files_dir: str):
        self.files_dir = files_dir

    def _conversion_format(self, data: Dict[str, Any]) -> Dict[int, Any]:
        converted_data = {}
        for key, value in data.items():
            converted_data[int(key)] = {'text': value['text']}
        return converted_data

    def load_data(self, model, dataset, 
                  temperature, frequency_penalty, presence_penalty,
                  prefix: str,
                  postfix: str,
                  appendix: str,
                  ) -> Dict[str, Any]:

        # finding
        model_path = self.files_dir + f'/{model}/{prefix}' if prefix else self.files_dir + f'/{model}'
        config_path = model_path + f'/{dataset}/temp_{float(temperature)}_fp_{float(frequency_penalty)}_pp_{float(presence_penalty)}/'
        post_path = config_path + f'{postfix}/{appendix}' if postfix else config_path + f'/{appendix}'
        
        
        with open(post_path, 'r') as f:
            data = json.load(f)

        return self._conversion_format(data)  



        