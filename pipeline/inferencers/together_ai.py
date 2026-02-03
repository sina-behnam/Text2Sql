import sys

sys.path.append('../..')

import os
from openai import OpenAI
from src.templates.json_template import JSONText2SQLTemplate
from loaders.dataloader import DataLoader, Text2SQLDataset
import argparse
from tqdm import tqdm
import json
from typing import List, Dict, Optional
import time


class TogetherAIRunner:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key or os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )
    
    def generate_single(self, system_message: str, user_message: str, 
                       sampling_params: Dict) -> Dict:
        """Generate response for single instance."""
        try:
            # Build request params, only include non-None values
            request_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            }
            
            # Add optional parameters only if they are not None
            if sampling_params.get('temperature') is not None:
                request_params['temperature'] = sampling_params['temperature']
            if sampling_params.get('max_tokens') is not None:
                request_params['max_tokens'] = sampling_params['max_tokens']
            if sampling_params.get('frequency_penalty') is not None:
                request_params['frequency_penalty'] = sampling_params['frequency_penalty']
            if sampling_params.get('presence_penalty') is not None:
                request_params['presence_penalty'] = sampling_params['presence_penalty']
            if sampling_params.get('response_format') is not None:
                request_params['response_format'] = sampling_params['response_format']
            
            response = self.client.chat.completions.create(**request_params)
            
            return {
                "text": response.choices[0].message.content,
                "finish_reason": response.choices[0].finish_reason
            }
        except Exception as e:
            return {"error": str(e), "text": None}
    
    def generate_batch(self, batched_instances: Dict, sampling_params: Dict) -> Dict:
        """Generate responses for batch sequentially."""
        instances_ids = batched_instances['instance_id'].detach().cpu().tolist()
        user_messages = batched_instances['user_message']
        system_messages = batched_instances['system_message']
        
        batch_responses = {}
        for inst_id, sys_msg, usr_msg in zip(instances_ids, system_messages, user_messages):
            response = self.generate_single(sys_msg, usr_msg, sampling_params)
            batch_responses[inst_id] = response
            time.sleep(0.1)
        
        return batch_responses


def save_batch_responses_json(batches_responses: List[Dict], save_path: str):
    all_responses = {}
    for batch_responses in batches_responses:
        all_responses.update(batch_responses)
    
    with open(save_path, 'w') as f:
        json.dump(all_responses, f, indent=2)
    
    print(f"Saved responses to {save_path}")


def argument_parser():
    parser = argparse.ArgumentParser(description="Together.ai Runner")
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--dialect', type=str, default='sqlite', help='SQL dialect')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle dataset')
    parser.add_argument('--temp', type=float, default=None, help='Temperature (default: API default)')
    parser.add_argument('--fp', type=float, default=None, help='Frequency penalty (default: API default)')
    parser.add_argument('--pp', type=float, default=None, help='Presence penalty (default: API default)')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Max tokens')
    parser.add_argument('--model-name', type=str, required=True, help='Together.ai model name')
    parser.add_argument('--save-dir', type=str, default='./together_results', help='Save directory')
    parser.add_argument('--api-key', type=str, default=None, help='Together.ai API key')
    
    return parser


def run_together(data_path: str, model_name: str, dialect: str = 'sqlite',
                batch_size: int = 4, temperature: Optional[float] = None,
                frequency_penalty: Optional[float] = None, 
                presence_penalty: Optional[float] = None,
                max_tokens: int = 2048,
                save_dir: str = './together_results', shuffle: bool = False,
                num_workers: int = 2, api_key: str = None):
    """Run Together.ai inference."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    prompt_template = JSONText2SQLTemplate()
    dataset = Text2SQLDataset(data_path, template=prompt_template, dialect=dialect)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    runner = TogetherAIRunner(model_name=model_name, api_key=api_key)
    
    sampling_params = {
        'temperature': temperature,
        'frequency_penalty': frequency_penalty,
        'presence_penalty': presence_penalty,
        'max_tokens': max_tokens,
        'response_format': prompt_template.response_format
    }
    
    print("Sampling Parameters:", {k: v for k, v in sampling_params.items() if v is not None})
    
    batches_responses = []
    
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
        batch_responses = runner.generate_batch(batch, sampling_params)
        batches_responses.append(batch_responses)
        
        save_path = os.path.join(save_dir, f"together_responses_batch_{batch_idx}.json")
        save_batch_responses_json([batch_responses], save_path)
    
    save_path = os.path.join(save_dir, "together_responses_all_batches.json")
    save_batch_responses_json(batches_responses, save_path)


if __name__ == "__main__":
    args = argument_parser().parse_args()
    run_together(
        data_path=args.data_path,
        model_name=args.model_name,
        dialect=args.dialect,
        batch_size=args.batch_size,
        temperature=args.temp,
        frequency_penalty=args.fp,
        presence_penalty=args.pp,
        max_tokens=args.max_tokens,
        save_dir=args.save_dir,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        api_key=args.api_key
    )