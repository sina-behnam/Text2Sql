import os
from typing import Dict, List, Tuple, Any, Optional, Union
import openai
import torch
from pydantic import BaseModel, Field

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

class ModelConfig(BaseModel):
    """Base model configuration"""
    seed: int = Field(default=42, description="Random seed for reproducibility")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Nucleus sampling probability")
    max_tokens: int = Field(default=1024, gt=0, description="Maximum tokens to generate")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")

    class Config:
        frozen = True
        extra = "forbid"

class TogetherAIConfig(ModelConfig):
    """Configuration for Together.ai models"""
    total_limit: int = Field(default=8193, gt=0, description="Total token limit for the model")

class ModelProvider:
    """Base class with config as class attribute"""
    config_class = ModelConfig  # Override in subclasses
    
    def __init__(self, model_name: str, config: Optional[ModelConfig] = None, **config_kwargs):
        self.model_name = model_name
        
        # Use provided config, or create from kwargs, or use defaults
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = self.config_class(**config_kwargs)
        else:
            self.config = self.config_class()  # Use defaults

    
    def generate(self, system_message: str, user_message: str, assistant_message: str = "") -> str:
        """
        Generate a response from the model.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def judge(self, user_message: str, judge_model : str = 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free', api_key : str=None) -> str:
        """
        LLM as a judge
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "To use the Together.ai provider, you need to install the openai package: "
                "pip install openai"
            )
        
        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key or os.getenv("TOGETHER_API_KEY")
        )
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        return response.choices[0].message.content
        
    
class TogetherAIProvider(ModelProvider):
    config_class = TogetherAIConfig  # Override!
    
    def __init__(self, model_name: str, api_key: str = None, config: Optional[TogetherAIConfig] = None, **config_kwargs):
        super().__init__(model_name, config=config, **config_kwargs)
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        # Now use self.config.temperature, self.config.total_limit, etc.

        # Initialize OpenAI client with Together.ai API
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
        )
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the Together.ai API.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        input_text = system_message + user_message
        estimated_input_tokens = len(input_text) // 4  # rough estimate
        max_tokens = self._model_param.max_tokens
        # Leave buffer and adjust max_tokens if needed
        available_tokens = 8100 - estimated_input_tokens - 100  # 100 token buffer
        actual_max_tokens = min(max_tokens, available_tokens)
        actual_max_tokens = max(actual_max_tokens, 50)  # Minimum 50
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens= actual_max_tokens,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=self._model_param.temperature,
            top_p=self._model_param.top_p,
            frequency_penalty=self._model_param.frequency_penalty,
            presence_penalty=self._model_param.presence_penalty,
        )
        
        return response.choices[0].message.content
    
class LocalHuggingFaceProvider(ModelProvider):
    """Improved provider for local HuggingFace models with proper configuration"""
    
    def __init__(self, model_path: str, device: str = "auto", max_new_tokens: int = 512, 
                 trust_remote_code: bool = True, extended_thinking: bool = False):
        """
        Initialize the local HuggingFace provider.
        
        Args:
            model_path: Path or name of the model
            device: Device to use ("cpu", "cuda", "auto")
            max_new_tokens: Maximum number of tokens to generate
            trust_remote_code: Whether to trust remote code
            extended_thinking: Enable Chain-of-Thought reasoning
        """

                # Optional imports for local models
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        except ImportError:
            raise ImportError(
                "To use local HuggingFace models, you need to install the transformers package: "
                "pip install transformers"
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.extended_thinking = extended_thinking
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model {model_path} on {self.device}...")
                
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=trust_remote_code
        )
        
        # Fix pad token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with proper device mapping
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None
        }
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Move to device if needed
        if self.device == "cpu":
            self.model = self.model.to(self.device)
            
        # Get context length
        self.context_length = getattr(self.model.config, 'max_position_embeddings', 
                                    getattr(self.model.config, 'max_sequence_length', 2048))
        
        print(f"Model loaded! Context length: {self.context_length}")
    
    def _format_prompt(self, system_message: str, user_message: str) -> str:
        """Format prompt using model's chat template or fallback"""
        
        # Try to use model's chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                pass  # Fall back to manual formatting
        
        # ! DOUBLE CHECK THIS PART ! #
        # Manual formatting based on model type 
        model_name_lower = self.model_path.lower()
        
        if "llama" in model_name_lower and "chat" in model_name_lower:
            return f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message} [/INST]"
        elif "mistral" in model_name_lower or "mixtral" in model_name_lower:
            return f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
        elif "gemma" in model_name_lower:
            return f"<start_of_turn>user\n{system_message}\n\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
        elif "qwen" in model_name_lower:
            return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Generic format
            return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
    
    def _generate_text(self, prompt: str, max_tokens: int = None) -> str:
        """Generate text with proper token management"""
        
        max_tokens = max_tokens or self.max_new_tokens
        
        # Tokenize and check length
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=self.context_length - max_tokens)
        
        if self.device == "cuda":
            inputs = inputs.to(self.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
            
            # Decode only the new tokens
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Reduce max_tokens and retry
                reduced_tokens = max_tokens // 2
                print(f"OOM error, retrying with {reduced_tokens} tokens...")
                return self._generate_text(prompt, reduced_tokens)
            else:
                raise e
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate response with optional extended thinking
        
        Args:
            system_message: System message to guide behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        
        if self.extended_thinking:
            # Chain-of-Thought approach
            thinking_system = (
                "You are an expert assistant. Think step by step before giving your final answer. "
                "First, analyze the problem and show your reasoning process."
            )
            
            # Step 1: Generate thinking
            thinking_prompt = self._format_prompt(
                thinking_system,
                f"Think step by step about this problem:\n\n{user_message}\n\nLet me think..."
            )
            
            thinking_response = self._generate_text(thinking_prompt, 150)
            
            # Step 2: Generate final answer with thinking context
            final_system = system_message + "\n\nUse the thinking process below to give your final answer."
            final_user = f"Problem: {user_message}\n\nThinking process: {thinking_response}\n\nFinal answer:"
            
            final_prompt = self._format_prompt(final_system, final_user)
            final_response = self._generate_text(final_prompt)
            
            # Format with thinking tags
            return f"<think>\n{thinking_response}\n</think>\n\n{final_response}"
        
        else:
            # Standard generation
            prompt = self._format_prompt(system_message, user_message)
            return self._generate_text(prompt)