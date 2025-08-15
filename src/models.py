import os
from typing import Dict, List, Tuple, Any, Optional, Union
import openai
import torch
# Optional import for Anthropic API
try:
    from anthropic import Anthropic, NOT_GIVEN
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Optional imports for local models
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Set your API key
API_KEY = 'your_api_key_here'
os.environ["TOGETHER_API_KEY"] = API_KEY

class ModelProvider:
    """Base class for model providers (API or local)"""
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response from the model.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        raise NotImplementedError("Subclasses must implement this method")

class TogetherAIProvider(ModelProvider):
    """Provider for Together.ai API-based models"""
    
    def __init__(self, model_name: str, api_key: str = None, max_tokens: int = 1024):
        """
        Initialize the Together.ai provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for Together.ai
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY", API_KEY)
        self.max_tokens = max_tokens
        self.total_limit = 8193  # Actual Together.ai limit
        
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
        
        # Leave buffer and adjust max_tokens if needed
        available_tokens = 8100 - estimated_input_tokens - 100  # 100 token buffer
        actual_max_tokens = min(self.max_tokens, available_tokens)
        actual_max_tokens = max(actual_max_tokens, 50)  # Minimum 50
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens= actual_max_tokens,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        
        return response.choices[0].message.content

class OpenAIProvider(ModelProvider):
    """Provider for OpenAI API-based models"""
    
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize the OpenAI provider.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for OpenAI
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key
        )
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        
        return response.choices[0].message.content
    
class AnthropicProvider(ModelProvider):
    """Provider for Anthropic API models (Claude)"""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None, max_tokens: int = 1024, extended_thinking: bool = False):
        """
        Initialize the Anthropic provider.
        
        Args:
            model_name: Name of the Claude model to use (e.g., "claude-3-opus-20240229")
            api_key: API key for Anthropic
            max_tokens: Maximum number of tokens to generate
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "To use Anthropic models, you need to install the anthropic package: "
                "pip install anthropic"
            )
            
        self.model_name = model_name
        self.api_key = api_key 
        self.max_tokens = max_tokens
        self.extended_thinking = extended_thinking
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key is required. Please provide it as a parameter "
                "or set the ANTHROPIC_API_KEY environment variable."
            )
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)

    @staticmethod
    def get_output_response(response):
        thinking_messages = None
        response_messages = None

        for block in response.content:
            if block.type == "thinking":
                thinking_messages = block.thinking
            elif block.type == "redacted_thinking":
                thinking_messages = 'IT IS REDACTED'
            elif block.type == "text":
                response_messages = block.text

        message = f'<think>\n{thinking_messages}\n</think>\n\n' if thinking_messages else ''
        message += response_messages

        return message
    
    def generate(self, system_message: str, user_message: str) -> str:
        """
        Generate a response using the Anthropic API.
        
        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            
        Returns:
            Model's response as a string
        """
        try:
            # Create message using Anthropic API format
            response = self.client.messages.create(
                model=self.model_name,
                system=system_message,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000,
                } if self.extended_thinking else NOT_GIVEN,
                messages=[
                    {"role": "user", "content": user_message},
                ],
                max_tokens= 4000 if self.extended_thinking else 1024, # It always should be higher than the budget tokens for thinking
            )
            
            # Extract the response message
            return AnthropicProvider.get_output_response(response)
    
        except Exception as e:
            # Handle API errors
            error_message = f"Anthropic API error: {str(e)}"
            print(error_message)
            return error_message

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