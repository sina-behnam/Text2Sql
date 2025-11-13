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
        extra = "forbid"

class TogetherAIConfig(ModelConfig):
    """Configuration for Together.ai models"""
    total_limit: int = Field(default=8193, gt=0, description="Total token limit for the model")

class ModelProvider:
    """Base class with config as class attribute"""
    config_class = ModelConfig  # Override in subclasses

    def __init__(self, model_name: str, **config_kwargs):
        self.model_name = model_name
        self.config = self.config_class(**config_kwargs)

    
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
        
    def update_config(self, **config_kwargs):
        """
        Update the model configuration with new parameters.

        Args:
            config_kwargs: Configuration parameters to update
        """
        for key, value in config_kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Invalid config parameter: {key}")

