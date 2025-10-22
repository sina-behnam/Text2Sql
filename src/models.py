"""
LangChain-based Model Providers for Text2SQL

This module provides a unified interface for different LLM providers using LangChain.
Supports: OpenAI, Anthropic, Together.ai, and local HuggingFace models.
"""

import os
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

# LangChain imports
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, SystemMessage


class ModelProvider(ABC):
    """Base class for LangChain model providers"""

    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize model provider.

        Args:
            model_name: Name/path of the model
            config: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = config or {}
        self.llm: Optional[BaseChatModel] = None

    @abstractmethod
    def _init_llm(self) -> BaseChatModel:
        """Initialize the LangChain LLM instance"""
        pass

    def get_llm(self) -> BaseChatModel:
        """Get the LangChain LLM instance"""
        if self.llm is None:
            self.llm = self._init_llm()
        return self.llm

    def invoke(self, messages: list) -> str:
        """
        Invoke the model with messages.

        Args:
            messages: List of LangChain messages

        Returns:
            Model response as string
        """
        llm = self.get_llm()
        response = llm.invoke(messages)
        return response.content


class OpenAIProvider(ModelProvider):
    """OpenAI model provider using LangChain"""

    def __init__(self, model_name: str, api_key: str = None, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _init_llm(self) -> BaseChatModel:
        """Initialize ChatOpenAI"""
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1024),
            top_p=self.config.get("top_p", 0.95),
            frequency_penalty=self.config.get("frequency_penalty", 0.0),
            presence_penalty=self.config.get("presence_penalty", 0.0)
        )


class AnthropicProvider(ModelProvider):
    """Anthropic (Claude) model provider using LangChain"""

    def __init__(self, model_name: str, api_key: str = None, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    def _init_llm(self) -> BaseChatModel:
        """Initialize ChatAnthropic"""
        return ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1024),
            top_p=self.config.get("top_p", 0.95)
        )


class TogetherAIProvider(ModelProvider):
    """Together.ai model provider using LangChain (via OpenAI-compatible API)"""

    def __init__(self, model_name: str, api_key: str = None, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")

    def _init_llm(self) -> BaseChatModel:
        """Initialize ChatOpenAI with Together.ai base URL"""
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://api.together.xyz/v1",
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 1024),
            top_p=self.config.get("top_p", 0.95),
            frequency_penalty=self.config.get("frequency_penalty", 0.0),
            presence_penalty=self.config.get("presence_penalty", 0.0)
        )


class LocalHuggingFaceProvider(ModelProvider):
    """Local HuggingFace model provider using LangChain"""

    def __init__(self, model_path: str, config: Dict[str, Any] = None):
        super().__init__(model_path, config)
        self.model_path = model_path

    def _init_llm(self) -> BaseChatModel:
        """Initialize HuggingFacePipeline"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        # Determine device
        device = self.config.get("device", "auto")
        if device == "auto":
            device = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            device = 0
        else:
            device = -1  # CPU

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        # Fix pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
            device_map="auto" if device >= 0 else None
        )

        # Create pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config.get("max_new_tokens", 512),
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.95),
            do_sample=True,
            device=device
        )

        return HuggingFacePipeline(pipeline=pipe)


def get_model_provider(model_config: Dict[str, Any]) -> ModelProvider:
    """
    Factory function to create model provider based on config.

    Args:
        model_config: Configuration dictionary with keys:
            - type: "openai", "anthropic", "together_ai", or "local"
            - name/path: Model identifier
            - api_key: API key (for API models)
            - Additional parameters passed to the provider

    Returns:
        Initialized ModelProvider instance
    """
    model_type = model_config.get("type", "openai").lower()

    # Extract common parameters
    config = {
        "temperature": model_config.get("temperature", 0.7),
        "max_tokens": model_config.get("max_tokens", 1024),
        "top_p": model_config.get("top_p", 0.95),
        "frequency_penalty": model_config.get("frequency_penalty", 0.0),
        "presence_penalty": model_config.get("presence_penalty", 0.0)
    }

    if model_type == "openai":
        return OpenAIProvider(
            model_name=model_config.get("name", "gpt-4"),
            api_key=model_config.get("api_key"),
            config=config
        )

    elif model_type == "anthropic":
        return AnthropicProvider(
            model_name=model_config.get("name", "claude-3-5-sonnet-20241022"),
            api_key=model_config.get("api_key"),
            config=config
        )

    elif model_type == "together_ai":
        return TogetherAIProvider(
            model_name=model_config.get("name"),
            api_key=model_config.get("api_key"),
            config=config
        )

    elif model_type == "local":
        config["device"] = model_config.get("device", "auto")
        config["max_new_tokens"] = model_config.get("max_new_tokens", 512)

        return LocalHuggingFaceProvider(
            model_path=model_config.get("path"),
            config=config
        )

    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Available options: 'openai', 'anthropic', 'together_ai', 'local'"
        )
