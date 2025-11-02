import os
from typing import Optional
import openai
from pydantic import Field
from src.models.models import ModelProvider, ModelConfig


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI-compatible APIs"""
    total_limit: int = Field(default=8193, gt=0, description="Total token limit for the model")
    stop_sequences: list[str] = Field(default_factory=list, description="Stop sequences for generation")


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI and OpenAI-compatible APIs (e.g., vLLM servers, LocalAI, etc.)"""

    config_class = OpenAIConfig

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
        **config_kwargs
    ):
        """
        Initialize the OpenAI provider.

        Args:
            model_name: Name of the model to use (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: API key for authentication. If None, reads from OPENAI_API_KEY env var
            base_url: Base URL for the API. If None, uses default OpenAI endpoint
                     Examples:
                     - "https://api.openai.com/v1" (OpenAI)
                     - "http://localhost:8000/v1" (vLLM local server)
                     - "https://api.together.xyz/v1" (Together.ai)
            config: Optional OpenAIConfig instance
            config_kwargs: Additional keyword arguments for configuration
        """
        super().__init__(model_name, config=config, **config_kwargs)

        # Handle config initialization (same pattern as Arctic)
        if config is not None:
            self.config = config
        elif config_kwargs:
            self.config = OpenAIConfig(**config_kwargs)
        else:
            self.config = OpenAIConfig()

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided either through the api_key parameter "
                "or OPENAI_API_KEY environment variable"
            )

        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self.base_url = base_url or "https://api.openai.com/v1"

    def generate(self, system_message: str, user_message: str, assistant_prefix: str = "") -> str:
        """
        Generate a response using the OpenAI-compatible API.

        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            assistant_prefix: Optional prefix for the assistant's response

        Returns:
            Model's response as a string
        """
        try:
            # Estimate input tokens (rough estimate: 4 chars ≈ 1 token)
            input_text = system_message + user_message + assistant_prefix
            estimated_input_tokens = len(input_text) // 4

            # Calculate available tokens
            available_tokens = self.config.total_limit - estimated_input_tokens - 100  # 100 token buffer
            actual_max_tokens = min(self.config.max_tokens, available_tokens)
            actual_max_tokens = max(actual_max_tokens, 50)  # Minimum 50 tokens

            # Prepare messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

            # Add assistant prefix if provided
            if assistant_prefix:
                messages.append({"role": "assistant", "content": assistant_prefix})

            # Build API call parameters
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": actual_max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "seed": self.config.seed,
            }

            # Add stop sequences if provided
            if self.config.stop_sequences:
                api_params["stop"] = self.config.stop_sequences

            # Make API call
            response = self.client.chat.completions.create(**api_params)

            generated_text = response.choices[0].message.content

            # If assistant_prefix was provided, prepend it to the response
            if assistant_prefix:
                return assistant_prefix + generated_text

            return generated_text

        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Error during generation: {e}") from e

    def generate_batch(self, prompts: list[tuple[str, str, str]]) -> list[str]:
        """
        Generate responses for multiple prompts (sequential for now).

        Args:
            prompts: List of (system_message, user_message, assistant_prefix) tuples

        Returns:
            List of generated responses
        """
        results = []
        for system_msg, user_msg, assistant_prefix in prompts:
            result = self.generate(system_msg, user_msg, assistant_prefix)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model_name}, base_url={self.base_url})"
