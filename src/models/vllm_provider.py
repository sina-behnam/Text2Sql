from vllm import LLM, SamplingParams
from pydantic import Field
from src.models.models import ModelProvider, ModelConfig
from typing import Optional, List, Dict
import re
import warnings

class VLLMConfig(ModelConfig):
    """Configuration for vLLM-based models"""
    gpu_memory_utilization: float = Field(default=0.9, ge=0.1, le=1.0, description="GPU memory utilization for vLLM")
    tensor_parallel_size: int = Field(default=1, gt=0, description="Tensor parallel size for vLLM")
    dtype: str = Field(default="auto", description="Data type for model weights (auto, float16, bfloat16, float32)")
    max_model_len: Optional[int] = Field(default=None, description="Maximum model context length (None for model default)")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences for generation")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code")
    enforce_eager: bool = Field(default=False, description="Use eager execution instead of CUDA graphs")

    class Config:
        extra = "forbid"


class VLLMProvider(ModelProvider):
    """General-purpose vLLM provider supporting multiple model families (Llama, Qwen, Mistral, etc.)"""

    config_class = VLLMConfig

    # Default stop sequences for different model families
    DEFAULT_STOP_SEQUENCES = {
        'qwen': ['<|im_end|>', '<|endoftext|>'],
        'llama': ['</s>'],
        'mistral': ['</s>'],
        'gemma': ['<end_of_turn>', '<eos>'],
        'phi': ['<|endoftext|>', '<|end|>'],
        'default': []
    }

    def __init__(self, model_name: str, **config_kwargs):
        """
        Initialize the vLLM model provider.

        Args:
            model_name: Name or path of the pre-trained model
            config_kwargs: Configuration parameters
        """
        super().__init__(model_name, **config_kwargs)

        # Detect model family
        self.model_family = self._detect_model_family(model_name)

        # Set default stop sequences if not provided
        if not self.config.stop_sequences:
            stop_sequences = self.DEFAULT_STOP_SEQUENCES.get(
                self.model_family,
                self.DEFAULT_STOP_SEQUENCES['default']
            )
            self.config.stop_sequences = stop_sequences

        # Initialize vLLM engine
        vllm_kwargs = {
            'model': model_name,
            'tensor_parallel_size': self.config.tensor_parallel_size,
            'gpu_memory_utilization': self.config.gpu_memory_utilization,
            'trust_remote_code': self.config.trust_remote_code,
            'dtype': self.config.dtype,
        }

        if self.config.max_model_len is not None:
            vllm_kwargs['max_model_len'] = self.config.max_model_len

        if self.config.enforce_eager:
            vllm_kwargs['enforce_eager'] = True

        self.llm = LLM(**vllm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()

        # Setup sampling parameters
        self.sampling_params = SamplingParams(
            seed=self.config.seed,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty
        )

        if self.config.stop_sequences:
            self.sampling_params.stop = self.config.stop_sequences

    def _detect_model_family(self, model_name: str) -> str:
        """Detect the model family from the model name"""
        model_name_lower = model_name.lower()

        if 'qwen' or 'arctic' in model_name_lower:
            return 'qwen'
        elif 'llama' in model_name_lower:
            return 'llama'
        elif 'mistral' in model_name_lower or 'mixtral' in model_name_lower:
            return 'mistral'
        elif 'gemma' in model_name_lower:
            return 'gemma'
        elif 'phi' in model_name_lower:
            return 'phi'
        else:
            return 'default'

    def _format_chat_messages(self, system_message: str, user_message: str,
                             assistant_prefix: str = "") -> List[Dict[str, str]]:
        """Format messages into chat format"""
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        if assistant_prefix:
            messages.append({"role": "assistant", "content": assistant_prefix})

        return messages

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply chat template based on model family.
        First tries to use tokenizer's chat template, then falls back to manual formatting.
        """
        # Try using tokenizer's built-in chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return prompt
            except Exception as e:
                print(f"Warning: Failed to use tokenizer chat template: {e}")
                print("Falling back to manual formatting...")

        # Manual formatting based on model family
        return self._manual_format_chat(messages)

    def _manual_format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Manually format chat messages based on model family"""

        if self.model_family == 'qwen':
            # Qwen chat format
            prompt_parts = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            return "\n".join(prompt_parts)

        elif self.model_family == 'llama':
            # Llama chat format
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            if assistant_msg:
                return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]{assistant_msg}"
            else:
                return f"<s>[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]"

        elif self.model_family == 'mistral':
            # Mistral chat format
            system_msg = messages[0]["content"] if messages[0]["role"] == "system" else ""
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            if assistant_msg:
                return f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]{assistant_msg}"
            else:
                return f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]"

        elif self.model_family == 'gemma':
            # Gemma chat format
            prompt_parts = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                if role == "user":
                    prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
                elif role == "assistant":
                    prompt_parts.append(f"<start_of_turn>model\n{content}")
                else:  # system
                    prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")

            if messages[-1]["role"] != "assistant":
                prompt_parts.append("<start_of_turn>model\n")

            return "\n".join(prompt_parts)

        elif self.model_family == 'phi':
            # Phi chat format
            prompt_parts = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                prompt_parts.append(f"<|{role}|>\n{content}")

            if messages[-1]["role"] != "assistant":
                prompt_parts.append("<|assistant|>\n")

            return "\n".join(prompt_parts)

        else:
            # Generic format
            prompt_parts = []
            for message in messages:
                role = message["role"].capitalize()
                content = message["content"]
                prompt_parts.append(f"{role}: {content}")

            prompt_parts.append("Assistant:")
            return "\n\n".join(prompt_parts)

    def generate(self, system_message: str, user_message: str, assistant_prefix: str = "") -> str:
        """
        Generate a response from the model.

        Args:
            system_message: System message to guide the model's behavior
            user_message: User message with the actual prompt
            assistant_prefix: Optional prefix for the assistant's response

        Returns:
            Model's response as a string
        """
        # Format messages
        messages = self._format_chat_messages(system_message, user_message, assistant_prefix)

        # Apply chat template
        prompt = self._apply_chat_template(messages)

        # Generate
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Combine with prefix if provided
        if assistant_prefix:
            full_response = assistant_prefix + generated_text
        else:
            full_response = generated_text

        return full_response

    def generate_batch(self, prompts: List[tuple[str, str]], assistant_prefixes: Optional[List[str]] = None) -> List[str]:
        """
        Generate responses for multiple prompts in batch (much faster!).

        Args:
            prompts: List of (system_message, user_message) tuples
            assistant_prefixes: Optional list of assistant prefixes for each prompt

        Returns:
            List of generated responses
        """
        if assistant_prefixes is None:
            assistant_prefixes = [""] * len(prompts)

        # Format all prompts
        formatted_prompts = []
        for (system_msg, user_msg), prefix in zip(prompts, assistant_prefixes):
            messages = self._format_chat_messages(system_msg, user_msg, prefix)
            prompt = self._apply_chat_template(messages)
            formatted_prompts.append(prompt)

        # Batch generate
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)

        # Extract and combine results
        results = []
        for output, prefix in zip(outputs, assistant_prefixes):
            generated_text = output.outputs[0].text
            if prefix:
                full_response = prefix + generated_text
            else:
                full_response = generated_text
            results.append(full_response)

        return results

    def update_sampling_params(self, **kwargs):
        """Update sampling parameters dynamically"""
        warnings.warn(
            "update_sampling_params is deprecated. Use update_config instead.",
            DeprecationWarning
        )
        self.update_config(**kwargs)

        # Recreate sampling params with updated config
        self.sampling_params = SamplingParams(
            seed=self.config.seed,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            presence_penalty=self.config.presence_penalty,
            frequency_penalty=self.config.frequency_penalty
        )

        if self.config.stop_sequences:
            self.sampling_params.stop = self.config.stop_sequences
        

