import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from pydantic import Field
from src.models.models import ModelProvider, ModelConfig
from typing import Optional, List, Dict
import warnings


class HuggingFaceConfig(ModelConfig):
    """Configuration for HuggingFace-based models"""
    device: str = Field(default="auto", description="Device to use (cpu, cuda, auto)")
    dtype: str = Field(default="auto", description="Data type for model weights (auto, float16, bfloat16, float32)")
    trust_remote_code: bool = Field(default=True, description="Whether to trust remote code")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences for generation")
    top_k: int = Field(default=50, ge=-1, description="Top-k sampling parameter (0 or None to disable)")
    repetition_penalty: float = Field(default=1.0, ge=1.0, le=2.0, description="Repetition penalty")
    use_cache: bool = Field(default=True, description="Whether to use KV cache for generation")
    do_sample: bool = Field(default=True, description="Whether to use sampling for generation")

    class Config:
        extra = "forbid"


class FrequencyPresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    Custom logits processor to apply frequency and presence penalties.

    Frequency penalty: Penalizes tokens based on how often they appear (linear with count)
    Presence penalty: Penalizes tokens that have appeared at least once (binary)

    Optimized version that uses vectorized operations and avoids CPU-GPU transfers.
    """

    def __init__(self, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, vocab_size: int = None):
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.vocab_size = vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply frequency and presence penalties to the logits.

        Args:
            input_ids: Previously generated token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]

        Returns:
            Modified logits with penalties applied
        """
        # Early exit if no penalties
        if self.frequency_penalty == 0.0 and self.presence_penalty == 0.0:
            return scores

        batch_size, seq_len = input_ids.shape
        vocab_size = scores.shape[-1]

        # Use vectorized operations on GPU/same device as input_ids
        device = input_ids.device

        # Create frequency count tensor [batch_size, vocab_size]
        # This is much faster than Python loops and stays on GPU
        token_counts = torch.zeros((batch_size, vocab_size), dtype=scores.dtype, device=device)

        # Use scatter_add for vectorized counting
        # This counts all tokens in one operation instead of looping
        token_counts.scatter_add_(
            dim=1,
            index=input_ids,
            src=torch.ones_like(input_ids, dtype=scores.dtype)
        )

        # Apply frequency penalty (vectorized)
        if self.frequency_penalty != 0.0:
            scores -= self.frequency_penalty * token_counts

        # Apply presence penalty (vectorized, binary mask)
        if self.presence_penalty != 0.0:
            presence_mask = (token_counts > 0).to(scores.dtype)
            scores -= self.presence_penalty * presence_mask

        return scores


class HuggingFaceProvider(ModelProvider):
    """General-purpose HuggingFace provider supporting multiple model families (Llama, Qwen, Mistral, etc.)"""

    config_class = HuggingFaceConfig

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
        Initialize the HuggingFace model provider.

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

        # Determine device
        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        print(f"Loading model {model_name} on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.trust_remote_code
        )

        # Fix pad token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine dtype
        torch_dtype = self._get_torch_dtype()

        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        # Move to device if needed
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        # Get context length
        self.context_length = getattr(
            self.model.config,
            'max_position_embeddings',
            getattr(self.model.config, 'max_sequence_length', 2048)
        )

        print(f"Model loaded! Context length: {self.context_length}")

        # Setup stop token IDs
        self.stop_token_ids = self._get_stop_token_ids()

    def _get_torch_dtype(self):
        """Get torch dtype from config"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.float16 if self.device == "cuda" else torch.float32
        }
        return dtype_map.get(self.config.dtype, torch.float16 if self.device == "cuda" else torch.float32)

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

    def _get_stop_token_ids(self) -> List[int]:
        """Convert stop sequences to token IDs"""
        stop_token_ids = []
        for stop_seq in self.config.stop_sequences:
            # Encode the stop sequence
            tokens = self.tokenizer.encode(stop_seq, add_special_tokens=False)
            if tokens:
                # For simplicity, we only use single-token stop sequences
                # Multi-token stop sequences would require custom stopping criteria
                if len(tokens) == 1:
                    stop_token_ids.append(tokens[0])

        # Always include EOS token
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)

        return list(set(stop_token_ids))  # Remove duplicates

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

            # Add generation prompt if last message is not assistant
            if messages[-1]["role"] != "assistant":
                prompt_parts.append("<|im_start|>assistant\n")

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

            if messages[-1]["role"] != "assistant":
                prompt_parts.append("Assistant:")

            return "\n\n".join(prompt_parts)

    def _remove_stop_sequences(self, text: str) -> str:
        """Remove stop sequences from generated text"""
        for stop_seq in self.config.stop_sequences:
            if stop_seq in text:
                text = text[:text.index(stop_seq)]
        return text.strip()

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

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.context_length - self.config.max_tokens
        )

        if self.device == "cuda":
            inputs = inputs.to(self.device)

        # Setup logits processors for frequency and presence penalties
        logits_processors = LogitsProcessorList()

        if self.config.frequency_penalty != 0.0 or self.config.presence_penalty != 0.0:
            logits_processors.append(
                FrequencyPresencePenaltyLogitsProcessor(
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty
                )
            )

        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature if self.config.do_sample else None,
                    do_sample=self.config.do_sample,
                    top_p=self.config.top_p if self.config.do_sample else None,
                    top_k=self.config.top_k if self.config.do_sample else None,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.stop_token_ids if self.stop_token_ids else self.tokenizer.eos_token_id,
                    use_cache=self.config.use_cache,
                    logits_processor=logits_processors if len(logits_processors) > 0 else None
                )

            # Decode only the new tokens
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            # Remove stop sequences
            generated_text = self._remove_stop_sequences(generated_text)

            # Combine with prefix if provided
            if assistant_prefix:
                full_response = assistant_prefix + generated_text
            else:
                full_response = generated_text

            return full_response

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error! Try reducing max_tokens or using a smaller model.")
                raise e
            else:
                raise e

    def batch_generate(
        self,
        system_messages: List[str],
        user_messages: List[str],
        assistant_prefixes: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts in a single batch for faster throughput.

        Args:
            system_messages: List of system messages (one per prompt)
            user_messages: List of user messages (one per prompt)
            assistant_prefixes: Optional list of assistant prefixes (one per prompt)

        Returns:
            List of model responses as strings

        Note:
            - All prompts in the batch will be padded to the same length
            - Batch size affects memory usage; reduce if you get OOM errors
            - Much faster than calling generate() in a loop for multiple prompts
        """
        if len(system_messages) != len(user_messages):
            raise ValueError("system_messages and user_messages must have the same length")

        batch_size = len(system_messages)

        if assistant_prefixes is None:
            assistant_prefixes = [""] * batch_size
        elif len(assistant_prefixes) != batch_size:
            raise ValueError("assistant_prefixes must have the same length as system_messages")

        # Format and prepare all prompts
        prompts = []
        for i in range(batch_size):
            messages = self._format_chat_messages(
                system_messages[i],
                user_messages[i],
                assistant_prefixes[i]
            )
            prompt = self._apply_chat_template(messages)
            prompts.append(prompt)

        # Tokenize all prompts with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,  # Pad to the same length
            truncation=True,
            max_length=self.context_length - self.config.max_tokens
        )

        if self.device == "cuda":
            inputs = inputs.to(self.device)

        # Setup logits processors for frequency and presence penalties
        logits_processors = LogitsProcessorList()

        if self.config.frequency_penalty != 0.0 or self.config.presence_penalty != 0.0:
            logits_processors.append(
                FrequencyPresencePenaltyLogitsProcessor(
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty
                )
            )

        # Generate for all prompts in batch
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature if self.config.do_sample else None,
                    do_sample=self.config.do_sample,
                    top_p=self.config.top_p if self.config.do_sample else None,
                    top_k=self.config.top_k if self.config.do_sample else None,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.stop_token_ids if self.stop_token_ids else self.tokenizer.eos_token_id,
                    use_cache=self.config.use_cache,
                    logits_processor=logits_processors if len(logits_processors) > 0 else None
                )

            # Decode each output in the batch
            responses = []
            input_lengths = inputs['attention_mask'].sum(dim=1)  # Get actual input length per sample

            for i in range(batch_size):
                # Decode only the new tokens for this sample
                generated_tokens = outputs[i][input_lengths[i]:]
                generated_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                # Remove stop sequences
                generated_text = self._remove_stop_sequences(generated_text)

                # Combine with prefix if provided
                if assistant_prefixes[i]:
                    full_response = assistant_prefixes[i] + generated_text
                else:
                    full_response = generated_text

                responses.append(full_response)

            return responses

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM error during batch generation! Try reducing batch size or max_tokens.")
                raise e
            else:
                raise e


    def update_generation_config(self, **kwargs):
        """
        Update generation parameters dynamically.

        Args:
            **kwargs: Parameters to update (temperature, top_p, max_tokens, etc.)
        """
        warnings.warn(
            "update_generation_config is deprecated. Use update_config instead.",
            DeprecationWarning
        )
        self.update_config(**kwargs)

        # Update stop token IDs if stop_sequences changed
        if 'stop_sequences' in kwargs:
            self.stop_token_ids = self._get_stop_token_ids()
