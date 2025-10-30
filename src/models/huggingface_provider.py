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
        frozen = True
        extra = "forbid"


class FrequencyPresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    Custom logits processor to apply frequency and presence penalties.

    Frequency penalty: Penalizes tokens based on how often they appear (linear with count)
    Presence penalty: Penalizes tokens that have appeared at least once (binary)
    """

    def __init__(self, frequency_penalty: float = 0.0, presence_penalty: float = 0.0):
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.token_counts = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply frequency and presence penalties to the logits.

        Args:
            input_ids: Previously generated token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]

        Returns:
            Modified logits with penalties applied
        """
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            # Count token frequencies in the current sequence
            token_counts = {}
            for token_id in input_ids[batch_idx].tolist():
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

            # Apply penalties
            for token_id, count in token_counts.items():
                if token_id < scores.shape[-1]:  # Safety check
                    # Frequency penalty: proportional to count
                    if self.frequency_penalty != 0.0:
                        scores[batch_idx, token_id] -= self.frequency_penalty * count

                    # Presence penalty: binary (token appeared or not)
                    if self.presence_penalty != 0.0:
                        scores[batch_idx, token_id] -= self.presence_penalty

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

    def __init__(self, model_name: str, config: Optional[HuggingFaceConfig] = None, **config_kwargs):
        """
        Initialize the HuggingFace model provider.

        Args:
            model_name: Name or path of the pre-trained model
            config: Optional HuggingFaceConfig instance
            config_kwargs: Additional keyword arguments for configuration
        """
        super().__init__(model_name, config=config, **config_kwargs)

        # Detect model family
        self.model_family = self._detect_model_family(model_name)

        # Set default stop sequences if not provided
        if not self.config.stop_sequences:
            stop_sequences = self.DEFAULT_STOP_SEQUENCES.get(
                self.model_family,
                self.DEFAULT_STOP_SEQUENCES['default']
            )
            # Create a new config with stop sequences
            config_dict = self.config.model_dump()
            config_dict['stop_sequences'] = stop_sequences
            self.config = HuggingFaceConfig(**config_dict)

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

        if 'qwen' in model_name_lower:
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
                    

    def update_generation_config(self, **kwargs):
        """
        Update generation parameters dynamically.

        Args:
            **kwargs: Parameters to update (temperature, top_p, max_tokens, etc.)
        """
        config_dict = self.config.model_dump()
        config_dict.update(kwargs)
        self.config = HuggingFaceConfig(**config_dict)

        # Update stop token IDs if stop_sequences changed
        if 'stop_sequences' in kwargs:
            self.stop_token_ids = self._get_stop_token_ids()
