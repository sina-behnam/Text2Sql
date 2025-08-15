from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import torch
import os

# model_name = "HuggingFaceTB/SmolLM3-3B"
# local_dir = "./models/SmolLM3-3B"

model_name = "Qwen/Qwen2.5-Coder-1.5B"
local_dir = "./models/Qwen2.5-Coder-1.5B"

# Step 1: Download the model explicitly
print("Downloading model...")
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True
)
print("Download complete!")