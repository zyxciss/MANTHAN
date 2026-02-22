# Unified Multimodal Model

This repository contains a unified multimodal model that combines a Vision-Language Model (VLM) and a Large Language Model (LLM) into a single `nn.Module` with a unified `generate()` interface.

## Architecture

The model is a two-stage system wrapped in a single HuggingFace-style repository:
1. **Vision Module**: Uses `Qwen/Qwen3-VL-4B-Instruct` to convert an input image into a highly detailed, structured text description.
2. **Reasoning Module**: Uses `openai/gpt-oss-20b` to process the structured image description alongside the user's prompt to generate the final response.

This approach allows for strong multimodal reasoning without requiring any training, fine-tuning, or projector alignment. The entire system is packaged as a single model checkpoint (`.safetensors`).

## Usage

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from PIL import Image

# 1. Register custom classes
from configuration_unified import UnifiedMultimodalConfig
from modeling_unified import UnifiedMultimodalModel

AutoConfig.register("unified_multimodal", UnifiedMultimodalConfig)
AutoModelForCausalLM.register(UnifiedMultimodalConfig, UnifiedMultimodalModel)

# 2. Load model and processors
model_path = "/tmp/my-unified-model"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    trust_remote_code=True
)

vlm_processor = AutoProcessor.from_pretrained(f"{model_path}/vlm_processor")
llm_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/llm_tokenizer")

# 3. Run inference
image = Image.open("test_image.jpg").convert("RGB")
prompt = "What is the main subject of this image?"

response = model.generate(
    images=image,
    text_prompt=prompt,
    vlm_processor=vlm_processor,
    llm_tokenizer=llm_tokenizer,
    max_new_tokens=1024
)

print(response)
```

## Packaging

The model is packaged with the following structure:
- `config.json`: Contains the configurations for both the VLM and LLM.
- `model.safetensors`: The combined weights of both models.
- `configuration_unified.py`: The custom configuration class.
- `modeling_unified.py`: The custom model class.
- `vlm_processor/`: The processor for the vision module.
- `llm_tokenizer/`: The tokenizer for the reasoning module.
