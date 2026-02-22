# Unified Multimodal Model

A unified multimodal model that combines a Vision-Language Model (VLM) and a Large Language Model (LLM) into a single `nn.Module` with a unified `generate()` interface.

**No training, no fine-tuning, no projector alignment.**

## Architecture

Two-stage pipeline wrapped as a single model:
1. **Vision Module** — `Qwen/Qwen3-VL-4B-Instruct` (BF16, ~8GB) converts image → structured text
2. **Reasoning Module** — `openai/gpt-oss-20b` (MXFP4 quantized, ~12GB) reasons over structured text + user prompt

Total repo size: ~20GB (preserving original MXFP4 quantization, NOT dequantized to BF16)

## Repository Layout

```
my-unified-model/
├── config.json              # Unified config (references both sub-models)
├── vlm/                     # VLM weights + config (BF16 safetensors)
├── llm/                     # LLM weights + config (MXFP4 safetensors, native)
├── vlm_processor/           # VLM processor (tokenizer + image processor)
└── llm_tokenizer/           # LLM tokenizer
```

## Usage

```python
import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from modeling_unified import UnifiedMultimodalModel

# 1. Load unified model (one call, handles both sub-models)
model = UnifiedMultimodalModel.from_pretrained(
    "/tmp/my-unified-model",
    dtype=torch.bfloat16,
    device_map="auto",
)

# 2. Load processors
vlm_processor = AutoProcessor.from_pretrained("/tmp/my-unified-model/vlm_processor")
llm_tokenizer = AutoTokenizer.from_pretrained("/tmp/my-unified-model/llm_tokenizer")

# 3. Single generate() call — no visible intermediate steps
image = Image.open("test_image.jpg").convert("RGB")

response = model.generate(
    images=image,
    text_prompt="What is the main subject of this image?",
    vlm_processor=vlm_processor,
    llm_tokenizer=llm_tokenizer,
    max_new_tokens=1024,
)

print(response)
```

## Creating the Model

```bash
python create_model.py
```

This will:
1. Download both models from HuggingFace
2. Package them into a single unified repo at `/tmp/my-unified-model`
3. Preserve the LLM's MXFP4 quantization (no weight bloat)
