# Manthan-M1

> *"Manthan" means churning — here it represents the churning of thoughts to produce deep, structured, and clear reasoning.*

**Manthan-M1** is the first generation of the Manthan reasoning model (~25B parameters). It combines a Vision-Language Model and a Large Language Model into a single `nn.Module` with a unified `generate()` interface.

Built by an independent developer — **no training, no fine-tuning, no projector alignment.**

## Architecture

Two-stage pipeline wrapped as a single model:
1. **Vision Module** — `Qwen/Qwen3-VL-4B-Instruct` (BF16, ~8GB) converts image → structured text
2. **Reasoning Module** — `openai/gpt-oss-20b` (MXFP4 quantized, ~12GB) reasons over structured text + user prompt

Total repo size: ~20GB (MXFP4 preserved, not dequantized)

## Repository Layout

```
Manthan-M1/
├── config.json                       # Unified Manthan-M1 config
├── model.safetensors.index.json      # Merged weight map (HF shows ~24B params)
├── vlm/                              # VLM weights + config (BF16 safetensors)
├── llm/                              # LLM weights + config (MXFP4 safetensors)
├── vlm_processor/                    # VLM processor (tokenizer + image processor)
└── llm_tokenizer/                    # LLM tokenizer
```

## Usage

```python
import torch
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image
from modeling_unified import ManthanM1

# 1. Load Manthan-M1 (one call, handles both sub-models)
model = ManthanM1.from_pretrained(
    "/tmp/Manthan-M1",
    dtype=torch.bfloat16,
    device_map="auto",
)

# 2. Load processors
vlm_processor = AutoProcessor.from_pretrained("/tmp/Manthan-M1/vlm_processor")
llm_tokenizer = AutoTokenizer.from_pretrained("/tmp/Manthan-M1/llm_tokenizer")

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

## Workflow

### Step 1: Create the model
```bash
python create_model.py
```
This downloads both models, packages them into `/tmp/Manthan-M1`, preserves MXFP4 quantization, and generates a merged `model.safetensors.index.json` for correct HF parameter count.

### Step 2: Upload to HuggingFace
```bash
huggingface-cli upload <your-username>/Manthan-M1 /tmp/Manthan-M1 .
```

### Step 3: Test inference
```bash
python test_inference.py
```
