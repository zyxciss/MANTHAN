import json
import os
import shutil
import torch
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from huggingface_hub import snapshot_download
from configuration_unified import ManthanM1Config
from modeling_unified import ManthanM1


def create_and_save_model(save_directory):
    vlm_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    llm_model_id = "openai/gpt-oss-20b"

    os.makedirs(save_directory, exist_ok=True)

    # ── 1. Build Manthan-M1 config ────────────────────────────────────
    print("Fetching sub-model configs...")
    vlm_config = AutoConfig.from_pretrained(vlm_model_id)
    llm_config = AutoConfig.from_pretrained(llm_model_id)

    config = ManthanM1Config(
        vlm_config=vlm_config.to_dict(),
        llm_config=llm_config.to_dict(),
        vlm_system_prompt=(
            "Analyze this image comprehensively. Describe the main subjects, "
            "background, text (OCR), spatial relationships, colors, and any "
            "notable details. Output in a structured format."
        ),
    )

    # Save unified config
    config_path = os.path.join(save_directory, "config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "model_type": config.model_type,
                "architectures": ["ManthanM1"],
                "vlm_config": config.vlm_config,
                "llm_config": config.llm_config,
                "vlm_system_prompt": config.vlm_system_prompt,
            },
            f,
            indent=2,
        )

    # ── 2. Copy VLM weights directly from HF cache (BF16, ~8GB) ──────
    #    No loading into GPU, no save_pretrained — just copy the originals.
    print(f"Downloading and copying VLM ({vlm_model_id})...")
    vlm_dir = os.path.join(save_directory, "vlm")
    _copy_from_hf_cache(vlm_model_id, vlm_dir)

    # ── 3. Copy LLM weights directly from HF cache (MXFP4, ~12GB) ────
    #    Preserves MXFP4 quantized safetensors as-is. No dequantization.
    print(f"Downloading and copying LLM ({llm_model_id})...")
    llm_dir = os.path.join(save_directory, "llm")
    _copy_from_hf_cache(llm_model_id, llm_dir)

    # ── 4. Generate top-level model.safetensors.index.json ────────────
    #    Merges weight maps from vlm/ and llm/ so HuggingFace Hub
    #    correctly reports the total parameter count (~24B).
    print("Generating top-level model.safetensors.index.json ...")
    ManthanM1._generate_merged_index(save_directory)

    # ── 5. Save processors / tokenizers ───────────────────────────────
    print("Saving VLM processor...")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm_processor.save_pretrained(f"{save_directory}/vlm_processor")

    print("Saving LLM tokenizer...")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_tokenizer.save_pretrained(f"{save_directory}/llm_tokenizer")

    # ── Done ──────────────────────────────────────────────────────────
    print(f"\n✓ Manthan-M1 packaged at: {save_directory}")
    _print_repo_size(save_directory)
    print("\nNext steps:")
    print(f"  1. huggingface-cli upload <your-username>/Manthan-M1 {save_directory} .")
    print(f"  2. python test_inference.py")


def _copy_from_hf_cache(model_id, dest_dir):
    """
    Download model files (or use HF cache) and copy into dest_dir.
    This preserves the original safetensors exactly as uploaded by the
    model author — no dequantization, no dtype conversion.
    """
    cache_dir = snapshot_download(model_id)

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.copytree(cache_dir, dest_dir)
    print(f"  Copied: {cache_dir} -> {dest_dir}")


def _print_repo_size(directory):
    """Print total size of the saved repo."""
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    print(f"Total repo size: {total / (1024**3):.2f} GB")


if __name__ == "__main__":
    create_and_save_model("/tmp/Manthan-M1")
