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

    # ── 2. Download VLM, move to dest, delete cache (~8GB) ───────────
    print(f"Downloading VLM ({vlm_model_id})...")
    vlm_dir = os.path.join(save_directory, "vlm")
    _download_move_and_cleanup(vlm_model_id, vlm_dir)

    # ── 3. Download LLM, move to dest, delete cache (~14GB) ──────────
    #    Preserves MXFP4 quantized safetensors as-is. No dequantization.
    print(f"Downloading LLM ({llm_model_id})...")
    llm_dir = os.path.join(save_directory, "llm")
    _download_move_and_cleanup(llm_model_id, llm_dir)

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

    # ── 6. Final cache cleanup ────────────────────────────────────────
    _purge_hf_cache()

    # ── Done ──────────────────────────────────────────────────────────
    print(f"\n✓ Manthan-M1 packaged at: {save_directory}")
    _print_repo_size(save_directory)
    print("\nNext steps:")
    print(f"  1. huggingface-cli upload <your-username>/Manthan-M1 {save_directory} .")
    print(f"  2. python test_inference.py")


def _download_move_and_cleanup(model_id, dest_dir):
    """
    Download model files from HF Hub, MOVE (not copy) into dest_dir,
    then delete the HF cache entry to free disk space immediately.
    
    Peak disk usage = 1x model size (download) + 1x model size (move target).
    But since we move, the cache is freed right after.
    """
    # Download to HF cache (or use existing cache)
    cache_dir = snapshot_download(model_id)
    
    # Remove dest if it already exists
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    
    # MOVE instead of copy — no duplication
    shutil.move(cache_dir, dest_dir)
    print(f"  Moved: {cache_dir} -> {dest_dir}")
    
    # Clean up any remaining cache artifacts for this model
    # (the blob store, refs, etc.)
    _cleanup_hf_cache_for_model(model_id)


def _cleanup_hf_cache_for_model(model_id):
    """Remove leftover HF cache entries for a specific model."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    # HF cache stores models as models--org--name
    folder_name = "models--" + model_id.replace("/", "--")
    cache_model_dir = os.path.join(cache_root, folder_name)
    if os.path.exists(cache_model_dir):
        shutil.rmtree(cache_model_dir, ignore_errors=True)
        print(f"  Cleaned HF cache: {cache_model_dir}")


def _purge_hf_cache():
    """Remove the entire HF cache to reclaim all disk space."""
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_root):
        shutil.rmtree(cache_root, ignore_errors=True)
        print("Purged HF cache.")


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
