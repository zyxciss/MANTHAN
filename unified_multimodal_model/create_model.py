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
    #    ONLY download root-level files (safetensors, config, tokenizer).
    #    Skip metal/ (~13.8GB .bin) and original/ (~13.8GB .safetensors)
    #    which are alternative format copies we don't need.
    print(f"Downloading LLM ({llm_model_id})...")
    llm_dir = os.path.join(save_directory, "llm")
    _download_move_and_cleanup(
        llm_model_id,
        llm_dir,
        ignore_patterns=["metal/*", "original/*"],
    )

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


def _download_move_and_cleanup(model_id, dest_dir, ignore_patterns=None):
    """
    Download only the needed model files from HF Hub, MOVE (not copy)
    into dest_dir, then delete the HF cache entry to free disk immediately.
    
    ignore_patterns: list of glob patterns to skip (e.g., ["metal/*", "original/*"])
    """
    # Download to HF cache — only the files we actually need
    kwargs = {}
    if ignore_patterns:
        kwargs["ignore_patterns"] = ignore_patterns

    cache_dir = snapshot_download(model_id, **kwargs)

    # Remove dest if it already exists
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    # HF cache stores files as symlinks (snapshots/<rev>/file -> blobs/<hash>).
    # We must resolve them into real files BEFORE deleting the cache,
    # otherwise the symlinks become dangling.
    _resolve_symlinks(cache_dir)

    # MOVE instead of copy — no duplication after resolving symlinks
    shutil.move(cache_dir, dest_dir)
    print(f"  Moved: {cache_dir} -> {dest_dir}")

    # Clean up any remaining cache artifacts for this model
    _cleanup_hf_cache_for_model(model_id)


def _resolve_symlinks(directory):
    """
    Walk *directory* and replace every symlink with a real copy of its target.

    HuggingFace's cache layout is:
        snapshots/<revision>/<file>  →  ../../blobs/<sha256>
    After resolving, the snapshot directory is self-contained and safe to
    move even after the blobs directory is deleted.
    """
    for dirpath, dirnames, filenames in os.walk(directory):
        for name in filenames:
            fpath = os.path.join(dirpath, name)
            if os.path.islink(fpath):
                real = os.path.realpath(fpath)
                os.remove(fpath)           # delete the symlink
                shutil.copy2(real, fpath)   # copy real file in its place


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
