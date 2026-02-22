import os
import shutil
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoProcessor, AutoTokenizer
from configuration_unified import ManthanM1Config
from modeling_unified import ManthanM1


def create_and_save_model(save_directory):
    vlm_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    llm_model_id = "openai/gpt-oss-20b"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── 1. Load VLM (BF16, ~8GB) ─────────────────────────────────────
    print("Loading VLM weights...")
    try:
        vlm = AutoModelForCausalLM.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16, device_map=device
        )
    except ValueError:
        vlm = AutoModel.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16, device_map=device
        )

    # ── 2. Load LLM (MXFP4 native, ~12GB) ────────────────────────────
    print("Loading LLM weights...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map=device
    )

    # ── 3. Build Manthan-M1 config ────────────────────────────────────
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

    # ── 4. Create wrapper and save ────────────────────────────────────
    model = ManthanM1(vlm=vlm, llm=llm, config=config)
    model.save_pretrained(save_directory)

    # ── 5. Copy original LLM safetensors to preserve MXFP4 ───────────
    #    save_pretrained above may dequantize MXFP4 → BF16 when saving.
    #    Overwrite with the original HF cache files to keep ~12GB.
    print("Copying original MXFP4 LLM weights from HF cache...")
    llm_dir = os.path.join(save_directory, "llm")
    _copy_original_hf_files(llm_model_id, llm_dir)

    # ── 6. Re-generate the merged index after copying originals ───────
    print("Re-generating top-level model.safetensors.index.json ...")
    ManthanM1._generate_merged_index(save_directory)

    # ── 7. Save processors / tokenizers ───────────────────────────────
    print("Saving VLM processor...")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm_processor.save_pretrained(f"{save_directory}/vlm_processor")

    print("Saving LLM tokenizer...")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_tokenizer.save_pretrained(f"{save_directory}/llm_tokenizer")

    print(f"\n✓ Manthan-M1 saved to: {save_directory}")
    _print_repo_size(save_directory)
    print("\nNext steps:")
    print(f"  1. cd {save_directory}")
    print(f"  2. huggingface-cli upload <your-username>/Manthan-M1 . .")


def _copy_original_hf_files(model_id, dest_dir):
    """Copy original safetensors from HF cache, preserving MXFP4."""
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(model_id)

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    shutil.copytree(cache_dir, dest_dir)
    print(f"  Copied original weights from {cache_dir} -> {dest_dir}")


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
