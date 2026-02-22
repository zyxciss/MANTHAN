import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel, AutoProcessor, AutoTokenizer
from configuration_unified import UnifiedMultimodalConfig
from modeling_unified import UnifiedMultimodalModel


def create_and_save_unified_model(save_directory):
    vlm_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    llm_model_id = "openai/gpt-oss-20b"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Download and load both sub-models with pretrained weights
    print("Loading VLM weights...")
    try:
        vlm = AutoModelForCausalLM.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except ValueError:
        vlm = AutoModel.from_pretrained(
            vlm_model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

    print("Loading LLM weights...")
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # 2. Build unified config (stores both sub-model configs)
    vlm_config = AutoConfig.from_pretrained(vlm_model_id)
    llm_config = AutoConfig.from_pretrained(llm_model_id)

    unified_config = UnifiedMultimodalConfig(
        vlm_config=vlm_config.to_dict(),
        llm_config=llm_config.to_dict(),
        vlm_system_prompt=(
            "Analyze this image comprehensively. Describe the main subjects, "
            "background, text (OCR), spatial relationships, colors, and any "
            "notable details. Output in a structured format."
        ),
    )

    # 3. Create the unified wrapper and save
    unified_model = UnifiedMultimodalModel(vlm=vlm, llm=llm, config=unified_config)
    unified_model.save_pretrained(save_directory)

    # 4. Save processors / tokenizers alongside
    print("Saving VLM processor...")
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm_processor.save_pretrained(f"{save_directory}/vlm_processor")

    print("Saving LLM tokenizer...")
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_tokenizer.save_pretrained(f"{save_directory}/llm_tokenizer")

    print("Done! Unified model saved to:", save_directory)


if __name__ == "__main__":
    create_and_save_unified_model("/tmp/my-unified-model")
