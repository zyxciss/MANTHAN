import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from configuration_unified import UnifiedMultimodalConfig
from modeling_unified import UnifiedMultimodalModel

def create_and_save_unified_model(save_directory):
    # 1. Load configs for both models
    vlm_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    llm_model_id = "openai/gpt-oss-20b"
    
    vlm_config = AutoConfig.from_pretrained(vlm_model_id)
    llm_config = AutoConfig.from_pretrained(llm_model_id)
    
    # 2. Create unified config
    unified_config = UnifiedMultimodalConfig(
        vlm_config=vlm_config.to_dict(),
        llm_config=llm_config.to_dict(),
        vlm_system_prompt="Analyze this image comprehensively. Describe the main subjects, background, text (OCR), spatial relationships, colors, and any notable details. Output in a structured format."
    )
    
    # 3. Initialize unified model
    unified_model = UnifiedMultimodalModel(unified_config)
    
    # 4. Load weights into sub-modules
    # Note: This requires downloading the weights of both models.
    # For a 20B model, this requires significant RAM.
    print("Loading VLM weights...")
    unified_model.vlm = AutoModelForCausalLM.from_pretrained(vlm_model_id, torch_dtype=torch.bfloat16)
    
    print("Loading LLM weights...")
    unified_model.llm = AutoModelForCausalLM.from_pretrained(llm_model_id, torch_dtype=torch.bfloat16)
    
    # 5. Save the unified model
    # This will save the combined state_dict as safetensors (default in newer transformers)
    print(f"Saving unified model to {save_directory}...")
    unified_model.save_pretrained(save_directory, safe_serialization=True)
    unified_config.save_pretrained(save_directory)
    
    # 6. Save processors/tokenizers
    # We need to save both so they can be loaded later.
    # A common pattern is to save them in subdirectories or with prefixes.
    vlm_processor = AutoProcessor.from_pretrained(vlm_model_id)
    vlm_processor.save_pretrained(f"{save_directory}/vlm_processor")
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_tokenizer.save_pretrained(f"{save_directory}/llm_tokenizer")
    
    print("Done!")

if __name__ == "__main__":
    create_and_save_unified_model("./my-unified-model")
