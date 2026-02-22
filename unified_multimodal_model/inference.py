import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from configuration_unified import UnifiedMultimodalConfig
from modeling_unified import UnifiedMultimodalModel

def load_and_run_unified_model(model_path, image_path, prompt):
    # 1. Register custom config and model classes
    AutoConfig.register("unified_multimodal", UnifiedMultimodalConfig)
    AutoModelForCausalLM.register(UnifiedMultimodalConfig, UnifiedMultimodalModel)
    
    # 2. Load the unified model
    print(f"Loading unified model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", # Automatically distribute across GPUs
        trust_remote_code=True
    )
    
    # 3. Load processors
    vlm_processor = AutoProcessor.from_pretrained(f"{model_path}/vlm_processor")
    llm_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/llm_tokenizer")
    
    # 4. Run inference
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    
    print("Generating response...")
    response = model.generate(
        images=image,
        text_prompt=prompt,
        vlm_processor=vlm_processor,
        llm_tokenizer=llm_tokenizer,
        max_new_tokens=1024
    )
    
    print("\n--- Final Response ---")
    print(response)

if __name__ == "__main__":
    # Example usage
    # load_and_run_unified_model("./my-unified-model", "test_image.jpg", "What is the main subject of this image?")
    pass
