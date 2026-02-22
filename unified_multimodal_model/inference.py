import torch
from transformers import AutoProcessor, AutoTokenizer
from modeling_unified import UnifiedMultimodalModel


def load_and_run_unified_model(model_path, image_path, prompt):
    # 1. Load the unified model (reads config.json + model.safetensors)
    print(f"Loading unified model from {model_path}...")
    model = UnifiedMultimodalModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Load processors
    vlm_processor = AutoProcessor.from_pretrained(f"{model_path}/vlm_processor")
    llm_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/llm_tokenizer")

    # 3. Run inference
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    print("Generating response...")
    response = model.generate(
        images=image,
        text_prompt=prompt,
        vlm_processor=vlm_processor,
        llm_tokenizer=llm_tokenizer,
        max_new_tokens=1024,
    )

    print("\n--- Final Response ---")
    print(response)


if __name__ == "__main__":
    # Example usage
    # load_and_run_unified_model("/tmp/my-unified-model", "board-361516_1280.jpg", "What is the main subject of this image?")
    pass
