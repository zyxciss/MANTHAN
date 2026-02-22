import torch
from transformers import AutoProcessor, AutoTokenizer
from modeling_unified import ManthanM1


def load_and_run(model_path, image_path, prompt):
    # 1. Load Manthan-M1 (reads config.json, loads vlm/ and llm/ sub-models)
    print(f"Loading Manthan-M1 from {model_path}...")
    model = ManthanM1.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # 2. Load processors
    vlm_processor = AutoProcessor.from_pretrained(f"{model_path}/vlm_processor")
    llm_tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/llm_tokenizer")

    # 3. Run inference — single generate() call
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

    print("\n--- Manthan-M1 Response ---")
    print(response)


if __name__ == "__main__":
    pass
