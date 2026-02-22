import os
from inference import load_and_run_unified_model

def main():
    model_path = "/tmp/my-unified-model"
    image_path = "board-361516_1280.jpg"
    
    prompt = "What are the main objects in this image? Please describe them."
    
    print(f"Running inference with model at {model_path}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}\n")
    
    try:
        load_and_run_unified_model(model_path, image_path, prompt)
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
