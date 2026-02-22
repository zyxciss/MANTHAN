import os
from inference import load_and_run

def main():
    model_path = "/tmp/Manthan-M1"
    image_path = "board-361516_1280.jpg"

    prompt = "What are the main objects in this image? Please describe them."

    print(f"Running Manthan-M1 inference")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}\n")

    try:
        load_and_run(model_path, image_path, prompt)
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
