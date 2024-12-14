import os
import json
from transformers import AutoProcessor, Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from tqdm import tqdm

# File paths
INPUT_FILE = "visualiztion_200.json"
IMAGES_DIR = "images"


def load_labels(label_file):
    """
    Load label data from the JSON file.
    """
    with open(label_file, "r") as f:
        return json.load(f)


def save_labels(labels, output_file):
    """
    Save the updated labels to a JSON file.
    """
    with open(output_file, "w") as f:
        json.dump(labels, f, indent=4)


def generate_captions_with_prompts(model, processor, labels, images_dir, device):
    """
    Generate captions and update label entries with generated text and prompts.
    """
    for entry in tqdm(labels):
        try:
            # Get the image file path
            image_name = entry.get("image")
            if not image_name:
                continue

            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                print(f"Image file {image_path} not found. Skipping.")
                continue

            # Load and preprocess the image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

            # Generate the caption
            generated_ids = model.generate(**inputs)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            generated_text += "."

            # Update the entry with generated text
            entry["generated_text"] = generated_text

            # Create prompt_w_label
            labels_text = ", ".join(entry.get("labels", []))
            height = entry.get("height", "unknown")
            width = entry.get("width", "unknown")
            prompt_w_label = f"{generated_text} {labels_text}, height: {height}, width: {width}"
            entry["prompt_w_label"] = prompt_w_label

            # Create prompt_w_suffix with a suffix
            suffix = "HD quality, highly detailed"
            prompt_w_suffix = f"{prompt_w_label}, {suffix}."
            entry["prompt_w_suffix"] = prompt_w_suffix

        except Exception as e:
            print(f"Error processing entry for image {entry.get('image')}: {e}")


def main():
    print("torch.cuda.is_available():", torch.cuda.is_available())

    # Load the model and processor
    # Salesforce/blip2-opt-2.7b
    # Salesforce/blip2-opt-6.7b-coco
    # Salesforce/blip2-opt-6.7b
    # Salesforce/blip2-flan-t5-xl
    # model_list = ["blip2-opt-2.7b", "blip2-opt-6.7b-coco", "blip2-opt-6.7b", "blip2-flan-t5-xl"]
    model_list = ["blip2-opt-6.7b-coco"]

    for model_name in model_list:
        if model_name in ["blip2-opt-2.7b", "blip2-opt-6.7b", "blip2-flan-t5-xl"]:
            processor = AutoProcessor.from_pretrained("Salesforce/{}".format(model_name))
        else:
            processor = Blip2Processor.from_pretrained("Salesforce/{}".format(model_name))
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/{}".format(model_name), torch_dtype=torch.float16)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Load label data
        labels = load_labels(INPUT_FILE)
        print(f"Loaded {len(labels)} entries from '{INPUT_FILE}'.")

        # Generate captions and update labels
        generate_captions_with_prompts(model, processor, labels, IMAGES_DIR, device)

        # Save the updated labels to a new JSON file
        OUTPUT_FILE = "visualiztion_200_with_{}.json".format(model_name)
        save_labels(labels, OUTPUT_FILE)
        print(f"Updated labels saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
