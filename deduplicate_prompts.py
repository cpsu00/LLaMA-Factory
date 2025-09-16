import json
from datasets import load_dataset

# --- Configuration ---
ORIGINAL_DATASET = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
PROMPT_COLUMN = "prompt"
OUTPUT_FILE = "data/tulu-3-deduplicated.jsonl"

def create_unique_prompt_dataset():
    print(f"Loading original dataset: {ORIGINAL_DATASET}...")
    dataset = load_dataset(ORIGINAL_DATASET, split="train")

    unique_prompts = set()
    print("Finding unique prompts...")
    for item in dataset:
        prompt_text = item[PROMPT_COLUMN].strip()
        # Skip empty prompts
        if not prompt_text:
            continue
        unique_prompts.add(prompt_text)

    print(f"Found {len(unique_prompts)} unique non-empty prompts out of {len(dataset)} total entries.")

    print(f"Saving unique prompts to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for prompt_text in unique_prompts:
            f.write(json.dumps({PROMPT_COLUMN: prompt_text}) + "\n")

    print("Deduplication complete! ✨")

if __name__ == "__main__":
    create_unique_prompt_dataset()
