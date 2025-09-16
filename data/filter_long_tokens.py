import json
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIGURE THESE VALUES ---
model_name = "allenai/Llama-3.1-Tulu-3-8B-SFT"
# Input file with the oversized samples
input_dataset_path = "delta-Qwen2.5-3B-1.5B.jsonl"
# Name for the new, cleaned output file
output_dataset_path = "delta-Qwen2.5-3B-1.5B_clean.jsonl"

# You can set this to 8192 since Llama 3 supports it, but 4096 is also fine.
# Let's use 8192 to keep more data, but you can change it back if you want.
cutoff_len = 2048
# -----------------------------

print(f"Loading tokenizer for '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded. Using cutoff_len = {cutoff_len}\n")

valid_samples = 0
removed_samples = 0

with open(input_dataset_path, 'r', encoding='utf-8') as infile, \
     open(output_dataset_path, 'w', encoding='utf-8') as outfile:
    
    # Get total line count for tqdm progress bar
    total_lines = sum(1 for line in open(input_dataset_path, 'r', encoding='utf-8'))
    infile.seek(0) # Reset file pointer
    
    for line in tqdm(infile, total=total_lines, desc="Processing samples"):
        try:
            example = json.loads(line)

            # Basic validation for content
            if not all(example.get(key) for key in ["prompt", "chosen", "rejected"]):
                removed_samples += 1
                continue

            # Check token length
            prompt_len = len(tokenizer.encode(example["prompt"]))
            chosen_len = len(tokenizer.encode(example["chosen"]))
            rejected_len = len(tokenizer.encode(example["rejected"]))

            if (prompt_len + chosen_len > cutoff_len) or (prompt_len + rejected_len > cutoff_len):
                removed_samples += 1
                continue
            
            # If valid, write to the new file
            outfile.write(json.dumps(example) + '\n')
            valid_samples += 1

        except json.JSONDecodeError:
            print(f"Warning: Skipping a line due to JSON decoding error.")
            removed_samples += 1

print("\n--- Cleaning Complete ---")
print(f"✅ Wrote {valid_samples} valid samples to '{output_dataset_path}'")
print(f"🗑️ Removed {removed_samples} oversized or invalid samples.")
