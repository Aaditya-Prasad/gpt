#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def main(args):
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    valid_prompts = []
    total_prompts = 0
    
    # Read JSONL file line by line
    with open(input_path, 'r') as f:
        for line in f:
            total_prompts += 1
            data = json.loads(line.strip())
            
            # Filter for English only
            if data.get("language", "").lower() == "english":
                # Create new format
                new_dict = {
                    "conversation_id": data["conversation_id"],
                    "prompt": data["prompt"]
                }
                valid_prompts.append(new_dict)

    
    
    # Write to output files with max size of out_size
    chunk_size = args.out_size
    num_chunks = (len(valid_prompts) + chunk_size - 1) // chunk_size  # Ceiling division
    
    base_path = output_path.with_suffix('')  # Remove extension
    extension = output_path.suffix
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(valid_prompts))
        chunk = valid_prompts[start_idx:end_idx]
        
        chunk_output_path = base_path.with_name(f"{base_path.name}_{i}{extension}")
        with open(chunk_output_path, 'w') as f:
            json.dump(chunk, f, indent=2)
        
        print(f"Saved chunk {i} with {len(chunk)} prompts to {chunk_output_path}")
    
    print(f"Processed {total_prompts} prompts")
    print(f"Found {len(valid_prompts)} English prompts")
    print(f"Split into {num_chunks} files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--out_size", type=int, default=102400)
    args = parser.parse_args()
    main(args)