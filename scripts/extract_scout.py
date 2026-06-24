import argparse
from pathlib import Path

from shatter_moe import (
    build_tensor_index, classify_scout_keys, extract_and_save_scout
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("scout_path", type=Path)
    args = parser.parse_args()

    print("Indexing tensors...")
    shards = sorted(args.model_dir.glob("*.safetensors"))
    tensor_index = build_tensor_index(shards)
    
    scout_keys = classify_scout_keys(tensor_index)
    print(f"Extracting Scout ({len(scout_keys)} tensors)...")
    extract_and_save_scout(tensor_index, scout_keys, args.scout_path)

if __name__ == "__main__":
    main()
