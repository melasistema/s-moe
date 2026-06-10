#!/usr/bin/env python3
"""
export_vocab.py — Export Tokenizer Vocabulary to Binary Format
═══════════════════════════════════════════════════════════════════
S-MoE Engine  │  Phase 5: Vocabulary Exporter
Author: S-MoE / ANDARTIS
═══════════════════════════════════════════════════════════════════

Reads tokenizer.json and exports the ordered vocabulary list to a
clean binary format (vault/vocab.bin) for fast loading in C++.
Format:
  For each token ID 0..102399:
    [uint32_t length][bytes token_string]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="S-MoE Vocabulary Exporter")
    parser.add_argument("tokenizer_json", type=str, help="Path to tokenizer.json")
    parser.add_argument("output_bin", type=str, help="Output path for vocab.bin")
    args = parser.parse_args()

    t_path = Path(args.tokenizer_json)
    out_path = Path(args.output_bin)

    if not t_path.exists():
        print(f"ERROR: '{t_path}' does not exist.")
        sys.exit(1)

    print(f"Loading '{t_path}' ...")
    with open(t_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract vocab
    # HF Tokenizer JSON usually has vocab under model.vocab
    vocab_dict = data.get("model", {}).get("vocab", {})
    if not vocab_dict:
        print("ERROR: Could not find 'model.vocab' in tokenizer.json")
        sys.exit(1)

    print(f"Found {len(vocab_dict)} tokens in tokenizer.json.")

    # Sort tokens by ID
    # DeepSeek vocab shape in the model is 102,400.
    vocab_size = 102400
    vocab_list = ["" for _ in range(vocab_size)]

    for token, token_id in vocab_dict.items():
        if token_id < vocab_size:
            # Replace SentencePiece space character ' ' with standard space
            # and replace other typical visual markers if needed
            token_clean = token.replace(" ", " ")
            vocab_list[token_id] = token_clean

    # Fill any missing/padding tokens
    for i in range(vocab_size):
        if not vocab_list[i]:
            vocab_list[i] = f"<pad_{i}>"

    # Save to binary file
    print(f"Writing to '{out_path}' ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "wb") as f:
        for token in vocab_list:
            token_bytes = token.encode("utf-8", errors="replace")
            f.write(struct.pack("<I", len(token_bytes)))
            f.write(token_bytes)

    print("✓ Vocabulary exported successfully!")

if __name__ == "__main__":
    main()
