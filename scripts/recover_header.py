import argparse
from pathlib import Path
import sys
import struct

from shatter_moe import (
    build_tensor_index, detect_moe_topology, extract_model_dimensions,
    load_expert_tensors, build_expert_blob, _align_up, PAGE_SIZE,
    HEADER_FMT, EXPERT_ENTRY_FMT, TENSOR_DESC_FMT, TENSORS_PER_EXPERT,
    SMOE_MAGIC, SMOE_VERSION, Q2_GROUP_SIZE, ExpertMeta
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("vault_file", type=Path)
    parser.add_argument("--bits", type=int, default=2)
    args = parser.parse_args()

    print("Indexing tensors...")
    shards = sorted(args.model_dir.glob("*.safetensors"))
    tensor_index = build_tensor_index(shards)
    topology = detect_moe_topology(tensor_index)
    dims = extract_model_dimensions(tensor_index)

    layers = topology["moe_layers"]
    total_experts = sum(len(topology["experts_per_layer"][lid]) for lid in layers)

    meta_bytes = (
        HEADER_FMT.size
        + total_experts * EXPERT_ENTRY_FMT.size
        + total_experts * TENSORS_PER_EXPERT * TENSOR_DESC_FMT.size
    )
    data_offset = _align_up(meta_bytes, PAGE_SIZE)
    print(f"Data offset calculated at: {hex(data_offset)}")

    # Load just ONE expert to get the exact descs and padded_size
    print("Loading one expert to sample dimensions...")
    lid = layers[0]
    eid = topology["experts_per_layer"][lid][0]
    tensors = load_expert_tensors(tensor_index, lid, eid)
    blob_bytes, sample_descs = build_expert_blob(tensors, Q2_GROUP_SIZE, args.bits)
    
    raw_size = len(blob_bytes)
    padded_size = _align_up(raw_size, PAGE_SIZE)
    total_groups = sum(d.num_groups for d in sample_descs)

    print(f"Sample Expert: raw={raw_size}, padded={padded_size}, groups={total_groups}")

    experts_meta = []
    cursor = data_offset
    for lid in layers:
        for eid in topology["experts_per_layer"][lid]:
            experts_meta.append(ExpertMeta(
                layer_id=lid,
                expert_id=eid,
                tensor_descs=sample_descs,
                raw_size=raw_size,
                padded_size=padded_size,
                total_groups=total_groups,
                file_offset=cursor
            ))
            cursor += padded_size

    print(f"Generated {len(experts_meta)} metadata entries. Total file size should be {cursor / 10**9:.2f} GB")

    # Patch the file
    print(f"Opening {args.vault_file} in r+b mode...")
    with open(args.vault_file, "r+b") as f_out:
        # Check actual file size
        f_out.seek(0, 2)
        actual_size = f_out.tell()
        print(f"Actual file size on disk: {actual_size / 10**9:.2f} GB")
        
        if actual_size < cursor:
            print("WARNING: File on disk is smaller than expected. The original script may have crashed early!")
            # We can still patch up to the experts that were written!
            written_experts = (actual_size - data_offset) // padded_size
            print(f"Actually completely written experts: {written_experts}")
            experts_meta = experts_meta[:written_experts]
            total_experts = len(experts_meta)
            # Recompute data_offset just in case, though it shouldn't change unless total_experts drops drastically
            # But the file was created with the ORIGINAL data_offset, so we MUST keep the original data_offset!
        
        f_out.seek(0)
        f_out.write(HEADER_FMT.pack(
            SMOE_MAGIC,
            SMOE_VERSION,
            topology["num_moe_layers"],
            topology["max_experts"],
            len(experts_meta),
            HEADER_FMT.size,    # table_offset
            data_offset,
            Q2_GROUP_SIZE,
            args.bits,
            dims.get("d_model", 0),
            dims.get("vocab_size", 0),
            dims.get("ffn_dim", 0),
            0  # reserved_ext
        ))

        for meta in experts_meta:
            f_out.write(EXPERT_ENTRY_FMT.pack(
                meta.layer_id,
                meta.expert_id,
                meta.file_offset,
                meta.raw_size,
                meta.padded_size,
                Q2_GROUP_SIZE,
                meta.total_groups,
            ))

        for meta in experts_meta:
            for desc in meta.tensor_descs:
                f_out.write(TENSOR_DESC_FMT.pack(
                    desc.tensor_type,
                    2,
                    desc.rows,
                    desc.cols,
                    desc.packed_offset,
                    desc.packed_size,
                    desc.scales_offset,
                    desc.scales_size,
                ))

    print("\nSUCCESS! The vault header has been permanently patched and recovered.")

if __name__ == "__main__":
    main()
