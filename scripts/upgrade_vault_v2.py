#!/usr/bin/env python3
"""
upgrade_vault_v2.py — stamp a v1 .smoe vault to format v2 in place.

Writes the 128-byte SARC arch block (built from the checkpoint's HF
config.json) into the existing zero-padding gap between the descriptor
tables and the first expert blob, then flips the header to version 2 with
reserved_ext pointing at the block. No expert data moves; a 112 GB vault
upgrades in milliseconds.

Crash-safe ordering: block → fsync → reserved_ext → version → fsync.
An interruption at any point leaves a valid v1 vault.

Usage:
    upgrade_vault_v2.py <config.json | model_dir> <vault.smoe> [--force]

With a model_dir containing *.safetensors shards, has_qk_norm and
has_dense_layer_0 come from tensor presence (authoritative). With a bare
config.json they are derived from model_type heuristics — the engine
cross-checks against the Scout tensors at load either way.

--force re-stamps the block on an already-v2 vault (same slot).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from shatter_moe import (  # noqa: E402
    HEADER_FMT, EXPERT_ENTRY_FMT, TENSOR_DESC_FMT, TENSORS_PER_EXPERT,
    SMOE_MAGIC, SMOE_VERSION, SMOE_VERSION_MIN, ARCH_FMT,
    _align_up, build_tensor_index, read_hf_arch_config,
    build_arch_block, unpack_arch_block,
)

# Header byte offsets patched in place (see HEADER_FMT / common.hpp)
HDR_OFF_VERSION      = 8
HDR_OFF_RESERVED_EXT = 60


def load_arch(config_src: Path) -> dict:
    """Build the arch dict from a model dir (tensor-presence flags) or a bare config.json."""
    if config_src.is_dir():
        shards = sorted(config_src.glob("*.safetensors"))
        if shards:
            print(f"  Indexing {len(shards)} shard(s) for tensor-presence flags…")
            return read_hf_arch_config(config_src, build_tensor_index(shards))
        config_path = config_src / "config.json"
    else:
        config_path = config_src

    arch = read_hf_arch_config(config_path, {})
    # No shards to inspect — derive the presence flags from the config itself.
    with open(config_path) as f:
        raw = json.load(f)
    arch["has_qk_norm"] = arch["model_type"].startswith("qwen3")
    arch["has_dense_layer_0"] = int(raw.get("first_k_dense_replace") or 0) > 0
    print("  No shards available — presence flags derived from model_type "
          f"(qk_norm={arch['has_qk_norm']}, dense_l0={arch['has_dense_layer_0']})")
    return arch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("config_src", type=Path,
                        help="config.json path or model directory containing it")
    parser.add_argument("vault", type=Path, help=".smoe vault to upgrade in place")
    parser.add_argument("--force", action="store_true",
                        help="re-stamp the arch block on an already-v2 vault")
    args = parser.parse_args()

    arch = load_arch(args.config_src.resolve())
    block = build_arch_block(arch)

    with open(args.vault, "r+b") as f:
        hdr = HEADER_FMT.unpack(f.read(HEADER_FMT.size))
        (magic, version, n_layers, max_exp, total_exp, tbl_off, data_off,
         _grp, _bits, _dm, _vocab, _ffn, reserved_ext) = hdr

        if magic != SMOE_MAGIC:
            sys.exit(f"ERROR: {args.vault} is not a .smoe vault (bad magic)")
        if not (SMOE_VERSION_MIN <= version <= SMOE_VERSION):
            sys.exit(f"ERROR: unsupported vault version {version}")

        if version >= 2:
            if not args.force:
                sys.exit(f"ERROR: vault is already v{version} — use --force to re-stamp the arch block")
            if reserved_ext == 0 or reserved_ext + ARCH_FMT.size > data_off:
                sys.exit(f"ERROR: v2 vault has invalid arch offset {reserved_ext:#x}")
            arch_offset = reserved_ext
            print(f"  Re-stamping arch block in existing slot @ {arch_offset:#x}")
        else:
            meta_bytes = (
                tbl_off
                + total_exp * EXPERT_ENTRY_FMT.size
                + total_exp * TENSORS_PER_EXPERT * TENSOR_DESC_FMT.size
            )
            arch_offset = _align_up(meta_bytes, 64)
            if arch_offset + ARCH_FMT.size > data_off:
                sys.exit(
                    f"ERROR: no room for the arch block — gap "
                    f"[{meta_bytes:#x}, {data_off:#x}) is "
                    f"{data_off - arch_offset} bytes, need {ARCH_FMT.size}. "
                    f"Re-shatter this vault instead."
                )
            f.seek(arch_offset)
            resident = f.read(ARCH_FMT.size)
            if resident.strip(b"\x00"):
                print(f"  ⚠ pad gap @ {arch_offset:#x} is not all zeros — overwriting anyway")

        # 1. Arch block into the gap, flushed to disk first.
        f.seek(arch_offset)
        f.write(block)
        f.flush()
        os.fsync(f.fileno())

        # 2. Header patch: reserved_ext before version, so an interruption
        #    between the two still reads as v1 (nonzero reserved_ext is
        #    ignored by v1 readers).
        f.seek(HDR_OFF_RESERVED_EXT)
        f.write(struct.pack("<I", arch_offset))
        f.seek(HDR_OFF_VERSION)
        f.write(struct.pack("<I", SMOE_VERSION))
        f.flush()
        os.fsync(f.fileno())

        # Read back and decode for the eyeball check.
        f.seek(arch_offset)
        decoded = unpack_arch_block(f.read(ARCH_FMT.size))

    print(f"\n  ✓ {args.vault.name} → v{SMOE_VERSION}, arch block @ {arch_offset:#x}")
    print(f"    {decoded['model_type']}  theta={decoded['rope_theta']:.0f}  "
          f"top_k={decoded['moe_top_k']}  "
          f"heads={decoded['num_heads']}/{decoded['num_kv_heads']}×{decoded['head_dim']}  "
          f"norm_topk={decoded['norm_topk_prob']}  qk_norm={decoded['has_qk_norm']}  "
          f"dense_l0={decoded['has_dense_layer_0']}  "
          f"shared_ffn={decoded['shared_expert_ffn_dim']}  "
          f"layers={decoded['num_hidden_layers']}")


if __name__ == "__main__":
    main()
