#!/usr/bin/env python3
"""
shatter_moe.py — The Seismic Weight Sculptor
═══════════════════════════════════════════════════════════════════
S-MoE Engine  │  Phase 1: Offline Weight Shaper
Author: S-MoE / ANDARTIS — Luca Visciola
═══════════════════════════════════════════════════════════════════

Slices a supported fine-grained MoE model from HuggingFace .safetensors
format into a custom page-aligned .smoe binary vault, optimised for
Direct I/O streaming on Apple Silicon NVMe hardware.

  Usage:
    python shatter_moe.py <model_dir> <output_dir> [options]

  Examples:
    python shatter_moe.py ./checkpoints/qwen3-235b-instruct/ ./vault/ --validate
    python shatter_moe.py ./checkpoints/qwen3-235b-instruct/ ./vault/ --dry-run
    python shatter_moe.py ./checkpoints/qwen3-235b-instruct/ ./vault/ --max-layers 2 --max-experts 4
    python shatter_moe.py ./checkpoints/qwen3-235b-instruct/ ./vault/ --measure-error

  .smoe Binary Format (v1):
  ─────────────────────────────────────────────────────────────────
  ┌───────────────────────────────────────────┐
  │  FILE HEADER       (64 bytes, fixed)      │  offset 0x0000
  ├───────────────────────────────────────────┤
  │  EXPERT TABLE      (N × 48 bytes)         │  offset 0x0040
  ├───────────────────────────────────────────┤
  │  TENSOR METADATA   (N × 3 × 44 bytes)     │
  ├───────────────────────────────────────────┤
  │  [zero padding to 16 KB boundary]         │
  ├───────────────────────────────────────────┤
  │  EXPERT BLOBS      (each 16 KB-aligned)   │
  │   ├─ gate_proj : 2-bit packed + scales    │
  │   ├─ up_proj   : 2-bit packed + scales    │
  │   └─ down_proj : 2-bit packed + scales    │
  └───────────────────────────────────────────┘

  Quantisation Scheme: SMOE-Q2  (Symmetric Group Absmax, 2-bit)
  ─────────────────────────────────────────────────────────────────
    Levels  : {-1.5, -0.5, +0.5, +1.5} × scale_per_group
    Codes   : {  00,   01,   10,   11 }  (2 bits, LSB-first within byte)
    Scale   : float16 absmax per group of GROUP_SIZE weights
    Packing : 4 weights per byte (little-endian bit order)

  Dequantisation (mirrored exactly in Metal kernel):
    scale  = scales_buffer[weight_idx / GROUP_SIZE]
    code   = (packed_byte >> ((weight_idx % 4) * 2)) & 0x3
    weight = ((float)code − 1.5) / 1.5 × scale
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Register bfloat16 data type in numpy
try:
    import ml_dtypes
except ImportError:
    print("ERROR: ml_dtypes not installed. Run: pip install ml_dtypes")
    sys.exit(1)

# ── safetensors (required) ────────────────────────────────────────
try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as safetensors_save
except ImportError:
    print("ERROR: safetensors not installed.  Run: pip install safetensors numpy")
    sys.exit(1)

# ── rich (optional, graceful fallback) ───────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                SpinnerColumn, TaskProgressColumn,
                                TextColumn, TimeRemainingColumn)
    from rich.table import Table
    from rich import box as rich_box
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False

    class _FallbackConsole:
        def print(self, *args, **kw):
            text = " ".join(str(a) for a in args)
            text = re.sub(r"\[/?[^\]]*\]", "", text)   # strip rich markup
            print(text)

    console = _FallbackConsole()


# ═════════════════════════════════════════════════════════════════
# BINARY FORMAT CONSTANTS
# These values are the single source of truth for the .smoe format.
# The C++ reader (src/common.hpp) mirrors these definitions exactly.
# ═════════════════════════════════════════════════════════════════

SMOE_MAGIC          = b"SMOE\xDE\xEA\x00\x01"  # 8 bytes: SMOE + DeepSeek tag + format v1
SMOE_VERSION        = 2
SMOE_VERSION_MIN    = 1                         # readers accept [MIN, VERSION]
PAGE_SIZE           = 16_384                    # 16 KB — Apple Silicon hardware page boundary
Q2_GROUP_SIZE       = 64                        # weights per quantisation group
TENSORS_PER_EXPERT  = 3                         # gate_proj, up_proj, down_proj

# Tensor type IDs
TENSOR_GATE = 0
TENSOR_UP   = 1
TENSOR_DOWN = 2

# ── Struct definitions (little-endian, packed) ─────────────────
#
# FILE HEADER — 64 bytes
#   magic(8s) + version(I) + num_moe_layers(I) + max_experts(I) +
#   total_experts(I) + table_offset(Q) + data_offset(Q) +
#   group_size(I) + bits(I) + d_model(I) + vocab_size(I) + ffn_dim(I) + reserved_ext(I)
HEADER_FMT = struct.Struct("<8sIIIIQQIIIIII")
assert HEADER_FMT.size == 64, f"Header size error: {HEADER_FMT.size}"

# EXPERT TABLE ENTRY — 48 bytes
#   layer_id(I) + expert_id(I) + byte_offset(Q) + raw_size(Q) +
#   padded_size(Q) + group_size(I) + num_groups(Q) + reserved(xxxx)
EXPERT_ENTRY_FMT = struct.Struct("<IIQQQIQxxxx")
assert EXPERT_ENTRY_FMT.size == 48, f"ExpertEntry size error: {EXPERT_ENTRY_FMT.size}"

# TENSOR DESCRIPTOR — 44 bytes
#   tensor_type(B) + ndim(B) + pad(xx) + rows(I) + cols(I) +
#   packed_offset(Q) + packed_size(Q) + scales_offset(Q) + scales_size(Q)
TENSOR_DESC_FMT = struct.Struct("<BBxxIIQQQQ")
assert TENSOR_DESC_FMT.size == 44, f"TensorDesc size error: {TENSOR_DESC_FMT.size}"

# ARCH BLOCK — 128 bytes (format v2; mirrored by SmoeArchBlock in common.hpp)
# Located via SmoeHeader.reserved_ext (byte offset; 0 = absent). Sits in the
# alignment gap between the descriptor tables and data_offset, 64-byte aligned.
#   magic(4s) + block_version(I) + block_size(I) + flags(I) +
#   rope_theta(f) + num_heads(I) + num_kv_heads(I) + head_dim(I) +
#   moe_top_k(I) + moe_ffn_dim(I) + dense_ffn_dim(I) + shared_expert_ffn_dim(I) +
#   activation(I) + num_hidden_layers(I) + rms_norm_eps(f) +
#   model_type(24s NUL-padded) + reserved(44x)
ARCH_MAGIC   = b"SARC"
ARCH_VERSION = 1
ARCH_FMT     = struct.Struct("<4sIIIfIIIIIIIIIf24s44x")
assert ARCH_FMT.size == 128, f"ArchBlock size error: {ARCH_FMT.size}"

ACT_SILU = 0                       # SiLU/SwiGLU — the only activation the kernels implement

ARCH_FLAG_NORM_TOPK     = 1 << 0   # renormalise selected top-k weights to sum 1
ARCH_FLAG_QK_NORM       = 1 << 1   # per-head Q/K RMS norm before RoPE (Qwen3)
ARCH_FLAG_DENSE_LAYER_0 = 1 << 2   # layer 0 is a dense MLP (DeepSeek)


# ═════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════

@dataclass
class TensorDescriptor:
    """Metadata for one quantised weight tensor within an expert blob."""
    tensor_type:   int    # TENSOR_GATE | TENSOR_UP | TENSOR_DOWN
    rows:          int
    cols:          int
    packed_offset: int    # byte offset within the expert blob
    packed_size:   int    # bytes of 2-bit packed weights
    scales_offset: int    # byte offset within the expert blob
    scales_size:   int    # bytes of float16 scale factors

    @property
    def num_weights(self) -> int:
        return self.rows * self.cols

    @property
    def num_groups(self) -> int:
        return math.ceil(self.num_weights / Q2_GROUP_SIZE)


@dataclass
class ExpertMeta:
    """Metadata for an expert, after it has been serialised to the vault."""
    layer_id:     int
    expert_id:    int
    tensor_descs: list[TensorDescriptor]
    raw_size:     int
    padded_size:  int
    total_groups: int
    file_offset:  int


def _align_up(value: int, alignment: int) -> int:
    """Round value up to the nearest multiple of alignment."""
    return math.ceil(value / alignment) * alignment


# ═════════════════════════════════════════════════════════════════
# QUANTISATION: SMOE-Q2
# ═════════════════════════════════════════════════════════════════

def quantize_smoeq2(
    weights: np.ndarray,
    group_size: int = Q2_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantise a weight tensor to SMOE-Q2 format.

    Scheme: symmetric group absmax, 4 levels, 2 bits per weight.
    Packing: 4 codes per byte, LSB-first (bits 0-1, 2-3, 4-5, 6-7).

    Args:
        weights:    float32 array of any shape (will be flattened internally)
        group_size: number of weights per quantisation group

    Returns:
        packed: uint8 ndarray, shape (ceil(n / 4),)   — 2-bit packed weights
        scales: float16 ndarray, shape (ceil(n / group_size),) — per-group absmax
    """
    flat = weights.astype(np.float32, copy=False).ravel()
    n    = len(flat)

    # Pad to exact multiple of group_size
    pad_g   = (group_size - (n % group_size)) % group_size
    padded  = np.pad(flat, (0, pad_g))

    n_groups = len(padded) // group_size
    grouped  = padded.reshape(n_groups, group_size)

    # ── MSE Scale Optimizer ──
    # Instead of naive absmax (which destroys sparsity when outliers exist),
    # we grid-search 64 candidate scales between 0.1*absmax and 1.0*absmax
    # to find the scale that minimizes the L2 error for each group.
    
    absmax = np.max(np.abs(grouped), axis=1, keepdims=True)  # shape (G, 1)
    safe_abs = np.where(absmax == 0.0, 1.0, absmax)
    
    # 64 candidate multipliers
    ratios = np.linspace(0.1, 1.0, 64, dtype=np.float32).reshape(64, 1, 1)
    candidate_scales = safe_abs * ratios  # (64, G, 1)
    
    # We broadcast W to (64, G, 64) virtually to compute MSE
    # (W / s) * 1.5 => round to codes
    # Decoded = (codes - 1.5) * (1.0 / 1.5) * s
    W = grouped[np.newaxis, :, :]  # (1, G, 64)
    code_float = (W / candidate_scales) * 1.5
    codes_search = np.clip(np.round(code_float + 1.5), 0, 3)
    decoded = (codes_search - 1.5) * (1.0 / 1.5) * candidate_scales
    
    # L2 Error (MSE)
    mse = np.sum((W - decoded) ** 2, axis=2)  # (64, G)
    
    # Best scale per group
    best_idx = np.argmin(mse, axis=0)  # (G,)
    best_scales = candidate_scales[best_idx, np.arange(n_groups), 0]  # (G,)
    scales = best_scales.astype(np.float16)

    # Now compute the final 2-bit codes using the optimal scales
    safe_best = np.where(best_scales == 0.0, 1.0, best_scales).reshape(n_groups, 1)
    final_code_float = (grouped / safe_best) * 1.5
    codes = np.clip(np.round(final_code_float + 1.5).astype(np.int32), 0, 3).astype(np.uint8)

    # Trim group-padding, then pack 4 codes per byte
    codes_flat = codes.ravel()[:n]
    pad_p      = (4 - (len(codes_flat) % 4)) % 4
    c4         = np.pad(codes_flat, (0, pad_p)).reshape(-1, 4)

    packed = (
          (c4[:, 0] & 0x3)
        | ((c4[:, 1] & 0x3) << 2)
        | ((c4[:, 2] & 0x3) << 4)
        | ((c4[:, 3] & 0x3) << 6)
    ).astype(np.uint8)

    return packed, scales


def dequantize_smoeq2(
    packed: np.ndarray,
    scales: np.ndarray,
    n_weights: int,
    group_size: int = Q2_GROUP_SIZE,
) -> np.ndarray:
    """
    Inverse of quantize_smoeq2. Mirrors the Metal kernel formula exactly.
    Used for reconstruction-error measurement and unit testing.
    """
    codes = np.empty(len(packed) * 4, dtype=np.float32)
    codes[0::4] = ( packed        & 0x3).astype(np.float32)
    codes[1::4] = ((packed >> 2)  & 0x3).astype(np.float32)
    codes[2::4] = ((packed >> 4)  & 0x3).astype(np.float32)
    codes[3::4] = ((packed >> 6)  & 0x3).astype(np.float32)
    codes = codes[:n_weights]

    pad     = (group_size - (n_weights % group_size)) % group_size
    codes_p = np.pad(codes, (0, pad)).reshape(-1, group_size)
    levels  = (codes_p - 1.5) / 1.5                                # ∈ [-1, +1]
    recon   = (levels * scales.astype(np.float32)[:, np.newaxis])

    return recon.ravel()[:n_weights]


def measure_q2_error(
    original: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
) -> dict:
    """Quantise quality metrics: RMSE, absmax error, and SNR."""
    orig_flat = original.astype(np.float32).ravel()
    recon     = dequantize_smoeq2(packed, scales, len(orig_flat))
    diff      = orig_flat - recon
    rmse      = float(np.sqrt(np.mean(diff ** 2)))
    absmax    = float(np.max(np.abs(diff)))
    var_s     = float(np.var(orig_flat))
    var_n     = float(np.var(diff))
    snr_db    = 10.0 * math.log10(max(var_s / max(var_n, 1e-30), 1e-12))
    return {"rmse": rmse, "absmax_err": absmax, "snr_db": snr_db}


# ═════════════════════════════════════════════════════════════════
# QUANTISATION: SMOE-Q4
# ═════════════════════════════════════════════════════════════════

def quantize_smoeq4(
    weights: np.ndarray,
    group_size: int = Q2_GROUP_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantise a weight tensor to SMOE-Q4 format.
    Scheme: symmetric group absmax, 16 levels, 4 bits per weight.
    Packing: 2 codes per byte, LSB-first (bits 0-3, 4-7).
    """
    flat = weights.astype(np.float32, copy=False).ravel()
    n    = len(flat)

    pad_g   = (group_size - (n % group_size)) % group_size
    padded  = np.pad(flat, (0, pad_g))

    n_groups = len(padded) // group_size
    grouped  = padded.reshape(n_groups, group_size)

    # ── Standard Absmax Optimizer for 4-bit ──
    # Unlike 2-bit, 4-bit has 16 bins, making standard absmax perfectly
    # sufficient and ~100x faster than a massive 64-way grid search.
    absmax = np.max(np.abs(grouped), axis=1, keepdims=True)
    best_scales = np.where(absmax == 0.0, 1.0, absmax)
    scales = best_scales.astype(np.float16).ravel()

    final_code_float = (grouped / best_scales) * 7.5
    codes = np.clip(np.round(final_code_float + 7.5).astype(np.int32), 0, 15).astype(np.uint8)

    codes_flat = codes.ravel()[:n]
    pad_p      = (2 - (len(codes_flat) % 2)) % 2
    c2         = np.pad(codes_flat, (0, pad_p)).reshape(-1, 2)

    packed = (
          (c2[:, 0] & 0xF)
        | ((c2[:, 1] & 0xF) << 4)
    ).astype(np.uint8)

    return packed, scales


def dequantize_smoeq4(
    packed: np.ndarray,
    scales: np.ndarray,
    n_weights: int,
    group_size: int = Q2_GROUP_SIZE,
) -> np.ndarray:
    codes = np.empty(len(packed) * 2, dtype=np.float32)
    codes[0::2] = ( packed       & 0xF).astype(np.float32)
    codes[1::2] = ((packed >> 4) & 0xF).astype(np.float32)
    codes = codes[:n_weights]

    pad     = (group_size - (n_weights % group_size)) % group_size
    codes_p = np.pad(codes, (0, pad)).reshape(-1, group_size)
    levels  = (codes_p - 7.5) / 7.5
    recon   = (levels * scales.astype(np.float32)[:, np.newaxis])

    return recon.ravel()[:n_weights]


def measure_q4_error(
    original: np.ndarray,
    packed: np.ndarray,
    scales: np.ndarray,
) -> dict:
    orig_flat = original.astype(np.float32).ravel()
    recon     = dequantize_smoeq4(packed, scales, len(orig_flat))
    diff      = orig_flat - recon
    rmse      = float(np.sqrt(np.mean(diff ** 2)))
    absmax    = float(np.max(np.abs(diff)))
    var_s     = float(np.var(orig_flat))
    var_n     = float(np.var(diff))
    snr_db    = 10.0 * math.log10(max(var_s / max(var_n, 1e-30), 1e-12))
    return {"rmse": rmse, "absmax_err": absmax, "snr_db": snr_db}


# ═════════════════════════════════════════════════════════════════
# TOPOLOGY DETECTION
# ═════════════════════════════════════════════════════════════════

_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def build_tensor_index(model_files: list[Path]) -> dict[str, Path]:
    """
    Scan all shards and build a {tensor_key → shard_path} index.
    Avoids re-scanning every shard for each individual tensor lookup.
    """
    index: dict[str, Path] = {}
    for fpath in model_files:
        with safe_open(str(fpath), framework="numpy") as sf:
            for key in sf.keys():
                index[key] = fpath
    return index


def detect_moe_topology(tensor_index: dict[str, Path]) -> dict:
    """
    Infer the full MoE layer / expert topology from tensor key names.

    Returns a dict with:
        moe_layers          — sorted list of layer indices with expert blocks
        experts_per_layer   — {layer_id: sorted list of expert_ids}
        num_moe_layers      — int
        max_experts         — int (largest expert count across all layers)
    """
    per_layer: dict[int, set[int]] = {}

    for key in tensor_index:
        m = _EXPERT_RE.match(key)
        if m:
            per_layer.setdefault(int(m.group(1)), set()).add(int(m.group(2)))

    sorted_layers = sorted(per_layer)
    return {
        "moe_layers":        sorted_layers,
        "experts_per_layer": {lid: sorted(per_layer[lid]) for lid in sorted_layers},
        "num_moe_layers":    len(sorted_layers),
        "max_experts":       max((len(v) for v in per_layer.values()), default=0),
    }

def extract_model_dimensions(tensor_index: dict[str, Path]) -> dict:
    dim = {"d_model": 2048, "vocab_size": 102400, "ffn_dim": 14336}
    
    embed_key = "model.embed_tokens.weight"
    if embed_key in tensor_index:
        with safe_open(str(tensor_index[embed_key]), framework="numpy") as sf:
            t = sf.get_slice(embed_key)
            dim["vocab_size"] = t.get_shape()[0]
            dim["d_model"] = t.get_shape()[1]
            
    expert_key = None
    for k in tensor_index:
        if _EXPERT_RE.match(k):
            expert_key = k
            break
            
    if expert_key:
        with safe_open(str(tensor_index[expert_key]), framework="numpy") as sf:
            t = sf.get_slice(expert_key)
            dim["ffn_dim"] = t.get_shape()[0]
            
    return dim

def read_hf_arch_config(model_dir: Path, tensor_index: dict[str, Path]) -> dict:
    """
    Read the checkpoint's HF config.json and map it to the arch-block fields.

    Per-family key mappings for qwen3_moe and the deepseek family; anything
    else falls back to the generic HF key names with a warning. has_qk_norm
    and has_dense_layer_0 come from tensor presence, not config — the shapes
    on disk are authoritative. `model_dir` may also be a direct path to a
    config.json (upgrade tool: no shards available; caller must then fix up
    the tensor-presence flags itself).
    """
    config_path = model_dir if model_dir.is_file() else model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"{config_path} not found — the arch block needs the checkpoint's "
            f"config.json (re-download with it included)."
        )
    with open(config_path) as f:
        raw = json.load(f)

    model_type = str(raw.get("model_type", "unknown"))
    known = ("qwen3_moe", "deepseek", "deepseek_v2", "deepseek_v3", "qwen2_moe")
    if model_type not in known:
        console.print(
            f"  [yellow]⚠ model_type '{model_type}' has no dedicated mapping — "
            f"using generic HF keys[/yellow]"
        )

    hidden_act = str(raw.get("hidden_act", "silu"))
    if hidden_act != "silu":
        raise ValueError(
            f"hidden_act '{hidden_act}' — the engine kernels implement SiLU/SwiGLU only."
        )

    num_heads = int(raw["num_attention_heads"])
    head_dim  = int(raw.get("head_dim") or raw["hidden_size"] // num_heads)

    arch = {
        "model_type":            model_type,
        "rope_theta":            float(raw.get("rope_theta", 10_000.0)),
        "num_heads":             num_heads,
        "num_kv_heads":          int(raw.get("num_key_value_heads") or num_heads),
        "head_dim":              head_dim,
        "moe_top_k":             int(raw.get("num_experts_per_tok") or raw.get("moe_top_k", 8)),
        "moe_ffn_dim":           int(raw.get("moe_intermediate_size") or raw.get("intermediate_size", 0)),
        "dense_ffn_dim":         int(raw.get("intermediate_size", 0)),
        "shared_expert_ffn_dim": 0,
        "activation":            ACT_SILU,
        "num_hidden_layers":     int(raw.get("num_hidden_layers", 0)),
        "rms_norm_eps":          float(raw.get("rms_norm_eps", 1e-6)),
        "norm_topk_prob":        bool(raw.get("norm_topk_prob", False)),
        # Tensor presence is authoritative for these two:
        "has_qk_norm":           "model.layers.0.self_attn.q_norm.weight" in tensor_index,
        "has_dense_layer_0":     "model.layers.0.mlp.gate_proj.weight" in tensor_index,
    }

    # DeepSeek family: shared experts + leading dense layers
    n_shared = int(raw.get("n_shared_experts") or 0)
    if n_shared and arch["moe_ffn_dim"]:
        arch["shared_expert_ffn_dim"] = n_shared * arch["moe_ffn_dim"]

    return arch


def build_arch_block(arch: dict) -> bytes:
    """Serialise the arch dict into the 128-byte SARC block."""
    flags = 0
    if arch["norm_topk_prob"]:      flags |= ARCH_FLAG_NORM_TOPK
    if arch["has_qk_norm"]:         flags |= ARCH_FLAG_QK_NORM
    if arch["has_dense_layer_0"]:   flags |= ARCH_FLAG_DENSE_LAYER_0

    return ARCH_FMT.pack(
        ARCH_MAGIC,
        ARCH_VERSION,
        ARCH_FMT.size,
        flags,
        arch["rope_theta"],
        arch["num_heads"],
        arch["num_kv_heads"],
        arch["head_dim"],
        arch["moe_top_k"],
        arch["moe_ffn_dim"],
        arch["dense_ffn_dim"],
        arch["shared_expert_ffn_dim"],
        arch["activation"],
        arch["num_hidden_layers"],
        arch["rms_norm_eps"],
        arch["model_type"].encode("utf-8")[:24],
    )


def unpack_arch_block(blob: bytes) -> dict:
    """Decode a 128-byte SARC block back into a dict (validator / tools)."""
    (magic, block_version, block_size, flags, rope_theta, num_heads,
     num_kv_heads, head_dim, moe_top_k, moe_ffn_dim, dense_ffn_dim,
     shared_ffn, activation, num_hidden_layers, rms_norm_eps,
     model_type) = ARCH_FMT.unpack(blob)
    assert magic == ARCH_MAGIC,          f"Arch magic mismatch: {magic!r}"
    assert block_size == ARCH_FMT.size,  f"Arch block_size mismatch: {block_size}"
    return {
        "block_version":         block_version,
        "model_type":            model_type.rstrip(b"\x00").decode("utf-8"),
        "rope_theta":            rope_theta,
        "num_heads":             num_heads,
        "num_kv_heads":          num_kv_heads,
        "head_dim":              head_dim,
        "moe_top_k":             moe_top_k,
        "moe_ffn_dim":           moe_ffn_dim,
        "dense_ffn_dim":         dense_ffn_dim,
        "shared_expert_ffn_dim": shared_ffn,
        "activation":            activation,
        "num_hidden_layers":     num_hidden_layers,
        "rms_norm_eps":          rms_norm_eps,
        "norm_topk_prob":        bool(flags & ARCH_FLAG_NORM_TOPK),
        "has_qk_norm":           bool(flags & ARCH_FLAG_QK_NORM),
        "has_dense_layer_0":     bool(flags & ARCH_FLAG_DENSE_LAYER_0),
    }


def classify_scout_keys(tensor_index: dict[str, Path]) -> list[str]:
    """
    Return all tensor keys that belong to the Surface Scout —
    i.e., everything EXCEPT routed expert weights.
    """
    return [k for k in tensor_index if not _EXPERT_RE.match(k)]


# ═════════════════════════════════════════════════════════════════
# EXPERT BLOCK BUILDER
# ═════════════════════════════════════════════════════════════════

def load_expert_tensors(
    tensor_index: dict[str, Path],
    layer_id: int,
    expert_id: int,
) -> dict[str, np.ndarray]:
    """
    Load {gate, up, down} weight arrays for one expert from the shard index.
    Raises KeyError if any weight tensor is missing.
    """
    tensors: dict[str, np.ndarray] = {}
    for suffix, name in [("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")]:
        key   = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}.weight"
        fpath = tensor_index.get(key)
        if fpath is None:
            raise KeyError(f"Missing tensor in index: {key}")
        with safe_open(str(fpath), framework="numpy") as sf:
            tensors[name] = sf.get_tensor(key)
    return tensors


def build_expert_blob(
    tensors: dict[str, np.ndarray],
    group_size: int = Q2_GROUP_SIZE,
    bits: int = 2,
) -> tuple[bytes, list[TensorDescriptor]]:
    """
    Quantise and pack the three expert weight tensors into a contiguous byte blob.

    Blob layout (no internal alignment padding between tensors):
        [gate packed bytes][gate scale bytes]
        [up   packed bytes][up   scale bytes]
        [down packed bytes][down scale bytes]

    Returns:
        blob_bytes   : raw bytes of the expert data block
        tensor_descs : [TensorDescriptor × 3] with byte offsets into blob
    """
    parts:  list[bytes]            = []
    descs:  list[TensorDescriptor] = []
    cursor: int                    = 0

    for type_id, key in [(TENSOR_GATE, "gate"), (TENSOR_UP, "up"), (TENSOR_DOWN, "down")]:
        w    = tensors[key].astype(np.float32, copy=False)
        rows = w.shape[0]
        cols = w.shape[1] if w.ndim > 1 else 1

        if bits == 4:
            packed, scales = quantize_smoeq4(w, group_size)
        else:
            packed, scales = quantize_smoeq2(w, group_size)
            
        pb = packed.tobytes()
        sb = scales.tobytes()   # float16 → 2 bytes per scale

        descs.append(TensorDescriptor(
            tensor_type=type_id,
            rows=rows,
            cols=cols,
            packed_offset=cursor,
            packed_size=len(pb),
            scales_offset=cursor + len(pb),
            scales_size=len(sb),
        ))

        parts.extend([pb, sb])
        cursor += len(pb) + len(sb)

    return b"".join(parts), descs





# ═════════════════════════════════════════════════════════════════
# SCOUT EXTRACTOR
# ═════════════════════════════════════════════════════════════════

def extract_and_save_scout(
    tensor_index: dict[str, Path],
    scout_keys: list[str],
    output_path: Path,
) -> None:
    """
    Load all Surface Scout tensors from the shard index and save them
    to a single consolidated .safetensors file.
    """
    scout_tensors: dict[str, np.ndarray] = {}
    for key in scout_keys:
        fpath = tensor_index[key]
        with safe_open(str(fpath), framework="numpy") as sf:
            scout_tensors[key] = sf.get_tensor(key)

    safetensors_save(scout_tensors, str(output_path))
    size_mb = output_path.stat().st_size / 1_048_576
    console.print(
        f"  [bold green]Scout  :[/bold green] {len(scout_tensors)} tensors "
        f"→ {output_path.name} ({size_mb:.1f} MB)"
    )


# ═════════════════════════════════════════════════════════════════
# VAULT VALIDATOR
# ═════════════════════════════════════════════════════════════════

def validate_vault(smoe_path: Path, sample_count: int = 3) -> None:
    """
    Sanity-check the written vault:
      1. Verify magic bytes and version.
      2. Verify that the first N expert blob offsets are 16 KB-aligned.
      3. Print key header fields.
    """
    with open(smoe_path, "rb") as f:
        # Header
        (magic, version, n_layers, max_exp, total_exp, tbl_off, data_off,
         grp_sz, bits, d_model, vocab_size, ffn_dim, reserved_ext) = \
            HEADER_FMT.unpack(f.read(HEADER_FMT.size))

        assert magic == SMOE_MAGIC, f"Magic mismatch: {magic!r}"
        assert SMOE_VERSION_MIN <= version <= SMOE_VERSION, \
            f"Unsupported version: {version}"

        console.print(
            f"  [cyan]✓[/cyan] Header valid — v{version}  "
            f"{n_layers} MoE layers  {max_exp} experts/layer  "
            f"{total_exp} total  group_size={grp_sz}  data@{data_off:#x}"
        )

        # Arch block (v2)
        if version >= 2:
            assert reserved_ext != 0, "v2 vault but reserved_ext (arch offset) is 0"
            assert reserved_ext + ARCH_FMT.size <= data_off, \
                f"Arch block at {reserved_ext:#x} overruns data_offset {data_off:#x}"
            f.seek(reserved_ext)
            arch = unpack_arch_block(f.read(ARCH_FMT.size))
            console.print(
                f"  [cyan]✓[/cyan] Arch block @ {reserved_ext:#x} — "
                f"{arch['model_type']}  theta={arch['rope_theta']:.0f}  "
                f"top_k={arch['moe_top_k']}  "
                f"heads={arch['num_heads']}/{arch['num_kv_heads']}×{arch['head_dim']}  "
                f"norm_topk={arch['norm_topk_prob']}  qk_norm={arch['has_qk_norm']}  "
                f"dense_l0={arch['has_dense_layer_0']}  "
                f"shared_ffn={arch['shared_expert_ffn_dim']}"
            )
        elif reserved_ext != 0:
            console.print(
                f"  [yellow]⚠ v1 vault with nonzero reserved_ext={reserved_ext:#x}[/yellow]"
            )

        # Expert table samples
        f.seek(tbl_off)
        for _ in range(min(sample_count, total_exp)):
            lid, eid, off, raw, pad, gs, ng = EXPERT_ENTRY_FMT.unpack(
                f.read(EXPERT_ENTRY_FMT.size)
            )
            assert off % PAGE_SIZE == 0, f"L{lid}E{eid} not 16 KB-aligned: {off:#x}"
            console.print(
                f"  [cyan]✓[/cyan] L{lid:02d}E{eid:03d} "
                f"off={off:#010x}  raw={raw // 1024} KB  "
                f"padded={pad // 1024} KB  groups={ng}"
            )

    console.print("  [bold green]✓ Vault validation passed[/bold green]" if _RICH
                  else "  ✓ Vault validation passed")


# ═════════════════════════════════════════════════════════════════
# RICH / PLAIN UI HELPERS
# ═════════════════════════════════════════════════════════════════

def _print_topology(topo: dict) -> None:
    if _RICH:
        t = Table(title="Detected MoE Topology", box=rich_box.SIMPLE_HEAVY)
        t.add_column("Property",    style="cyan",       no_wrap=True)
        t.add_column("Value",       style="bold green")
        t.add_row("MoE Layers",           str(topo["num_moe_layers"]))
        t.add_row("Experts / Layer",      str(topo["max_experts"]))
        t.add_row("Total Expert Blocks",  str(topo["num_moe_layers"] * topo["max_experts"]))
        t.add_row("Layer Range",
                  f"[{topo['moe_layers'][0]} … {topo['moe_layers'][-1]}]"
                  if topo["moe_layers"] else "N/A")
        console.print(t)
    else:
        print(f"  MoE layers   : {topo['num_moe_layers']}")
        print(f"  Experts/layer: {topo['max_experts']}")
        print(f"  Total blocks : {topo['num_moe_layers'] * topo['max_experts']}")


def _print_summary(
    experts:         list[ExpertMeta],
    total_size:      int,
    total_raw_bytes: int,
    total_q2_bytes:  int,
    t_quant:         float,
    t_write:         float,
) -> None:
    ratio   = total_raw_bytes / total_q2_bytes if total_q2_bytes else 0.0
    avg_kb  = total_size / max(len(experts), 1) / 1024

    if _RICH:
        t = Table(title="⚡  Shatter Complete", box=rich_box.DOUBLE_EDGE, border_style="green")
        t.add_column("Metric",  style="cyan",       no_wrap=True)
        t.add_column("Value",   style="bold white")
        t.add_row("Experts written",    f"{len(experts):,}")
        t.add_row("Quantisation time",  f"{t_quant:.1f} s")
        t.add_row("Disk write time",    f"{t_write:.1f} s")
        t.add_row("Raw weight volume",  f"{total_raw_bytes  / 1_073_741_824:.3f} GB")
        t.add_row("Compressed volume",  f"{total_q2_bytes   / 1_073_741_824:.3f} GB")
        t.add_row("Compression ratio",  f"{ratio:.2f}×")
        t.add_row("Avg expert block",   f"{avg_kb:.1f} KB")
        t.add_row("Vault total size",   f"{total_size / 1_073_741_824:.3f} GB")
        t.add_row("Page alignment",     "✓ 16 KB")
        console.print(t)
    else:
        print(f"\n  Experts: {len(experts)} | Ratio: {ratio:.2f}× | Avg: {avg_kb:.1f} KB")
        print(f"  Raw: {total_raw_bytes/1e9:.3f} GB  → Compressed: {total_q2_bytes/1e9:.3f} GB")


# ═════════════════════════════════════════════════════════════════
# CLI ENTRYPOINT
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="shatter_moe.py",
        description="S-MoE Seismic Weight Sculptor — Supported MoE → .smoe vault",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model_dir",       type=Path,
                        help="HuggingFace model directory (contains *.safetensors shards)")
    parser.add_argument("output_dir",      type=Path,
                        help="Output directory for .smoe vault and .scout.safetensors")
    parser.add_argument("--group-size",    type=int, default=Q2_GROUP_SIZE,
                        help=f"Q2 quantisation group size (default: {Q2_GROUP_SIZE})")
    parser.add_argument("--validate",      action="store_true",
                        help="Validate the output vault after writing")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Detect topology and estimate output sizes without writing")
    parser.add_argument("--max-layers",    type=int, default=None,
                        help="Process only the first N MoE layers (for quick testing)")
    parser.add_argument("--max-experts",   type=int, default=None,
                        help="Process only the first N experts per layer (for quick testing)")
    parser.add_argument("--bits",          type=int, default=4, choices=[2, 4],
                        help="Quantisation bits per parameter (default: 4)")
    parser.add_argument("--measure-error", action="store_true",
                        help="Compute RMSE / SNR for the first expert's gate_proj (slow)")

    args = parser.parse_args()

    # ── Banner ─────────────────────────────────────────────────────
    if _RICH:
        console.print(Panel.fit(
            "[bold magenta]⚡  S-MoE  ·  Seismic Weight Sculptor[/bold magenta]\n"
            "[dim]shatter_moe.py   Phase 1   SMOE-Q4   4-bit vault builder[/dim]",
            border_style="magenta",
        ))
    else:
        print("\n" + "═" * 60)
        print("   S-MoE — Seismic Weight Sculptor")
        print("═" * 60)

    # ── Discover model shards ─────────────────────────────────────
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        console.print(f"[red]ERROR: model_dir not found: {model_dir}[/red]")
        sys.exit(1)

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        console.print("[red]ERROR: No .safetensors files found in model directory.[/red]")
        sys.exit(1)

    console.print(f"\n  Model  : [bold]{model_dir.name}[/bold]")
    console.print(f"  Shards : {len(shards)} .safetensors file(s)")

    # ── Build tensor index ────────────────────────────────────────
    console.print("  Indexing tensors…")
    tensor_index = build_tensor_index(shards)
    console.print(f"  Tensors: [bold]{len(tensor_index):,}[/bold] total keys indexed")

    # ── Detect MoE topology ───────────────────────────────────────
    topology = detect_moe_topology(tensor_index)
    dims = extract_model_dimensions(tensor_index)
    if topology["num_moe_layers"] == 0:
        console.print(
            "[red]ERROR: No MoE expert layers detected. "
            "Is this a supported fine-grained MoE model?[/red]"
        )
        sys.exit(1)
    _print_topology(topology)

    # ── Arch block source: the checkpoint's config.json ───────────
    try:
        arch = read_hf_arch_config(model_dir, tensor_index)
    except (FileNotFoundError, ValueError, KeyError) as e:
        console.print(f"[red]ERROR: {e}[/red]")
        sys.exit(1)
    console.print(
        f"  Arch    : [bold]{arch['model_type']}[/bold]"
        f"  theta={arch['rope_theta']:.0f}"
        f"  top_k={arch['moe_top_k']}"
        f"  heads={arch['num_heads']}/{arch['num_kv_heads']}×{arch['head_dim']}"
        f"  norm_topk={arch['norm_topk_prob']}"
    )

    # ── Dry run early exit ────────────────────────────────────────
    if args.dry_run:
        # Estimate output sizes without loading any weights
        n_exp    = topology["num_moe_layers"] * topology["max_experts"]
        console.print(f"\n  [yellow]Dry run — no files written.[/yellow]")
        console.print(f"  Estimated expert blocks to shatter: {n_exp:,}")
        return

    # ── Setup output ──────────────────────────────────────────────
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem       = model_dir.name
    smoe_path  = output_dir / f"{stem}.smoe"
    scout_path = output_dir / f"{stem}.scout.safetensors"

    # ── Determine work scope ──────────────────────────────────────
    layers = topology["moe_layers"]
    if args.max_layers:
        layers = layers[: args.max_layers]

    total_work = sum(
        len((topology["experts_per_layer"][lid][: args.max_experts]
             if args.max_experts else topology["experts_per_layer"][lid]))
        for lid in layers
    )
    console.print(f"\n  Shattering [bold]{total_work:,}[/bold] expert block(s)…\n")

    # ── Quantise all experts ──────────────────────────────────────
    experts_meta:    list[ExpertMeta] = []
    total_raw_bytes: int = 0
    total_q2_bytes:  int = 0
    error_metrics:   Optional[dict] = None

    t_quant_start = time.perf_counter()

    meta_bytes = (
        HEADER_FMT.size
        + total_work * EXPERT_ENTRY_FMT.size
        + total_work * TENSORS_PER_EXPERT * TENSOR_DESC_FMT.size
    )
    # Arch block lives 64-byte aligned after the descriptor tables,
    # inside the page-alignment gap before the first expert blob.
    arch_offset = _align_up(meta_bytes, 64)
    data_offset = _align_up(arch_offset + ARCH_FMT.size, PAGE_SIZE)

    console.print(
        f"  Layout  : header={HEADER_FMT.size}B"
        f" + table={total_work * EXPERT_ENTRY_FMT.size}B"
        f" + meta={total_work * TENSORS_PER_EXPERT * TENSOR_DESC_FMT.size}B"
        f" + arch@{arch_offset:#x}"
        f"  →  data@{data_offset:#x}"
    )

    with open(smoe_path, "wb") as f_out:
        # Seek past the metadata block to start writing blobs immediately
        f_out.seek(data_offset)

        def _run(progress=None, task=None):
            nonlocal total_raw_bytes, total_q2_bytes, error_metrics

            for lid in layers:
                eids = topology["experts_per_layer"][lid]
                if args.max_experts:
                    eids = eids[: args.max_experts]

                for eid in eids:
                    if progress and task is not None:
                        progress.update(task, description=f"L{lid:02d} E{eid:03d}")

                    tensors = load_expert_tensors(tensor_index, lid, eid)

                    for arr in tensors.values():
                        total_raw_bytes += arr.nbytes

                    blob_bytes, descs = build_expert_blob(tensors, args.group_size, args.bits)
                    total_q2_bytes += len(blob_bytes)

                    # Measure reconstruction error for the very first expert
                    if args.measure_error and not experts_meta:
                        d      = descs[0]    # gate_proj descriptor
                        packed = np.frombuffer(blob_bytes[d.packed_offset: d.packed_offset + d.packed_size], np.uint8)
                        scales = np.frombuffer(blob_bytes[d.scales_offset: d.scales_offset + d.scales_size], np.float16)
                        if args.bits == 4:
                            error_metrics = measure_q4_error(tensors["gate"].astype(np.float32), packed, scales)
                        else:
                            error_metrics = measure_q2_error(tensors["gate"].astype(np.float32), packed, scales)

                    raw_size = len(blob_bytes)
                    padded_size = _align_up(raw_size, PAGE_SIZE)
                    file_offset = f_out.tell()

                    # Write blob to disk immediately
                    f_out.write(blob_bytes)
                    f_out.write(b"\x00" * (padded_size - raw_size))

                    experts_meta.append(ExpertMeta(
                        layer_id=lid,
                        expert_id=eid,
                        tensor_descs=descs,
                        raw_size=raw_size,
                        padded_size=padded_size,
                        total_groups=sum(d.num_groups for d in descs),
                        file_offset=file_offset,
                    ))

                    # Drop tensors explicitly to force GC and avoid memory spikes
                    del tensors
                    del blob_bytes

                    if progress and task is not None:
                        progress.advance(task)
                    else:
                        print(f"  Processed L{lid:02d} E{eid:03d}")

        if _RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}", justify="left"),
                BarColumn(bar_width=40),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=False,
            ) as prog:
                task = prog.add_task("Shattering…", total=total_work)
                _run(prog, task)
        else:
            _run()

        t_quant = time.perf_counter() - t_quant_start

        # ── Write metadata block at the beginning ─────────────────────
        console.print(f"\n  Writing metadata header → [bold]{smoe_path.name}[/bold]")
        t_write_start = time.perf_counter()
        
        total_size = f_out.tell()
        f_out.seek(0)
        
        f_out.write(HEADER_FMT.pack(
            SMOE_MAGIC,
            SMOE_VERSION,
            topology["num_moe_layers"],
            topology["max_experts"],
            len(experts_meta),
            HEADER_FMT.size,    # table_offset: table starts immediately after header
            data_offset,
            args.group_size,
            args.bits,
            dims["d_model"],
            dims["vocab_size"],
            dims["ffn_dim"],
            arch_offset,        # reserved_ext: v2 arch-block offset
        ))

        for meta in experts_meta:
            f_out.write(EXPERT_ENTRY_FMT.pack(
                meta.layer_id,
                meta.expert_id,
                meta.file_offset,
                meta.raw_size,
                meta.padded_size,
                args.group_size,
                meta.total_groups,
            ))

        for meta in experts_meta:
            for desc in meta.tensor_descs:
                f_out.write(TENSOR_DESC_FMT.pack(
                    desc.tensor_type,
                    2,                  # ndim — always 2 for weight matrices
                    desc.rows,
                    desc.cols,
                    desc.packed_offset,
                    desc.packed_size,
                    desc.scales_offset,
                    desc.scales_size,
                ))
        
        # Zero-pad to the arch block slot, write it, then pad to data_offset
        gap = arch_offset - f_out.tell()
        assert gap >= 0, "Metadata overflow!"
        f_out.write(b"\x00" * gap)
        f_out.write(build_arch_block(arch))
        pad = data_offset - f_out.tell()
        assert pad >= 0, "Arch block overflow!"
        f_out.write(b"\x00" * pad)

        t_write = time.perf_counter() - t_write_start

    # ── Extract Scout weights ─────────────────────────────────────
    scout_keys = classify_scout_keys(tensor_index)
    console.print(f"\n  Extracting Scout ({len(scout_keys)} tensors)…")
    extract_and_save_scout(tensor_index, scout_keys, scout_path)

    # ── Validate (optional) ───────────────────────────────────────
    if args.validate:
        console.print(
            "\n[bold cyan]  Validating vault…[/bold cyan]" if _RICH
            else "\n  Validating vault…"
        )
        validate_vault(smoe_path)

    # ── Error metrics (optional) ──────────────────────────────────
    if error_metrics:
        console.print(
            f"\n  [cyan]Q2 error (L0E0 gate_proj):[/cyan]"
            f"  RMSE={error_metrics['rmse']:.6f}"
            f"  absmax={error_metrics['absmax_err']:.6f}"
            f"  SNR={error_metrics['snr_db']:.1f} dB"
        )

    # ── Final telemetry table ─────────────────────────────────────
    _print_summary(experts_meta, total_size, total_raw_bytes, total_q2_bytes, t_quant, t_write)

    if _RICH:
        console.print(f"\n  [bold green]→ {smoe_path}[/bold green]")
        console.print(f"  [bold green]→ {scout_path}[/bold green]\n")
    else:
        print(f"\n  → {smoe_path}")
        print(f"  → {scout_path}\n")


if __name__ == "__main__":
    main()
