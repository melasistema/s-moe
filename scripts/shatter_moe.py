#!/usr/bin/env python3
"""
shatter_moe.py — The Seismic Weight Sculptor
═══════════════════════════════════════════════════════════════════
S-MoE Engine  │  Phase 1: Offline Weight Shaper
Author: S-MoE / ANDARTIS — Luca Visciola
═══════════════════════════════════════════════════════════════════

Slices a DeepSeek fine-grained MoE model from HuggingFace .safetensors
format into a custom page-aligned .smoe binary vault, optimised for
Direct I/O streaming on Apple Silicon NVMe hardware.

  Usage:
    python shatter_moe.py <model_dir> <output_dir> [options]

  Examples:
    python shatter_moe.py ./deepseek-moe-16b/ ./vault/ --validate
    python shatter_moe.py ./deepseek-moe-16b/ ./vault/ --dry-run
    python shatter_moe.py ./deepseek-moe-16b/ ./vault/ --max-layers 2 --max-experts 4
    python shatter_moe.py ./deepseek-moe-16b/ ./vault/ --measure-error

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
SMOE_VERSION        = 1
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
#   group_size(I) + reserved(20s)
HEADER_FMT = struct.Struct("<8sIIIIQQI20s")
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
class ExpertBlob:
    """A fully quantised expert, ready to be serialised to the vault."""
    layer_id:     int
    expert_id:    int
    tensor_descs: list[TensorDescriptor]
    data:         bytes
    file_offset:  int = field(default=0, repr=False)  # set during layout pass

    @property
    def raw_size(self) -> int:
        return len(self.data)

    @property
    def padded_size(self) -> int:
        return _align_up(len(self.data), PAGE_SIZE)

    @property
    def total_groups(self) -> int:
        return sum(d.num_groups for d in self.tensor_descs)


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

    # Per-group absmax scale (stored as float16 to halve scale array size)
    absmax = np.max(np.abs(grouped), axis=1, keepdims=True)  # shape (G, 1)
    scales = absmax.squeeze(1).astype(np.float16)             # shape (G,)

    # Normalise ∈ [-1, +1], scale to code space [-1.5, +1.5]
    safe_abs   = np.where(absmax == 0.0, 1.0, absmax)
    code_float = (grouped / safe_abs) * 1.5                   # ∈ [-1.5, +1.5]

    # Quantise: round to nearest integer offset, clamp to [0, 3]
    #   code 0 → -1.5  code 1 → -0.5  code 2 → +0.5  code 3 → +1.5
    codes = np.clip(np.round(code_float + 1.5).astype(np.int32), 0, 3).astype(np.uint8)

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
# DeepSeek MoE TOPOLOGY DETECTION
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
# SMOE VAULT WRITER
# ═════════════════════════════════════════════════════════════════

def _layout_experts(experts: list[ExpertBlob]) -> tuple[int, int]:
    """
    Compute and assign file_offset for every expert blob.
    The data section begins at the first 16 KB-aligned byte
    after all metadata (header + expert table + tensor descriptors).

    Returns (data_start_offset, total_file_size).
    """
    meta_bytes = (
        HEADER_FMT.size
        + len(experts) * EXPERT_ENTRY_FMT.size
        + len(experts) * TENSORS_PER_EXPERT * TENSOR_DESC_FMT.size
    )
    data_start = _align_up(meta_bytes, PAGE_SIZE)

    cursor = data_start
    for blob in experts:
        blob.file_offset = cursor
        cursor += blob.padded_size

    return data_start, cursor   # (data_offset, total_file_size)


def write_smoe_vault(
    output_path: Path,
    experts: list[ExpertBlob],
    topology: dict,
) -> dict:
    """
    Serialise all expert blobs into the .smoe binary vault.

    Every expert blob is guaranteed to start on a 16 KB page boundary.
    The file is fully deterministic given the same input experts list.

    Returns a telemetry dict.
    """
    data_offset, total_size = _layout_experts(experts)

    console.print(
        f"  Layout  : header={HEADER_FMT.size}B"
        f" + table={len(experts) * EXPERT_ENTRY_FMT.size}B"
        f" + meta={len(experts) * TENSORS_PER_EXPERT * TENSOR_DESC_FMT.size}B"
        f"  →  data@{data_offset:#x}"
        f"  |  total={total_size / 1_073_741_824:.3f} GB"
    )

    with open(output_path, "wb") as f:

        # ── FILE HEADER (64 bytes) ───────────────────────────────
        f.write(HEADER_FMT.pack(
            SMOE_MAGIC,
            SMOE_VERSION,
            topology["num_moe_layers"],
            topology["max_experts"],
            len(experts),
            HEADER_FMT.size,    # table_offset: table starts immediately after header
            data_offset,
            Q2_GROUP_SIZE,
            b"\x00" * 20,       # reserved
        ))

        # ── EXPERT TABLE (N × 48 bytes) ───────────────────────────
        for blob in experts:
            f.write(EXPERT_ENTRY_FMT.pack(
                blob.layer_id,
                blob.expert_id,
                blob.file_offset,
                blob.raw_size,
                blob.padded_size,
                Q2_GROUP_SIZE,
                blob.total_groups,
            ))

        # ── TENSOR DESCRIPTORS (N × 3 × 44 bytes) ─────────────────
        for blob in experts:
            for desc in blob.tensor_descs:
                f.write(TENSOR_DESC_FMT.pack(
                    desc.tensor_type,
                    2,                  # ndim — always 2 for weight matrices
                    desc.rows,
                    desc.cols,
                    desc.packed_offset,
                    desc.packed_size,
                    desc.scales_offset,
                    desc.scales_size,
                ))

        # ── ZERO-PAD TO DATA SECTION ──────────────────────────────
        pad = data_offset - f.tell()
        assert pad >= 0, f"Metadata overflow: wrote {f.tell()} bytes, data_offset={data_offset}"
        f.write(b"\x00" * pad)

        # ── EXPERT BLOBS (each 16 KB-page-aligned) ────────────────
        for blob in experts:
            pos = f.tell()
            assert pos == blob.file_offset, (
                f"Offset mismatch L{blob.layer_id}E{blob.expert_id}: "
                f"expected {blob.file_offset:#x}, got {pos:#x}"
            )
            assert pos % PAGE_SIZE == 0, f"Expert blob not 16 KB-aligned: {pos:#x}"

            f.write(blob.data)
            f.write(b"\x00" * (blob.padded_size - blob.raw_size))

        assert f.tell() == total_size, \
            f"Final size mismatch: wrote {f.tell()}, expected {total_size}"

    return {
        "total_experts":    len(experts),
        "total_size_bytes": total_size,
        "data_offset":      data_offset,
    }


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
        hdr = HEADER_FMT.unpack(f.read(HEADER_FMT.size))
        magic, version, n_layers, max_exp, total_exp, tbl_off, data_off, grp_sz, _ = hdr

        assert magic   == SMOE_MAGIC,    f"Magic mismatch: {magic!r}"
        assert version == SMOE_VERSION,  f"Version mismatch: {version}"

        console.print(
            f"  [cyan]✓[/cyan] Header valid — v{version}  "
            f"{n_layers} MoE layers  {max_exp} experts/layer  "
            f"{total_exp} total  group_size={grp_sz}  data@{data_off:#x}"
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
    experts:         list[ExpertBlob],
    telemetry:       dict,
    total_raw_bytes: int,
    total_q2_bytes:  int,
    t_quant:         float,
    t_write:         float,
) -> None:
    ratio   = total_raw_bytes / total_q2_bytes if total_q2_bytes else 0.0
    avg_kb  = telemetry["total_size_bytes"] / max(len(experts), 1) / 1024

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
        t.add_row("Vault total size",   f"{telemetry['total_size_bytes'] / 1_073_741_824:.3f} GB")
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
        description="S-MoE Seismic Weight Sculptor — DeepSeek MoE → .smoe vault",
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
    parser.add_argument("--measure-error", action="store_true",
                        help="Compute Q2 RMSE / SNR for the first expert's gate_proj (slow)")

    args = parser.parse_args()

    # ── Banner ─────────────────────────────────────────────────────
    if _RICH:
        console.print(Panel.fit(
            "[bold magenta]⚡  S-MoE  ·  Seismic Weight Sculptor[/bold magenta]\n"
            "[dim]shatter_moe.py   Phase 1   SMOE-Q2   2-bit vault builder[/dim]",
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
    if topology["num_moe_layers"] == 0:
        console.print(
            "[red]ERROR: No MoE expert layers detected. "
            "Is this a DeepSeek fine-grained MoE model?[/red]"
        )
        sys.exit(1)
    _print_topology(topology)

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
    experts:         list[ExpertBlob] = []
    total_raw_bytes: int = 0
    total_q2_bytes:  int = 0
    error_metrics:   Optional[dict] = None

    t_quant_start = time.perf_counter()

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

                blob_bytes, descs = build_expert_blob(tensors, args.group_size)
                total_q2_bytes += len(blob_bytes)

                # Measure reconstruction error for the very first expert
                if args.measure_error and not experts:
                    d      = descs[0]    # gate_proj descriptor
                    packed = np.frombuffer(blob_bytes[d.packed_offset: d.packed_offset + d.packed_size], np.uint8)
                    scales = np.frombuffer(blob_bytes[d.scales_offset: d.scales_offset + d.scales_size], np.float16)
                    error_metrics = measure_q2_error(tensors["gate"].astype(np.float32), packed, scales)

                experts.append(ExpertBlob(
                    layer_id=lid,
                    expert_id=eid,
                    tensor_descs=descs,
                    data=blob_bytes,
                ))

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

    # ── Write vault ───────────────────────────────────────────────
    console.print(f"\n  Writing vault → [bold]{smoe_path.name}[/bold]")
    t_write_start = time.perf_counter()
    telemetry = write_smoe_vault(smoe_path, experts, topology)
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
    _print_summary(experts, telemetry, total_raw_bytes, total_q2_bytes, t_quant, t_write)

    if _RICH:
        console.print(f"\n  [bold green]→ {smoe_path}[/bold green]")
        console.print(f"  [bold green]→ {scout_path}[/bold green]\n")
    else:
        print(f"\n  → {smoe_path}")
        print(f"  → {scout_path}\n")


if __name__ == "__main__":
    main()
