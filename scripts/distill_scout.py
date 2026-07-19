#!/usr/bin/env python3
"""
distill_scout.py — The Surface Scout Distiller
═══════════════════════════════════════════════════════════════════
S-MoE Engine  │  Phase 5: Scout Weight Extractor & Router Distiller
Author: S-MoE / ANDARTIS
═══════════════════════════════════════════════════════════════════

Extracts all non-expert tensors (Surface Scout backbone) from a 
HuggingFace supported MoE model shard directory, and optionally
distills the routing gate weights using a calibration dataset.

Modes:
  1. Extraction (Default):
     Zero-shot extraction of embedding, attention-1, shared MLP,
     layer norms, and original router gates. Runs in seconds,
     requires < 1 GB RAM, and needs only NumPy + Safetensors.

  2. Distillation (--distill):
     Trains the routing gates to predict subsequent MoE layers'
     expert activations from the Layer 1 hidden state. Requires
     PyTorch, Transformers, and sufficient RAM to load the model.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Register bfloat16 data type in numpy
try:
    import ml_dtypes
except ImportError:
    print("ERROR: ml_dtypes not installed. Run: pip install ml_dtypes")
    sys.exit(1)

# safetensors (required)
try:
    from safetensors import safe_open
    from safetensors.numpy import save_file as safetensors_save
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors")
    sys.exit(1)

# rich (optional, graceful fallback)
try:
    from rich import box as rich_box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    _RICH = True
    console = Console()
except ImportError:
    _RICH = False
    class _FallbackConsole:
        def print(self, *args, **kw):
            text = " ".join(str(a) for a in args)
            text = re.sub(r"\[/?[^\]]*\]", "", text)
            print(text)
    console = _FallbackConsole()


# Regex to detect routed expert tensors (to be excluded from the scout)
_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def build_tensor_index(model_files: List[Path]) -> Dict[str, Path]:
    """Scan all shards and build a {tensor_key: shard_path} map."""
    index = {}
    for fpath in model_files:
        with safe_open(str(fpath), framework="numpy") as sf:
            for key in sf.keys():
                index[key] = fpath
    return index


def extract_scout_weights(
    tensor_index: Dict[str, Path],
    output_path: Path,
) -> Dict[str, np.ndarray]:
    """
    Extracts all non-expert tensors from the original model shards.
    This includes embeddings, attention layers, shared MLP experts,
    norms, and original gates.
    """
    scout_tensors = {}
    total_tensors = 0
    ignored_experts = 0

    keys = sorted(tensor_index.keys())
    
    # We load tensors one-by-one to keep memory usage under 1 GB
    for key in keys:
        if _EXPERT_RE.match(key):
            ignored_experts += 1
            continue
        
        shard_path = tensor_index[key]
        with safe_open(str(shard_path), framework="numpy") as sf:
            scout_tensors[key] = sf.get_tensor(key)
        total_tensors += 1

    console.print(f"[green]✓[/green] Extracted {total_tensors} non-expert tensors.")
    console.print(f"[dim]Excluded {ignored_experts} routed expert tensors.[/dim]")
    return scout_tensors


def run_distillation(
    model_dir: Path,
    scout_tensors: Dict[str, np.ndarray],
    dataset_path: Optional[Path],
    epochs: int,
    lr: float,
    batch_size: int,
    seq_len: int,
    device_str: str,
) -> Dict[str, np.ndarray]:
    """
    Runs active PyTorch distillation of the MoE gates.
    Trains gate_w[L] to predict layer L expert activation from Layer 1 hidden state.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        console.print("[red]ERROR: Distillation requires torch and transformers.[/red]")
        console.print("Please install them: [bold]pip install torch transformers[/bold]")
        sys.exit(1)

    device = torch.device(device_str)
    console.print(f"Running distillation training on: [bold cyan]{device}[/bold cyan]")

    # 1. Prepare calibration dataset
    class TextDataset(Dataset):
        def __init__(self, texts: List[str], tokenizer, max_len: int):
            self.encodings = tokenizer(
                texts, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt"
            )

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}

    # Load tokenizer
    console.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build calibration texts
    calibration_texts = []
    if dataset_path and dataset_path.exists():
        console.print(f"Loading calibration dataset from {dataset_path}...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            calibration_texts = [line.strip() for line in f if line.strip()]
    else:
        console.print("No calibration dataset provided. Using default fallback sentences...")
        calibration_texts = [
            "Deep learning and mixture of experts models are highly efficient.",
            "Apple Silicon Unified Memory Architecture enables direct zero-copy GPU access.",
            "Surface Scout listens to the acoustic phonons of the routing decisions.",
            "Synthetic Aperture Radar techniques map the subsurface deep strata.",
            "Asynchronous POSIX prefetching avoids NVMe storage latency stalls.",
            "The heavy frontier model executes FFN layers on the GPU thread.",
            "Zero runtime heap allocations are enforced inside the token generation loop.",
            "Softmax and top-k operations identify the active experts per token."
        ] * 8

    dataset = TextDataset(calibration_texts, tokenizer, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Load the original model
    console.print("Loading full frontier model in float16/bfloat16 for trace collection...")
    # Load on CPU/device with appropriate dtype
    torch_dtype = torch.bfloat16 if device.type == "cuda" or (device.type == "mps") else torch.float32
    
    # We load with device_map auto if CUDA is available, else standard load
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # 3. Trace and collect hidden states & gate activations
    console.print("Collecting activation traces ...")
    x1_list = []
    # activation labels for each layer L (1..27)
    labels_list = {L: [] for L in range(1, 28)}

    # We register a hook on Layer 1 output to get x1
    x1_cache = []
    def hook_x1(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        x1_cache.append(hidden.detach())

    # Locate the transformer layers block and register hooks
    # DeepSeek-MoE structure: model.model.layers[L]
    layers = model.model.layers
    layers[1].register_forward_hook(hook_x1)

    # Collect data by running forward passes
    with torch.no_grad():
        for batch in loader:
            x1_cache.clear()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # x1 has shape (batch_size, seq_len, hidden_dim)
            if not x1_cache:
                continue
            x1 = x1_cache[0]
            
            # Apply final model norm to replicate C++ environment
            x1_normed = model.model.norm(x1)
            
            # Flatten batch and seq dimensions
            x1_flat = x1_normed.view(-1, x1_normed.size(-1)).to(torch.float32)
            x1_list.append(x1_flat.cpu())

            # Capture routing decisions for MoE layers 1..27
            for L in range(1, 28):
                # MLP layer is at layers[L].mlp
                mlp = layers[L].mlp
                
                # Get the hidden states entering the MLP layer (which is index 10 of hidden_states)
                # hidden_states contains outputs of all layers. MLP input is the post-attention norm output
                # Alternatively, we can calculate routing from layer L input directly.
                x_L = outputs.hidden_states[L].to(device)
                
                # Apply input/post norms appropriately or use MLP gate directly
                # DeepSeek-MoE gate scoring:
                # gate_logits = MLP.gate(x_L)
                gate_w = mlp.gate.weight  # shape (64, 2048)
                
                # Compute scores
                scores = torch.matmul(x_L.view(-1, x_L.size(-1)), gate_w.t())
                # Top-8 expert selection
                _, top_indices = torch.topk(scores, k=8, dim=-1)
                
                # Create multi-hot binary label (1 if selected, 0 otherwise)
                labels = torch.zeros_like(scores, dtype=torch.float32)
                labels.scatter_(1, top_indices, 1.0)
                labels_list[L].append(labels.cpu())

    # Concatenate features and labels
    X = torch.cat(x1_list, dim=0) # shape (N, 2048)
    Y = {L: torch.cat(labels_list[L], dim=0) for L in range(1, 28)} # shape (N, 64)
    N_samples = X.size(0)
    console.print(f"Collected [bold cyan]{N_samples}[/bold cyan] token activation samples.")

    # 4. Train the 27 gates
    console.print(f"Distilling routing gates (epochs={epochs}, lr={lr}) ...")
    
    # We move X to device
    X = X.to(device)
    
    for L in range(1, 28):
        y_L = Y[L].to(device)
        
        # Linear projection without bias: (2048 -> 64)
        gate_net = nn.Linear(2048, 64, bias=False).to(device)
        # Initialize with original gate weights for fast convergence
        orig_gate_np = scout_tensors[f"model.layers.{L}.mlp.gate.weight"]
        with torch.no_grad():
            gate_net.weight.copy_(torch.from_numpy(orig_gate_np.astype(np.float32)))

        optimizer = optim.AdamW(gate_net.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        # Simple training loop
        dataset_l = torch.utils.data.TensorDataset(X, y_L)
        loader_l = DataLoader(dataset_l, batch_size=256, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for feat, lbl in loader_l:
                optimizer.zero_grad()
                pred = gate_net(feat)
                loss = criterion(pred, lbl)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * feat.size(0)
            
            # Print intermediate progress for the first/last layers
            if L in [1, 14, 27] and epoch == epochs - 1:
                avg_loss = total_loss / N_samples
                console.print(f"  Layer {L:02d} | Final BCE Loss: {avg_loss:.5f}")

        # Update the scout tensor weights with the trained weights (converted back to bf16)
        trained_gate_np = gate_net.weight.detach().cpu().numpy().astype(ml_dtypes.bfloat16)
        scout_tensors[f"model.layers.{L}.mlp.gate.weight"] = trained_gate_np

    console.print("[green]✓[/green] Distillation complete.")
    return scout_tensors


def main():
    parser = argparse.ArgumentParser(
        description="S-MoE Surface Scout Distillation & Extraction Pipeline"
    )
    parser.add_argument("model_dir", type=str, help="Directory containing HF model shards")
    parser.add_argument("output_file", type=str, help="Output path for scout.safetensors")
    
    # Distillation arguments
    parser.add_argument("--distill", action="store_true", help="Run active neural gate distillation")
    parser.add_argument("--dataset", type=str, default=None, help="Path to text calibration file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, mps, cuda)")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_file = Path(args.output_file)

    if _RICH:
        console.print(
            Panel.fit(
                "[bold cyan]⚡ S-MoE  ·  Surface Scout Distiller[/bold cyan]\n"
                "[dim]Phase 5  ·  Neural Router Sculptor[/dim]",
                box=rich_box.DOUBLE,
            )
        )

    # 1. Locate safetensors files
    if not model_dir.exists():
        console.print(f"[red]ERROR: Model directory '{model_dir}' does not exist.[/red]")
        sys.exit(1)
        
    shards = sorted(list(model_dir.glob("*.safetensors")))
    if not shards:
        console.print(f"[red]ERROR: No .safetensors files found in '{model_dir}'.[/red]")
        sys.exit(1)

    console.print(f"Indexing {len(shards)} shards ...")
    tensor_index = build_tensor_index(shards)
    console.print(f"Tensors: {len(tensor_index)} keys found.")

    # 2. Extract Scout Backbone
    console.print("Extracting Surface Scout backbone tensors (Zero-shot base) ...")
    scout_tensors = extract_scout_weights(tensor_index, output_file)

    # 3. Distill Router Gates (optional)
    if args.distill:
        console.print("Starting active distillation...")
        scout_tensors = run_distillation(
            model_dir=model_dir,
            scout_tensors=scout_tensors,
            dataset_path=Path(args.dataset) if args.dataset else None,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device_str=args.device,
        )
    else:
        console.print("[yellow]Extraction mode (zero-shot) — skipping distillation training.[/yellow]")

    # 4. Save file
    console.print(f"Saving scout weights to [bold cyan]{output_file}[/bold cyan] ...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the dictionary of tensors using safetensors
    safetensors_save(scout_tensors, str(output_file))
    console.print("[green]✓[/green] Scout weights saved successfully!")


if __name__ == "__main__":
    main()
