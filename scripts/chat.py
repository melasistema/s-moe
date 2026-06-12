#!/usr/bin/env python3
"""
chat.py — Interactive Chat Wrapper for S-MoE Engine
═══════════════════════════════════════════════════════════════════
S-MoE Engine  │  Phase 5: User & Developer Experience Console
Author: S-MoE / ANDARTIS
═══════════════════════════════════════════════════════════════════

Provides a beautiful interactive terminal console to chat with the
S-MoE engine. It tokenises your prompts using the official DeepSeek
BPE tokenizer, passes them as token IDs to the C++ smoe-engine, 
and streams the detokenised outputs in real-time with live telemetry.
"""

import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import subprocess
import sys
from pathlib import Path

# Verify transformers is installed
try:
    from transformers import AutoTokenizer
    from rich.console import Console
    from rich.panel import Panel
    from rich import box as rich_box
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False

def main():
    if not _IMPORTS_OK:
        print("ERROR: Missing required Python dependencies (transformers, torch, rich).")
        print("Please install them in your virtual environment:")
        print("  .venv/bin/pip install transformers torch rich")
        sys.exit(1)

    console = Console()

    parser = argparse.ArgumentParser(description="S-MoE Interactive Chat Console")
    parser.add_argument("--vault", type=str, default="vault/deepseek-chat.smoe", help="Path to .smoe vault")
    parser.add_argument("--scout", type=str, default="vault/deepseek-chat.scout.safetensors", help="Path to scout safetensors")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-MoE-16B-chat", help="HuggingFace model ID for tokenizer")
    parser.add_argument("--tokens", type=int, default=100, help="Max tokens to generate per response")
    parser.add_argument("--ring", type=int, default=1024, help="Ring buffer slot count")
    parser.add_argument("--workers", type=int, default=4, help="I/O worker thread count")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    parser.add_argument("--rep-penalty", type=float, default=1.1, help="Repetition penalty")

    args = parser.parse_args()

    # Verify paths
    vault_path = Path(args.vault)
    scout_path = Path(args.scout)
    engine_bin = Path("build/smoe-engine")

    if not engine_bin.exists():
        console.print("[red]ERROR: C++ binary 'build/smoe-engine' not found.[/red]")
        console.print("Please compile the engine first: [bold]make all[/bold]")
        sys.exit(1)

    if not vault_path.exists():
        console.print(f"[red]ERROR: Vault file '{vault_path}' not found.[/red]")
        sys.exit(1)

    # Load tokenizer
    console.print(f"Loading BPE Tokenizer from '{args.model}' ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    except Exception as e:
        console.print(f"[red]Failed to load tokenizer: {e}[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold cyan]⚡ Welcome to S-MoE Interactive Chat Console ⚡[/bold cyan]\n"
            "[dim]Week 5: Live BPE Tokenization + SSD Streaming + GPU Execution[/dim]\n\n"
            "Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to end the session.",
            box=rich_box.DOUBLE,
            border_style="cyan"
        )
    )

    while True:
        # Keep conversational history stateless for now since engine restarts per prompt
        messages = []
        try:
            try:
                prompt = console.input("\n[bold green]User[/bold green] ❯ ")
            except EOFError:
                break
            if not prompt.strip():
                continue
            if prompt.strip().lower() in ["exit", "quit"]:
                console.print("[yellow]Exiting S-MoE chat session.[/yellow]")
                break

            # Append to conversational history
            messages.append({"role": "user", "content": prompt})

            # 1. BPE Tokenise input prompt. DeepSeek-MoE is highly sensitive
            # to formatting. We use a single newline before Assistant: to prevent degradation.
            prompt_str = f"User: {prompt}\nAssistant:"
            token_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
            token_str = ",".join(map(str, token_ids))

            console.print("\n[bold cyan]S-MoE Engine[/bold cyan] ❯", end="")
            sys.stdout.flush()

            # 2. Execute C++ engine passing the tokenized prompt
            cmd = [
                str(engine_bin),
                "--vault", str(vault_path),
                "--scout", str(scout_path),
                "--tokens-in", token_str,
                "--tokens", str(args.tokens),
                "--ring", str(args.ring),
                "--workers", str(args.workers),
                "--temperature", str(args.temperature),
                "--top-p", str(args.top_p),
                "--rep-penalty", str(args.rep_penalty)
            ]

            # Run and let C++ write directly to stdout and stderr, intercepting to clean BPE chars
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
                text=True,
                bufsize=1
            )
            for char in iter(lambda: process.stdout.read(1), ''):
                if char == 'Ġ':
                    sys.stdout.write(' ')
                elif char == 'Ċ':
                    sys.stdout.write('\n')
                elif char == 'ĉ':
                    sys.stdout.write('\t')
                else:
                    sys.stdout.write(char)
                sys.stdout.flush()
            process.wait()
            print() # Print newline after output completion

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted.[/yellow]")
            break

if __name__ == "__main__":
    main()
