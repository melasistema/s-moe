#!/usr/bin/env python3
import os
import sys
import subprocess
from transformers import AutoTokenizer

# ANSI aesthetic colors for the Utopian experience
CYAN = '\033[96m'
MAGENTA = '\033[95m'
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print(f"{CYAN}{BOLD}=== S-MoE Utopia Console ==={RESET}")
    print(f"{CYAN}Initializing Qwen3-235B tokenizer...{RESET}", end="", flush=True)
    # Must match the shattered checkpoint: Qwen3-235B-A22B-Instruct-2507 (non-thinking).
    # The original Qwen3-235B-A22B (thinking) template injects an empty <think></think>
    # block whose token embeddings (151667/151668) are untrained in the 2507 weights —
    # feeding them mid-prompt collapses the residual stream into degenerate output.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507", trust_remote_code=True)
    print(f" {GREEN}[OK]{RESET}\n")
    
    print(f"{MAGENTA}Welcome to the democratic frontier of AI.{RESET}")
    print(f"{MAGENTA}Type 'quit' or 'exit' to escape.{RESET}")
    print("-" * 50)
    
    messages = [
        {"role": "system", "content": "You are a helpful, respectful, and honest AI coding assistant named S-MoE."}
    ]
    
    while True:
        try:
            user_input = input(f"\n{BOLD}You:{RESET} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
            
        if user_input.strip().lower() in ['quit', 'exit']:
            break
        if not user_input.strip():
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        # Instruct-2507 is a non-thinking model: its template has no <think> block
        # and no enable_thinking switch.
        token_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        token_str = ",".join(map(str, token_ids))
        
        # Ring sizing is delegated to the engine (--ring 0 = auto): it scans the
        # vault for the real slot size (Q2 ~5MB vs Q4 ~10MB blobs) and budgets
        # 25% of free RAM, so a fixed slot count here would over-allocate on Q4.
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)

        eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids_str = str(eos_ids)
        elif isinstance(eos_ids, list):
            eos_ids_str = ",".join(map(str, eos_ids))
        else:
            eos_ids_str = "100001"

        cmd = [
            "build/smoe-engine",
            "--vault", "vault/qwen3-235b-q4.smoe",
            "--scout", "vault/qwen3-235b-q4.scout.safetensors",
            "--tokens-in", token_str,
            "--tokens", "256",
            "--ring", "0",
            "--eos-ids", eos_ids_str,
            "--workers", "4",
            "--temperature", "0.6",
            "--top-p", "0.95",
            "--top-k", "50",
            "--rep-penalty", "1.1",
            "--raw-ids"
        ]
        
        print(f"{CYAN}{BOLD}S-MoE [RAM: {ram_gb:.1f}GB | Ring: auto]:{RESET} ", end="", flush=True)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        response_tokens = []
        token_buffer = ""
        printed_text = ""
        try:
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    if token_buffer:
                        clean_buf = token_buffer.replace('[', '').replace(']', '').strip()
                        if clean_buf.isdigit():
                            tok_id = int(clean_buf)
                            response_tokens.append(tok_id)
                            full_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                            if not full_text.endswith('\ufffd'):
                                new_text = full_text[len(printed_text):]
                                print(new_text, end="", flush=True)
                                printed_text = full_text
                    break
                if char:
                    if char.isspace() or char == '\n':
                        if token_buffer:
                            # Strip brackets if main.cpp uses them, like "[151644]"
                            clean_buf = token_buffer.replace('[', '').replace(']', '').strip()
                            if clean_buf.isdigit():
                                tok_id = int(clean_buf)
                                response_tokens.append(tok_id)
                                # Decode using BPE delta to handle multi-byte chars and space prefixes correctly
                                full_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                                if not full_text.endswith('\ufffd'):
                                    new_text = full_text[len(printed_text):]
                                    print(new_text, end="", flush=True)
                                    printed_text = full_text
                            token_buffer = ""
                    else:
                        token_buffer += char
        except KeyboardInterrupt:
            process.kill()
            print(f"\n{MAGENTA}[Generation Interrupted]{RESET}")
            continue
            
        print()
        process.wait()
        if process.returncode != 0 or not response_tokens:
            stderr_text = process.stderr.read().strip()
            if stderr_text:
                print(f"{MAGENTA}[engine exited with code {process.returncode}]{RESET}")
                print(stderr_text, file=sys.stderr)
            if not response_tokens:
                messages.pop()  # drop the user turn so history stays consistent
                continue

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text.strip()})

if __name__ == "__main__":
    main()
