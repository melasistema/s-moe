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
    print(f"{CYAN}Initializing DeepSeek-MoE-16B tokenizer...{RESET}", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-MoE-16B-chat", trust_remote_code=True)
    print(f" {GREEN}[OK]{RESET}\n")
    
    print(f"{MAGENTA}Welcome to the democratic frontier of AI.{RESET}")
    print(f"{MAGENTA}Type 'quit' or 'exit' to escape.{RESET}")
    print("-" * 50)
    
    messages = []
    
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
        
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        token_str = ",".join(map(str, token_ids))
        
        # Auto-tune ring size based on system RAM
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
        if ram_gb > 36:
            ring_size = "4096"
        elif ram_gb > 20:
            ring_size = "2048"
        else:
            ring_size = "512"

        cmd = [
            "build/smoe-engine",
            "--vault", "vault/deepseek-chat.smoe",
            "--scout", "vault/deepseek-chat.scout.safetensors",
            "--tokens-in", token_str,
            "--tokens", "256",
            "--ring", ring_size,
            "--workers", "4",
            "--temperature", "0.3",
            "--top-p", "0.95",
            "--rep-penalty", "1.0",
            "--raw-ids" # Most robust decoding via python
        ]
        
        print(f"{CYAN}{BOLD}S-MoE [RAM: {ram_gb:.1f}GB | Ring: {ring_size}]:{RESET} ", end="", flush=True)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        response_tokens = []
        token_buffer = ""
        try:
            while True:
                char = process.stdout.read(1)
                if not char and process.poll() is not None:
                    break
                if char:
                    if char.isspace():
                        if token_buffer:
                            tok_id = int(token_buffer)
                            token_buffer = ""
                            response_tokens.append(tok_id)
                            # Decode and print safely
                            print(tokenizer.decode([tok_id], skip_special_tokens=True), end="", flush=True)
                    else:
                        token_buffer += char
        except KeyboardInterrupt:
            process.kill()
            print(f"\n{MAGENTA}[Generation Interrupted]{RESET}")
            continue
            
        print()
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text.strip()})

if __name__ == "__main__":
    main()
