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

MAX_NEW_TOKENS = 256


def spawn_engine():
    # Persistent server mode: the engine stays alive across turns, keeping
    # the KV cache, expert ring and scout state warm. Each turn only
    # prefills the token suffix beyond what the engine has already seen.
    cmd = [
        "build/smoe-engine",
        "--vault", "vault/qwen3-235b-q4.smoe",
        "--scout", "vault/qwen3-235b-q4.scout.safetensors",
        "--serve",
        "--ring", "0",
        "--workers", "4",
        "--temperature", "0.6",
        "--top-p", "0.95",
        "--top-k", "50",
        "--rep-penalty", "1.1",
        "--raw-ids",
    ]
    # stderr inherited: boot info ([ring]/[scout]) goes to the terminal.
    return subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=0,
    )


def stream_reply(process, tokenizer):
    """Read token IDs from the engine until <<DONE>>/<<ERR>>, printing
    BPE-delta-decoded text as it arrives. Returns (tokens, err)."""
    response_tokens = []
    token_buffer = ""
    printed_text = ""

    def flush_token(buf):
        nonlocal printed_text
        clean = buf.replace('[', '').replace(']', '').strip()
        if clean.isdigit():
            response_tokens.append(int(clean))
            full_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
            # Hold back while the tail is an incomplete UTF-8 sequence
            if not full_text.endswith('�'):
                new_text = full_text[len(printed_text):]
                print(new_text, end="", flush=True)
                printed_text = full_text

    while True:
        char = process.stdout.read(1)
        if not char:
            return response_tokens, "engine_exit"
        if char.isspace():
            if token_buffer:
                if token_buffer.startswith("<<"):
                    if token_buffer == "<<DONE>>":
                        return response_tokens, None
                    return response_tokens, token_buffer  # <<ERR ...>> fragment
                flush_token(token_buffer)
                token_buffer = ""
        else:
            token_buffer += char


def main():
    print(f"{CYAN}{BOLD}=== S-MoE Utopia Console ==={RESET}")
    print(f"{CYAN}Initializing Qwen3-235B tokenizer...{RESET}", end="", flush=True)
    # Must match the shattered checkpoint: Qwen3-235B-A22B-Instruct-2507 (non-thinking).
    # The original Qwen3-235B-A22B (thinking) template injects an empty <think></think>
    # block whose token embeddings (151667/151668) are untrained in the 2507 weights —
    # feeding them mid-prompt collapses the residual stream into degenerate output.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507", trust_remote_code=True)
    print(f" {GREEN}[OK]{RESET}\n")

    print(f"{CYAN}Starting persistent S-MoE engine...{RESET}", flush=True)
    process = spawn_engine()

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

        # Instruct-2507 is a non-thinking model: its template has no <think>
        # block and no enable_thinking switch. The FULL conversation is sent
        # every turn; the engine prefix-matches against its stored stream and
        # prefills only the new suffix (KV cache reuse).
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        request = f"GEN {MAX_NEW_TOKENS} " + ",".join(map(str, token_ids)) + "\n"

        print(f"{CYAN}{BOLD}S-MoE:{RESET} ", end="", flush=True)

        response_tokens = []
        for attempt in range(2):
            if process.poll() is not None:
                print(f"\n{MAGENTA}[engine died (exit {process.returncode}) — restarting]{RESET}")
                process = spawn_engine()
            try:
                process.stdin.write(request)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                process.kill()
                process = spawn_engine()
                continue

            try:
                response_tokens, err = stream_reply(process, tokenizer)
            except KeyboardInterrupt:
                # No clean way to interrupt one request of a persistent
                # engine yet — restart it so the next turn starts fresh.
                process.kill()
                print(f"\n{MAGENTA}[Generation Interrupted — engine restarted]{RESET}")
                process = spawn_engine()
                response_tokens = []
                break

            if err is None:
                break
            if err == "engine_exit":
                continue  # respawned at top of retry loop
            print(f"\n{MAGENTA}[engine error: {err}]{RESET}")
            break

        print()
        if not response_tokens:
            messages.pop()  # drop the user turn so history stays consistent
            continue

        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response_text.strip()})

    if process.poll() is None:
        process.stdin.close()
        process.wait()


if __name__ == "__main__":
    main()
