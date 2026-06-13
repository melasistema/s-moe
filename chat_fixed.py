import subprocess
import sys
import os
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-MoE-16B-chat", trust_remote_code=True)
    messages = []
    
    while True:
        try:
            user_input = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
            
        if not user_input.strip():
            continue
            
        messages.append({"role": "user", "content": user_input})
        
        # USE tokenize=True directly
        token_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # The engine expects comma-separated
        token_str = ",".join(map(str, token_ids))
        
        cmd = [
            "build/smoe-engine",
            "--vault", "vault/deepseek-chat.smoe",
            "--scout", "vault/deepseek-chat.scout.safetensors",
            "--tokens-in", token_str,
            "--tokens", "256",
            "--ring", "2048",
            "--workers", "4",
            "--temperature", "0.6",
            "--top-p", "0.95"
        ]
        
        print(f"Assistant: ", end="", flush=True)
        
        # Run subprocess and stream stdout
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        assistant_reply = ""
        while True:
            char = process.stdout.read(1)
            if not char:
                break
            sys.stdout.write(char)
            sys.stdout.flush()
            assistant_reply += char
            
        process.wait()
        
        # Append the assistant's reply so we can continue chatting!
        messages.append({"role": "assistant", "content": assistant_reply.strip()})

if __name__ == "__main__":
    main()
