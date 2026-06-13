import os
import sys
import subprocess
import signal
from transformers import AutoTokenizer

print("Loading tokenizer...", end="", flush=True)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-MoE-16B-chat", trust_remote_code=True)
print(" OK")

def chat_loop():
    messages = []
    print("\nS-MoE Chat Console (DeepSeek-MoE-16B)")
    print("Type 'quit' or 'exit' to leave.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            break
            
        if user_input.strip().lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt_text)
        tokens_str = ",".join(map(str, tokens))
        
        cmd = [
            "./build/smoe-engine",
            "--vault", "./vault/deepseek-chat.smoe",
            "--scout", "./vault/deepseek-chat.scout.safetensors",
            "--tokens-in", tokens_str,
            "--tokens", "256",
            "--raw-ids"
        ]
        
        print("S-MoE: ", end="", flush=True)
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        response_tokens = []
        token_str = ""
        try:
            while True:
                output = process.stdout.read(1)
                if output == '' and process.poll() is not None:
                    break
                if output:
                    if output.isspace():
                        if token_str:
                            tok_id = int(token_str)
                            token_str = ""
                            response_tokens.append(tok_id)
                            # decode only the last token and print
                            print(tokenizer.decode([tok_id]), end="", flush=True)
                    else:
                        token_str += output
        except KeyboardInterrupt:
            process.kill()
            print("\n[Interrupted]")
            continue
            
        print()
        response_text = tokenizer.decode(response_tokens)
        messages.append({"role": "assistant", "content": response_text})
            
chat_loop()
