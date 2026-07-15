#!/usr/bin/env python3
"""
serve_openai.py — OpenAI-compatible HTTP front-end for the S-MoE engine.

Wraps the persistent `--serve` engine (GEN/<<DONE>> stdin protocol) in the
OpenAI Chat Completions API, so any OpenAI-compatible client (Open WebUI,
LibreChat, editor plugins, the `openai` SDK, curl) can talk to S-MoE.

Endpoints:
    GET  /                     → webchat.html, a built-in same-origin test UI
    GET  /health               → {"status":"ok", "engine":"up|down"}
    GET  /v1/models            → the single served model
    POST /v1/chat/completions  → streaming (SSE) or non-streaming completion

Design constraints (dictated by the engine, see chat.py):
  • Chat template + tokenization happen HERE (transformers), matching chat.py
    exactly — the engine only ever sees token IDs.
  • The engine holds ONE conversation context and LCP-prefix-matches each
    request against it, so generations must be SERIALIZED. A single lock
    queues concurrent HTTP requests; continued conversations reuse the KV
    cache (fast warm TTFT), unrelated ones re-prefill from the divergence
    point (correct, just slower).
  • Per-request sampling: `temperature`, `top_p`, `top_k` and
    `repetition_penalty` are forwarded via the extended GEN protocol
    (`GEN <max> t=.. p=.. k=.. r=.. <csv>`); the engine's launch values are
    the defaults for anything a request omits. If the engine binary predates
    the extension (replies <<ERR bad_request>>), the server warns and retries
    once without overrides. `max_tokens` / `max_completion_tokens` are
    honoured and capped.
  • `finish_reason` is real: the engine's fin= trailer distinguishes an EOS
    ("stop") from an exhausted token budget ("length"). Old engine binaries
    without the trailer report "stop" for everything.

No new dependencies: Python stdlib + transformers (already used by chat.py).
"""
import argparse
import json
import os
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = os.path.dirname(os.path.abspath(__file__))


class Engine:
    """Owns the persistent smoe-engine subprocess and serializes access."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.proc = None
        # Imported lazily so `--help` / import works without transformers.
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        self.spawn()

    def spawn(self):
        cmd = [
            self.args.engine,
            "--vault", self.args.vault,
            "--scout", self.args.scout,
            "--serve",
            "--ring", str(self.args.ring),
            "--workers", str(self.args.workers),
            "--temperature", str(self.args.temperature),
            "--top-p", str(self.args.top_p),
            "--top-k", str(self.args.top_k),
            "--rep-penalty", str(self.args.rep_penalty),
            "--raw-ids",
        ]
        # stderr inherited → engine boot/telemetry goes to the server console.
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=0, cwd=ROOT,
        )

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    @staticmethod
    def _override_fields(sampling):
        """Map OpenAI-style sampling params onto GEN protocol key=value
        fields (t/p/k/r). Only params the request actually set are sent —
        the engine's launch values cover the rest."""
        if not sampling:
            return ""
        out = []
        for name, tag, cast in (("temperature", "t", float),
                                ("top_p", "p", float),
                                ("top_k", "k", int),
                                ("repetition_penalty", "r", float)):
            v = sampling.get(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = cast(v)
                out.append(f"{tag}={v:.6g}" if isinstance(v, float) else f"{tag}={v}")
        return (" " + " ".join(out)) if out else ""

    def generate(self, messages, max_tokens, on_delta, sampling=None):
        """Serialized generation. Applies the chat template, sends one GEN
        request (with per-request sampling overrides when given), and streams
        decoded text deltas to on_delta(str). Returns (full_text,
        prompt_tokens, completion_tokens, finish_reason). Raises on engine
        error. finish_reason is "stop" (EOS) or "length" (max_tokens cap),
        taken from the engine's fin= trailer; old binaries without the
        trailer default to "stop".

        The incremental BPE-delta decode (re-decode all tokens, diff against
        what was already emitted, hold back trailing incomplete UTF-8) mirrors
        chat.py's proven logic."""
        with self.lock:
            if not self.alive():
                self.spawn()

            token_ids = self.tok.apply_chat_template(messages, add_generation_prompt=True)
            prompt_n = len(token_ids)
            csv = ",".join(map(str, token_ids))
            overrides = self._override_fields(sampling)
            request = f"GEN {max_tokens}{overrides} {csv}\n"
            try:
                self.proc.stdin.write(request)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self.spawn()
                raise RuntimeError(f"engine_write_failed: {e}")

            resp_tokens = []
            emitted = ""      # text already handed to on_delta
            buf = ""
            finish = "stop"   # engines without the fin= trailer → "stop"

            def flush_final():
                nonlocal emitted
                text = self.tok.decode(resp_tokens, skip_special_tokens=True)
                if len(text) > len(emitted):
                    on_delta(text[len(emitted):])
                    emitted = text
                return emitted

            while True:
                ch = self.proc.stdout.read(1)
                if not ch:
                    raise RuntimeError("engine_exit")
                if not ch.isspace():
                    buf += ch
                    continue
                if not buf:
                    continue
                if buf.startswith("fin="):
                    finish = "length" if buf[4:] == "length" else "stop"
                    buf = ""
                    continue
                if buf.startswith("<<"):
                    if buf == "<<DONE>>":
                        full = flush_final()
                        return full, prompt_n, len(resp_tokens), finish
                    # "<<ERR reason>>" splits on its internal space — pull the
                    # rest of the line so the pipe is clean for the next request.
                    err = (buf + " " + self.proc.stdout.readline().strip()).rstrip()
                    if "bad_request" in err and overrides and not resp_tokens:
                        # Engine binary predates the sampling extension: fall
                        # back to the plain protocol and retry this request.
                        print("[serve] WARNING: engine rejected sampling overrides "
                              "(old GEN protocol) — retrying without them. Rebuild "
                              "with `make all` for per-request sampling.", flush=True)
                        overrides = ""
                        buf = ""
                        self.proc.stdin.write(f"GEN {max_tokens} {csv}\n")
                        self.proc.stdin.flush()
                        continue
                    raise RuntimeError(err)
                clean = buf.replace("[", "").replace("]", "").strip()
                buf = ""
                if not clean.isdigit():
                    continue
                resp_tokens.append(int(clean))
                text = self.tok.decode(resp_tokens, skip_special_tokens=True)
                # Hold back while the tail is an incomplete UTF-8 sequence.
                if text.endswith("�"):
                    continue
                if len(text) > len(emitted):
                    on_delta(text[len(emitted):])
                    emitted = text


def _messages_from(body):
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("'messages' must be a non-empty array")
    out = []
    for m in msgs:
        role = m.get("role")
        content = m.get("content", "")
        if isinstance(content, list):  # OpenAI content-parts → concat text parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        out.append({"role": role, "content": content})
    return out


def _max_tokens_from(body, default, cap):
    v = body.get("max_completion_tokens") or body.get("max_tokens") or default
    try:
        v = int(v)
    except (TypeError, ValueError):
        v = default
    return max(1, min(v, cap))


def _sampling_from(body):
    """Collect the sampling params a request explicitly set. `top_k` and
    `repetition_penalty` are common OpenAI-compatible extensions (also
    accepted under vLLM's `rep_penalty` alias)."""
    out = {}
    for src, dst in (("temperature", "temperature"),
                     ("top_p", "top_p"),
                     ("top_k", "top_k"),
                     ("repetition_penalty", "repetition_penalty"),
                     ("rep_penalty", "repetition_penalty")):
        v = body.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and dst not in out:
            out[dst] = v
    return out


class Handler(BaseHTTPRequestHandler):
    engine = None          # set in main()
    model_id = "s-moe"
    default_max = 256
    max_cap = 4096

    def log_message(self, fmt, *a):  # concise access log to stderr
        print(f"[serve] {self.address_string()} {fmt % a}", flush=True)

    # ── helpers ──────────────────────────────────────────────
    def _json(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _sse_open(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def _file(self, name, ctype):
        """Serve a sibling static file (read from disk each hit — it's a
        low-traffic manual-test asset, and this way editing it needs no
        server restart)."""
        try:
            with open(os.path.join(ROOT, name), "rb") as f:
                payload = f.read()
        except OSError:
            self._json(404, {"error": {"message": f"{name} not found",
                                       "type": "invalid_request_error"}})
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    # ── routes ───────────────────────────────────────────────
    def do_GET(self):
        # Same-origin test console (see webchat.html) — no CORS needed because
        # it is served from the very origin its fetch() calls target.
        if self.path in ("/", "/index.html"):
            self._file("webchat.html", "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200, {"status": "ok",
                             "engine": "up" if self.engine.alive() else "down"})
        elif self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [
                {"id": self.model_id, "object": "model",
                 "created": 0, "owned_by": "s-moe"}]})
        else:
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            messages = _messages_from(body)
        except Exception as e:
            self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            return

        max_tokens = _max_tokens_from(body, self.default_max, self.max_cap)
        sampling = _sampling_from(body)
        stream = bool(body.get("stream", False))
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if stream:
            self._stream(cid, created, messages, max_tokens, sampling)
        else:
            self._complete(cid, created, messages, max_tokens, sampling)

    def _stream(self, cid, created, messages, max_tokens, sampling):
        self._sse_open()
        base = {"id": cid, "object": "chat.completion.chunk",
                "created": created, "model": self.model_id}

        def chunk(delta, finish=None):
            return {**base, "choices": [{"index": 0, "delta": delta,
                                         "finish_reason": finish}]}
        try:
            self._sse(chunk({"role": "assistant"}))

            def on_delta(text):
                self._sse(chunk({"content": text}))

            _, _, _, finish = self.engine.generate(messages, max_tokens, on_delta, sampling)
            self._sse(chunk({}, finish=finish))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up mid-stream
        except Exception as e:
            try:
                self._sse(chunk({"content": f"\n[engine error: {e}]"}, finish="stop"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception:
                pass

    def _complete(self, cid, created, messages, max_tokens, sampling):
        parts = []
        try:
            full, p_n, c_n, finish = self.engine.generate(
                messages, max_tokens, lambda t: parts.append(t), sampling)
        except Exception as e:
            self._json(503, {"error": {"message": f"engine error: {e}",
                                       "type": "server_error"}})
            return
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": self.model_id,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": full}}],
            "usage": {"prompt_tokens": p_n, "completion_tokens": c_n,
                      "total_tokens": p_n + c_n},
        })


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible server for S-MoE.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--engine", default=os.path.join(ROOT, "build/smoe-engine"))
    ap.add_argument("--vault",  default=os.path.join(ROOT, "vault/qwen3-235b-q4.smoe"))
    ap.add_argument("--scout",  default=os.path.join(ROOT, "vault/qwen3-235b-q4.scout.safetensors"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-235B-A22B-Instruct-2507")
    ap.add_argument("--model-id", default="s-moe")
    ap.add_argument("--ring", type=int, default=0)      # 0 = auto-tune
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--max-tokens", type=int, default=256, help="default when a request omits it")
    args = ap.parse_args()

    print(f"[serve] loading tokenizer {args.tokenizer} …", flush=True)
    engine = Engine(args)
    print("[serve] engine spawned (booting in background — first request waits for it)", flush=True)

    Handler.engine = engine
    Handler.model_id = args.model_id
    Handler.default_max = args.max_tokens

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] listening on http://{args.host}:{args.port}  "
          f"(POST /v1/chat/completions, GET /v1/models)", flush=True)
    print(f"[serve] web console: open http://{args.host}:{args.port}/ in a browser", flush=True)
    print("[serve] per-request sampling: temperature/top_p/top_k/repetition_penalty "
          "forwarded to the engine; launch values are the defaults.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] shutting down", flush=True)
    finally:
        if engine.alive():
            engine.proc.stdin.close()
            engine.proc.terminate()


if __name__ == "__main__":
    main()
