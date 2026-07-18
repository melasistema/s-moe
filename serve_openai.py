#!/usr/bin/env python3
"""
serve_openai.py — OpenAI-compatible HTTP front-end for the S-MoE engine.

Wraps the persistent `--serve` engine (GEN/<<DONE>> stdin protocol) in the
OpenAI Chat Completions API, so any OpenAI-compatible client (Open WebUI,
LibreChat, editor plugins, the `openai` SDK, curl) can talk to S-MoE.

Endpoints:
    GET  /                     → webchat.html, a built-in same-origin test UI
    GET  /health               → {"status":"ok", "engine":"up|down"}
    GET  /sysmem               → macOS physical-memory snapshot for the console
    GET  /v1/models            → every vault discovered in --vault-dir
    POST /v1/chat/completions  → streaming (SSE) or non-streaming completion

Model switching: every *.smoe with a sibling *.scout.safetensors in
--vault-dir is a servable model (id = filename stem); identity (quant bits,
model_type) is read from the vault's own header + arch block. A request
whose `model` names another discovered vault makes the server shut down the
engine and boot that vault before generating — one engine at a time, the
machine can't hold two. Unknown/absent `model` values run on whatever is
loaded, which keeps old clients working.

Design constraints (dictated by the engine, see chat.py):
  • Chat template + tokenization happen HERE (transformers), matching chat.py
    exactly — the engine only ever sees token IDs. The tokenizer for a vault
    is checkpoints/<stem>/ when that directory exists (the shatter convention
    names the vault after the checkpoint dir), else the --tokenizer fallback.
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
import re
import struct
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── vault discovery ──────────────────────────────────────────────────
# Minimal .smoe header/arch-block peek, mirroring src/common.hpp — kept
# inline so the server stays stdlib+transformers (no scripts/ import).
SMOE_MAGIC = b"SMOE\xde\xea\x00\x01"


def _vault_meta(path):
    """Read a vault's identity from its own bytes: quant bits from the
    64-byte header, model_type from the v2 SARC arch block when present.
    Returns a dict, or None when the file is not a valid .smoe vault."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(64)
            if len(hdr) != 64 or hdr[:8] != SMOE_MAGIC:
                return None
            version, layers, experts = struct.unpack_from("<III", hdr, 8)
            bits, = struct.unpack_from("<I", hdr, 44)
            arch_off, = struct.unpack_from("<I", hdr, 60)
            meta = {"version": version, "moe_layers": layers,
                    "experts_per_layer": experts,
                    "quant": f"Q{bits}", "model_type": ""}
            if version >= 2 and arch_off:
                f.seek(arch_off)
                blk = f.read(128)
                if len(blk) == 128 and blk[:4] == b"SARC":
                    meta["model_type"] = (
                        blk[60:84].split(b"\x00")[0].decode(errors="replace"))
            return meta
    except OSError:
        return None


def discover_models(vault_dir, default_tokenizer):
    """Scan vault_dir for servable models: every *.smoe with a sibling
    *.scout.safetensors. Returns {stem: {vault, scout, tokenizer, meta}}."""
    models = {}
    try:
        names = sorted(os.listdir(vault_dir))
    except OSError:
        return models
    for name in names:
        if not name.endswith(".smoe"):
            continue
        stem = name[:-len(".smoe")]
        vault = os.path.join(vault_dir, name)
        scout = os.path.join(vault_dir, stem + ".scout.safetensors")
        if not os.path.isfile(scout):
            print(f"[serve] skipping {name}: no sibling {stem}.scout.safetensors", flush=True)
            continue
        meta = _vault_meta(vault)
        if meta is None:
            print(f"[serve] skipping {name}: not a valid .smoe vault", flush=True)
            continue
        # The shatter names the vault after the checkpoint dir, so the
        # matching tokenizer normally lives at checkpoints/<stem>/.
        ckpt = os.path.join(ROOT, "checkpoints", stem)
        if os.path.isfile(os.path.join(ckpt, "tokenizer_config.json")):
            tokenizer = ckpt
        else:
            tokenizer = default_tokenizer
            print(f"[serve] {stem}: no checkpoints/{stem}/ tokenizer — "
                  f"falling back to {default_tokenizer}", flush=True)
        models[stem] = {"vault": vault, "scout": scout,
                        "tokenizer": tokenizer, "meta": meta}
    return models


class Engine:
    """Owns the persistent smoe-engine subprocess and serializes access.
    Holds the discovered model catalogue; exactly one vault is loaded at a
    time, and _ensure() swaps the subprocess when a request names another."""

    def __init__(self, args, models, active):
        self.args = args
        self.models = models
        self.active = active
        self.lock = threading.Lock()
        self.proc = None
        self.toks = {}     # tokenizer source → loaded tokenizer (switch-back is free)
        # Imported lazily so `--help` / import works without transformers.
        from transformers import AutoTokenizer
        self._autotok = AutoTokenizer
        self.tok = self._tokenizer(active)
        self.spawn()

    def _tokenizer(self, model_id):
        src = self.models[model_id]["tokenizer"]
        if src not in self.toks:
            print(f"[serve] loading tokenizer {src} …", flush=True)
            self.toks[src] = self._autotok.from_pretrained(src, trust_remote_code=True)
        return self.toks[src]

    def _ensure(self, model_id):
        """Switch the loaded vault when a request names another discovered
        model. Caller holds self.lock, so no generation is in flight; the
        old engine is torn down before the new one boots — the machine
        cannot hold two. Unknown ids run on the active model (old clients
        send made-up labels like "s-moe")."""
        models = getattr(self, "models", None)   # absent on test doubles
        if not models or not model_id or model_id == self.active:
            return
        if model_id not in models:
            return
        print(f"[serve] switching model: {self.active} → {model_id} "
              f"(engine restart — first request pays the cold boot)", flush=True)
        if self.alive():
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.active = model_id
        self.tok = self._tokenizer(model_id)
        self.spawn()

    def model_cards(self):
        """/v1/models payload: one card per discovered vault, active first
        (older clients read data[0] as *the* model)."""
        cards = []
        for mid, m in self.models.items():
            meta = m["meta"] or {}
            cards.append({
                "id": mid, "object": "model", "created": 0, "owned_by": "s-moe",
                # Non-standard fields (OpenAI clients ignore extras): which
                # vault the engine currently has loaded, the tokenizer
                # checkpoint lineage, and identity read from the vault itself.
                "active": mid == self.active,
                "lineage": m["tokenizer"],
                "quant": meta.get("quant", ""),
                "model_type": meta.get("model_type", ""),
                "moe_layers": meta.get("moe_layers"),
                "experts_per_layer": meta.get("experts_per_layer"),
            })
        cards.sort(key=lambda c: not c["active"])
        return cards

    def spawn(self):
        m = self.models[self.active]
        cmd = [
            self.args.engine,
            "--vault", m["vault"],
            "--scout", m["scout"],
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

    def _drain(self):
        """The engine is mid-response but its reader gave up (typically the
        client hung up mid-SSE and the socket write raised out of on_delta).
        The engine knows nothing of this and keeps streaming; without a drain
        the NEXT request would read this response's leftover tokens as its
        own. Consume until the request terminator so the pipe is back at a
        request boundary. May take as long as the abandoned generation runs —
        the engine is busy anyway, so queued requests would be waiting on it
        regardless."""
        try:
            while self.alive():
                line = self.proc.stdout.readline()
                if not line or "<<DONE>>" in line or "<<ERR" in line:
                    break
        except Exception:
            pass

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

    def generate(self, messages, max_tokens, on_delta, sampling=None, on_prompt=None,
                 model=None):
        """Serialized generation. Applies the chat template, sends one GEN
        request (with per-request sampling overrides when given), and streams
        decoded text deltas to on_delta(str). Returns (full_text,
        prompt_tokens, completion_tokens, finish_reason). Raises on engine
        error. finish_reason is "stop" (EOS) or "length" (max_tokens cap),
        taken from the engine's fin= trailer; old binaries without the
        trailer default to "stop". When `model` names another discovered
        vault, the engine is swapped before generating (see _ensure).

        The incremental BPE-delta decode (re-decode all tokens, diff against
        what was already emitted, hold back trailing incomplete UTF-8) mirrors
        chat.py's proven logic."""
        with self.lock:
            self._ensure(model)
            if not self.alive():
                self.spawn()

            token_ids = self.tok.apply_chat_template(messages, add_generation_prompt=True)
            prompt_n = len(token_ids)
            if on_prompt:
                on_prompt(prompt_n)
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
            at_boundary = False  # pipe consumed up to a <<DONE>>/<<ERR>> line

            def flush_final():
                nonlocal emitted
                text = self.tok.decode(resp_tokens, skip_special_tokens=True)
                if len(text) > len(emitted):
                    on_delta(text[len(emitted):])
                    emitted = text
                return emitted

            try:
                while True:
                    ch = self.proc.stdout.read(1)
                    if not ch:
                        at_boundary = True  # engine gone — nothing left to drain
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
                            at_boundary = True
                            full = flush_final()
                            return full, prompt_n, len(resp_tokens), finish
                        # "<<ERR reason>>" splits on its internal space — pull the
                        # rest of the line so the pipe is clean for the next request.
                        err = (buf + " " + self.proc.stdout.readline().strip()).rstrip()
                        at_boundary = True
                        if "bad_request" in err and overrides and not resp_tokens:
                            # Engine binary predates the sampling extension: fall
                            # back to the plain protocol and retry this request.
                            print("[serve] WARNING: engine rejected sampling overrides "
                                  "(old GEN protocol) — retrying without them. Rebuild "
                                  "with `make all` for per-request sampling.", flush=True)
                            overrides = ""
                            buf = ""
                            at_boundary = False  # the retried GEN is in flight again
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
            finally:
                # Any exit before the terminator (an on_delta socket write
                # raising on client disconnect, a decode error) leaves the
                # engine mid-stream; drain before the lock is released so the
                # next request can't read this response's tokens.
                if not at_boundary:
                    self._drain()


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


_MEMSIZE = None    # hw.memsize never changes at runtime — fetch it once.


def _memsize():
    global _MEMSIZE
    if _MEMSIZE is None:
        _MEMSIZE = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
    return _MEMSIZE


def _sysmem(pid=None):
    """macOS physical-memory snapshot for the dev console, all in bytes:

        total       installed RAM (hw.memsize)
        used        Activity-Monitor sense: app (anon−purgeable) + wired + compressed
        available   total − used, the headroom S-MoE's expert-streaming ring can grow into
        wired,
        compressed  the two non-reclaimable components of `used`
        pressure    macOS VM pressure level ("normal"|"warning"|"critical"|None)
        rss         the engine process's resident set, when a pid is given

    Best-effort and macOS-specific: any failure returns None so the client
    simply hides the meter (e.g. on Linux)."""
    try:
        total = _memsize()
        vm = subprocess.check_output(["vm_stat"]).decode()
    except Exception:
        return None
    m = re.search(r"page size of (\d+) bytes", vm)
    ps = int(m.group(1)) if m else 4096

    def pages(label):
        mm = re.search(re.escape(label) + r":\s+(\d+)", vm)
        return int(mm.group(1)) if mm else 0

    wired = pages("Pages wired down") * ps
    compressed = pages("Pages occupied by compressor") * ps
    app = max(pages("Anonymous pages") - pages("Pages purgeable"), 0) * ps
    used = min(app + wired + compressed, total)
    out = {"total": total, "used": used, "available": total - used,
           "wired": wired, "compressed": compressed}
    try:
        lvl = int(subprocess.check_output(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]))
        out["pressure"] = {1: "normal", 2: "warning", 4: "critical"}.get(lvl)
    except Exception:
        out["pressure"] = None
    if pid:
        try:  # ps reports RSS in KiB
            out["rss"] = int(subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)]).strip()) * 1024
        except Exception:
            out["rss"] = None
    return out


class Handler(BaseHTTPRequestHandler):
    engine = None          # set in main()
    model_id = "s-moe"     # legacy alias — requests naming it run on the active model
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
        # The SSE body has no Content-Length and no chunked framing, so the
        # ONLY way the client learns the stream ended is the connection
        # closing. "keep-alive" here would leave readers blocked forever
        # after [DONE].
        self.send_header("Connection", "close")
        self.close_connection = True
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
        elif self.path == "/sysmem":
            # System RAM for the console's memory meter. Pass the engine's live
            # pid (when up) so the snapshot can include S-MoE's own resident set.
            proc = getattr(self.engine, "proc", None)
            pid = proc.pid if proc and self.engine.alive() else None
            self._json(200, _sysmem(pid) or {"error": "unavailable"})
        elif self.path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": self.engine.model_cards()})
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
        # The request's model field selects the vault. The legacy alias and
        # unknown labels resolve to whatever is loaded — the response then
        # reports the model that actually generated.
        model = body.get("model")
        if model == self.model_id or model not in getattr(self.engine, "models", {}):
            model = getattr(self.engine, "active", self.model_id)

        if stream:
            self._stream(cid, created, messages, max_tokens, sampling, model)
        else:
            self._complete(cid, created, messages, max_tokens, sampling, model)

    def _stream(self, cid, created, messages, max_tokens, sampling, model):
        self._sse_open()
        base = {"id": cid, "object": "chat.completion.chunk",
                "created": created, "model": model}

        def chunk(delta, finish=None):
            return {**base, "choices": [{"index": 0, "delta": delta,
                                         "finish_reason": finish}]}
        try:
            self._sse(chunk({"role": "assistant"}))

            def on_delta(text):
                self._sse(chunk({"content": text}))

            def on_prompt(n):
                # The prompt's exact token count exists the moment the chat
                # template is tokenized — before prefill even starts. Surface
                # it right away (empty delta + partial usage) so clients can
                # relate the prefill wait to the prompt size in real time.
                early = chunk({})
                early["usage"] = {"prompt_tokens": n}
                self._sse(early)

            _, p_n, c_n, finish = self.engine.generate(
                messages, max_tokens, on_delta, sampling, on_prompt=on_prompt,
                model=model)
            # Exact token counts ride on the final chunk (the include_usage
            # convention): clients computing tok/s shouldn't have to infer
            # them from delta counts, which under-count UTF-8 hold-backs.
            final = chunk({}, finish=finish)
            final["usage"] = {"prompt_tokens": p_n, "completion_tokens": c_n,
                              "total_tokens": p_n + c_n}
            self._sse(final)
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

    def _complete(self, cid, created, messages, max_tokens, sampling, model):
        parts = []
        try:
            full, p_n, c_n, finish = self.engine.generate(
                messages, max_tokens, lambda t: parts.append(t), sampling,
                model=model)
        except Exception as e:
            self._json(503, {"error": {"message": f"engine error: {e}",
                                       "type": "server_error"}})
            return
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": model,
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
    ap.add_argument("--vault-dir", default=os.path.join(ROOT, "vault"),
                    help="every *.smoe + *.scout.safetensors pair here is servable")
    ap.add_argument("--vault", default=os.path.join(ROOT, "vault/qwen3-235b-q4.smoe"),
                    help="the vault loaded at startup")
    ap.add_argument("--scout", default=None,
                    help="scout for --vault (default: derived from the vault stem)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-235B-A22B-Instruct-2507",
                    help="fallback tokenizer for vaults without a checkpoints/<stem>/ dir")
    ap.add_argument("--model-id", default="s-moe",
                    help="legacy alias: requests naming it run on the active model")
    ap.add_argument("--ring", type=int, default=0)      # 0 = auto-tune
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--max-tokens", type=int, default=256, help="default when a request omits it")
    args = ap.parse_args()

    models = discover_models(args.vault_dir, args.tokenizer)
    # An explicit --vault outside --vault-dir is still servable: register it.
    stem = os.path.splitext(os.path.basename(args.vault))[0]
    if stem not in models and os.path.isfile(args.vault):
        scout = args.scout or os.path.splitext(args.vault)[0] + ".scout.safetensors"
        models[stem] = {"vault": args.vault, "scout": scout,
                        "tokenizer": args.tokenizer, "meta": _vault_meta(args.vault) or {}}
    if not models:
        raise SystemExit(f"[serve] no servable models: {args.vault_dir} has no "
                         "*.smoe + *.scout.safetensors pair and --vault does not exist")
    active = stem if stem in models else next(iter(models))
    for mid, m in models.items():
        meta = m["meta"] or {}
        print(f"[serve] model: {mid}  {meta.get('quant', '?')}  "
              f"{meta.get('model_type') or 'v1 vault'}"
              f"{'  ← active' if mid == active else ''}", flush=True)

    engine = Engine(args, models, active)
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
    if len(models) > 1:
        print("[serve] model switching: request `model` = any id from /v1/models "
              "swaps the loaded vault (engine restart, cold boot).", flush=True)
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
