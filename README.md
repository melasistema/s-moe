# S-MoE: Seismic Mixture of Experts Engine

> **Rock, Paper, Silicon.** A rebellious attempt to shatter the "Monolithic Delusion" of Silicon Valley gatekeepers. S-MoE enables high-fidelity frontier LLM inference completely within the democratic boundaries of a standard 32/48 GB Apple Silicon MacBook — the 235B frontier's dense backbone alone weighs ~16 GB, the price of admission just to turn it on. On 16 GB machines, the same architecture rules over smaller fine-grained MoE models. And it is **model-agnostic**: any fine-grained MoE, one shatter command, one self-describing vault — switchable at runtime.

<div align="center">

![s-moe-seismic-mixture-of-experts](https://docs.s-moe.com/images/smoe-seismic-mixture-of-experts.png)

### The Cartography of the Mountain

Every byte, every struct field, every atomic flag — mapped, explained, and public.
No secrets. No priesthood. A door in the memory wall, and the map to walk through it.

**[docs.s-moe.com](https://docs.s-moe.com/)**

*The Guide · The Phonon Metaphor · The Internals · The Manifesto*

</div>

---

## 235B, Streamed From a Laptop SSD

No cloud. No H100. No half-terabyte workstation. A 235-billion-parameter frontier model, answering on hardware you can close and put in a bag.

| | |
|---|---|
| **Model** | Qwen3-235B — 235B params, fine-grained MoE (128 experts/layer) |
| **Hardware** | Standard 48 GB Apple Silicon MacBook |
| **Quantization** | Q4 — 4-bit, unpacked directly in GPU registers |
| **Resident RAM** | ~16 GB dense trunk + a small streaming ring — *not* the whole model |
| **Cold time-to-first-token** | **~26 s** (45-token prompt) — down from 92.7 s |
| **Warm follow-up turn** | **~14 s**, flat — no longer grows with conversation length |
| **Decode speed** | **1.84 tok/s** — up from 0.48 t/s at the start of the campaign |
| **Interfaces** | Native console · **OpenAI-compatible HTTP server** · built-in web chat |


The mountain rests cold on the SSD; only the handful of experts each word actually needs is ever pulled into memory. See how the wall came down in [Current Status](#current-status).

---

## One Engine, Any Fine-Grained MoE

The 235B is the proof of force, not the whole story. S-MoE is **model-agnostic infrastructure**: point the shatter script at any compatible fine-grained MoE checkpoint and it produces a *self-describing* vault — the model's own mathematics (RoPE theta, routing top-k, attention geometry, lineage) travel inside the `.smoe` file's arch block, so the engine reads what it is given instead of assuming what it was built for. No recompilation, no per-model configuration.

Every vault in `vault/` becomes a servable model, and they trade places live:

| | Qwen3-235B-A22B | Qwen3-30B-A3B |
|---|---|---|
| **Role** | The frontier — maximum intelligence | The daily driver — speed |
| **Vault (Q4)** | ~112 GB, streamed from SSD | ~14 GB, fits almost entirely in the RAM ring |
| **Decode** | 1.84 tok/s | **~13.7 tok/s** |

| Qwen 235B | Qwen 30B |
|---|---|
| [![Qwen 235B](https://img.youtube.com/vi/lbf4aqNezus/default.jpg)](https://www.youtube.com/watch?v=lbf4aqNezus) | [![Qwen 30B](https://img.youtube.com/vi/xDjyxQ-GHcY/default.jpg)](https://www.youtube.com/watch?v=xDjyxQ-GHcY) |

Same engine, same binary, same laptop. The smaller the model, the more of it lives in memory — and the streaming cost that bounds the frontier simply vanishes. The full model matrix and hardware projections live in the [official documentation](https://docs.s-moe.com/).

---

## The Monolithic Delusion

Silicon Valley has a bucket problem. 

To run a modern frontier AI model—like Qwen3 with its **235 billion parameters**—the corporate high priests of compute insist that you must worship at the altar of the unified memory wall. They tell us we must host the entire mathematical mountain in RAM simultaneously, forcing developers to rent $15,000 H100 cloud instances or buy half-terabyte workstations.

**The truth they hide:** In a true Mixture of Experts (MoE) architecture, 95% of the model is completely silent at any given millisecond. The weights are just cold stone; only a tiny fraction of "experts" fire for any single word. Yet, standard runtimes keep gigabytes of dead parameters sitting in precious RAM just in case.

Co-designed with an AI pair-programming agent, **S-MoE** breaks this paradigm. We don't fight the memory wall by trying to cram a massive model into RAM. Instead, we treat your Mac’s ultra-fast NVMe SSD ($~7.4\text{ GB/s}$) as the primary solid repository, and turn unified memory into a transient, fluid execution medium.

---

## The Phonon Metaphor: Acoustic JIT Streaming

Standard models blast inputs into monolithic weights and hope they fit in RAM. We took inspiration from Filippo Biondi's unorthodox application of Synthetic Aperture Radar (SAR) for deep subsurface tomography. When an electromagnetic wave strikes a pyramid's surface, it generates mechanical sound waves (phonons) that ripple through the rock, revealing the hidden voids beneath.

S-MoE splits monolithic models into a similar acoustic sensor array:

1. **The Resident Backbone (The EM Strike):** The model's own dense trunk — embeddings, attention, norms, routing gates (~16GB in bfloat16 for Qwen3-235B) — sits permanently in Unified Memory. It once ran ahead as a speculative "Scout," *guessing* which experts the mountain would summon next. Measurement retired that prophet: the backbone now evaluates the router gate **exactly**, on the true hidden state, at every layer of every token — one small matrix multiply that names the precise experts to stream. The mountain no longer needs predicting; it reads its own next move.
2. **The Deep Strata (The Rock):** The massive 235B parameters rest cold on the SSD, shattered into custom page-aligned binary segments perfectly aligned to the Apple Silicon 16KB hardware boundary.
3. **The Acoustic Receiver (The JIT Pump):** Guided by that exact routing, specialized background workers read the named experts asynchronously using Direct I/O (`F_NOCACHE`), completely bypassing sluggish OS page caches. Experts are packed into a cyclic ring buffer fractions of a second before the Metal GPU executes them — and the ring doubles as an LRU cache, so the ~46% of experts an adjacent token reuses are already resident and never re-read. Total system RAM footprint stays incredibly low.

---

## It Speaks Every Protocol That Matters

S-MoE isn't a walled garden. Beyond the native console, it exposes a **multi-protocol HTTP server** — anything that speaks OpenAI *or* Anthropic talks to your laptop's vault fleet, unmodified:

```bash
.venv/bin/python serve.py --port 8000
```

- `POST /v1/chat/completions` — **OpenAI** Chat Completions (streaming + non-streaming), with per-request `temperature`, `top_p`, `top_k`, and an honest `finish_reason`.
- `POST /v1/messages` — **Anthropic** Messages API (streaming + non-streaming) — Claude Code, the Anthropic SDK, and any Anthropic-speaking client work natively.
- `GET /v1/models` — every vault discovered on disk, identity read from the vault's own bytes. Naming another one in the standard `model` field **switches the engine** — so Open WebUI's model picker switches vaults with zero S-MoE-specific code.
- `GET /health` — the plumbing supervisors expect.
- `GET /` — a **built-in, dependency-free web chat console** with live turn timings, context and RAM meters, and a model dropdown. Open `http://127.0.0.1:8000/` and start typing.

Point Open WebUI, LibreChat, an editor plugin, or the `openai` SDK at `http://127.0.0.1:8000/v1` and it cannot tell S-MoE from OpenAI. Point Claude Code at the same origin (`ANTHROPIC_BASE_URL=http://127.0.0.1:8000 claude`) and it cannot tell S-MoE from Anthropic. One file, one process, one port, both protocols, zero new dependencies.

---

## Tech Stack Decisions
* **System Core:** C++20 for absolute precision over threads, direct memory-mapped spaces, and low-overhead POSIX execution primitives.
* **Compute Layer:** Native Apple Metal Performance Shaders (`.metal`) wrapped in an Objective-C++ bridge to achieve direct, zero-copy pointer access within Apple’s Unified Memory Architecture (UMA).
* **Quantization & Layout:** Custom page-aligned binary layouts ($16\text{KB}$ boundaries) paired with register-space 2-bit and 4-bit weight compression — Q4 is the reliable default, unpacked directly inside the GPU registers.

---

## Architectural Layout Overview
* `scripts/shatter_moe.py`: Offline pre-processing utility that slices monolithic model weights into aligned, discrete expert components — and stamps each vault with a self-describing arch block read straight from the model's own `config.json`.
* `src/io/streamer.cpp`: High-speed asynchronous storage pipeline that handles multi-threaded, unbuffered file reads — its ring buffer doubles as a true LRU expert cache.
* `src/prefill.cpp`: Layer-major batched prefill — prompt chunks traverse the model layer by layer, each layer's deduplicated expert union read from the vault exactly once.
* `src/compute/metal_bridge.mm`: The operational execution nexus, swapping active memory pointers directly into the Apple GPU ring, with token-batch fused FFN kernels for prefill.
* `chat.py`: The thin Python console — tokenization and display only. It speaks to one persistent engine process per session over a line protocol.
* `serve.py`: The multi-protocol HTTP front-end. Owns HTTP, tokenization, the chat template, and the vault fleet — discovery, `/v1/models`, and live model switching; drives the same persistent engine over the same line protocol. Speaks both OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) natively. Zero new dependencies.
* `webchat.html`: A single self-contained, dependency-free browser chat console, served same-origin at `GET /` for hands-on testing.

---

## Current Status
S-MoE runs a fully operational, **model-agnostic** MoE pipeline on a purely Data-Oriented C++ architecture — a verified two-model fleet today (Qwen3-235B and Qwen3-30B-A3B), switchable at runtime. Since the latency campaign began it has moved the needle on every axis that matters: on the 235B frontier, cold **time-to-first-token 92.7 s → ~26 s (>3×)** and **decode 0.48 → 1.84 tok/s (>3.8×)**, every step verified bit-exact or coherence-checked against the reference path. It is no longer just *running* the frontier — it is running it much faster, on both ends, and it is no longer *only* running the frontier.

Where the wins came from:

- **Self-Describing Vaults:** The `.smoe` format carries the model's own architecture — RoPE theta, routing top-k, head geometry, normalization flags, lineage — inside an arch block written at shatter time from the checkpoint's `config.json`. The engine's per-model hardcodes are gone; older vaults upgrade in place.
- **The Model Fleet:** The server discovers every vault on disk, lists them all on `/v1/models`, and swaps the engine when a request names another one — from the web console's dropdown or any OpenAI client's model picker. One shatter command turns a downloaded MoE checkpoint into a member of the fleet.
- **Layer-Major Batched Prefill:** Prompt chunks traverse the model layer by layer with exact router-gate evaluation; each layer's deduplicated expert union is read from NVMe once per chunk instead of once per token. Cold time-to-first-token on a 45-token prompt: **92.7s → 32.4s**, bit-identical output.
- **Persistent Serve Mode:** One engine process per *session*, not per turn. The KV-cache, expert ring, and Scout state survive across turns; each message prefills only its new suffix (longest-common-prefix contract). Follow-up turns answer in **~14s flat** — no longer growing with conversation length.
- **The Ring is a Cache:** Released expert slots are retained with valid data and re-claimed as LRU cache hits; eviction respects live GPU references. The old regime evicted the entire ring every prompt token.
- **GPU-Resident Hot Path:** The whole decode attention block runs in one command buffer per layer, coalesced simdgroup-per-row Metal kernels stream both expert and dense weights at ~150+ GB/s (a 7× dequant-bandwidth jump), and prompt positions that cannot emit a token skip the LM head and sampler entirely — decode **0.48 → 1.84 t/s**.
- **Dynamic Topology:** Safely routes 128 fine-grained MoE experts per layer with true Qwen3 routing — the real router gate is evaluated on the heavy hidden state at every layer, in prefill *and* decode — streaming exactly what is needed and nothing more.
- **Q4 Quantisation:** 4-bit Apple Metal shaders efficiently unpack weights directly in the GPU registers, avoiding the mathematical collapse of 2-bit uniform quantization on fine-grained architectures.
- **Zero-Allocation:** KV-caches, context windows, and all prefill activation planes are managed entirely in static/aligned memory, strictly preserving the **zero runtime heap allocation** invariant.
- **Self-Learning Cache:** Expert claim frequencies persist to `vault/expert_freq.bin`; an idle-time prewarm streams the historically hottest experts into the ring while the user types — machinery that comes into its own as ring capacity grows (Q2 vaults).

---

## License

S-MoE is open-source software licensed under the [MIT license](LICENSE).

The code lives here. The philosophy, the architecture, and the honest mistakes live at **[docs.s-moe.com](https://docs.s-moe.com/)**.

Built by [Luca Visciola](https://github.com/melasistema) and an AI agent who hopefully knows C++ and low level programming, because i do not 😅.