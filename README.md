# S-MoE: Seismic Mixture of Experts Engine

> **Rock, Paper, Silicon.** A rebellious attempt to shatter the "Monolithic Delusion" of Silicon Valley gatekeepers. S-MoE enables high-fidelity frontier LLM inference completely within the democratic boundaries of a standard 32/48 GB Apple Silicon MacBook — the 235B frontier's dense backbone alone weighs ~16 GB, the price of admission just to turn it on. On 16 GB machines, the same architecture rules over smaller fine-grained MoE models.

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

1. **The Surface Scout (The EM Strike):** The model's own dense backbone — embeddings, attention, norms, routing gates (~16GB in bfloat16 for Qwen3-235B) — sitting permanently in Unified Memory. During generation it acts as a temporal oracle, running speculatively ahead to predict which specialized experts the heavy model will summon next. During prompt prefill it stands down entirely: the routing is computed *exactly* from the true hidden state, because a known future needs no prophet.
2. **The Deep Strata (The Rock):** The massive 235B parameters rest cold on the SSD, shattered into custom page-aligned binary segments perfectly aligned to the Apple Silicon 16KB hardware boundary.
3. **The Acoustic Receiver (The JIT Pump):** Guided by the Scout's phonon map, specialized background workers read data asynchronously using Direct I/O (`F_NOCACHE`), completely bypassing sluggish OS page caches. The incoming experts are packed into a cyclic ring buffer fractions of a second before the Metal GPU executes them, keeping total system RAM footprint incredibly low.

---

## Tech Stack Decisions
* **System Core:** C++20 for absolute precision over threads, direct memory-mapped spaces, and low-overhead POSIX execution primitives.
* **Compute Layer:** Native Apple Metal Performance Shaders (`.metal`) wrapped in an Objective-C++ bridge to achieve direct, zero-copy pointer access within Apple’s Unified Memory Architecture (UMA).
* **Quantization & Layout:** Custom page-aligned binary layouts ($16\text{KB}$ boundaries) paired with high-efficiency 2-bit weight compression strategies.

---

## Architectural Layout Overview
* `scripts/shatter_moe.py`: Offline pre-processing utility that slices monolithic model weights into aligned, discrete expert components.
* `src/io/streamer.cpp`: High-speed asynchronous storage pipeline that handles multi-threaded, unbuffered file reads — its ring buffer doubles as a true LRU expert cache.
* `src/prefill.cpp`: Layer-major batched prefill — prompt chunks traverse the model layer by layer, each layer's deduplicated expert union read from the vault exactly once.
* `src/compute/metal_bridge.mm`: The operational execution nexus, swapping active memory pointers directly into the Apple GPU ring, with token-batch fused FFN kernels for prefill.
* `chat.py`: The thin Python console — tokenization and display only. It speaks to one persistent engine process per session over a line protocol.

---

## Current Status
S-MoE has reached a fully operational **Qwen3-235B** pipeline with a purely Data-Oriented C++ architecture — and, after the utopian latency refactor, it is no longer just *running* the frontier; it is running it **3× faster to first token**.

- **Layer-Major Batched Prefill:** Prompt chunks traverse the model layer by layer with exact router-gate evaluation; each layer's deduplicated expert union is read from NVMe once per chunk instead of once per token. Cold time-to-first-token on a 45-token prompt: **92.7s → 32.4s**, bit-identical output.
- **Persistent Serve Mode:** One engine process per *session*, not per turn. The KV-cache, expert ring, and Scout state survive across turns; each message prefills only its new suffix (longest-common-prefix contract). Follow-up turns answer in **~14s flat** — no longer growing with conversation length.
- **The Ring is a Cache:** Released expert slots are retained with valid data and re-claimed as LRU cache hits; eviction respects live GPU references. The old regime evicted the entire ring every prompt token.
- **GPU-Resident Hot Path:** The heavy LM head runs on Metal (decode **0.48 → 0.70 t/s**), token-batch fused FFN kernels apply one expert to a whole chunk in a single dispatch, and prompt positions that cannot emit a token skip the LM head and sampler entirely.
- **Dynamic Topology:** Safely routes 128 fine-grained MoE experts per layer — exact routing during prefill, Scout speculation during generation — streaming exactly what is needed for Qwen3-235B.
- **Q4 Quantisation:** 4-bit Apple Metal shaders efficiently unpack weights directly in the GPU registers, avoiding the mathematical collapse of 2-bit uniform quantization on fine-grained architectures.
- **Zero-Allocation:** KV-caches, context windows, and all prefill activation planes are managed entirely in static/aligned memory, strictly preserving the **zero runtime heap allocation** invariant.
- **Self-Learning Cache:** Expert claim frequencies persist to `vault/expert_freq.bin`; an idle-time prewarm streams the historically hottest experts into the ring while the user types — machinery that comes into its own as ring capacity grows (Q2 vaults).

---

## License

S-MoE is open-source software licensed under the [MIT license](LICENSE).

Built by [Luca Visciola](https://github.com/melasistema) and an AI agent who hopefully knows C++ and low level programming, because i do not 😅.