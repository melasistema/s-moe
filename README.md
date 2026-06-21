# S-MoE: Seismic Mixture of Experts Engine

> **Rock, Paper, Silicon.** A rebellious attempt to shatter the "Monolithic Delusion" of Silicon Valley gatekeepers. S-MoE enables high-fidelity frontier LLM inference completely within the democratic boundaries of a standard 16GB/32GB Apple Silicon MacBook.

---

## The Monolithic Delusion

Silicon Valley has a bucket problem. 

To run a modern frontier AI model—like DeepSeek with its **284 billion parameters**—the corporate high priests of compute insist that you must worship at the altar of the unified memory wall. They tell us we must host the entire mathematical mountain in RAM simultaneously, forcing developers to rent $15,000 H100 cloud instances or buy half-terabyte workstations.

**The truth they hide:** In a true Mixture of Experts (MoE) architecture, 95% of the model is completely silent at any given millisecond. The weights are just cold stone; only a tiny fraction of "experts" fire for any single word. Yet, standard runtimes keep gigabytes of dead parameters sitting in precious RAM just in case.

Co-designed with an AI pair-programming agent, **S-MoE** breaks this paradigm. We don't fight the memory wall by trying to cram a massive model into RAM. Instead, we treat your Mac’s ultra-fast NVMe SSD ($~7.4\text{ GB/s}$) as the primary solid repository, and turn unified memory into a transient, fluid execution medium.

---

## The Phonon Metaphor: Acoustic JIT Streaming

Standard models blast inputs into monolithic weights and hope they fit in RAM. We took inspiration from Filippo Biondi's unorthodox application of Synthetic Aperture Radar (SAR) for deep subsurface tomography. When an electromagnetic wave strikes a pyramid's surface, it generates mechanical sound waves (phonons) that ripple through the rock, revealing the hidden voids beneath.

S-MoE splits monolithic models into a similar acoustic sensor array:

1. **The Surface Scout (The EM Strike):** A lightweight, highly efficient 1.5B parameter model sitting permanently in memory (~3GB). As it processes your prompt, it acts as a temporal oracle—calculating the semantic trajectory to predict exactly which specialized experts will be needed 5 to 10 tokens into the future. It computes the "sound" the model will make before the generation loop even gets there.
2. **The Deep Strata (The Rock):** The massive 284B parameters rest cold on the SSD, shattered into custom page-aligned binary segments perfectly aligned to the Apple Silicon 16KB hardware boundary.
3. **The Acoustic Receiver (The JIT Pump):** Guided by the Scout's phonon map, specialized background workers read data asynchronously using Direct I/O (`F_NOCACHE`), completely bypassing sluggish OS page caches. The incoming experts are packed into a cyclic ring buffer fractions of a second before the Metal GPU executes them, keeping total system RAM footprint incredibly low.

---

## Tech Stack Decisions
* **System Core:** C++20 for absolute precision over threads, direct memory-mapped spaces, and low-overhead POSIX execution primitives.
* **Compute Layer:** Native Apple Metal Performance Shaders (`.metal`) wrapped in an Objective-C++ bridge to achieve direct, zero-copy pointer access within Apple’s Unified Memory Architecture (UMA).
* **Quantization & Layout:** Custom page-aligned binary layouts ($16\text{KB}$ boundaries) paired with high-efficiency 2-bit weight compression strategies.

---

## Architectural Layout Overview
* `scripts/shatter_moe.py`: Offline pre-processing utility that slices monolithic model weights into aligned, discrete expert components.
* `src/io/streamer.cpp`: High-speed asynchronous storage pipeline that handles multi-threaded, unbuffered file reads.
* `src/compute/metal_bridge.mm`: The operational execution nexus, swapping active memory pointers directly into the Apple GPU ring.

---

## Current Status
S-MoE has achieved **Milestone 6**, featuring a fully integrated 28-layer sequential execution loop. 
- The engine successfully coordinates CPU-based Attention and Shared Expert processing with high-speed GPU-accelerated Routed Expert dispatches (`smoe_metal_fused_ffn`).
- Multi-layer KV-caches are managed entirely in static/aligned memory, strictly preserving the **zero runtime heap allocation** invariant.
- Sustains active streaming and execution at ~1.6 tokens/second natively on Apple Silicon.

---

## License

S-MoE is open-source software licensed under the [MIT license](LICENSE).

Built by [Luca Visciola](https://github.com/melasistema) and an AI agent who hopefully knows C++ and low level programming, because i do not 😅.