# S-MoE: Seismic Mixture of Experts Engine

> **Utopia on consumer hardware.** High-fidelity local frontier LLM inference running completely within the democratic boundaries of a standard 16GB/32GB Apple Silicon MacBook.

---

## The Vision
Traditional local LLM deployment is trapped behind a financial paywall. Running a 284-billion parameter frontier model natively requires an enterprise machine with hundreds of gigabytes of Unified Memory.

**S-MoE** breaks this paradigm by rethinking the computational medium. Instead of holding the entire model graph statically in memory, S-MoE treats your Mac’s ultra-fast NVMe flash storage ($~7.4\text{ GB/s}$) as the primary model repository, using predictive streaming mechanics to pipe weights into a tiny, hyper-active memory buffer fractions of a second before they are executed by the GPU.

---

## How It Works: The Phonon Metaphor
S-MoE splits monolithic models into two distinct, high-performance layers:

1. **The Surface Scout:** A lightweight, highly efficient 1.5B parameter model sitting permanently in memory (~3GB). As it begins processing your prompt, it listens to the semantic trajectory of the text and projects a "seismic map" forecasting exactly which specialized experts will be needed up to 10 tokens into the future.
2. **The Sub-Surface Core:** The massive collection of hundreds of deep experts rests cold on the SSD, shattered into custom page-aligned binary segments.
3. **The Asynchronous JIT Pump:** Guided by the Scout's predictive map, specialized background workers read data asynchronously from the storage controller using Direct I/O (`F_NOCACHE`), packing the upcoming experts into a cyclic execution ring buffer. The GPU processes the weights, flushes them instantly, and hops to the next incoming wave.

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

## License

S-MoE is open-source software licensed under the [MIT license](LICENSE).

Built with ⚡️ by [Luca Visciola](https://github.com/melasistema).