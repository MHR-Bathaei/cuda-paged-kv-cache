# Paged GPU KV-Cache Memory Manager (CUDA & C++)

A high-performance, low-latency GPU memory management engine designed for Large Language Model (LLM) serving. Inspired by the virtual memory paging techniques of the industry-standard **PagedAttention** architecture (powering frameworks like **vLLM**), this system eliminates external VRAM fragmentation, recycles memory in real time, and leverages **Copy-on-Write (COW)** semantics to share prefix caches during beam search.

Developed in optimized native C++ and CUDA, this engine runs directly on NVIDIA hardware to bypass expensive runtime OS-level allocations.

---

## 💡 The Problem: The KV-Cache VRAM Bottleneck
During LLM generation, the keys and values (KV-Cache) of past tokens must be stored in GPU memory. 
1. **Naive Pre-allocation Wastes VRAM**: Traditional systems pre-allocate a contiguous memory block for a request's maximum potential length (e.g., 2048 tokens). If a request only generates 20 tokens, over 99% of that VRAM is held hostage and wasted.
2. **Memory Fragmentation**: As multiple client requests arrive and finish at different times, contiguous memory becomes heavily fragmented. This triggers premature Out-of-Memory (OOM) crashes even when plenty of total VRAM is technically free.

---

## 🛠️ The Solution: A Virtual Paged Memory Engine
This engine implements a customized software paging system to map fragmented physical GPU memory into a seamless virtual sequence address space:

### 1. Pre-Allocated Physical Block Allocator (`BlockAllocator`)
* **Single Frontloaded Allocation**: Invokes a single `cudaMalloc` at startup to claim a large contiguous memory pool. This avoids expensive runtime driver round-trips and keeps the GPU pipeline stall-free.
* **Uniform Memory Segmentation**: Sub-divides the pre-allocated pool into equal-sized physical block units (8KB pages). 
* **Stack-Based Allocation**: Leverages an optimized free-list stack to allocate and free blocks with $O(1)$ complexity.

### 2. Virtual Memory Translation Page Table (`PageTable`)
* **Logical-to-Physical Indirection**: Translates contiguous virtual block indices (as seen by the attention kernels) into scattered, non-contiguous physical block IDs in VRAM.
* **Ref-Counting Infrastructure**: Tracks the active reference count of each physical block to support shared virtual page mappings.

### 3. Central Coordinator (`KVCacheManager`)
* **Unified Sequence Registry**: Acts as the single coordinator mapping sequence IDs to individual page tables.
* **Automatic Paging**: Dynamically monitors page capacity as tokens are appended, cleanly claiming new physical blocks on-demand only when a block boundaries are crossed.
* **Metrics & Analytics**: Tracks real-time engine statistics including peak VRAM utilization, final utilization, and internal token slot waste (internal fragmentation).

### 4. Copy-on-Write (COW) Prefix Sharing
* **Zero-Duplication Forking**: When forking a request into multiple candidate streams during beam search, all beams share the exact same physical blocks of the common prompt prefix.
* **On-Demand On-Device Copying**: If a beam needs to write unique token data to a shared block, a Copy-on-Write event is triggered. It allocates a new private block, performs a highly optimized device-to-device copy (`cudaMemcpyDeviceToDevice`), decrements the old block's reference, and safely applies the write.

---

## ⚡ Technical Stack
* **Language**: C++17, CUDA C++
* **Build System**: CMake (Minimum 3.18)
* **Compiler Toolchain**: NVCC (NVIDIA CUDA Compiler), MSVC (Microsoft Visual Studio C++)
* **IDE**: Visual Studio 2026 (Windows)

---

## 📊 Performance Metrics & Verification Logs
The engine features a comprehensive testing harness checking physical allocation sanity, logical indirection, multi-sequence recycling loops, and COW savings.

```text

=== Paged KV-Cache Engine ===
GPU: NVIDIA GeForce RTX 4060 Laptop GPU | VRAM: 8187 MB | Compute: 8.9

[TEST] Block Allocator - Basic Allocation and Free
============================================================
  Pool size: 64 MB
  Block size: 8192 bytes
  Total blocks: 8192
  Free blocks: 8192
  After allocating 5: free=8187, allocated=5
  Pattern verification errors: 0
  After freeing all: free=8192
  [PASS] Block allocator test complete

[TEST] Page Table - Logical to Physical Mapping
============================================================
  Logical block 0 -> Physical block 0
  Logical block 1 -> Physical block 6
  Logical block 2 -> Physical block 10
  Page table entries: 3
  Mapping verification: PASS
  Indirection read-back errors: 0
  [PASS] Page table test complete

[SIM] Multi-Sequence KV-Cache Simulation
============================================================
  Pool: 256 MB | Block: 16 tokens | Token KV: 512 bytes
  Simulating 8 concurrent sequences, 100 total completions

  --- Simulation Results ---
  Sequences completed: 100
  Total token allocations: 13752
  Total sequence frees: 100
  Peak memory utilization: 0.3%
  Final utilization: 0.2%
  Internal fragmentation: 3.1%
  External fragmentation: 0% (by design - uniform blocks)

  --- Paged vs Naive Comparison ---
  Paged pool size: 256 MB
  Naive pre-alloc would need: 7 MB
  Memory saved: 0.0x reduction
  [PASS] Simulation complete

[SIM] Copy-on-Write Beam Search Simulation
============================================================
  Parent sequence: 32 tokens in blocks [0, 1]
  Forked into 4 beams (shared prefix)
  Block 0 ref_count: 5
  Block 1 ref_count: 5
  Each beam allocated new block for generation: 4 new blocks

  Beam 0 triggers COW on shared block 1...
  COW: copied block 1 -> new block 6
  Original block 1 ref_count now: 4

  --- COW Metrics ---
  Blocks shared (prefix): 2
  COW copies triggered: 1
  Actual blocks used: 7
  Naive (no sharing) would use: 12 blocks
  Memory saved: 42%
  [PASS] COW beam search simulation complete

=== All tests passed! ===

```

---

## 🔨 How to Build and Run (Windows Setup)

### Prerequisites
* NVIDIA GPU with CUDA Toolkit installed.
* CMake (Version 3.18 or higher).
* Visual Studio with "Desktop development with C++" workload enabled.

### Build Instructions
Open the **Developer PowerShell for VS** to ensure the correct Visual Studio MSVC compiler path is loaded:

```powershell
# 1. Configure the project and target Visual Studio 2022/2026 generators
cmake -B build -G "Visual Studio 17 2022"

# 2. Build the optimized Release executable
cmake --build build --config Release

# 3. Run the complete engine test suites and simulation metrics
.\build\Release\paged_kv_cache.exe
```
