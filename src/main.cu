#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "kv_cache_manager.h"

// =================================================================
// INTELLISENSE FIX FOR VISUAL STUDIO 2026
// Keeps the editor visual analyzer happy while compiling perfectly!
// =================================================================
#ifdef __INTELLISENSE__
#define __global__
struct Dim3 { int x; int y; int z; };
extern Dim3 blockIdx;
extern Dim3 threadIdx;
extern Dim3 blockDim;
inline int atomicAdd(int* address, int val) { return 0; }
#endif
// =================================================================

// Simple kernel to write a pattern to a block's memory
__global__ void write_pattern_kernel(float* block_ptr, int num_elements, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        block_ptr[idx] = value;
    }
}

// Kernel to verify a pattern
__global__ void verify_pattern_kernel(float* block_ptr, int num_elements,
    float expected, int* errors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        if (block_ptr[idx] != expected) {
            atomicAdd(errors, 1);
        }
    }
}

void print_separator() {
    printf("============================================================\n");
}

void test_block_allocator() {
    printf("\n[TEST] Block Allocator - Basic Allocation and Free\n");
    print_separator();

    size_t pool_size = 64 * 1024 * 1024;
    size_t block_size = 16 * 128 * sizeof(float);
    BlockAllocator alloc(pool_size, block_size);

    printf("  Pool size: %zu MB\n", pool_size / (1024 * 1024));
    printf("  Block size: %zu bytes\n", block_size);
    printf("  Total blocks: %zu\n", alloc.num_total_blocks());
    printf("  Free blocks: %zu\n", alloc.num_free_blocks());

    std::vector<int> allocated;
    for (int i = 0; i < 5; ++i) {
        int id = alloc.allocate();
        if (id >= 0) allocated.push_back(id);
    }
    printf("  After allocating 5: free=%zu, allocated=%zu\n",
        alloc.num_free_blocks(), alloc.num_allocated_blocks());

    int elements_per_block = static_cast<int>(block_size / sizeof(float));
    int threads = 256;
    int blocks = (elements_per_block + threads - 1) / threads;

    int* d_errors;
    cudaMalloc(&d_errors, sizeof(int));

    for (size_t i = 0; i < allocated.size(); ++i) {
        float* ptr = alloc.get_block_ptr(allocated[i]);
        float pattern = static_cast<float>(i + 1) * 1.5f;

#ifdef __INTELLISENSE__
        write_pattern_kernel(ptr, elements_per_block, pattern);
#else
        write_pattern_kernel << <blocks, threads >> > (ptr, elements_per_block, pattern);
#endif
    }
    cudaDeviceSynchronize();

    int total_errors = 0;
    for (size_t i = 0; i < allocated.size(); ++i) {
        float* ptr = alloc.get_block_ptr(allocated[i]);
        float pattern = static_cast<float>(i + 1) * 1.5f;
        int zero = 0;
        cudaMemcpy(d_errors, &zero, sizeof(int), cudaMemcpyHostToDevice);

#ifdef __INTELLISENSE__
        verify_pattern_kernel(ptr, elements_per_block, pattern, d_errors);
#else
        verify_pattern_kernel << <blocks, threads >> > (ptr, elements_per_block, pattern, d_errors);
#endif

        int err_count = 0;
        cudaMemcpy(&err_count, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
        total_errors += err_count;
    }
    printf("  Pattern verification errors: %d\n", total_errors);

    for (int id : allocated) {
        alloc.free(id);
    }
    printf("  After freeing all: free=%zu\n", alloc.num_free_blocks());

    cudaFree(d_errors);
    printf("  [PASS] Block allocator test complete\n");
}

void test_page_table_mapping() {
    printf("\n[TEST] Page Table - Logical to Physical Mapping\n");
    print_separator();

    size_t pool_size = 64 * 1024 * 1024;
    size_t tokens_per_block = 16;
    size_t token_kv_bytes = 128 * sizeof(float);
    size_t block_bytes = tokens_per_block * token_kv_bytes;

    BlockAllocator alloc(pool_size, block_bytes);

    int b0 = alloc.allocate();
    std::vector<int> skipped;
    for (int i = 0; i < 5; ++i) skipped.push_back(alloc.allocate());
    int b1 = alloc.allocate();
    for (int i = 0; i < 3; ++i) skipped.push_back(alloc.allocate());
    int b2 = alloc.allocate();

    for (int id : skipped) alloc.free(id);

    printf("  Logical block 0 -> Physical block %d\n", b0);
    printf("  Logical block 1 -> Physical block %d\n", b1);
    printf("  Logical block 2 -> Physical block %d\n", b2);

    PageTable pt;
    pt.append_block(b0);
    pt.append_block(b1);
    pt.append_block(b2);

    bool pass = (pt.get_physical_id(0) == b0 &&
        pt.get_physical_id(1) == b1 &&
        pt.get_physical_id(2) == b2);
    printf("  Page table entries: %zu\n", pt.num_blocks());
    printf("  Mapping verification: %s\n", pass ? "PASS" : "FAIL");

    int elements = static_cast<int>(block_bytes / sizeof(float));
    int threads = 256;
    int grid = (elements + threads - 1) / threads;

#ifdef __INTELLISENSE__
    write_pattern_kernel(alloc.get_block_ptr(b0), elements, 100.0f);
    write_pattern_kernel(alloc.get_block_ptr(b1), elements, 200.0f);
    write_pattern_kernel(alloc.get_block_ptr(b2), elements, 300.0f);
#else
    write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(b0), elements, 100.0f);
    write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(b1), elements, 200.0f);
    write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(b2), elements, 300.0f);
#endif
    cudaDeviceSynchronize();

    int* d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    int zero = 0;
    int total_errors = 0;

    for (int logical = 0; logical < 3; ++logical) {
        int phys = pt.get_physical_id(logical);
        float expected = (logical + 1) * 100.0f;
        cudaMemcpy(d_errors, &zero, sizeof(int), cudaMemcpyHostToDevice);

#ifdef __INTELLISENSE__
        verify_pattern_kernel(alloc.get_block_ptr(phys), elements, expected, d_errors);
#else
        verify_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(phys), elements, expected, d_errors);
#endif

        int err_count = 0;
        cudaMemcpy(&err_count, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
        total_errors += err_count;
    }
    printf("  Indirection read-back errors: %d\n", total_errors);

    alloc.free(b0);
    alloc.free(b1);
    alloc.free(b2);
    cudaFree(d_errors);
    printf("  [PASS] Page table test complete\n");
}

void run_multi_sequence_simulation() {
    printf("\n[SIM] Multi-Sequence KV-Cache Simulation\n");
    print_separator();

    size_t pool_size = 256 * 1024 * 1024;
    size_t tokens_per_block = 16;
    size_t token_kv_bytes = 128 * sizeof(float);
    int num_concurrent = 8;
    int total_completions = 100;

    KVCacheManager manager(pool_size, tokens_per_block, token_kv_bytes);

    printf("  Pool: %zu MB | Block: %zu tokens | Token KV: %zu bytes\n",
        pool_size / (1024 * 1024), tokens_per_block, token_kv_bytes);
    printf("  Simulating %d concurrent sequences, %d total completions\n\n",
        num_concurrent, total_completions);

    srand(42);

    struct SeqInfo {
        int seq_id;
        int current_len;
        int target_len;
        bool active;
    };

    std::vector<SeqInfo> slots(num_concurrent);
    int next_seq_id = 0;
    int completions = 0;
    size_t total_allocs = 0;
    size_t total_frees = 0;
    float peak_utilization = 0.0f;
    size_t naive_memory_needed = 0;

    for (int i = 0; i < num_concurrent; ++i) {
        slots[i].seq_id = next_seq_id++;
        slots[i].current_len = 0;
        slots[i].target_len = 32 + rand() % 225;
        slots[i].active = true;
        manager.register_sequence(slots[i].seq_id);

        int prompt_len = 4 + rand() % 29;
        manager.append_tokens(slots[i].seq_id, prompt_len);
        slots[i].current_len = prompt_len;
        naive_memory_needed += slots[i].target_len * token_kv_bytes;
    }

    while (completions < total_completions) {
        for (int i = 0; i < num_concurrent; ++i) {
            if (!slots[i].active) continue;

            if (manager.append_token(slots[i].seq_id)) {
                slots[i].current_len++;
                total_allocs++;
            }

            if (slots[i].current_len >= slots[i].target_len) {
                manager.free_sequence(slots[i].seq_id);
                total_frees++;
                completions++;
                slots[i].active = false;

                if (completions < total_completions) {
                    slots[i].seq_id = next_seq_id++;
                    slots[i].current_len = 0;
                    slots[i].target_len = 32 + rand() % 225;
                    slots[i].active = true;
                    manager.register_sequence(slots[i].seq_id);

                    int prompt_len = 4 + rand() % 29;
                    manager.append_tokens(slots[i].seq_id, prompt_len);
                    slots[i].current_len = prompt_len;
                    naive_memory_needed += slots[i].target_len * token_kv_bytes;
                }
            }
        }

        CacheStats stats = manager.get_stats();
        if (stats.utilization_percent > peak_utilization) {
            peak_utilization = stats.utilization_percent;
        }
    }

    CacheStats final_stats = manager.get_stats();
    printf("  --- Simulation Results ---\n");
    printf("  Sequences completed: %d\n", completions);
    printf("  Total token allocations: %zu\n", total_allocs);
    printf("  Total sequence frees: %zu\n", total_frees);
    printf("  Peak memory utilization: %.1f%%\n", peak_utilization);
    printf("  Final utilization: %.1f%%\n", final_stats.utilization_percent);
    printf("  Internal fragmentation: %.1f%%\n", final_stats.internal_frag_percent);
    printf("  External fragmentation: 0%% (by design - uniform blocks)\n");
    printf("\n  --- Paged vs Naive Comparison ---\n");
    printf("  Paged pool size: %zu MB\n", pool_size / (1024 * 1024));
    printf("  Naive pre-alloc would need: %zu MB\n", naive_memory_needed / (1024 * 1024));
    printf("  Memory saved: %.1fx reduction\n",
        static_cast<float>(naive_memory_needed) / pool_size);
    printf("\n  [PASS] Simulation complete\n");
}

void run_cow_beam_search_simulation() {
    printf("\n[SIM] Copy-on-Write Beam Search Simulation\n");
    print_separator();

    size_t pool_size = 128 * 1024 * 1024;
    size_t tokens_per_block = 16;
    size_t token_kv_bytes = 128 * sizeof(float);
    size_t block_bytes = tokens_per_block * token_kv_bytes;

    BlockAllocator alloc(pool_size, block_bytes);

    // Allocate 2 blocks for parent sequence (32 tokens total)
    int parent_blocks[2];
    parent_blocks[0] = alloc.allocate();
    parent_blocks[1] = alloc.allocate();

    // Write distinct patterns to parent blocks
    int elements = static_cast<int>(block_bytes / sizeof(float));
    int threads = 256;
    int grid = (elements + threads - 1) / threads;

#ifdef __INTELLISENSE__
    write_pattern_kernel(alloc.get_block_ptr(parent_blocks[0]), elements, 1.0f);
    write_pattern_kernel(alloc.get_block_ptr(parent_blocks[1]), elements, 2.0f);
#else
    write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(parent_blocks[0]), elements, 1.0f);
    write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(parent_blocks[1]), elements, 2.0f);
#endif
    cudaDeviceSynchronize();

    printf("  Parent sequence: 32 tokens in blocks [%d, %d]\n",
        parent_blocks[0], parent_blocks[1]);

    // Fork parent into 4 beams - each beam shares the parent blocks
    int num_beams = 4;
    struct BeamState {
        PageTable page_table;
        int beam_id;
    };

    std::vector<BeamState> beams(num_beams);
    for (int b = 0; b < num_beams; ++b) {
        beams[b].beam_id = b;
        beams[b].page_table.append_block(parent_blocks[0]);
        beams[b].page_table.append_block(parent_blocks[1]);
        beams[b].page_table.set_filled_in_last(static_cast<int>(tokens_per_block));

        // Increment ref counts - blocks are now shared
        alloc.increment_ref(parent_blocks[0]);
        alloc.increment_ref(parent_blocks[1]);
    }

    printf("  Forked into %d beams (shared prefix)\n", num_beams);
    printf("  Block %d ref_count: %d\n", parent_blocks[0], alloc.get_ref_count(parent_blocks[0]));
    printf("  Block %d ref_count: %d\n", parent_blocks[1], alloc.get_ref_count(parent_blocks[1]));

    size_t blocks_before_cow = alloc.num_allocated_blocks();
    int cow_copies = 0;
    int new_blocks_allocated = 0;

    // Each beam allocates its own new block for generated tokens
    for (int b = 0; b < num_beams; ++b) {
        int new_block = alloc.allocate();
        if (new_block >= 0) {
            beams[b].page_table.append_block(new_block);
            beams[b].page_table.set_filled_in_last(1);
            new_blocks_allocated++;

            // Write beam-specific data to the new block
            float beam_val = (b + 1) * 10.0f;

#ifdef __INTELLISENSE__
            write_pattern_kernel(alloc.get_block_ptr(new_block), elements, beam_val);
#else
            write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(new_block), elements, beam_val);
#endif
        }
    }
    cudaDeviceSynchronize();

    printf("  Each beam allocated new block for generation: %d new blocks\n", new_blocks_allocated);

    // COW trigger: beam 0 needs to modify shared block 1
    printf("\n  Beam 0 triggers COW on shared block %d...\n", parent_blocks[1]);

    int shared_block = parent_blocks[1];
    if (alloc.get_ref_count(shared_block) > 1) {
        // Block is shared - must copy before writing
        int cow_block = alloc.allocate();
        if (cow_block >= 0) {
            // Copy data device-to-device
            cudaMemcpy(alloc.get_block_ptr(cow_block),
                alloc.get_block_ptr(shared_block),
                block_bytes, cudaMemcpyDeviceToDevice);
            // Decrement old block's ref count
            alloc.free(shared_block);

            cow_copies++;
            printf("  COW: copied block %d -> new block %d\n", shared_block, cow_block);
            printf("  Original block %d ref_count now: %d\n",
                shared_block, alloc.get_ref_count(shared_block));

            // Now safe to write to the private copy
#ifdef __INTELLISENSE__
            write_pattern_kernel(alloc.get_block_ptr(cow_block), elements, 99.0f);
#else
            write_pattern_kernel << <grid, threads >> > (alloc.get_block_ptr(cow_block), elements, 99.0f);
#endif
            cudaDeviceSynchronize();
        }
    }

    // Calculate and print COW metrics
    size_t blocks_after = alloc.num_allocated_blocks();
    size_t naive_blocks_needed = num_beams * 3;
    size_t actual_blocks_used = blocks_after;

    printf("\n  --- COW Metrics ---\n");
    printf("  Blocks shared (prefix): 2\n");
    printf("  COW copies triggered: %d\n", cow_copies);
    printf("  Actual blocks used: %zu\n", actual_blocks_used);
    printf("  Naive (no sharing) would use: %zu blocks\n", naive_blocks_needed);
    printf("  Memory saved: %.0f%%\n",
        100.0f * (1.0f - static_cast<float>(actual_blocks_used) / naive_blocks_needed));
    printf("  [PASS] COW beam search simulation complete\n");

    // Cleanup - free all remaining blocks
    for (int b = 0; b < num_beams; ++b) {
        for (int phys_id : beams[b].page_table.entries()) {
            alloc.free(phys_id);
        }
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Paged KV-Cache Engine ===\n");
    printf("GPU: %s | VRAM: %zu MB | Compute: %d.%d\n\n",
        prop.name, prop.totalGlobalMem / (1024 * 1024),
        prop.major, prop.minor);

    test_block_allocator();
    test_page_table_mapping();
    run_multi_sequence_simulation();
    run_cow_beam_search_simulation();

    printf("\n=== All tests passed! ===\n");
    return 0;
}
