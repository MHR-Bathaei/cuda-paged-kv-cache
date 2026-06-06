#ifndef KV_CACHE_MANAGER_H
#define KV_CACHE_MANAGER_H

#include "block_allocator.cuh"
#include "page_table.h"
#include <unordered_map>
#include <cstdio>

struct CacheStats {
    size_t total_blocks;
    size_t free_blocks;
    size_t allocated_blocks;
    size_t active_sequences;
    float utilization_percent;
    float internal_frag_percent;
};

class KVCacheManager {
public:
    KVCacheManager(size_t total_pool_bytes, size_t block_size_tokens,
        size_t token_kv_size_bytes)
        : block_size_tokens_(block_size_tokens),
        token_kv_size_bytes_(token_kv_size_bytes),
        allocator_(total_pool_bytes, block_size_tokens* token_kv_size_bytes) {
    }

    bool register_sequence(int seq_id) {
        if (sequences_.count(seq_id)) return false;
        sequences_[seq_id] = PageTable();
        return true;
    }

    // Returns true if successful, false if out of memory
    bool append_token(int seq_id) {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return false;

        PageTable& pt = it->second;

        // Check if we need a new block
        if (pt.num_blocks() == 0 || pt.filled_in_last() >= static_cast<int>(block_size_tokens_)) {
            int new_block = allocator_.allocate();
            if (new_block < 0) return false; // OOM
            pt.append_block(new_block);
        }

        pt.set_filled_in_last(pt.filled_in_last() + 1);
        return true;
    }

    // Append multiple tokens at once (e.g., prompt prefill)
    bool append_tokens(int seq_id, int count) {
        for (int i = 0; i < count; ++i) {
            if (!append_token(seq_id)) return false;
        }
        return true;
    }

    void free_sequence(int seq_id) {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return;

        const PageTable& pt = it->second;
        for (int phys_id : pt.entries()) {
            allocator_.free(phys_id);
        }
        sequences_.erase(it);
    }

    const PageTable* get_page_table(int seq_id) const {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return nullptr;
        return &it->second;
    }

    float* get_token_ptr(int seq_id, int token_index) const {
        auto it = sequences_.find(seq_id);
        if (it == sequences_.end()) return nullptr;

        const PageTable& pt = it->second;
        size_t logical_block = token_index / block_size_tokens_;
        size_t offset_in_block = token_index % block_size_tokens_;

        int phys_id = pt.get_physical_id(logical_block);
        if (phys_id < 0) return nullptr;

        float* block_ptr = allocator_.get_block_ptr(phys_id);
        if (!block_ptr) return nullptr;

        size_t float_offset = offset_in_block * (token_kv_size_bytes_ / sizeof(float));
        return block_ptr + float_offset;
    }

    CacheStats get_stats() const {
        CacheStats stats;
        stats.total_blocks = allocator_.num_total_blocks();
        stats.free_blocks = allocator_.num_free_blocks();
        stats.allocated_blocks = allocator_.num_allocated_blocks();
        stats.active_sequences = sequences_.size();

        stats.utilization_percent = (stats.total_blocks > 0)
            ? 100.0f * stats.allocated_blocks / stats.total_blocks
            : 0.0f;

        // Internal fragmentation: wasted slots in partially-filled last blocks
        size_t total_slots_allocated = stats.allocated_blocks * block_size_tokens_;
        size_t total_slots_used = 0;
        for (const auto& pair : sequences_) {
            const PageTable& pt = pair.second;
            if (pt.num_blocks() > 0) {
                total_slots_used += (pt.num_blocks() - 1) * block_size_tokens_ + pt.filled_in_last();
            }
        }
        stats.internal_frag_percent = (total_slots_allocated > 0)
            ? 100.0f * (total_slots_allocated - total_slots_used) / total_slots_allocated
            : 0.0f;

        return stats;
    }

    size_t block_size_tokens() const { return block_size_tokens_; }
    const BlockAllocator& allocator() const { return allocator_; }

private:
    size_t block_size_tokens_;
    size_t token_kv_size_bytes_;
    BlockAllocator allocator_;
    std::unordered_map<int, PageTable> sequences_;
};

#endif // KV_CACHE_MANAGER_H