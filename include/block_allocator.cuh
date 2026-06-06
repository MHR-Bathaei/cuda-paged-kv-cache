#ifndef BLOCK_ALLOCATOR_CUH
#define BLOCK_ALLOCATOR_CUH

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

struct PhysicalBlock {
    int block_id;
    int ref_count;
    float* device_ptr;
};

class BlockAllocator {
public:
    BlockAllocator(size_t total_pool_bytes, size_t block_size_bytes)
        : block_size_bytes_(block_size_bytes), pool_base_(nullptr) {

        num_blocks_ = total_pool_bytes / block_size_bytes;
        if (num_blocks_ == 0) {
            throw std::runtime_error("Pool too small for even one block");
        }

        cudaError_t err = cudaMalloc(&pool_base_, num_blocks_ * block_size_bytes_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }

        // Initialize free list with all block IDs
        free_list_.reserve(num_blocks_);
        for (int i = static_cast<int>(num_blocks_) - 1; i >= 0; --i) {
            free_list_.push_back(i);
        }

        // Initialize block metadata
        blocks_.resize(num_blocks_);
        for (size_t i = 0; i < num_blocks_; ++i) {
            blocks_[i].block_id = static_cast<int>(i);
            blocks_[i].ref_count = 0;
            blocks_[i].device_ptr = reinterpret_cast<float*>(
                reinterpret_cast<uint8_t*>(pool_base_) + i * block_size_bytes_
                );
        }
    }

    ~BlockAllocator() {
        if (pool_base_) {
            cudaFree(pool_base_);
        }
    }

    // Non-copyable
    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;

    int allocate() {
        if (free_list_.empty()) {
            return -1; // No free blocks
        }
        int block_id = free_list_.back();
        free_list_.pop_back();
        blocks_[block_id].ref_count = 1;
        return block_id;
    }

    void free(int block_id) {
        if (block_id < 0 || block_id >= static_cast<int>(num_blocks_)) {
            return;
        }
        blocks_[block_id].ref_count--;
        if (blocks_[block_id].ref_count <= 0) {
            blocks_[block_id].ref_count = 0;
            free_list_.push_back(block_id);
        }
    }

    void increment_ref(int block_id) {
        if (block_id >= 0 && block_id < static_cast<int>(num_blocks_)) {
            blocks_[block_id].ref_count++;
        }
    }

    int get_ref_count(int block_id) const {
        if (block_id < 0 || block_id >= static_cast<int>(num_blocks_)) return 0;
        return blocks_[block_id].ref_count;
    }

    float* get_block_ptr(int block_id) const {
        if (block_id < 0 || block_id >= static_cast<int>(num_blocks_)) return nullptr;
        return blocks_[block_id].device_ptr;
    }

    size_t num_free_blocks() const { return free_list_.size(); }
    size_t num_total_blocks() const { return num_blocks_; }
    size_t num_allocated_blocks() const { return num_blocks_ - free_list_.size(); }
    size_t block_size_bytes() const { return block_size_bytes_; }
    void* pool_base() const { return pool_base_; }

private:
    void* pool_base_;
    size_t num_blocks_;
    size_t block_size_bytes_;
    std::vector<int> free_list_;
    std::vector<PhysicalBlock> blocks_;
};

#endif // BLOCK_ALLOCATOR_CUH