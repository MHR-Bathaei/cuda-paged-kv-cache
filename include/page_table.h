#ifndef PAGE_TABLE_H
#define PAGE_TABLE_H

#include <vector>
#include <cstddef>

class PageTable {
public:
    PageTable() : num_filled_in_last_block_(0) {}

    void append_block(int physical_block_id) {
        entries_.push_back(physical_block_id);
        num_filled_in_last_block_ = 0;
    }

    int get_physical_id(size_t logical_index) const {
        if (logical_index >= entries_.size()) return -1;
        return entries_[logical_index];
    }

    size_t num_blocks() const { return entries_.size(); }

    void set_filled_in_last(int count) { num_filled_in_last_block_ = count; }
    int filled_in_last() const { return num_filled_in_last_block_; }

    const std::vector<int>& entries() const { return entries_; }

    void clear() {
        entries_.clear();
        num_filled_in_last_block_ = 0;
    }

private:
    std::vector<int> entries_;
    int num_filled_in_last_block_;
};

#endif // PAGE_TABLE_H