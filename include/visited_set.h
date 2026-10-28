#pragma once
//
// BitSet needs N bits per set (125 MB at 1B, 12.5 MB at 100M). Each thread holds
// several of them, so every visited/expanded check is a random DRAM + TLB miss.
// A query only touches O(L * degree) ids (~1e5), so an open-addressing hash set
// of a few MB stays in L2/L3. Mode is chosen at reserve():
//   bitmap : historical BitSet (dirty-word clear)
//   hash   : linear-probing table, grows x2 at 50% load, O(touched) clear
//   auto   : bitmap if max_id <= g_visited_auto_bitmap_max, else hash
#include <cstdint>
#include <cstring>
#include <vector>
#include "types.h"

enum class VisitedMode : int { Bitmap = 0, Hash = 1, Auto = 2 };
inline VisitedMode g_visited_mode = VisitedMode::Auto;
inline size_t g_visited_auto_bitmap_max = size_t{1} << 24;  // 16M ids -> 2 MB bitmap
inline size_t g_visited_hash_initial = size_t{1} << 16;

class VisitedSet {
public:
    void reserve(size_t max_id) {
        const bool use_hash = g_visited_mode == VisitedMode::Hash ||
            (g_visited_mode == VisitedMode::Auto && max_id > g_visited_auto_bitmap_max);
        hash_ = use_hash;
        count_ = 0;
        if (!hash_) {
            bits_.reserve(max_id);
            table_.clear(); table_.shrink_to_fit();
            slots_.clear();
        } else {
            init_table(g_visited_hash_initial);
            slots_.clear();
            slots_.reserve(4096);
        }
    }

    // Always use the hash table regardless of g_visited_mode (for sparse id spaces).
    void reserve_hash(size_t initial = g_visited_hash_initial) {
        hash_ = true; count_ = 0;
        init_table(initial);
        slots_.clear();
        slots_.reserve(4096);
    }

    inline bool insert(uint32_t id) {
        if (!hash_) return bits_.insert(id);
        const uint32_t key = id + 1u;
        size_t pos = hash_pos(id);
        while (true) {
            const uint32_t cur = table_[pos];
            if (cur == key) return false;
            if (cur == 0) break;
            pos = (pos + 1) & mask_;
        }
        table_[pos] = key;
        slots_.push_back(static_cast<uint32_t>(pos));
        ++count_;
        if (count_ * 2 >= table_.size()) grow();
        return true;
    }

    inline bool contains(uint32_t id) const {
        if (!hash_) return bits_.contains(id);
        const uint32_t key = id + 1u;
        size_t pos = hash_pos(id);
        while (true) {
            const uint32_t cur = table_[pos];
            if (cur == key) return true;
            if (cur == 0) return false;
            pos = (pos + 1) & mask_;
        }
    }

    void clear() {
        if (!hash_) { bits_.clear(); return; }
        if (slots_.size() * 8 > table_.size()) {
            std::memset(table_.data(), 0, table_.size() * sizeof(uint32_t));
        } else {
            for (uint32_t s : slots_) table_[s] = 0;
        }
        slots_.clear();
        count_ = 0;
    }

    size_t size() const { return hash_ ? count_ : bits_.size(); }
    bool empty() const { return size() == 0; }
    bool is_hash() const { return hash_; }

private:
    inline size_t hash_pos(uint32_t id) const {
        return static_cast<size_t>((static_cast<uint64_t>(id) * 0x9E3779B97F4A7C15ull) >> shift_);
    }
    void init_table(size_t cap) {
        size_t c = 16; int bits = 4;
        while (c < cap) { c <<= 1; ++bits; }
        table_.assign(c, 0u);
        mask_ = c - 1;
        shift_ = 64 - bits;
    }
    void grow() {
        std::vector<uint32_t> old_keys;
        old_keys.reserve(slots_.size());
        for (uint32_t s : slots_) old_keys.push_back(table_[s]);
        init_table(table_.size() * 2);
        slots_.clear();
        for (uint32_t key : old_keys) {
            size_t pos = hash_pos(key - 1u);
            while (table_[pos] != 0) pos = (pos + 1) & mask_;
            table_[pos] = key;
            slots_.push_back(static_cast<uint32_t>(pos));
        }
    }

    BitSet bits_;
    std::vector<uint32_t> table_;
    std::vector<uint32_t> slots_;
    size_t mask_ = 0;
    int shift_ = 64;
    size_t count_ = 0;
    bool hash_ = false;
};
