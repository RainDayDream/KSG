#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cstring>

using GID_Type = uint32_t;
using GL_Type = uint32_t;
using DisType = float;
using GraphType = std::vector<std::vector<GID_Type>>;

struct Neighbor
{
    GID_Type id;
    DisType distance;

    Neighbor() = default;

    Neighbor(GID_Type id, DisType distance) : id{id}, distance{distance}
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
    inline bool operator>(const Neighbor &other) const
    {
        return distance > other.distance || (distance == other.distance && id > other.id);
    }
};

class BitSet {
public:
    void reserve(size_t max_id) {
        size_t words = (max_id + 63) / 64;
        bits.assign(words, 0);
        dirty_marks.assign((words + 63) / 64, 0);
        dirty_words.clear();
        dirty_words.reserve(std::min<size_t>(words, 4096));
        count_ = 0;
    }

    bool insert(uint32_t id) {
        size_t word = id >> 6;
        uint64_t mask = 1ULL << (id & 63);
        if (bits[word] & mask) return false;
        bits[word] |= mask;
        mark_dirty(word);
        ++count_;
        return true;
    }

    bool contains(uint32_t id) const {
        size_t word = id >> 6;
        uint64_t mask = 1ULL << (id & 63);
        return bits[word] & mask;
    }

    void clear() {
        for (size_t w : dirty_words) {
            bits[w] = 0;
            dirty_marks[w >> 6] &= ~(1ULL << (w & 63));
        }
        dirty_words.clear();
        count_ = 0;
    }

    size_t size() const { return count_; }

    bool empty() const { return count_ == 0; }

private:
    void mark_dirty(size_t word) {
        size_t mark_word = word >> 6;
        uint64_t mark_mask = 1ULL << (word & 63);
        if (dirty_marks[mark_word] & mark_mask) return;
        dirty_marks[mark_word] |= mark_mask;
        dirty_words.push_back(word);
    }

    std::vector<uint64_t> bits;
    std::vector<uint64_t> dirty_marks;
    std::vector<size_t> dirty_words;
    size_t count_ = 0;
};


template<int D>
struct dary_max_heap {
    std::vector<Neighbor> data;
    void push(const Neighbor& v) {
        data.push_back(v);
        sift_up(data.size() - 1);
    }
    const Neighbor& top() const { return data[0]; }
    void pop() {
        data[0] = data.back();
        data.pop_back();
        if (!data.empty()) sift_down(0);
    }
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    void reserve(size_t n) { data.reserve(n); }
    void clear() { data.clear(); }
private:
    void sift_up(size_t idx) {
        while (idx > 0) {
            size_t parent = (idx - 1) / D;
            if (!(data[parent] < data[idx])) break;
            std::swap(data[parent], data[idx]);
            idx = parent;
        }
    }
    void sift_down(size_t idx) {
        while (true) {
            size_t best = idx;
            size_t first = idx * D + 1;
            for (int i = 0; i < D && first + i < data.size(); ++i) {
                if (data[best] < data[first + i]) best = first + i;
            }
            if (best == idx) break;
            std::swap(data[idx], data[best]);
            idx = best;
        }
    }
};

template<int D>
struct dary_min_heap {
    std::vector<Neighbor> data;
    void push(const Neighbor& v) {
        data.push_back(v);
        sift_up(data.size() - 1);
    }
    const Neighbor& top() const { return data[0]; }
    void pop() {
        data[0] = data.back();
        data.pop_back();
        if (!data.empty()) sift_down(0);
    }
    size_t size() const { return data.size(); }
    bool empty() const { return data.empty(); }
    void reserve(size_t n) { data.reserve(n); }
    void clear() { data.clear(); }
private:
    void sift_up(size_t idx) {
        while (idx > 0) {
            size_t parent = (idx - 1) / D;
            if (data[idx] < data[parent]) {
                std::swap(data[parent], data[idx]);
                idx = parent;
            } else break;
        }
    }
    void sift_down(size_t idx) {
        while (true) {
            size_t best = idx;
            size_t first = idx * D + 1;
            for (int i = 0; i < D && first + i < data.size(); ++i) {
                if (data[first + i] < data[best]) best = first + i;
            }
            if (best == idx) break;
            std::swap(data[idx], data[best]);
            idx = best;
        }
    }
};
