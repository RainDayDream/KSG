#pragma once
// Compressed, heap-resident KSG adjacency (low-memory variant, --mem-graph-compact): every neighbour list is sorted and stored as varints
// (first: zigzag(id0 - node), then id_k - id_{k-1} - 1). Node -> byte offset is two-level: a u64 base per block of B nodes plus a u16
// offset of each node inside its block (B is the largest power of two <= 64 whose block spans all fit in 16 bits; chosen by the builder).
// File: [u32 n][u32 B][u32 magic 'KSGC'][u32 0][u64 stream bytes] base u64[n/B + 1], rel u16[n + 1], stream bytes.
// Memory: stream + 8 n / B + 2 n bytes, read into heap memory (no mmap: counted as anonymous memory, nothing for the kernel to reclaim).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace ksg {

class CompactAdj {
public:
    static constexpr uint32_t kMagic = 0x4347534B;   // "KSGC"
    explicit CompactAdj(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open compact KSG: " + path);
        uint32_t h[4]; uint64_t sb;
        if (std::fread(h, 4, 4, f) != 4 || std::fread(&sb, 8, 1, f) != 1 || h[2] != kMagic) throw std::runtime_error("bad compact KSG header");
        n_ = h[0]; B_ = h[1];
        base_.resize(n_ / B_ + 1); rel_.resize(size_t(n_) + 1); s_.resize(sb);
        read(f, base_.data(), base_.size() * 8); read(f, rel_.data(), rel_.size() * 2); read(f, s_.data(), sb);
        std::fclose(f);
        int fd = ::open(path.c_str(), O_RDONLY); if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }   // drop the page cache copy
    }
    uint32_t size() const { return n_; }
    size_t bytes() const { return s_.size() + base_.size() * 8 + rel_.size() * 2; }
    inline size_t begin(uint32_t i) const { return base_[i / B_] + rel_[i]; }
    // decode node i into a thread-local buffer; the returned pointer is valid until the next call on this thread
    inline const uint32_t* neighbors(uint32_t i, uint32_t& count) const {
        thread_local std::vector<uint32_t> buf;
        const uint8_t* p = s_.data() + begin(i); const uint8_t* e = s_.data() + begin(i + 1);
        buf.clear();
        int64_t prev = 0; bool first = true;
        while (p < e) {
            uint64_t v = 0; int sh = 0; uint8_t b;
            do { b = *p++; v |= uint64_t(b & 0x7F) << sh; sh += 7; } while (b & 0x80);
            int64_t id;
            if (first) { id = int64_t(i) + (int64_t(v >> 1) ^ -int64_t(v & 1)); first = false; } else id = prev + int64_t(v) + 1;
            buf.push_back(uint32_t(id)); prev = id;
        }
        count = static_cast<uint32_t>(buf.size());
        return buf.data();
    }
private:
    static void read(FILE* f, void* dst, size_t bytes) {
        const size_t CH = size_t(1) << 26; char* d = static_cast<char*>(dst);
        for (size_t o = 0; o < bytes; o += CH) { const size_t c = std::min(CH, bytes - o); if (std::fread(d + o, 1, c, f) != c) throw std::runtime_error("short compact KSG file"); }
    }
    uint32_t n_ = 0, B_ = 1;
    std::vector<uint64_t> base_; std::vector<uint16_t> rel_; std::vector<uint8_t> s_;
};

}  // namespace ksg
