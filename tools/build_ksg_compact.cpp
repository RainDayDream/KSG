// build_ksg_compact: variable-degree KSG/KSG graph ([u32 n][u32 -1] then per node [u32 count][u32 ids]) -> compressed adjacency
// (include/compact_adj.h). Lists are sorted (search order inside a list does not matter: every neighbour is scored and pushed).
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -Iinclude -o build_ksg_compact tools/build_ksg_compact.cpp
// Usage: build_ksg_compact in.graph out.ksgc [max_degree base.bin(uint8) new2old]
//   with max_degree: keep only the max_degree nearest neighbours of every node (exact L2 on the uint8 base; graph ids are layout ids,
//   mapped to base rows through new2old) -> sparser KSG for low-memory budgets.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
static uint32_t g_k = 0, g_d = 0; static const uint8_t* g_base = nullptr; static const uint32_t* g_n2o = nullptr;
static inline uint32_t l2(uint32_t a, uint32_t b) {
    const uint8_t* x = g_base + size_t(g_n2o[a]) * g_d; const uint8_t* y = g_base + size_t(g_n2o[b]) * g_d; uint32_t s = 0;
    for (uint32_t j = 0; j < g_d; ++j) { const int t = int(x[j]) - int(y[j]); s += uint32_t(t * t); }
    return s;
}
static size_t enc(uint32_t node, const uint32_t* ids, uint32_t c, uint8_t* out, std::vector<uint32_t>& tmp) {
    tmp.assign(ids, ids + c);
    if (g_k && tmp.size() > g_k) {   // keep the g_k nearest (ties by id, deterministic)
        std::sort(tmp.begin(), tmp.end()); tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
        std::vector<std::pair<uint32_t, uint32_t>> dv; dv.reserve(tmp.size());
        for (uint32_t v : tmp) dv.emplace_back(l2(node, v), v);
        std::partial_sort(dv.begin(), dv.begin() + std::min<size_t>(g_k, dv.size()), dv.end());
        tmp.clear(); for (size_t j = 0; j < std::min<size_t>(g_k, dv.size()); ++j) tmp.push_back(dv[j].second);
    }
    std::sort(tmp.begin(), tmp.end()); tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());
    size_t k = 0; int64_t prev = 0;
    for (size_t j = 0; j < tmp.size(); ++j) {
        uint64_t v;
        if (j == 0) { const int64_t d = int64_t(tmp[0]) - int64_t(node); v = (uint64_t(d) << 1) ^ uint64_t(d >> 63); }
        else v = uint64_t(int64_t(tmp[j]) - prev - 1);
        prev = tmp[j];
        do { const uint8_t b = v & 0x7F; v >>= 7; if (out) out[k] = b | (v ? 0x80 : 0); ++k; } while (v);
    }
    return k;
}
int main(int argc, char** argv) {
    if (argc != 3 && argc != 6) { std::fprintf(stderr, "usage: %s in.graph out.ksgc [max_degree base.bin new2old]\n", argv[0]); return 2; }
    if (argc == 6) {
        g_k = std::atoi(argv[3]);
        int bf = ::open(argv[4], O_RDONLY); struct stat bs; fstat(bf, &bs);
        const uint8_t* bm = static_cast<const uint8_t*>(mmap(nullptr, bs.st_size, PROT_READ, MAP_PRIVATE, bf, 0)); uint32_t bn; std::memcpy(&bn, bm, 4); std::memcpy(&g_d, bm + 4, 4);
        if (size_t(bs.st_size) != 8 + size_t(bn) * g_d) { std::fprintf(stderr, "max_degree needs a uint8 base\n"); return 2; }
        g_base = bm + 8;
        int pf = ::open(argv[5], O_RDONLY); struct stat ps; fstat(pf, &ps); g_n2o = static_cast<const uint32_t*>(mmap(nullptr, ps.st_size, PROT_READ, MAP_PRIVATE, pf, 0));
    }
    int fd = ::open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t* m = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    madvise(const_cast<uint8_t*>(m), st.st_size, MADV_SEQUENTIAL);
    uint32_t n; std::memcpy(&n, m, 4);
    std::vector<uint64_t> foff(size_t(n) + 1); uint64_t o = 8; uint64_t edges = 0;
    for (uint32_t i = 0; i < n; ++i) { foff[i] = o; uint32_t c; std::memcpy(&c, m + o, 4); o += 4 + 4ull * c; edges += c; }
    foff[n] = o;
    if (o != uint64_t(st.st_size)) { std::fprintf(stderr, "size mismatch: parsed %llu, file %lld\n", (unsigned long long)o, (long long)st.st_size); return 1; }
    std::vector<uint32_t> len(n);
    #pragma omp parallel
    { std::vector<uint32_t> tmp;
      #pragma omp for schedule(dynamic, 65536)
      for (int64_t i = 0; i < int64_t(n); ++i) { uint32_t c; std::memcpy(&c, m + foff[i], 4); len[i] = uint32_t(enc(uint32_t(i), reinterpret_cast<const uint32_t*>(m + foff[i] + 4), c, nullptr, tmp)); } }
    uint32_t B = 64;
    for (;; B /= 2) {
        bool ok = true;
        for (uint64_t b = 0; b * B < n && ok; ++b) { uint64_t s = 0; for (uint64_t i = b * B; i < std::min<uint64_t>(n, (b + 1) * B); ++i) s += len[i]; ok = s < 65536; }
        if (ok || B == 1) break;
    }
    std::vector<uint64_t> start(size_t(n) + 1); start[0] = 0; for (uint32_t i = 0; i < n; ++i) start[i + 1] = start[i] + len[i];
    std::vector<uint64_t> base(n / B + 1); std::vector<uint16_t> rel(size_t(n) + 1);
    for (uint64_t b = 0; b < base.size(); ++b) base[b] = start[std::min<uint64_t>(n, b * B)];
    for (uint64_t i = 0; i <= n; ++i) {
        const uint64_t r = start[i] - base[i / B];
        if (r >= 65536) { std::fprintf(stderr, "relative offset overflow at %llu (B=%u)\n", (unsigned long long)i, B); return 1; }
        rel[i] = uint16_t(r);
    }
    std::vector<uint8_t> s(start[n]);
    #pragma omp parallel
    { std::vector<uint32_t> tmp;
      #pragma omp for schedule(dynamic, 65536)
      for (int64_t i = 0; i < int64_t(n); ++i) { uint32_t c; std::memcpy(&c, m + foff[i], 4); enc(uint32_t(i), reinterpret_cast<const uint32_t*>(m + foff[i] + 4), c, s.data() + start[i], tmp); } }
    FILE* f = std::fopen(argv[2], "wb"); const uint32_t h[4] = {n, B, 0x4347534B, 0}; const uint64_t sb = s.size();
    std::fwrite(h, 4, 4, f); std::fwrite(&sb, 8, 1, f); std::fwrite(base.data(), 8, base.size(), f); std::fwrite(rel.data(), 2, rel.size(), f);
    std::fwrite(s.data(), 1, s.size(), f); std::fclose(f);
    std::fprintf(stderr, "build_ksg_compact: max_degree=%u n=%u edges_in=%llu (avg %.2f) B=%u stream=%.3f GB (%.2f B/edge) offsets=%.3f GB, total %.3f GB vs graph %.3f GB + offsets %.3f GB\n",
                 g_k, n, (unsigned long long)edges, double(edges) / n, B, sb / 1e9, double(sb) / edges, (base.size() * 8.0 + rel.size() * 2.0) / 1e9,
                 (sb + base.size() * 8.0 + rel.size() * 2.0) / 1e9, st.st_size / 1e9, (n + 1) * 8.0 / 1e9);
    return 0;
}
