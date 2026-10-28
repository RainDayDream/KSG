// shard_split: partition a uint8 base into P shards for the sharded SSG build.
//   primary shard = nearest k-means centre (sample-trained); a node is also a member of shard X ("halo") when one of its KSG
//   neighbours has primary shard X. Writes labels.u8 (primary label per node) and, per shard k, shard<k>.ids ([u32 m] + sorted global ids
//   of all members) and shard<k>.prim (m bytes, 1 = primary member).
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o shard_split tools/shard_split.cpp
// Usage: shard_split base.bin ksg_graph.graph P out_dir
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
static const uint8_t* map_ro(const char* p, size_t& n) {
    int fd = ::open(p, O_RDONLY); if (fd < 0) { std::perror(p); std::exit(1); }
    struct stat st; fstat(fd, &st); n = st.st_size; void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { std::perror("mmap"); std::exit(1); } madvise(m, n, MADV_RANDOM); return static_cast<const uint8_t*>(m);
}
int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: shard_split base.bin ksg_graph.graph P out_dir\n"); return 2; }
    size_t bb, pb; const uint8_t* base = map_ro(argv[1], bb); const uint8_t* ksg_graph = map_ro(argv[2], pb);
    const uint32_t P = std::atoi(argv[3]); const std::string out = argv[4];
    uint32_t n, d; std::memcpy(&n, base, 4); std::memcpy(&d, base + 4, 4); const uint8_t* B = base + 8;
    std::vector<uint64_t> off(size_t(n) + 1); { uint64_t o = 8; for (uint32_t i = 0; i < n; ++i) { off[i] = o; uint32_t c; std::memcpy(&c, ksg_graph + o, 4); o += 4 + 4ull * c; } off[n] = o; }
    auto nbrs = [&](uint32_t i, const uint32_t*& p, uint32_t& c) { const uint32_t* r = reinterpret_cast<const uint32_t*>(ksg_graph + off[i]); c = r[0]; p = r + 1; };
    std::mt19937_64 rng(7); const uint32_t NSMP = 2000000; std::vector<float> C(size_t(P) * d);
    for (uint32_t k = 0; k < P; ++k) { const uint64_t id = rng() % n; for (uint32_t j = 0; j < d; ++j) C[k * d + j] = B[id * d + j]; }
    for (int it = 0; it < 10; ++it) {
        std::vector<double> sum(size_t(P) * d, 0.0); std::vector<uint64_t> cnt(P, 0);
        #pragma omp parallel
        { std::vector<double> ls(size_t(P) * d, 0.0); std::vector<uint64_t> lc(P, 0);
          #pragma omp for schedule(static)
          for (uint32_t s = 0; s < NSMP; ++s) { const uint64_t id = uint64_t(s) * (n / NSMP); const uint8_t* v = B + id * d; uint32_t best = 0; float bd = 1e30f;
            for (uint32_t k = 0; k < P; ++k) { float dd = 0; for (uint32_t j = 0; j < d; ++j) { float x = v[j] - C[k * d + j]; dd += x * x; } if (dd < bd) { bd = dd; best = k; } }
            lc[best]++; for (uint32_t j = 0; j < d; ++j) ls[best * d + j] += v[j]; }
          #pragma omp critical
          { for (uint32_t k = 0; k < P; ++k) { cnt[k] += lc[k]; for (uint32_t j = 0; j < d; ++j) sum[k * d + j] += ls[k * d + j]; } } }
        for (uint32_t k = 0; k < P; ++k) if (cnt[k]) for (uint32_t j = 0; j < d; ++j) C[k * d + j] = float(sum[k * d + j] / cnt[k]);
    }
    std::vector<uint8_t> lab(n);
    #pragma omp parallel for schedule(static, 4096)
    for (uint32_t i = 0; i < n; ++i) { const uint8_t* v = B + uint64_t(i) * d; float b1 = 1e30f; uint32_t k1 = 0;
        for (uint32_t k = 0; k < P; ++k) { float dd = 0; for (uint32_t j = 0; j < d; ++j) { float x = v[j] - C[k * d + j]; dd += x * x; } if (dd < b1) { b1 = dd; k1 = k; } }
        lab[i] = k1; }
    std::vector<uint64_t> mask(n);
    #pragma omp parallel for schedule(static, 65536)
    for (uint32_t i = 0; i < n; ++i) { const uint32_t* p; uint32_t c; nbrs(i, p, c); uint64_t mk = 1ull << lab[i]; for (uint32_t x = 0; x < c; ++x) mk |= 1ull << lab[p[x]]; mask[i] = mk; }
    { FILE* f = std::fopen((out + "/labels.u8").c_str(), "wb"); std::fwrite(lab.data(), 1, n, f); std::fclose(f); }
    for (uint32_t k = 0; k < P; ++k) {
        std::vector<uint32_t> ids; std::vector<uint8_t> prim;
        for (uint32_t i = 0; i < n; ++i) if ((mask[i] >> k) & 1) { ids.push_back(i); prim.push_back(lab[i] == k); }
        uint32_t m = ids.size(); uint64_t np = 0; for (uint8_t x : prim) np += x;
        FILE* f = std::fopen((out + "/shard" + std::to_string(k) + ".ids").c_str(), "wb"); std::fwrite(&m, 4, 1, f); std::fwrite(ids.data(), 4, m, f); std::fclose(f);
        f = std::fopen((out + "/shard" + std::to_string(k) + ".prim").c_str(), "wb"); std::fwrite(prim.data(), 1, m, f); std::fclose(f);
        std::printf("shard %u: members %u (primary %llu, halo %llu)\n", k, m, (unsigned long long)np, (unsigned long long)(m - np));
    }
    return 0;
}
