// indegree_count: global in-degree of every node of a fixed-degree graph file (streaming, parallel) -> n x u32 file.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o indegree_count tools/indegree_count.cpp
// Usage: indegree_count graph.ssg_graph out.indeg [cap]
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s graph.ssg_graph out.indeg [cap]\n", argv[0]); return 2; }
    int fd = ::open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t* g = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0));
    uint32_t n, R; std::memcpy(&n, g, 4); std::memcpy(&R, g + 4, 4);
    const uint32_t cap = argc > 3 ? std::atoi(argv[3]) : R;
    int ofd = ::open(argv[2], O_RDWR | O_CREAT | O_TRUNC, 0644); const size_t ob = size_t(n) * 4;
    if (ftruncate(ofd, ob) != 0) { std::perror("ftruncate"); return 1; }
    std::atomic<uint32_t>* ind = static_cast<std::atomic<uint32_t>*>(mmap(nullptr, ob, PROT_READ | PROT_WRITE, MAP_SHARED, ofd, 0));
    #pragma omp parallel for schedule(static, 65536)
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t* r = reinterpret_cast<const uint32_t*>(g + 8 + size_t(i) * 4ull * (R + 1));
        const uint32_t c = std::min(r[0], cap);
        for (uint32_t x = 0; x < c; ++x) if (r[1 + x] < n) ind[r[1 + x]].fetch_add(1, std::memory_order_relaxed);
    }
    msync(ind, ob, MS_SYNC);
    uint64_t zero = 0, lt48 = 0, edges = 0; uint32_t mx = 0;
    for (uint32_t i = 0; i < n; ++i) { const uint32_t v = ind[i].load(std::memory_order_relaxed); zero += v == 0; lt48 += v < 48; edges += v; mx = std::max(mx, v); }
    std::fprintf(stderr, "n=%u edges=%llu mean=%.2f max=%u zero=%.4f%% <48=%.3f%%\n", n, (unsigned long long)edges, double(edges) / n, mx, 100.0 * zero / n, 100.0 * lt48 / n);
    return 0;
}
