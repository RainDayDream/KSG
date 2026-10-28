#include <atomic>
// boost_shard: in-degree boost for the PRIMARY nodes of one shard, working directly on the global graph file (shared mmap) and a
// global in-degree array (shared mmap, atomic updates). Sources are restricted to primaries of the same shard, so shards can be
// processed concurrently (each process only modifies the records of its own primaries).
// For every primary p with in-degree < T: p's nearest out-neighbours s (same shard) get p written into a tail slot (>= protect)
// that currently points to a node whose in-degree is still > donor.
// Parallel within a shard (OMP threads; striped locks on source records).
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o boost_shard tools/boost_shard.cpp
// Usage: boost_shard --base b.bin --graph g.ssg_graph --indeg g.indeg --ids shardK.ids --prim shardK.prim [--T 80] [--donor 88] [--src 48] [--protect 16]
#include "../include/ssg_search_core.h"

int main(int argc, char** argv) {
    std::string base, graph, indeg, idsf, primf; uint32_t T = 80, donor = 88, nsrc = 48, protect = 16;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--base") base = v; else if (a == "--graph") graph = v; else if (a == "--indeg") indeg = v; else if (a == "--ids") idsf = v; else if (a == "--prim") primf = v;
        else if (a == "--T") T = std::stoul(v); else if (a == "--donor") donor = std::stoul(v); else if (a == "--src") nsrc = std::stoul(v); else if (a == "--protect") protect = std::stoul(v);
    }
    auto t0 = std::chrono::steady_clock::now();
    size_t bb, ib, pb; const uint8_t* bm = map_ro(base, bb); const uint8_t* im = map_ro(idsf, ib); const uint8_t* pm = map_ro(primf, pb);
    uint32_t nG, dim, m; std::memcpy(&nG, bm, 4); std::memcpy(&dim, bm + 4, 4); std::memcpy(&m, im, 4);
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(im + 4);
    std::vector<uint32_t> prim; prim.reserve(m);   // global ids of the shard's primaries (ascending)
    for (uint32_t i = 0; i < m; ++i) if (pm[i]) prim.push_back(ids[i]);
    const uint32_t mp = prim.size();
    std::vector<uint8_t> buf(size_t(mp) * dim);
    #pragma omp parallel for schedule(static, 4096)
    for (uint32_t i = 0; i < mp; ++i) std::memcpy(&buf[size_t(i) * dim], bm + 8 + size_t(prim[i]) * dim, dim);
    Data D(std::move(buf), mp);
    std::vector<uint32_t> g2l(nG, UINT32_MAX);
    #pragma omp parallel for schedule(static, 65536)
    for (uint32_t i = 0; i < mp; ++i) g2l[prim[i]] = i;
    // shared, writable mappings
    int gfd = ::open(graph.c_str(), O_RDWR); struct stat gst; fstat(gfd, &gst);
    uint8_t* gm = static_cast<uint8_t*>(mmap(nullptr, gst.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, gfd, 0));
    uint32_t gn, R; std::memcpy(&gn, gm, 4); std::memcpy(&R, gm + 4, 4);
    const size_t rec_bytes = 4ull * (R + 1);
    int dfd = ::open(indeg.c_str(), O_RDWR);
    std::atomic<uint32_t>* ind = static_cast<std::atomic<uint32_t>*>(mmap(nullptr, size_t(nG) * 4, PROT_READ | PROT_WRITE, MAP_SHARED, dfd, 0));
    auto rec = [&](uint32_t gid) { return reinterpret_cast<uint32_t*>(gm + 8 + size_t(gid) * rec_bytes); };
    uint64_t low = 0, added = 0, failed = 0;
    // Load-modify-store: copy the records of the shard's primaries into memory (sources are primaries of this shard),
    // edit them there under striped locks (in-degrees stay global atomics), then write back only the modified records.
    madvise(gm, gst.st_size, MADV_RANDOM);
    const size_t RW = R + 1;
    std::vector<uint32_t> recs(size_t(mp) * RW);
    std::vector<uint8_t> dirty(mp, 0);
    #pragma omp parallel for schedule(dynamic, 4096)
    for (uint32_t li = 0; li < mp; ++li) std::memcpy(&recs[size_t(li) * RW], rec(prim[li]), rec_bytes);
    const double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    constexpr size_t NL = size_t(1) << 20;
    std::vector<std::atomic_flag> locks(NL);
    for (auto& l : locks) l.clear();
    #pragma omp parallel reduction(+:low,added,failed)
    {
        std::vector<std::pair<int32_t, uint32_t>> src;   // (dist, local id)
        #pragma omp for schedule(dynamic, 1024)
        for (uint32_t li = 0; li < mp; ++li) {
            const uint32_t p = prim[li];
            if (ind[p].load(std::memory_order_relaxed) >= T) continue;
            ++low;
            src.clear();
            const uint32_t* pr = &recs[size_t(li) * RW]; const uint32_t pc = std::min(pr[0], nsrc);
            for (uint32_t x = 0; x < pc; ++x) { const uint32_t sx = pr[1 + x]; const uint32_t ls = sx < nG ? g2l[sx] : UINT32_MAX; if (ls != UINT32_MAX) src.emplace_back(D.dist(li, ls), ls); }
            std::sort(src.begin(), src.end());
            for (auto& sp : src) {
                if (ind[p].load(std::memory_order_relaxed) >= T) break;
                const uint32_t ls = sp.second;
                std::atomic_flag& lk = locks[(size_t(ls) * 0x9E3779B97F4A7C15ull) >> 44];
                while (lk.test_and_set(std::memory_order_acquire)) {}
                uint32_t* sr = &recs[size_t(ls) * RW]; const uint32_t c = sr[0];
                bool has = false; for (uint32_t x = 0; x < c; ++x) if (sr[1 + x] == p) { has = true; break; }
                int slot = -1;
                if (!has) for (int x = int(c) - 1; x >= int(protect); --x) if (sr[1 + x] < nG && ind[sr[1 + x]].load(std::memory_order_relaxed) > donor) { slot = x; break; }
                if (slot >= 0) { ind[sr[1 + slot]].fetch_sub(1, std::memory_order_relaxed); sr[1 + slot] = p; ind[p].fetch_add(1, std::memory_order_relaxed); dirty[ls] = 1; ++added; }
                lk.clear(std::memory_order_release);
            }
            if (ind[p].load(std::memory_order_relaxed) < T) ++failed;
        }
    }
    uint64_t written = 0;
    #pragma omp parallel for schedule(dynamic, 4096) reduction(+:written)
    for (uint32_t li = 0; li < mp; ++li) if (dirty[li]) { std::memcpy(rec(prim[li]), &recs[size_t(li) * RW], rec_bytes); ++written; }
    std::fprintf(stderr, "boost_shard: load %.1f s, %llu records written back\n", t_load, (unsigned long long)written);
    msync(gm, gst.st_size, MS_ASYNC); msync(ind, size_t(nG) * 4, MS_ASYNC);
    std::fprintf(stderr, "boost_shard: %u primaries, %llu below T, %llu edges replaced, %llu still below T, %.1f s\n", mp, (unsigned long long)low, (unsigned long long)added, (unsigned long long)failed,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
