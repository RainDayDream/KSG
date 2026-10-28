// build_ssg_shard: build the SSG neighbour lists of ONE shard (members = primary + halo) and write the lists of its primary nodes
// into the global fixed-degree graph file (records at their global ids; global neighbour ids).
// Steps: gather member vectors -> initial graph = old global SSG restricted to the shard -> one search-based round (Vamana-style) ->
//        write primary records. Run once per shard; the output file is created on first use.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o build_ssg_shard tools/build_ssg_shard.cpp
// Usage: build_ssg_shard --base b.bin --old old.ssg_graph|random --ids shardK.ids --prim shardK.prim --out global.ssg_graph [--R 96] [--Lc 64] [--C 512] [--alpha 1.2] [--rounds n]
//   --old random: no previous graph; start from a random R=32 graph inside the shard and refine for --rounds rounds (default 2).
#include "../include/ssg_search_core.h"

int main(int argc, char** argv) {
    std::string base, old, idsf, primf, out; Opt o; o.rounds = 1; o.backlink = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--base") base = v; else if (a == "--old") old = v; else if (a == "--ids") idsf = v; else if (a == "--prim") primf = v; else if (a == "--out") out = v;
        else if (a == "--R") o.R = std::stoul(v); else if (a == "--Lc") o.Lc = std::stoul(v); else if (a == "--C") o.C = std::stoul(v);
        else if (a == "--alpha") o.alpha = std::stof(v); else if (a == "--rounds") o.rounds = std::stoul(v); else if (a == "--entries") o.entries = std::stoul(v); else if (a == "--rev") o.rev = std::stoul(v);
    }
    auto t0 = std::chrono::steady_clock::now();
    const bool rnd = old == "random";
    size_t bb, ob = 0, ib, pb; const uint8_t* bm = map_ro(base, bb); const uint8_t* om = rnd ? nullptr : map_ro(old, ob);
    const uint8_t* im = map_ro(idsf, ib); const uint8_t* pm = map_ro(primf, pb);
    uint32_t nG, dim, oN, oR, m; std::memcpy(&nG, bm, 4); std::memcpy(&dim, bm + 4, 4); if (rnd) { oN = nG; oR = 32; if (o.rounds < 2) o.rounds = 2; } else { std::memcpy(&oN, om, 4); std::memcpy(&oR, om + 4, 4); } std::memcpy(&m, im, 4);
    const uint32_t* ids = reinterpret_cast<const uint32_t*>(im + 4);
    if (dim != 128 || oN != nG) { std::fprintf(stderr, "bad inputs\n"); return 1; }
    std::vector<uint8_t> buf(size_t(m) * dim);
    #pragma omp parallel for schedule(static, 4096)
    for (uint32_t i = 0; i < m; ++i) std::memcpy(&buf[size_t(i) * dim], bm + 8 + size_t(ids[i]) * dim, dim);
    Data D(std::move(buf), m);
    std::vector<uint32_t> g2l(nG, UINT32_MAX);
    #pragma omp parallel for schedule(static, 65536)
    for (uint32_t i = 0; i < m; ++i) g2l[ids[i]] = i;
    Graph G(m, oR);   // initial graph: old SSG edges that stay inside the shard (or random edges)
    uint64_t kept = 0, total = 0;
    if (rnd) {
        #pragma omp parallel for schedule(static, 4096)
        for (uint32_t i = 0; i < m; ++i) {
            uint64_t x = 0x9E3779B97F4A7C15ull * (uint64_t(ids[i]) + 7); uint32_t* rec = G.rec(i); rec[0] = oR;
            for (uint32_t k = 0; k < oR; ++k) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; rec[1 + k] = uint32_t(x % m); }
        }
        kept = total = uint64_t(m) * oR;
    } else {
    #pragma omp parallel for schedule(static, 4096) reduction(+:kept,total)
    for (uint32_t i = 0; i < m; ++i) {
        const uint32_t* r = reinterpret_cast<const uint32_t*>(om + 8 + size_t(ids[i]) * 4ull * (oR + 1));
        uint32_t c = std::min(r[0], oR), w = 0; uint32_t* rec = G.rec(i);
        for (uint32_t x = 0; x < c; ++x) { const uint32_t l = g2l[r[1 + x]]; ++total; if (l != UINT32_MAX) { rec[1 + w++] = l; ++kept; } }
        rec[0] = w;
    }
    }
    std::fprintf(stderr, "shard: %u members ; initial edges kept inside the shard: %.1f%%\n", m, 100.0 * kept / std::max<uint64_t>(1, total));
    Graph N = std::move(G);
    for (uint32_t r = 0; r < std::max<uint32_t>(1, o.rounds); ++r) { Graph rev = reverse_graph(N, o.rev); N = one_round(D, N, o, &rev); }
    // write the primary records into the global file
    const size_t rec_bytes = 4ull * (o.R + 1), file_bytes = 8 + size_t(nG) * rec_bytes;
    int fd = ::open(out.c_str(), O_RDWR | O_CREAT, 0644); if (fd < 0) { std::perror("open out"); return 1; }
    struct stat st; fstat(fd, &st);
    if (static_cast<size_t>(st.st_size) != file_bytes) {
        if (ftruncate(fd, file_bytes) != 0) { std::perror("ftruncate"); return 1; }
        uint32_t h[2] = {nG, o.R}; if (pwrite(fd, h, 8, 0) != 8) { std::perror("pwrite"); return 1; }
    }
    uint8_t* om2 = static_cast<uint8_t*>(mmap(nullptr, file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (om2 == MAP_FAILED) { std::perror("mmap out"); return 1; }
    uint64_t written = 0;
    #pragma omp parallel for schedule(static, 4096) reduction(+:written)
    for (uint32_t i = 0; i < m; ++i) {
        if (!pm[i]) continue;
        uint32_t* rec = reinterpret_cast<uint32_t*>(om2 + 8 + size_t(ids[i]) * rec_bytes);
        const uint32_t c = N.cnt(i); rec[0] = c;
        for (uint32_t x = 0; x < c; ++x) rec[1 + x] = ids[N.nb(i)[x]];
        for (uint32_t x = c; x < o.R; ++x) rec[1 + x] = 0;
        ++written;
    }
    msync(om2, file_bytes, MS_SYNC); munmap(om2, file_bytes); ::close(fd);
    std::fprintf(stderr, "wrote %llu primary records in %.1f s\n", (unsigned long long)written, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}
