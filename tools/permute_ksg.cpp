// permute_ksg: renumber a variable-length KSG graph (and an entry index) into the page-locality id order, so that the whole
// in-memory side (KSG, SQ sidecar, entry representatives) works in layout ids and no id maps are needed at search time.
//   new KSG: node i = old node new2old[i], neighbour ids mapped through old2new.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o permute_ksg tools/permute_ksg.cpp
// Usage: permute_ksg ksg_graph.graph perm_prefix out.graph [entry_in entry_out]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
static const uint8_t* map_ro(const char* p, size_t& n) {
    int fd = ::open(p, O_RDONLY); if (fd < 0) { std::perror(p); std::exit(1); }
    struct stat st; fstat(fd, &st); n = st.st_size; void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { std::perror("mmap"); std::exit(1); } return static_cast<const uint8_t*>(m);
}
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s ksg_graph.graph perm_prefix out.graph [entry_in entry_out]\n", argv[0]); return 2; }
    size_t pb, a, b; const uint8_t* ksg_graph = map_ro(argv[1], pb);
    const uint32_t* n2o = reinterpret_cast<const uint32_t*>(map_ro((std::string(argv[2]) + ".new2old").c_str(), a));
    const uint32_t* o2n = reinterpret_cast<const uint32_t*>(map_ro((std::string(argv[2]) + ".old2new").c_str(), b));
    uint32_t n, md; std::memcpy(&n, ksg_graph, 4); std::memcpy(&md, ksg_graph + 4, 4);
    std::vector<uint64_t> off(size_t(n) + 1); { uint64_t o = 8; for (uint32_t i = 0; i < n; ++i) { off[i] = o; uint32_t c; std::memcpy(&c, ksg_graph + o, 4); o += 4 + 4ull * c; } off[n] = o; }
    FILE* f = std::fopen(argv[3], "wb"); std::fwrite(ksg_graph, 4, 2, f);
    const uint32_t CH = 1 << 20; std::vector<uint32_t> buf; std::vector<uint64_t> pos(CH + 1);
    for (uint64_t s0 = 0; s0 < n; s0 += CH) {
        const uint32_t c = static_cast<uint32_t>(std::min<uint64_t>(CH, n - s0));
        pos[0] = 0; for (uint32_t i = 0; i < c; ++i) { const uint32_t o = n2o[s0 + i]; uint32_t cnt; std::memcpy(&cnt, ksg_graph + off[o], 4); pos[i + 1] = pos[i] + 1 + cnt; }
        buf.resize(pos[c]);
        #pragma omp parallel for schedule(static, 4096)
        for (uint32_t i = 0; i < c; ++i) {
            const uint32_t* r = reinterpret_cast<const uint32_t*>(ksg_graph + off[n2o[s0 + i]]); uint32_t* w = &buf[pos[i]];
            w[0] = r[0]; for (uint32_t x = 0; x < r[0]; ++x) w[1 + x] = o2n[r[1 + x]];
        }
        std::fwrite(buf.data(), 4, buf.size(), f);
    }
    std::fclose(f);
    if (argc > 5) {   // entry index: representatives -> layout ids
        size_t eb; const uint8_t* e = map_ro(argv[4], eb); std::vector<uint8_t> out(e, e + eb);
        uint32_t C, d, r; std::memcpy(&C, e, 4); std::memcpy(&d, e + 4, 4); const size_t ro = 8 + size_t(C) * d * 4; std::memcpy(&r, e + ro, 4);
        uint32_t* reps = reinterpret_cast<uint32_t*>(out.data() + ro + 4);
        for (size_t i = 0; i < size_t(C) * r; ++i) if (reps[i] != UINT32_MAX) reps[i] = o2n[reps[i]];
        FILE* g = std::fopen(argv[5], "wb"); std::fwrite(out.data(), 1, out.size(), g); std::fclose(g);
    }
    std::fprintf(stderr, "permuted KSG written: n=%u (%.2f GB)\n", n, double(off[n]) / 1e9);
    return 0;
}
