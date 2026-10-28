// build_colocated: fixed-degree SSG file + base.bin -> co-located graph (vector + neighbours per record, see include/colo_graph.h).
// uint8 base: vector stored as is (d bytes); float32 base (any d): vector stored as fp16 (2d bytes), header field vbytes = 2.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o build_colocated tools/build_colocated.cpp
// Usage: build_colocated base.bin graph.ssg_graph out.colo [perm_prefix]   (perm: record order new2old, neighbour ids rewritten via old2new)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <vector>
#include <omp.h>
#include <immintrin.h>
static const uint8_t* map_ro(const char* path, size_t& bytes) {
    int fd = ::open(path, O_RDONLY); if (fd < 0) { std::perror(path); std::exit(1); }
    struct stat st; fstat(fd, &st); bytes = st.st_size;
    void* m = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0); if (m == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    return static_cast<const uint8_t*>(m);
}
int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) { std::fprintf(stderr, "usage: %s base.bin graph.ssg_graph out.colo [perm_prefix]\n", argv[0]); return 2; }
    size_t bb, gb; const uint8_t* base = map_ro(argv[1], bb); const uint8_t* g = map_ro(argv[2], gb);
    uint32_t n, d, gn, R; std::memcpy(&n, base, 4); std::memcpy(&d, base + 4, 4); std::memcpy(&gn, g, 4); std::memcpy(&R, g + 4, 4);
    if (n != gn) { std::fprintf(stderr, "node count mismatch\n"); return 1; }
    const uint32_t* n2o = nullptr; const uint32_t* o2n = nullptr;
    if (argc == 5) { size_t a, b; n2o = reinterpret_cast<const uint32_t*>(map_ro((std::string(argv[4]) + ".new2old").c_str(), a)); o2n = reinterpret_cast<const uint32_t*>(map_ro((std::string(argv[4]) + ".old2new").c_str(), b)); }
    const bool store_orig = n2o != nullptr && std::getenv("COLO_NO_ORIG") == nullptr;   // permuted layouts carry the original id
    const bool f32 = bb == 8 + size_t(n) * d * 4;   // float32 base -> fp16 records
    const uint32_t vb = f32 ? 2 : 1, vbytes = d * vb;
    const uint32_t rec_len = vbytes + 4 + 4 * R + (store_orig ? 4 : 0), per = 4096 / rec_len;
    if (per == 0) { std::fprintf(stderr, "record does not fit a sector\n"); return 1; }
    const uint64_t sectors = (uint64_t(n) + per - 1) / per;
    FILE* f = std::fopen(argv[3], "wb");
    std::vector<uint8_t> hdr(4096, 0); uint32_t h[6] = {n, R, rec_len, per, d, vb}; std::memcpy(hdr.data(), h, sizeof(h)); std::fwrite(hdr.data(), 1, 4096, f);
    const uint64_t CH = 1 << 14;   // sectors per chunk
    std::vector<uint8_t> buf(CH * 4096);
    for (uint64_t s0 = 0; s0 < sectors; s0 += CH) {
        const uint64_t c = std::min<uint64_t>(CH, sectors - s0);
        std::memset(buf.data(), 0, c * 4096);
        #pragma omp parallel for
        for (uint64_t s = 0; s < c; ++s)
            for (uint32_t k = 0; k < per; ++k) {
                const uint64_t id = (s0 + s) * per + k; if (id >= n) break;
                const uint64_t src = n2o ? n2o[id] : id;   // record position id holds original node src
                uint8_t* r = buf.data() + s * 4096 + k * rec_len;
                if (f32) { const float* fv = reinterpret_cast<const float*>(base + 8) + src * d; uint16_t* hv = reinterpret_cast<uint16_t*>(r);
                           for (uint32_t j = 0; j < d; ++j) hv[j] = _cvtss_sh(fv[j], 0); }
                else std::memcpy(r, base + 8 + src * d, d);
                const uint8_t* gr = g + 8 + src * (4ull * (R + 1));
                std::memcpy(r + vbytes, gr, 4 * (R + 1));   // count + ids (padded to R)
                if (store_orig) { const uint32_t o = static_cast<uint32_t>(src); std::memcpy(r + vbytes + 4 + 4ull * R, &o, 4); }
                if (o2n) { uint32_t* ids = reinterpret_cast<uint32_t*>(r + vbytes + 4); const uint32_t c = std::min(ids[-1], R); for (uint32_t x = 0; x < c; ++x) ids[x] = o2n[ids[x]]; }
            }
        std::fwrite(buf.data(), 4096, c, f);
    }
    std::fclose(f);
    std::fprintf(stderr, "n=%u R=%u rec_len=%u per_sector=%u sectors=%llu size=%.2f GB\n", n, R, rec_len, per, (unsigned long long)sectors, (sectors + 1) * 4096.0 / 1e9);
    return 0;
}
