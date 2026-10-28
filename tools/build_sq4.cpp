// build_sq4: base.bin (uint8, d=128) -> 4-bit scalar-quantised sidecar (see include/sq4.h). Memory = 64 B / vector.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o build_sq4 tools/build_sq4.cpp ; Usage: build_sq4 base.bin out.sq4
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s base.bin out.sq4 [extra_bits 0|1|2] [new2old|-] [affine_permille]  (total bits = 4 + extra)\n", argv[0]); return 2; }
    // affine_permille > 0 (only with extra_bits 1): per-dimension 5-bit code over [min, permille-th percentile] (see include/sq4.h)
    const int affine_pm = argc > 5 ? std::atoi(argv[5]) : 0;
    const uint32_t extra = argc > 3 ? std::atoi(argv[3]) : 0;
    const uint32_t* n2o = nullptr;   // optional 4th arg: new2old permutation file -> record i encodes base[new2old[i]]
    if (argc > 4 && std::string(argv[4]) != "-") { int pf = ::open(argv[4], O_RDONLY); struct stat ps; fstat(pf, &ps); n2o = static_cast<const uint32_t*>(mmap(nullptr, ps.st_size, PROT_READ, MAP_PRIVATE, pf, 0)); }
    if (extra > 2) { std::fprintf(stderr, "extra bits must be 0, 1 or 2\n"); return 2; }
    const size_t rec = 64 + 16 * extra;
    int fd = ::open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t* m = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    uint32_t n, d; std::memcpy(&n, m, 4); std::memcpy(&d, m + 4, 4);
    if (d != 128) { std::fprintf(stderr, "dim must be 128\n"); return 1; }
    if (affine_pm > 0 && extra != 1) { std::fprintf(stderr, "affine needs extra_bits 1\n"); return 2; }
    float lo[128], stp[128];
    if (affine_pm > 0) for (uint32_t j = 0; j < d; ++j) {   // same rule as tools/quant_eval.cpp A5p<permille>
        uint64_t hst[256] = {0}, tot = 0;
        for (size_t i = 0; i < n; i += 7) { hst[m[8 + i * 128 + j]]++; tot++; }
        int l = 0; while (l < 255 && hst[l] == 0) ++l;
        int hi = 255; uint64_t acc = 0; for (int v = 0; v < 256; ++v) { acc += hst[v]; if (acc * 1000 >= tot * uint64_t(affine_pm)) { hi = v; break; } }
        lo[j] = float(l); stp[j] = std::max(1.0f, float(hi - l + 1) / 32);
    }
    FILE* f = std::fopen(argv[2], "wb"); uint32_t h[3] = {n, d, extra | (affine_pm > 0 ? 0x100u : 0u)}; std::fwrite(h, 4, extra ? 3 : 2, f);
    if (affine_pm > 0) { std::fwrite(lo, 4, 128, f); std::fwrite(stp, 4, 128, f); }
    const size_t CH = 1 << 20;
    uint8_t* buf = new uint8_t[CH * rec];
    for (size_t b = 0; b < n; b += CH) {
        const size_t c = std::min<size_t>(CH, n - b);
        #pragma omp parallel for
        for (size_t i = 0; i < c; ++i) {
            const uint8_t* v = m + 8 + size_t(n2o ? n2o[b + i] : (b + i)) * 128;
            uint8_t av[128];
            if (affine_pm > 0) { for (int j = 0; j < 128; ++j) { const int c = std::min(31, std::max(0, int((v[j] - lo[j]) / stp[j]))); av[j] = uint8_t(c << 3); } v = av; }
            uint8_t* o = buf + i * rec;
            for (int j = 0; j < 64; ++j) o[j] = (v[j] >> 4) | ((v[j + 64] >> 4) << 4);
            if (extra == 2) {   // byte k: dims k+32m, bits 2m..2m+1 = the two bits after the top nibble
                for (int k = 0; k < 32; ++k) { uint8_t x = 0; for (int m = 0; m < 4; ++m) x |= ((v[k + 32 * m] >> 2) & 3) << (2 * m); o[64 + k] = x; }
            } else if (extra == 1) {   // byte k: dims k+16m, bit m = the bit after the top nibble
                for (int k = 0; k < 16; ++k) { uint8_t x = 0; for (int m = 0; m < 8; ++m) x |= ((v[k + 16 * m] >> 3) & 1) << m; o[64 + k] = x; }
            }
        }
        std::fwrite(buf, rec, c, f);
    }
    std::fclose(f); return 0;
}
