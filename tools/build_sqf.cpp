// build_sqf: float32 base (any d) -> affine SQ sidecar (include/sqf.h), one byte per dimension (bits <= 4: two per byte), 2^bits levels over
// [min, permille-th percentile] of each dimension (estimated on every 7th vector); records in layout order (new2old) if given.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o build_sqf tools/build_sqf.cpp
// Usage: build_sqf base.bin out.sqf [bits=8] [new2old|-] [permille=999]
#include <algorithm>
#include <cmath>
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
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s base.bin out.sqf [bits=8] [new2old|-] [permille=999]\n", argv[0]); return 2; }
    const uint32_t bits = argc > 3 ? std::atoi(argv[3]) : 8; const int pm = argc > 5 ? std::atoi(argv[5]) : 999;
    if (bits < 1 || bits > 8) { std::fprintf(stderr, "bits must be 1..8\n"); return 2; }
    int fd = ::open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t* m = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    uint32_t n, d; std::memcpy(&n, m, 4); std::memcpy(&d, m + 4, 4);
    if (size_t(st.st_size) != 8 + size_t(n) * d * 4) { std::fprintf(stderr, "not a float32 base\n"); return 1; }
    const float* X = reinterpret_cast<const float*>(m + 8);
    const uint32_t* n2o = nullptr;
    if (argc > 4 && std::string(argv[4]) != "-") { int pf = ::open(argv[4], O_RDONLY); struct stat ps; fstat(pf, &ps); n2o = static_cast<const uint32_t*>(mmap(nullptr, ps.st_size, PROT_READ, MAP_PRIVATE, pf, 0)); }
    std::vector<float> lo(d), step(d); const uint32_t L = 1u << bits;
    #pragma omp parallel for
    for (uint32_t j = 0; j < d; ++j) {
        std::vector<float> v; v.reserve(n / 7 + 1);
        for (size_t i = 0; i < n; i += 7) v.push_back(X[i * d + j]);
        std::sort(v.begin(), v.end());
        const float mn = v.front(), hi = v[std::min(v.size() - 1, size_t(double(v.size()) * pm / 1000.0))];
        lo[j] = mn; step[j] = std::max((hi - mn) / L, 1e-12f);
    }
    FILE* f = std::fopen(argv[2], "wb"); const uint32_t h[4] = {n, d, bits, 0x31465153};
    std::fwrite(h, 4, 4, f); std::fwrite(lo.data(), 4, d, f); std::fwrite(step.data(), 4, d, f);
    const bool packed = bits <= 4; const size_t rec = packed ? (d + 1) / 2 : d;
    const size_t CH = 1 << 16; std::vector<uint8_t> buf(CH * rec); std::vector<uint8_t> tmp;
    for (size_t b = 0; b < n; b += CH) {
        const size_t c = std::min<size_t>(CH, n - b);
        #pragma omp parallel for
        for (size_t i = 0; i < c; ++i) {
            const float* x = X + size_t(n2o ? n2o[b + i] : (b + i)) * d;
            uint8_t* o = &buf[i * rec]; if (packed) std::memset(o, 0, rec);
            for (uint32_t j = 0; j < d; ++j) {
                const int q = int(std::floor((x[j] - lo[j]) / step[j])); const uint8_t v = uint8_t(std::min<int>(L - 1, std::max(0, q)));
                if (packed) o[j / 2] |= uint8_t(v << ((j & 1) * 4)); else o[j] = v;
            }
        }
        std::fwrite(buf.data(), rec, c, f);
    }
    std::fclose(f);
    std::fprintf(stderr, "build_sqf: n=%u d=%u bits=%u permille=%d -> %.2f GB\n", n, d, bits, pm, (16 + 8.0 * d + double(n) * rec) / 1e9);
    return 0;
}
