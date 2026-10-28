// build_entry: in-memory entry index for ONLINE starters: k-means centroids over a uniform sample + the r sample points nearest to
// every centroid (their global ids). Per query the search computes distances to all C centroids (inside the timing window) and uses
// the representatives of the nearest centroids as starters. Memory: C*d*4 + C*r*4 bytes (C=16384, d=128, r=4: 8.6 MB).
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o build_entry tools/build_entry.cpp
// Usage: build_entry base.bin out.entry [C=16384] [sample=2000000] [r=4] [iters=8]   (uint8 or float32 base, detected from the file size)
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <random>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s base.bin out.entry [C] [sample] [r] [iters]\n", argv[0]); return 2; }
    const uint32_t C = argc > 3 ? std::atoi(argv[3]) : 16384, S = argc > 4 ? std::atoi(argv[4]) : 2000000, R = argc > 5 ? std::atoi(argv[5]) : 4, IT = argc > 6 ? std::atoi(argv[6]) : 8;
    int fd = ::open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t* m = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    uint32_t n, d; std::memcpy(&n, m, 4); std::memcpy(&d, m + 4, 4); const uint8_t* B = m + 8;
    std::mt19937_64 rng(42); std::vector<uint32_t> sid(S); for (auto& x : sid) x = static_cast<uint32_t>(rng() % n);
    std::sort(sid.begin(), sid.end()); sid.erase(std::unique(sid.begin(), sid.end()), sid.end());
    const size_t ns = sid.size();
    std::vector<float> X(ns * d);
    const bool f32 = size_t(st.st_size) == 8 + size_t(n) * d * 4; const float* BF = reinterpret_cast<const float*>(B);
    #pragma omp parallel for
    for (size_t i = 0; i < ns; ++i) for (uint32_t j = 0; j < d; ++j) X[i * d + j] = f32 ? BF[size_t(sid[i]) * d + j] : float(B[size_t(sid[i]) * d + j]);
    std::vector<float> Cn(size_t(C) * d), cn(C); std::vector<uint32_t> asg(ns);
    for (uint32_t c = 0; c < C; ++c) std::memcpy(&Cn[size_t(c) * d], &X[(rng() % ns) * d], d * 4);
    for (uint32_t it = 0; it <= IT; ++it) {
        for (uint32_t c = 0; c < C; ++c) { float s = 0; for (uint32_t j = 0; j < d; ++j) s += Cn[size_t(c) * d + j] * Cn[size_t(c) * d + j]; cn[c] = s; }
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = 0; i < ns; ++i) {
            const float* x = &X[i * d]; float best = std::numeric_limits<float>::max(); uint32_t bc = 0;
            for (uint32_t c = 0; c < C; ++c) { const float* v = &Cn[size_t(c) * d]; float dot = 0; for (uint32_t j = 0; j < d; ++j) dot += x[j] * v[j]; const float s = cn[c] - 2 * dot; if (s < best) { best = s; bc = c; } }
            asg[i] = bc;
        }
        if (it == IT) break;
        std::vector<double> sum(size_t(C) * d, 0.0); std::vector<uint32_t> cnt(C, 0);
        for (size_t i = 0; i < ns; ++i) { cnt[asg[i]]++; for (uint32_t j = 0; j < d; ++j) sum[size_t(asg[i]) * d + j] += X[i * d + j]; }
        uint32_t empty = 0;
        for (uint32_t c = 0; c < C; ++c) { if (!cnt[c]) { ++empty; std::memcpy(&Cn[size_t(c) * d], &X[(rng() % ns) * d], d * 4); continue; } for (uint32_t j = 0; j < d; ++j) Cn[size_t(c) * d + j] = float(sum[size_t(c) * d + j] / cnt[c]); }
        std::fprintf(stderr, "k-means iteration %u done (%u empty clusters re-seeded)\n", it + 1, empty);
    }
    // representatives: the R sample points nearest to each centroid (among its members)
    std::vector<std::vector<std::pair<float, uint32_t>>> mem(C);
    for (size_t i = 0; i < ns; ++i) {
        const uint32_t c = asg[i]; float s = 0; for (uint32_t j = 0; j < d; ++j) { const float t = X[i * d + j] - Cn[size_t(c) * d + j]; s += t * t; }
        mem[c].emplace_back(s, sid[i]);
    }
    std::vector<uint32_t> reps(size_t(C) * R, UINT32_MAX); uint32_t filled = 0;
    for (uint32_t c = 0; c < C; ++c) { auto& v = mem[c]; std::sort(v.begin(), v.end()); for (uint32_t k = 0; k < R && k < v.size(); ++k) { reps[size_t(c) * R + k] = v[k].second; ++filled; } }
    FILE* f = std::fopen(argv[2], "wb");
    std::fwrite(&C, 4, 1, f); std::fwrite(&d, 4, 1, f); std::fwrite(Cn.data(), 4, Cn.size(), f); std::fwrite(&R, 4, 1, f); std::fwrite(reps.data(), 4, reps.size(), f); std::fclose(f);
    std::fprintf(stderr, "entry index: C=%u d=%u sample=%zu r=%u, %u representatives filled\n", C, d, ns, R, filled);
    return 0;
}
