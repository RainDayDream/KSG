#pragma once
// Product-quantised sidecar (heap-resident) as a smaller replacement for the SQ codes in the KSG search (low-memory variant):
// M sub-quantisers of 256 centroids over d/M dimensions each, one byte per sub-quantiser (M bytes / vector), records in layout order.
// File: [u32 n][u32 d][u32 M][u32 magic 'PQC1'][f32 bias] centroids f32[M][256][d/M], then n records of M codes.
// distance(q, id) = sum_m LUT_m[c_m] - bias, LUT_m[k] = ||q_m - centroid_{m,k}||^2 computed once per query (prepare, per thread);
// bias = mean squared quantisation error (from the build sample), so the estimate is unbiased like the affine SQ.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ksg {

class PQC {
public:
    static constexpr uint32_t kMagic = 0x31435150;   // "PQC1"
    explicit PQC(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open pqc file: " + path);
        uint32_t h[4]; float b;
        if (std::fread(h, 4, 4, f) != 4 || std::fread(&b, 4, 1, f) != 1 || h[3] != kMagic) throw std::runtime_error("bad pqc header");
        n_ = h[0]; d_ = h[1]; M_ = h[2]; bias_ = b; ds_ = d_ / M_;
        if (M_ == 0 || d_ % M_ || M_ % 16) throw std::runtime_error("pqc: M must divide d and be a multiple of 16");
        cent_.resize(size_t(M_) * 256 * ds_); codes_.resize(size_t(n_) * M_);   // heap: memory is explicit (counted as anonymous)
        if (std::fread(cent_.data(), 4, cent_.size(), f) != cent_.size()) throw std::runtime_error("pqc: short centroids");
        const size_t CH = size_t(1) << 26;
        for (size_t o = 0; o < codes_.size(); o += CH) {
            const size_t c = std::min(CH, codes_.size() - o);
            if (std::fread(codes_.data() + o, 1, c, f) != c) throw std::runtime_error("pqc: short codes");
        }
        std::fclose(f);
    }
    uint32_t size() const { return n_; }
    uint32_t dim() const { return d_; }
    uint32_t bytes_per_vector() const { return M_; }
    // per-query lookup table (thread-local; call before any distance() for this query on this thread)
    template <class T> void prepare(const T* q) const {
        lut().resize(size_t(M_) * 256);
        for (uint32_t m = 0; m < M_; ++m)
            for (uint32_t k = 0; k < 256; ++k) {
                const float* c = &cent_[(size_t(m) * 256 + k) * ds_]; float s = 0;
                for (uint32_t j = 0; j < ds_; ++j) { const float t = float(q[m * ds_ + j]) - c[j]; s += t * t; }
                lut()[size_t(m) * 256 + k] = s;
            }
    }
    inline void prefetch(uint64_t id) const { __builtin_prefetch(codes_.data() + id * M_, 0, 1); }
    template <class T> inline float distance(const T*, uint64_t id) const {
        const uint8_t* c = codes_.data() + id * M_; const float* L = lut().data();
        __m512 acc = _mm512_setzero_ps();
        const __m512i base = _mm512_mullo_epi32(_mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), _mm512_set1_epi32(256));
        for (uint32_t m = 0; m < M_; m += 16) {
            const __m512i idx = _mm512_add_epi32(_mm512_add_epi32(base, _mm512_set1_epi32(int(m) * 256)),
                                                 _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c + m))));
            acc = _mm512_add_ps(acc, _mm512_i32gather_ps(idx, L, 4));
        }
        const float s = _mm512_reduce_add_ps(acc) - bias_;
        return s < 0.f ? 0.f : s;
    }
private:
    static std::vector<float>& lut() { thread_local std::vector<float> t; return t; }
    uint32_t n_ = 0, d_ = 0, M_ = 0, ds_ = 0; float bias_ = 0;
    std::vector<float> cent_; std::vector<uint8_t> codes_;
};

}  // namespace ksg
