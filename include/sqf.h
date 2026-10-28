#pragma once
// Affine scalar-quantised sidecar for float vectors of any dimension (e.g. GIST1M, d = 960): one byte per dimension,
// code c_j = clamp(floor((x_j - lo_j) / step_j), 0, 2^bits - 1); reconstruction lo_j + (c_j + 0.5) step_j.
// File: [u32 n][u32 d][u32 bits][u32 magic 'SQF1'] lo[d] f32, step[d] f32, then n records of codes:
//   bits 5..8: one byte per dimension (d bytes); bits <= 4: packed, byte k = dim 2k (low nibble) | dim 2k+1 (high nibble), (d+1)/2 bytes.
// distance(q, id) = sum_j step_j^2 ((q_j - lo_j) / step_j - c_j - 0.5)^2 - sum_j step_j^2 / 12   (same units as exact ||q - x||^2)
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ksg {

class SQF {
public:
    static constexpr uint32_t kMagic = 0x31465153;   // "SQF1"
    explicit SQF(const std::string& path, bool resident) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open sqf file: " + path);
        struct stat st; fstat(fd, &st); bytes_ = static_cast<size_t>(st.st_size);
        map_ = static_cast<uint8_t*>(mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE | (resident ? MAP_POPULATE : 0), fd, 0));
        ::close(fd);
        if (map_ == MAP_FAILED) throw std::runtime_error("mmap sqf failed");
        uint32_t h[4]; std::memcpy(h, map_, 16); n_ = h[0]; d_ = h[1]; bits_ = h[2];
        packed_ = bits_ <= 4; rec_ = packed_ ? (d_ + 1) / 2 : d_;
        if (h[3] != kMagic || bytes_ != 16 + 8ull * d_ + uint64_t(n_) * rec_) throw std::runtime_error("bad sqf file");
        const float* lo = reinterpret_cast<const float*>(map_ + 16); const float* st_ = lo + d_;
        dp_ = (d_ + 15) / 16 * 16;
        a_.assign(dp_, 0.f); b_.assign(dp_, 0.f); w_.assign(dp_, 0.f);
        double bias = 0;
        for (uint32_t j = 0; j < d_; ++j) { a_[j] = 1.f / st_[j]; b_[j] = -lo[j] / st_[j] - 0.5f; w_[j] = st_[j] * st_[j]; bias += st_[j] * double(st_[j]) / 12.0; }
        bias_ = static_cast<float>(bias);
        codes_ = map_ + 16 + 8ull * d_;
    }
    ~SQF() { if (map_ && map_ != MAP_FAILED) munmap(map_, bytes_); }
    uint32_t size() const { return n_; }
    uint32_t dim() const { return d_; }
    uint32_t bytes_per_vector() const { return rec_; }
    inline void prefetch(uint64_t id) const { const uint8_t* c = codes_ + id * rec_; for (uint32_t o = 0; o < rec_; o += 64) __builtin_prefetch(c + o, 0, 1); }
    inline float distance(const float* q, uint64_t id) const {
        if (packed_) return distance_packed(q, codes_ + id * rec_);
        const uint8_t* c = codes_ + id * rec_;
        __m512 acc = _mm512_setzero_ps(); uint32_t j = 0;
        for (; j + 16 <= d_; j += 16) {
            const __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(c + j))));
            const __m512 t = _mm512_sub_ps(_mm512_fmadd_ps(_mm512_loadu_ps(q + j), _mm512_loadu_ps(&a_[j]), _mm512_loadu_ps(&b_[j])), cf);
            acc = _mm512_fmadd_ps(_mm512_mul_ps(t, t), _mm512_loadu_ps(&w_[j]), acc);
        }
        float s = _mm512_reduce_add_ps(acc);
        for (; j < d_; ++j) { const float t = q[j] * a_[j] + b_[j] - float(c[j]); s += w_[j] * t * t; }
        s -= bias_;
        return s < 0.f ? 0.f : s;
    }
private:
    // 4-bit packed: 8 bytes -> 16 dimensions (low nibble = even dim, high nibble = odd dim)
    inline float distance_packed(const float* q, const uint8_t* c) const {
        __m512 acc = _mm512_setzero_ps(); uint32_t j = 0; const __m128i m4 = _mm_set1_epi8(0x0F);
        for (; j + 16 <= d_; j += 16) {
            const __m128i x = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(c + j / 2));
            const __m128i codes = _mm_unpacklo_epi8(_mm_and_si128(x, m4), _mm_and_si128(_mm_srli_epi16(x, 4), m4));
            const __m512 cf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes));
            const __m512 t = _mm512_sub_ps(_mm512_fmadd_ps(_mm512_loadu_ps(q + j), _mm512_loadu_ps(&a_[j]), _mm512_loadu_ps(&b_[j])), cf);
            acc = _mm512_fmadd_ps(_mm512_mul_ps(t, t), _mm512_loadu_ps(&w_[j]), acc);
        }
        float s = _mm512_reduce_add_ps(acc);
        for (; j < d_; ++j) { const float cj = float((c[j / 2] >> ((j & 1) * 4)) & 0x0F); const float t = q[j] * a_[j] + b_[j] - cj; s += w_[j] * t * t; }
        s -= bias_;
        return s < 0.f ? 0.f : s;
    }
    bool packed_ = false; uint32_t rec_ = 0;
    uint8_t* map_ = nullptr; const uint8_t* codes_ = nullptr; size_t bytes_ = 0;
    uint32_t n_ = 0, d_ = 0, bits_ = 8, dp_ = 0; float bias_ = 0; std::vector<float> a_, b_, w_;
};

}  // namespace ksg
