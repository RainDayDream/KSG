#pragma once
// 4-bit scalar-quantised sidecar for 128-d uint8 vectors (64 B / vector). Used as a cheap in-memory "second look":
// refine decisions are made on SQ4 distances (no SSD read); exact vectors are only read for the final rerank.
//   code byte j (0..63): low nibble = floor(x[j]/16), high nibble = floor(x[j+64]/16); reconstruction = 16*c + 8.
//   E||q - rec||^2 = ||q - x||^2 + 128 * (16^2-1)/12 (uniform error) -> that constant is subtracted.
// Affine variant (header extra field | 0x100, then lo[128] and step[128] as float): the 5-bit code c of dimension j is
//   floor((x - lo_j) / step_j) clamped to 0..31 and stored as v = 8c in the same packed layout; reconstruction
//   lo_j - 0.5 + (c + 0.5) step_j. Per-dimension ranges ([min, 99.9th percentile]) give U6-like ranking accuracy at 80 B.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ksg {

class SQ4 {
public:
    // bias constant: 128 dims * (step^2-1)/12 with step = 2^(4-extra_bits)
    int32_t bias_ = 2720;
    // heap = true: copy the codes into anonymous memory (pread) and drop the file's page cache, so the memory is explicit
    // (counted as anonymous by a cgroup, nothing for the kernel to reclaim/re-read); otherwise a read-only file mapping.
    explicit SQ4(const std::string& path, bool resident, bool heap = false) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open sq4 file: " + path);
        struct stat st; fstat(fd, &st); bytes_ = static_cast<size_t>(st.st_size);
        if (heap) {
            map_ = static_cast<uint8_t*>(mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            if (map_ == MAP_FAILED) throw std::runtime_error("sq4 heap alloc failed");
            for (size_t o = 0; o < bytes_;) { const ssize_t r = ::pread(fd, map_ + o, std::min<size_t>(bytes_ - o, size_t(1) << 26), static_cast<off_t>(o)); if (r <= 0) throw std::runtime_error("sq4 heap read failed"); o += size_t(r); }
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        } else map_ = static_cast<uint8_t*>(mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE | (resident ? MAP_POPULATE : 0), fd, 0));
        ::close(fd);
        if (map_ == MAP_FAILED) throw std::runtime_error("mmap sq4 failed");
        std::memcpy(&n_, map_, 4); std::memcpy(&d_, map_ + 4, 4);
        extra_ = 0; hdr_ = 8;
        if (d_ == 128 && bytes_ != 8 + static_cast<size_t>(n_) * 64) {   // new format: [n][d][extra_bits]
            std::memcpy(&extra_, map_ + 8, 4); hdr_ = 12;
            if (extra_ & 0x100) {   // affine: lo[128], step[128] follow
                extra_ &= 0xFF; affine_ = true;
                const float* lo = reinterpret_cast<const float*>(map_ + 12); const float* st = reinterpret_cast<const float*>(map_ + 12 + 512);
                double bias = 0;
                for (int j = 0; j < 128; ++j) { a_[j] = 1.0f / st[j]; b_[j] = (0.5f - lo[j]) / st[j]; w_[j] = st[j] * st[j]; bias += st[j] * st[j] / 12.0; }
                fbias_ = static_cast<float>(bias); hdr_ = 12 + 1024;
            }
        }
        if (d_ != 128 || (extra_ != 0 && extra_ != 1 && extra_ != 2)) throw std::runtime_error("bad sq4 file header");
        rec_ = 64 + 16 * extra_;
        if (bytes_ != hdr_ + static_cast<size_t>(n_) * rec_) throw std::runtime_error("bad sq4 file size");
        codes_ = map_ + hdr_;
        const int step = 16 >> extra_;
        bias_ = static_cast<int32_t>(128 * (step * step - 1) / 12.0 + 0.5);
    }
    uint32_t size() const { return n_; }
    inline int32_t distance(const uint8_t* q, uint64_t id) const {
        const uint8_t* c = codes_ + id * rec_;
        const __m512i x = _mm512_loadu_si512(c);
        const __m512i m = _mm512_set1_epi8(0x0F);
        const __m512i lo = _mm512_and_si512(x, m);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(x, 4), m);
        __m512i r0 = _mm512_slli_epi16(lo, 4), r1 = _mm512_slli_epi16(hi, 4);
        if (extra_ == 0) {
            const __m512i eight = _mm512_set1_epi8(8);
            r0 = _mm512_or_si512(r0, eight); r1 = _mm512_or_si512(r1, eight);
        } else if (extra_ == 2) {   // 32 B plane: byte k holds dims k+32m (m=0..3) in bits 2m..2m+1
            const __m256i X = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c + 64));
            const __m512i XX = _mm512_inserti64x4(_mm512_castsi256_si512(X), X, 1);
            const __m512i m3 = _mm512_set1_epi8(3), mid = _mm512_set1_epi8(2);
            const __m512i e0 = _mm512_and_si512(_mm512_srlv_epi64(XX, _mm512_set_epi64(2, 2, 2, 2, 0, 0, 0, 0)), m3);   // dims 0..31 | 32..63
            const __m512i e1 = _mm512_and_si512(_mm512_srlv_epi64(XX, _mm512_set_epi64(6, 6, 6, 6, 4, 4, 4, 4)), m3);   // dims 64..95 | 96..127
            r0 = _mm512_or_si512(r0, _mm512_or_si512(_mm512_slli_epi16(e0, 2), mid));
            r1 = _mm512_or_si512(r1, _mm512_or_si512(_mm512_slli_epi16(e1, 2), mid));
        } else {   // extra_ == 1: 16 B plane: byte k holds dims k+16m (m=0..7) in bit m
            const __m128i X = _mm_loadu_si128(reinterpret_cast<const __m128i*>(c + 64));
            const __m512i XX = _mm512_broadcast_i32x4(X);
            const __m512i m1 = _mm512_set1_epi8(1), mid = _mm512_set1_epi8(4);
            const __m512i e0 = _mm512_and_si512(_mm512_srlv_epi64(XX, _mm512_set_epi64(3, 3, 2, 2, 1, 1, 0, 0)), m1);   // dims 0..15|16..31|32..47|48..63
            const __m512i e1 = _mm512_and_si512(_mm512_srlv_epi64(XX, _mm512_set_epi64(7, 7, 6, 6, 5, 5, 4, 4)), m1);   // dims 64..127
            r0 = _mm512_or_si512(r0, _mm512_or_si512(_mm512_slli_epi16(e0, 3), mid));
            r1 = _mm512_or_si512(r1, _mm512_or_si512(_mm512_slli_epi16(e1, 3), mid));
        }
        if (affine_) return affine_distance(q, r0, r1);
        __m512i acc = sqdiff(_mm512_loadu_si512(q), r0);
        acc = _mm512_add_epi32(acc, sqdiff(_mm512_loadu_si512(q + 64), r1));
        const int32_t d = _mm512_reduce_add_epi32(acc) - bias_;
        return d < 0 ? 0 : d;
    }
    inline void prefetch(uint64_t id) const { __builtin_prefetch(codes_ + id * rec_, 0, 1); if (extra_) __builtin_prefetch(codes_ + id * rec_ + 64, 0, 1); }
private:
    // sum_j step_j^2 * ((q_j - lo_j + 0.5) / step_j - r_j / 8)^2 - sum_j step_j^2 / 12, r_j = 8c + 4 from the shared decoder
    inline int32_t affine_distance(const uint8_t* q, __m512i r0, __m512i r1) const {
        __m512 acc = _mm512_setzero_ps();
        const __m512 eighth = _mm512_set1_ps(0.125f);
        for (int h = 0; h < 2; ++h) {
            const __m512i r = h ? r1 : r0;
            for (int k = 0; k < 4; ++k) {
                const int j = h * 64 + k * 16;
                const __m128i rb = k == 0 ? _mm512_castsi512_si128(r) : k == 1 ? _mm512_extracti32x4_epi32(r, 1)
                                 : k == 2 ? _mm512_extracti32x4_epi32(r, 2) : _mm512_extracti32x4_epi32(r, 3);
                const __m512 rf = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(rb)), eighth);
                const __m512 qf = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + j))));
                const __m512 t = _mm512_sub_ps(_mm512_fmadd_ps(qf, _mm512_loadu_ps(a_ + j), _mm512_loadu_ps(b_ + j)), rf);
                acc = _mm512_fmadd_ps(_mm512_mul_ps(t, t), _mm512_loadu_ps(w_ + j), acc);
            }
        }
        const float d = _mm512_reduce_add_ps(acc) - fbias_;
        return d < 0 ? 0 : static_cast<int32_t>(d + 0.5f);
    }
    static inline __m512i sqdiff(__m512i a, __m512i b) {
        const __m512i d = _mm512_or_si512(_mm512_subs_epu8(a, b), _mm512_subs_epu8(b, a));
        const __m512i l = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(d));
        const __m512i h = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(d, 1));
        return _mm512_add_epi32(_mm512_madd_epi16(l, l), _mm512_madd_epi16(h, h));
    }
    bool affine_ = false; float fbias_ = 0; alignas(64) float a_[128], b_[128], w_[128];
    uint8_t* map_ = nullptr; const uint8_t* codes_ = nullptr; size_t bytes_ = 0, hdr_ = 8; uint32_t n_ = 0, d_ = 0, extra_ = 0, rec_ = 64;
};

}  // namespace ksg
