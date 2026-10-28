#pragma once
//
// DenseIVFPQReader::distance() reconstructs every vector (coarse centroid + 16 PQ
// sub-centroids, 128-d float arithmetic, ~8 random cache lines per call). For a
// residual IVFPQ code x = (list l, codes k_1..k_M):
//   ||q - c_l - r||^2 = ||q - c_l||^2 + beta(x) - 2 * sum_m <q_m, p_{m,k_m}>
//   beta(x)          = ||r||^2 + 2 <c_l, r>          (per vector, precomputed once)
// so a distance becomes: one cached per-(query, list) term, one float load and
// M lookups into a per-query 16 KB table. Mathematically identical to the
// reconstruction distance; differs only by float rounding.
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <omp.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include "dense_ivfpq.h"

namespace ksg {

class DenseADC {
public:
    // Precompute beta for all vectors (anonymous memory, count * 4 bytes).
    explicit DenseADC(const DenseIVFPQReader& r, int threads) : r_(r) {
        if (!r.header().by_residual) throw std::runtime_error("DenseADC expects a residual IVFPQ sidecar");
        M_ = r.chunks(); ksub_ = r.ksub(); dsub_ = r.dsub(); dim_ = r.dim(); nlist_ = r.nlist();
        // ||p_{m,k}||^2 table
        pnorm_.resize(static_cast<size_t>(M_) * ksub_);
        for (uint32_t m = 0; m < M_; ++m)
            for (uint32_t k = 0; k < ksub_; ++k) {
                const float* p = r.pq_centroid(m, k);
                float s = 0; for (uint32_t j = 0; j < dsub_; ++j) s += p[j] * p[j];
                pnorm_[static_cast<size_t>(m) * ksub_ + k] = s;
            }
        const uint64_t n = r.size();
        beta_.reset(new float[n]);
        #pragma omp parallel for schedule(static, 1 << 16) num_threads(std::max(1, threads))
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const uint32_t l = r.list_id(static_cast<uint64_t>(i));
            if (l >= nlist_) { beta_[i] = 0; continue; }
            const float* c = r.coarse_centroid(l);
            const uint8_t* code = r.code(static_cast<uint64_t>(i));
            float s = 0;
            for (uint32_t m = 0; m < M_; ++m) {
                const float* p = r.pq_centroid(m, code[m]);
                const float* cm = c + static_cast<size_t>(m) * dsub_;
                float dot = 0; for (uint32_t j = 0; j < dsub_; ++j) dot += cm[j] * p[j];
                s += pnorm_[static_cast<size_t>(m) * ksub_ + code[m]] + 2.0f * dot;
            }
            beta_[i] = s;
        }
    }

    // Per-query state; call once per query on the thread that searches it.
    template <class T>
    void prepare(const T* q) const {
        State& st = state();
        if (st.qf.size() != dim_) {
            st.qf.resize(dim_); st.lut.resize(static_cast<size_t>(M_) * ksub_);
            st.a.assign(nlist_, 0.0f); st.stamp.assign(nlist_, 0u); st.epoch = 0;
        }
        for (uint32_t j = 0; j < dim_; ++j) st.qf[j] = static_cast<float>(q[j]);
        for (uint32_t m = 0; m < M_; ++m) {
            const float* qm = st.qf.data() + static_cast<size_t>(m) * dsub_;
            for (uint32_t k = 0; k < ksub_; ++k) {
                const float* p = r_.pq_centroid(m, k);
                float dot = 0; for (uint32_t j = 0; j < dsub_; ++j) dot += qm[j] * p[j];
                st.lut[static_cast<size_t>(m) * ksub_ + k] = -2.0f * dot;
            }
        }
        if (++st.epoch == 0) { std::fill(st.stamp.begin(), st.stamp.end(), 0u); st.epoch = 1; }
    }

    inline float distance(uint64_t id) const {
        State& st = state();
        const uint32_t l = r_.list_id(id);
        if (l >= nlist_) return std::numeric_limits<float>::infinity();
        if (st.stamp[l] != st.epoch) {
            const float* c = r_.coarse_centroid(l);
            float s = 0; for (uint32_t j = 0; j < dim_; ++j) { const float d = st.qf[j] - c[j]; s += d * d; }
            st.a[l] = s; st.stamp[l] = st.epoch;
        }
        const uint8_t* code = r_.code(id);
        const float* lut = st.lut.data();
        float s = st.a[l] + beta_[id];
        for (uint32_t m = 0; m < M_; ++m) s += lut[static_cast<size_t>(m) * ksub_ + code[m]];
        return s;
    }

    // Batched distance(): same arithmetic per id (bit-identical results), but the thread-local
    // state is resolved once and the random loads of id i+D are prefetched while scoring id i.
    template <class Id>
    void distance_many(const Id* ids, size_t n, float* out) const {
        State& st = state();
        const float* lut = st.lut.data();
        constexpr size_t D = 8;
        for (size_t i = 0; i < n && i < D; ++i) prefetch(ids[i]);
        for (size_t i = 0; i < n; ++i) {
            if (i + D < n) prefetch(ids[i + D]);
            const uint64_t id = ids[i];
            const uint32_t l = r_.list_id(id);
            if (l >= nlist_) { out[i] = std::numeric_limits<float>::infinity(); continue; }
            if (st.stamp[l] != st.epoch) {
                const float* c = r_.coarse_centroid(l);
                float s = 0; for (uint32_t j = 0; j < dim_; ++j) { const float d = st.qf[j] - c[j]; s += d * d; }
                st.a[l] = s; st.stamp[l] = st.epoch;
            }
            const uint8_t* code = r_.code(id);
#if defined(__AVX512F__)
            if (gather_ && M_ == 16) {
                // one 16-lane gather instead of 16 dependent scalar loads (summation order differs from distance())
                const __m512i idx = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(code))),
                                                     _mm512_mullo_epi32(_mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
                                                                        _mm512_set1_epi32(static_cast<int>(ksub_))));
                out[i] = st.a[l] + beta_[id] + _mm512_reduce_add_ps(_mm512_i32gather_ps(idx, lut, 4));
                continue;
            }
#endif
            float s = st.a[l] + beta_[id];
            for (uint32_t m = 0; m < M_; ++m) s += lut[static_cast<size_t>(m) * ksub_ + code[m]];
            out[i] = s;
        }
    }

    void set_gather(bool v) { gather_ = v; }

    // Software prefetch of the three random-access structures a distance() call touches.
    inline void prefetch(uint64_t id) const {
        __builtin_prefetch(r_.code(id), 0, 1);
        __builtin_prefetch(beta_.get() + id, 0, 1);
        __builtin_prefetch(r_.list_id_ptr(id), 0, 1);
    }

    size_t beta_bytes() const { return static_cast<size_t>(r_.size()) * sizeof(float); }

private:
    struct State {
        std::vector<float> qf, lut, a;
        std::vector<uint32_t> stamp;
        uint32_t epoch = 0;
    };
    static State& state() { thread_local State s; return s; }

    const DenseIVFPQReader& r_;
    bool gather_ = false;
    uint32_t M_ = 0, ksub_ = 0, dsub_ = 0, dim_ = 0, nlist_ = 0;
    std::vector<float> pnorm_;
    std::unique_ptr<float[]> beta_;
};

}  // namespace ksg
