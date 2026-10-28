#ifndef DISTANCE_H
#define DISTANCE_H
#include <vector>
#include <cmath>
#include <stdexcept>
#include <immintrin.h>
#include <omp.h>
#include <type_traits>
#include <cstdint>
#include "types.h"


template<typename T>
inline DisType euclidean_distance(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Euclidean_distance calculation: vectors must be of the same dimension");
    }
    DisType sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        DisType diff = static_cast<DisType>(a[i]) - static_cast<DisType>(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}


template<typename T>
inline DisType euclidean_distance_avx512(const T* a, const T* b, size_t n) {
    if (!a || !b) {
        throw std::invalid_argument("Euclidean_distance calculation: null vector pointer");
    }

    DisType sum = 0.0f;
    size_t i = 0;

    if constexpr (std::is_same<T, float>::value) {
        __m512 sum_vec = _mm512_setzero_ps();
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
        }
        alignas(64) float temp[16];
        _mm512_store_ps(temp, sum_vec);
        for (float v : temp) sum += v;
    }
    else if constexpr (std::is_same<T, uint8_t>::value) {
        __m512i sum_vec = _mm512_setzero_si512();
        const __m512i zero = _mm512_setzero_si512();
        for (; i + 64 <= n; i += 64) {
            __m256i a_lo_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            __m256i b_lo_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            __m256i a_hi_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
            __m256i b_hi_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));

            __m512i a_lo = _mm512_cvtepu8_epi16(a_lo_u8);
            __m512i b_lo = _mm512_cvtepu8_epi16(b_lo_u8);
            __m512i a_hi = _mm512_cvtepu8_epi16(a_hi_u8);
            __m512i b_hi = _mm512_cvtepu8_epi16(b_hi_u8);

            __m512i diff_lo = _mm512_sub_epi16(a_lo, b_lo);
            __m512i diff_hi = _mm512_sub_epi16(a_hi, b_hi);
            sum_vec = _mm512_add_epi32(sum_vec, _mm512_madd_epi16(diff_lo, diff_lo));
            sum_vec = _mm512_add_epi32(sum_vec, _mm512_madd_epi16(diff_hi, diff_hi));
        }
        alignas(64) int32_t temp[16];
        _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_vec);
        for (int32_t v : temp) sum += static_cast<DisType>(v);
        (void)zero;
    }

    for (; i < n; ++i) {
        DisType diff = static_cast<DisType>(a[i]) - static_cast<DisType>(b[i]);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

template<typename T>
inline DisType squared_l2_distance_avx512(const T* a, const T* b, size_t n) {
    if (!a || !b) {
        throw std::invalid_argument("Squared L2 distance calculation: null vector pointer");
    }

    DisType sum = 0.0f;
    size_t i = 0;

    if constexpr (std::is_same<T, float>::value) {
        __m512 sum_vec = _mm512_setzero_ps();
        for (; i + 16 <= n; i += 16) {
            __m512 va = _mm512_loadu_ps(a + i);
            __m512 vb = _mm512_loadu_ps(b + i);
            __m512 diff = _mm512_sub_ps(va, vb);
            sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
        }
        alignas(64) float temp[16];
        _mm512_store_ps(temp, sum_vec);
        for (float v : temp) sum += v;
    }
    else if constexpr (std::is_same<T, uint8_t>::value) {
        __m512i sum_vec = _mm512_setzero_si512();
        for (; i + 64 <= n; i += 64) {
            __m256i a_lo_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
            __m256i b_lo_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
            __m256i a_hi_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
            __m256i b_hi_u8 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));

            __m512i a_lo = _mm512_cvtepu8_epi16(a_lo_u8);
            __m512i b_lo = _mm512_cvtepu8_epi16(b_lo_u8);
            __m512i a_hi = _mm512_cvtepu8_epi16(a_hi_u8);
            __m512i b_hi = _mm512_cvtepu8_epi16(b_hi_u8);

            __m512i diff_lo = _mm512_sub_epi16(a_lo, b_lo);
            __m512i diff_hi = _mm512_sub_epi16(a_hi, b_hi);
            sum_vec = _mm512_add_epi32(sum_vec, _mm512_madd_epi16(diff_lo, diff_lo));
            sum_vec = _mm512_add_epi32(sum_vec, _mm512_madd_epi16(diff_hi, diff_hi));
        }
        alignas(64) int32_t temp[16];
        _mm512_store_si512(reinterpret_cast<__m512i*>(temp), sum_vec);
        for (int32_t v : temp) sum += static_cast<DisType>(v);
    }

    for (; i < n; ++i) {
        DisType diff = static_cast<DisType>(a[i]) - static_cast<DisType>(b[i]);
        sum += diff * diff;
    }
    return sum;
}

template<typename T>
inline DisType euclidean_distance_avx512(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Euclidean_distance calculation: vectors must be of the same dimension");
    }
    return euclidean_distance_avx512(a.data(), b.data(), a.size());
}




template<typename T>
inline void squared_l2_batch2(
    const T* query, const T* vec0, const T* vec1, size_t dim,
    DisType& dist0, DisType& dist1) {

    dist0 = 0.0f;
    dist1 = 0.0f;
    size_t i = 0;

    if constexpr (std::is_same<T, float>::value) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        for (; i + 16 <= dim; i += 16) {
            __m512 q = _mm512_loadu_ps(query + i);
            __m512 v0 = _mm512_loadu_ps(vec0 + i);
            __m512 v1 = _mm512_loadu_ps(vec1 + i);
            __m512 d0 = _mm512_sub_ps(q, v0);
            __m512 d1 = _mm512_sub_ps(q, v1);
            sum0 = _mm512_fmadd_ps(d0, d0, sum0);     // sum0 += (q-v0)^2
            sum1 = _mm512_fmadd_ps(d1, d1, sum1);     // sum1 += (q-v1)^2
        }
        alignas(64) float tmp0[16], tmp1[16];
        _mm512_store_ps(tmp0, sum0);
        _mm512_store_ps(tmp1, sum1);
        for (int j = 0; j < 16; ++j) {
            dist0 += tmp0[j];
            dist1 += tmp1[j];
        }
    } else if constexpr (std::is_same<T, uint8_t>::value) {
        __m512i sum0 = _mm512_setzero_si512();
        __m512i sum1 = _mm512_setzero_si512();
        for (; i + 64 <= dim; i += 64) {
            __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + i));
            __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + i + 32));
            __m512i ql = _mm512_cvtepu8_epi16(q_lo);
            __m512i qh = _mm512_cvtepu8_epi16(q_hi);

            __m256i v0l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vec0 + i));
            __m256i v0h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vec0 + i + 32));
            __m512i v0_lo = _mm512_cvtepu8_epi16(v0l);
            __m512i v0_hi = _mm512_cvtepu8_epi16(v0h);

            __m256i v1l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vec1 + i));
            __m256i v1h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(vec1 + i + 32));
            __m512i v1_lo = _mm512_cvtepu8_epi16(v1l);
            __m512i v1_hi = _mm512_cvtepu8_epi16(v1h);

            __m512i d0l = _mm512_sub_epi16(ql, v0_lo);
            __m512i d0h = _mm512_sub_epi16(qh, v0_hi);
            __m512i d1l = _mm512_sub_epi16(ql, v1_lo);
            __m512i d1h = _mm512_sub_epi16(qh, v1_hi);

            sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(d0l, d0l));
            sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(d0h, d0h));
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(d1l, d1l));
            sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(d1h, d1h));
        }
        alignas(64) int32_t t0[16], t1[16];
        _mm512_store_si512(reinterpret_cast<__m512i*>(t0), sum0);
        _mm512_store_si512(reinterpret_cast<__m512i*>(t1), sum1);
        for (int j = 0; j < 16; ++j) {
            dist0 += static_cast<DisType>(t0[j]);
            dist1 += static_cast<DisType>(t1[j]);
        }
    }

    for (; i < dim; ++i) {
        DisType d0 = static_cast<DisType>(query[i]) - static_cast<DisType>(vec0[i]);
        DisType d1 = static_cast<DisType>(query[i]) - static_cast<DisType>(vec1[i]);
        dist0 += d0 * d0;
        dist1 += d1 * d1;
    }
}

#endif // DISTANCE_H
