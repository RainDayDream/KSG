#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <type_traits>
#include <unistd.h>

namespace ksg {

// index.  It is neither a graph nor ground truth; exact refine/rerank still
// read the base vecbin.  Header provenance is intentionally limited to source
// size/mtime, not a content hash.

constexpr uint32_t kDenseIVFPQVersion = 1;
constexpr size_t kDenseIVFPQHeaderBytes = 4096;

struct DenseIVFPQHeader {
    char magic[8] = {'K', 'S', 'G', 'D', 'P', 'Q', '0', '1'};
    uint32_t version = kDenseIVFPQVersion;
    uint32_t header_bytes = kDenseIVFPQHeaderBytes;
    uint64_t count = 0;
    uint32_t dim = 0;
    uint32_t nlist = 0;
    uint32_t chunks = 0;
    uint32_t nbits = 0;
    uint32_t code_size = 0;
    uint32_t list_id_bytes = 0;
    uint32_t dsub = 0;
    uint32_t ksub = 0;
    uint32_t by_residual = 0;
    uint32_t metric_type = 0;
    uint64_t source_bytes = 0;
    int64_t source_mtime = 0;
    uint64_t coarse_offset = 0;
    uint64_t pq_offset = 0;
    uint64_t list_ids_offset = 0;
    uint64_t codes_offset = 0;
    uint64_t file_bytes = 0;
    uint64_t assigned_count = 0;
    uint64_t checksum = 0;
};
static_assert(sizeof(DenseIVFPQHeader) <= kDenseIVFPQHeaderBytes);

inline uint64_t align_up(uint64_t value, uint64_t alignment = 4096) {
    return (value + alignment - 1) / alignment * alignment;
}

inline uint64_t dense_pq_checksum(const DenseIVFPQHeader& h) {
    uint64_t value = 1469598103934665603ULL;
    auto mix = [&](uint64_t x) { value ^= x; value *= 1099511628211ULL; };
    mix(h.version); mix(h.count); mix(h.dim); mix(h.nlist); mix(h.chunks); mix(h.nbits);
    mix(h.code_size); mix(h.list_id_bytes); mix(h.dsub); mix(h.ksub); mix(h.by_residual);
    mix(h.source_bytes); mix(static_cast<uint64_t>(h.source_mtime)); mix(h.assigned_count);
    return value;
}

class DenseIVFPQReader {
public:
    // Reads the optional ID-addressable distance sidecar. Constructor checks
    // sidecar structure, but source-index identity is metadata, not a hash.
    explicit DenseIVFPQReader(const std::string& path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open dense IVFPQ: " + path + ": " + std::strerror(errno));
        struct stat st{};
        if (fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat dense IVFPQ: " + path);
        bytes_ = static_cast<size_t>(st.st_size);
        map_ = static_cast<const uint8_t*>(mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (map_ == MAP_FAILED) { map_ = nullptr; throw std::runtime_error("cannot mmap dense IVFPQ: " + path); }
        if (bytes_ < kDenseIVFPQHeaderBytes) throw std::runtime_error("dense IVFPQ is truncated");
        std::memcpy(&header_, map_, sizeof(header_));
        validate();
        point_to_base(map_);
        (void)madvise(const_cast<uint8_t*>(map_), bytes_, MADV_RANDOM);
    }
    ~DenseIVFPQReader() { if (map_) munmap(const_cast<uint8_t*>(map_), bytes_); if (fd_ >= 0) close(fd_); }
    DenseIVFPQReader(const DenseIVFPQReader&) = delete;
    DenseIVFPQReader& operator=(const DenseIVFPQReader&) = delete;

    uint64_t size() const { return header_.count; }
    uint32_t dim() const { return header_.dim; }
    uint32_t code_size() const { return header_.code_size; }
    const DenseIVFPQHeader& header() const { return header_; }
    size_t bytes() const { return bytes_; }
    bool resident() const { return !resident_bytes_.empty(); }

    double make_resident(int threads) {
        if (resident()) return 0.0;
        if (!map_ || bytes_ == 0) throw std::runtime_error("dense IVFPQ mmap is not initialized");
        resident_bytes_.resize(bytes_);
        const int worker_count = std::max(1, threads);
        const size_t chunk = (bytes_ + static_cast<size_t>(worker_count) - 1) /
                             static_cast<size_t>(worker_count);
        const auto start = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int t = 0; t < worker_count; ++t) {
            const size_t begin = std::min(bytes_, static_cast<size_t>(t) * chunk);
            const size_t end = std::min(bytes_, begin + chunk);
            if (begin < end) {
                std::memcpy(resident_bytes_.data() + begin, map_ + begin, end - begin);
            }
        }
        point_to_base(resident_bytes_.data());
        (void)madvise(const_cast<uint8_t*>(map_), bytes_, MADV_DONTNEED);
        const auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }

    uint32_t list_id(uint64_t id) const {
        if (header_.list_id_bytes == 2) {
            uint16_t value; std::memcpy(&value, list_ids_ + id * 2, 2); return value;
        }
        uint32_t value; std::memcpy(&value, list_ids_ + id * 4, 4); return value;
    }

    const float* coarse_centroid(uint32_t list) const {
        if (list >= header_.nlist) throw std::out_of_range("dense IVFPQ list id out of range");
        return coarse_ + static_cast<uint64_t>(list) * header_.dim;
    }

    const void* list_id_ptr(uint64_t id) const { return list_ids_ + id * header_.list_id_bytes; }
    const uint8_t* code(uint64_t id) const { return codes_ + id * header_.code_size; }
    uint32_t nlist() const { return header_.nlist; }
    uint32_t chunks() const { return header_.chunks; }
    uint32_t ksub() const { return header_.ksub; }
    uint32_t dsub() const { return header_.dsub; }
    // PQ sub-centroid k of chunk m (dsub floats), shared by all lists.
    const float* pq_centroid(uint32_t m, uint32_t k) const {
        return pq_ + (static_cast<uint64_t>(m) * header_.ksub + k) * header_.dsub;
    }

    template<class T>
    float distance(const T* query, uint64_t id) const {
        const uint32_t list = list_id(id);
        if (list >= header_.nlist) return std::numeric_limits<float>::infinity();
        const float* coarse = coarse_ + static_cast<uint64_t>(list) * header_.dim;
        const uint8_t* encoded = code(id);
        if (header_.dsub == 8 && (std::is_same<T, uint8_t>::value || std::is_same<T, float>::value)) {
            __m256 sum = _mm256_setzero_ps();
            for (uint32_t m = 0; m < header_.chunks; ++m) {
                const uint32_t base = m * 8;
                const float* centroid = pq_ + (static_cast<uint64_t>(m) * header_.ksub + encoded[m]) * 8;
                __m256 q;
                if constexpr (std::is_same<T, uint8_t>::value) {
                    const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(query + base));
                    q = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(q8));
                } else {
                    q = _mm256_loadu_ps(query + base);
                }
                const __m256 reconstructed = _mm256_add_ps(
                    _mm256_loadu_ps(coarse + base), _mm256_loadu_ps(centroid));
                const __m256 delta = _mm256_sub_ps(q, reconstructed);
                sum = _mm256_fmadd_ps(delta, delta, sum);
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, sum);
            return lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
        }
        float result = 0.0f;
        for (uint32_t m = 0; m < header_.chunks; ++m) {
            const float* centroid = pq_ +
                (static_cast<uint64_t>(m) * header_.ksub + encoded[m]) * header_.dsub;
            const uint32_t base = m * header_.dsub;
            for (uint32_t j = 0; j < header_.dsub; ++j) {
                const float delta = static_cast<float>(query[base + j]) - (coarse[base + j] + centroid[j]);
                result += delta * delta;
            }
        }
        return result;
    }

private:
    void point_to_base(const uint8_t* base) {
        coarse_ = reinterpret_cast<const float*>(base + header_.coarse_offset);
        pq_ = reinterpret_cast<const float*>(base + header_.pq_offset);
        list_ids_ = base + header_.list_ids_offset;
        codes_ = base + header_.codes_offset;
    }

    void validate() const {
        const std::array<char, 8> expected{'K','S','G','D','P','Q','0','1'};
        if (!std::equal(expected.begin(), expected.end(), header_.magic)) throw std::runtime_error("invalid dense IVFPQ magic");
        if (header_.version != kDenseIVFPQVersion || header_.header_bytes != kDenseIVFPQHeaderBytes)
            throw std::runtime_error("unsupported dense IVFPQ version");
        if (!header_.count || !header_.dim || !header_.nlist || !header_.chunks || header_.nbits != 8 ||
            header_.code_size != header_.chunks || header_.dim != header_.chunks * header_.dsub ||
            (header_.list_id_bytes != 2 && header_.list_id_bytes != 4) || !header_.by_residual)
            throw std::runtime_error("invalid dense IVFPQ metadata");
        if (header_.file_bytes != bytes_ || header_.codes_offset + header_.count * header_.code_size > bytes_ ||
            header_.assigned_count != header_.count || header_.checksum != dense_pq_checksum(header_))
            throw std::runtime_error("dense IVFPQ size, coverage, or checksum mismatch");
    }

    std::string path_;
    int fd_ = -1;
    size_t bytes_ = 0;
    const uint8_t* map_ = nullptr;
    std::vector<uint8_t> resident_bytes_;
    DenseIVFPQHeader header_{};
    const float* coarse_ = nullptr;
    const float* pq_ = nullptr;
    const uint8_t* list_ids_ = nullptr;
    const uint8_t* codes_ = nullptr;
};

} // namespace ksg
