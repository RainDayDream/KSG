#pragma once
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <memory>
#include <atomic>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include "types.h"
#include "compact_adj.h"

// search. They define what the binary artifacts mean on disk.
//

inline void advise_random_read_access(int fd, void* ptr, size_t len) noexcept {
#ifdef POSIX_FADV_RANDOM
    if (fd >= 0) {
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
    }
#endif
#ifdef MADV_RANDOM
    if (ptr && ptr != MAP_FAILED && len > 0) {
        (void)madvise(ptr, len, MADV_RANDOM);
    }
#endif
}

// --data_type, so review base/query/GT identity together with their type and
// header.  This reader is the common data source for builders and search.
template<typename T>
class VecMMapReader {
public:
    VecMMapReader(const VecMMapReader&) = delete;
    VecMMapReader& operator=(const VecMMapReader&) = delete;
    VecMMapReader(VecMMapReader&& other) noexcept {
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        filename_ = std::move(other.filename_);
        dim_ = other.dim_;
        num_vectors_ = other.num_vectors_;
        vector_bytes_ = other.vector_bytes_;
        metadata_bytes_ = other.metadata_bytes_;
        vectors_ = std::move(other.vectors_);
        block_cache_enabled_ = other.block_cache_enabled_;
        block_cache_vectors_ = other.block_cache_vectors_;
        block_cache_blocks_ = other.block_cache_blocks_;
    }
    VecMMapReader& operator=(VecMMapReader&& other) noexcept {
        cleanup();
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        filename_ = std::move(other.filename_);
        dim_ = other.dim_;
        num_vectors_ = other.num_vectors_;
        vector_bytes_ = other.vector_bytes_;
        metadata_bytes_ = other.metadata_bytes_;
        vectors_ = std::move(other.vectors_);
        block_cache_enabled_ = other.block_cache_enabled_;
        block_cache_vectors_ = other.block_cache_vectors_;
        block_cache_blocks_ = other.block_cache_blocks_;
        return *this;
    }

    explicit VecMMapReader(const std::string& filename) : filename_(filename) {
        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ == -1) throw std::runtime_error("[VecMMapReader]Failed to open file: " + filename);
        struct stat sb;
        if (fstat(fd_, &sb) == -1) { close(fd_); throw std::runtime_error("[VecMMapReader]Failed to stat file: " + filename); }
        mmap_len_ = sb.st_size;
        mmap_ptr_ = static_cast<char*>(mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mmap_ptr_ == MAP_FAILED) { close(fd_); mmap_ptr_ = nullptr; throw std::runtime_error("[VecMMapReader]Failed to mmap file: " + filename); }
        uint32_t num_vectors= 0;
        uint32_t dim = 0;
        std::memcpy(&num_vectors, mmap_ptr_, sizeof(uint32_t));
        std::memcpy(&dim, mmap_ptr_ + sizeof(uint32_t), sizeof(uint32_t));
        if (dim == 0) throw std::runtime_error("[VecMMapReader]Invalid dim in file: " + filename);
        dim_ = dim;
        num_vectors_ = num_vectors;
        metadata_bytes_ = sizeof(num_vectors) + sizeof(dim);
        vector_bytes_ =  dim_ * sizeof(T);
        if ((mmap_len_ - metadata_bytes_) % vector_bytes_ != 0)
            throw std::runtime_error("[VecMMapReader]File size does not match vector dimensions: " + filename);
    }
    ~VecMMapReader() { cleanup(); }

    std::vector<T> read_vector(size_t index) const {
        if (!mmap_ptr_) throw std::runtime_error("[read_vector]MMap not initialized");
        if (index >= num_vectors_) throw std::out_of_range("[read_vector]Index out of range: " + std::to_string(index));
        try{
            if(vectors_.size() > 0) {
                size_t vec_offset = index * dim_;
                if (vec_offset + dim_ > vectors_.size()) throw std::runtime_error("[read_vector]Offset exceeds vector size: " + filename_+", vector's index="+std::to_string(index));
                std::vector<T> vec(dim_);
                std::memcpy(vec.data(), vectors_.data() + vec_offset, vector_bytes_);
                return vec;
            }
            size_t file_offset = metadata_bytes_ + index * vector_bytes_;
            if (file_offset + vector_bytes_ > mmap_len_) throw std::runtime_error("[read_vector]Offset exceeds file size: " + filename_);
            std::vector<T> vec(dim_);
            const char* data_ptr = mmap_ptr_ + file_offset;
            std::memcpy(vec.data(), data_ptr, dim_ * sizeof(T));
            return vec;
        }
        catch(const std::exception& e)
        {
            std::cerr << "[read_vector]Error: " << e.what() << '\n';
            throw;
        }
        
    }

    const T* vector_data(size_t index) const {
        if (!mmap_ptr_) throw std::runtime_error("[vector_data]MMap not initialized");
        if (index >= num_vectors_) throw std::out_of_range("[vector_data]Index out of range: " + std::to_string(index));
        if (!vectors_.empty()) {
            size_t vec_offset = index * dim_;
            if (vec_offset + dim_ > vectors_.size()) throw std::runtime_error("[vector_data]Offset exceeds vector cache size: " + filename_);
            return vectors_.data() + vec_offset;
        }
        size_t file_offset = metadata_bytes_ + index * vector_bytes_;
        if (file_offset + vector_bytes_ > mmap_len_) throw std::runtime_error("[vector_data]Offset exceeds file size: " + filename_);
        return reinterpret_cast<const T*>(mmap_ptr_ + file_offset);
    }

    const T* vector_data_unchecked(size_t index) const {
        if (!vectors_.empty()) {
            return vectors_.data() + index * dim_;
        }
        if (block_cache_enabled_ && block_cache_blocks_ > 0 && block_cache_vectors_ > 0) {
            return cached_vector_data(index);
        }
        return reinterpret_cast<const T*>(mmap_ptr_ + metadata_bytes_ + index * vector_bytes_);
    }

    void prefetch_vector(size_t index) const noexcept {
        if (!mmap_ptr_ || index >= num_vectors_) return;
        const char* ptr = nullptr;
        if (!vectors_.empty()) {
            ptr = reinterpret_cast<const char*>(vectors_.data() + index * dim_);
        } else {
            ptr = mmap_ptr_ + metadata_bytes_ + index * vector_bytes_;
        }
        __builtin_prefetch(ptr, 0, 0);
        if (vector_bytes_ > 64) {
            __builtin_prefetch(ptr + 64, 0, 0);
        }
    }


    void read_all_vectors_flat(int start = 0, int end = -1) const {
        if (end < 0 || static_cast<size_t>(end) > num_vectors_) {
            end = num_vectors_;
        }
        if (start < 0 || static_cast<size_t>(start) >= num_vectors_ || start > end) {
            throw std::invalid_argument("[read_all_vectors]Invalid vector range specified");
        }
        
        const size_t num = end - start;
        size_t total_num = num * dim_;
        if (total_num == 0) return;
        vectors_.resize(total_num);
        const int available_threads = omp_get_max_threads();
        const size_t chunk_size = std::max<size_t>(1, num / (available_threads * 4));
        #pragma omp parallel for schedule(dynamic, chunk_size) num_threads(available_threads)
        for (size_t i = 0; i < num; ++i) {
            const size_t file_offset = metadata_bytes_ + (start + i) * vector_bytes_;
            if (file_offset + vector_bytes_ > mmap_len_) continue;
            const char* data_ptr = mmap_ptr_ + file_offset;
            std::memcpy(vectors_.data() + i * dim_, data_ptr, dim_ * sizeof(T));
        }
    }



    std::vector<std::vector<T>> read_all_vectors(int start = 0, int end = -1) const {
        if (end < 0 || static_cast<size_t>(end) > num_vectors_) {
            end = num_vectors_;
        }
        if (start < 0 || static_cast<size_t>(start) >= num_vectors_ || start > end) {
            throw std::invalid_argument("[read_all_vectors]Invalid vector range specified");
        }
        
        const size_t num = end - start;
        std::vector<std::vector<T>> vectors(num);
        for(auto & vec : vectors) {
            vec.resize(dim_);
        }
        const int available_threads = omp_get_max_threads();
        const size_t chunk_size = std::max<size_t>(1, num / (available_threads * 4));

        #pragma omp parallel for schedule(dynamic, chunk_size) num_threads(available_threads)
        for (size_t i = 0; i < num; ++i) {
            const size_t file_offset = metadata_bytes_ + (start + i) * vector_bytes_;
            if (file_offset + vector_bytes_ > mmap_len_) continue;
            const char* data_ptr = mmap_ptr_ + file_offset;
            std::memcpy(vectors[i].data(), data_ptr, dim_ * sizeof(T));   
        }
        return vectors;
    }

    

    std::string filename() const { return filename_; }
    size_t file_size() const { return mmap_len_; }
    size_t dim() const { return dim_; }
    size_t num_vectors() const { return num_vectors_; }
    size_t vector_bytes() const { return vector_bytes_; }
    void enable_block_cache(size_t block_vectors, size_t block_count) const {
        block_cache_vectors_ = std::max<size_t>(1, block_vectors);
        block_cache_blocks_ = block_count;
        block_cache_enabled_ = block_count > 0;
        if (block_cache_enabled_) advise_random_read_access(fd_, mmap_ptr_, mmap_len_);
    }
    bool block_cache_enabled() const { return block_cache_enabled_; }
    void advise_random() const { advise_random_read_access(fd_, mmap_ptr_, mmap_len_); }
private:
    struct ThreadVecBlock {
        const VecMMapReader<T>* owner = nullptr;
        size_t block_id = std::numeric_limits<size_t>::max();
        size_t valid_vectors = 0;
        std::vector<T> data;
        bool valid = false;
    };
    struct ThreadVecCache {
        size_t configured_blocks = 0;
        size_t configured_block_vectors = 0;
        size_t next_slot = 0;
        std::vector<ThreadVecBlock> blocks;
    };

    int fd_ = -1;
    char* mmap_ptr_ = nullptr;
    size_t mmap_len_ = 0;
    std::string filename_;
    uint32_t dim_ = 0;
    uint32_t num_vectors_ = 0;
    uint32_t vector_bytes_ = 0;
    uint32_t metadata_bytes_ = 0;
    mutable std::vector<T> vectors_;
    mutable bool block_cache_enabled_ = false;
    mutable size_t block_cache_vectors_ = 0;
    mutable size_t block_cache_blocks_ = 0;

    bool read_exact(void* dst, size_t bytes, size_t offset) const noexcept {
        char* out = static_cast<char*>(dst);
        size_t done = 0;
        while (done < bytes) {
            ssize_t ret = pread(fd_, out + done, bytes - done, static_cast<off_t>(offset + done));
            if (ret <= 0) return false;
            done += static_cast<size_t>(ret);
        }
        return true;
    }

    const T* cached_vector_data(size_t index) const {
        if (index >= num_vectors_) return reinterpret_cast<const T*>(mmap_ptr_ + metadata_bytes_ + index * vector_bytes_);
        static thread_local std::unordered_map<const VecMMapReader<T>*, ThreadVecCache> thread_caches;
        ThreadVecCache& cache = thread_caches[this];
        if (cache.configured_blocks != block_cache_blocks_ || cache.configured_block_vectors != block_cache_vectors_) {
            cache.configured_blocks = block_cache_blocks_;
            cache.configured_block_vectors = block_cache_vectors_;
            cache.next_slot = 0;
            cache.blocks.clear();
            cache.blocks.resize(block_cache_blocks_);
        }
        if (cache.blocks.empty()) {
            return reinterpret_cast<const T*>(mmap_ptr_ + metadata_bytes_ + index * vector_bytes_);
        }

        const size_t block_id = index / block_cache_vectors_;
        for (auto& block : cache.blocks) {
            if (block.valid && block.owner == this && block.block_id == block_id) {
                return block.data.data() + (index % block_cache_vectors_) * dim_;
            }
        }

        ThreadVecBlock& block = cache.blocks[cache.next_slot++ % cache.blocks.size()];
        const size_t start_vector = block_id * block_cache_vectors_;
        const size_t valid_vectors = std::min(block_cache_vectors_, static_cast<size_t>(num_vectors_) - start_vector);
        const size_t values = valid_vectors * static_cast<size_t>(dim_);
        const size_t bytes = valid_vectors * static_cast<size_t>(vector_bytes_);
        block.data.resize(values);
        const size_t offset = metadata_bytes_ + start_vector * static_cast<size_t>(vector_bytes_);
        if (!read_exact(block.data.data(), bytes, offset)) {
            block.valid = false;
            return reinterpret_cast<const T*>(mmap_ptr_ + metadata_bytes_ + index * vector_bytes_);
        }
        block.owner = this;
        block.block_id = block_id;
        block.valid_vectors = valid_vectors;
        block.valid = true;
        return block.data.data() + (index - start_vector) * dim_;
    }

    void cleanup() {
        if (mmap_ptr_) munmap(mmap_ptr_, mmap_len_);
        if (fd_ != -1) close(fd_);
        mmap_ptr_ = nullptr; fd_ = -1; mmap_len_ = 0;
    }
};

template<typename T>
class VecTXTMapReader {
public:
    VecTXTMapReader(const VecTXTMapReader&) = delete;
    VecTXTMapReader& operator=(const VecTXTMapReader&) = delete;
    VecTXTMapReader(VecTXTMapReader&& other) noexcept {
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        filename_ = std::move(other.filename_);
        dim_ = other.dim_;
        num_vectors_ = other.num_vectors_;
    }
    VecTXTMapReader& operator=(VecTXTMapReader&& other) noexcept {
        cleanup();
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        filename_ = std::move(other.filename_);
        dim_ = other.dim_;
        num_vectors_ = other.num_vectors_;
        return *this;
    }

    explicit VecTXTMapReader(const std::string& filename) : filename_(filename) {
        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ == -1) throw std::runtime_error("[VecTXTMapReader]Failed to open file: " + filename);
        struct stat sb;
        if (fstat(fd_, &sb) == -1) { close(fd_); throw std::runtime_error("[VecTXTMapReader]Failed to stat file: " + filename); }
        mmap_len_ = sb.st_size;
        mmap_ptr_ = static_cast<char*>(mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mmap_ptr_ == MAP_FAILED) { close(fd_); mmap_ptr_ = nullptr; throw std::runtime_error("[VecTXTMapReader]Failed to mmap file: " + filename); }
        uint32_t d0 = 0;
        std::memcpy(&d0, mmap_ptr_, sizeof(uint32_t));
        if (d0 == 0) throw std::runtime_error("[VecTXTMapReader]Invalid dim in file: " + filename);
        dim_ = d0;
        vector_bytes_ = sizeof(uint32_t) + dim_ * sizeof(T);
        if (mmap_len_ % vector_bytes_ != 0)
            throw std::runtime_error("[VecTXTMapReader]File size does not match vector dimensions: " + filename);
        num_vectors_ = mmap_len_ / vector_bytes_;
    }
    ~VecTXTMapReader() { cleanup(); }

    std::vector<T> read_vector(size_t index) const {
        if (!mmap_ptr_) throw std::runtime_error("[read_vector]MMap not initialized");
        if (index >= num_vectors_) throw std::out_of_range("[read_vector]Index out of range: " + std::to_string(index));
        try{
            size_t file_offset = index * vector_bytes_;
            if (file_offset + vector_bytes_ > mmap_len_) throw std::runtime_error("[read_vector]Offset exceeds file size: " + filename_);
            uint32_t read_dim;
            std::memcpy(&read_dim, mmap_ptr_ + file_offset, sizeof(uint32_t));
            if (read_dim != dim_) throw std::runtime_error("[read_vector]Dimension mismatch in file: " + filename_);
            std::vector<T> vec(dim_);
            const char* data_ptr = mmap_ptr_ + file_offset + sizeof(uint32_t);
            std::memcpy(vec.data(), data_ptr, dim_ * sizeof(T));
            return vec;
        }
        catch(const std::exception& e)
        {
            std::cerr << "[read_vector]Error: " << e.what() << '\n';
            throw;
        }
        
    }

    std::vector<std::vector<T>> read_all_vectors(int start = 0, int end = -1) const {
        if (end < 0 || static_cast<size_t>(end) > num_vectors_) {
            end = num_vectors_;
        }
        if (start < 0 || static_cast<size_t>(start) >= num_vectors_ || start > end) {
            throw std::invalid_argument("[read_all_vectors]Invalid vector range specified");
        }
        
        const size_t num = end - start;
        std::vector<std::vector<T>> vectors(num);
        for(auto & vec : vectors) {
            vec.resize(dim_);
        }
        // std::vector<T> flat_data(num * dim_);
        const int available_threads = omp_get_max_threads();
        const size_t chunk_size = std::max<size_t>(1, num / (available_threads * 4));
        #pragma omp parallel for schedule(dynamic, chunk_size) num_threads(available_threads)
        for (size_t i = 0; i < num; ++i) {
            const size_t file_offset = (start + i) * vector_bytes_;
            if (file_offset + vector_bytes_ > mmap_len_) continue;
            uint32_t read_dim;
            std::memcpy(&read_dim, mmap_ptr_ + file_offset, sizeof(uint32_t));
            if (read_dim != dim_) {
                #pragma omp critical
                {
                    throw std::runtime_error("[read_all_vectors] Dimension mismatch in file: " + 
                                            filename_ + " at index: " + std::to_string(start + i));
                }
            }
            const char* data_ptr = mmap_ptr_ + file_offset + sizeof(uint32_t);
            std::memcpy(vectors[i].data(), data_ptr, dim_ * sizeof(T));   

        }
        return vectors;
    }
    std::string filename() const { return filename_; }
    size_t file_size() const { return mmap_len_; }
    size_t dim() const { return dim_; }
    size_t num_vectors() const { return num_vectors_; }
    size_t vector_bytes() const { return vector_bytes_; }
private:
    int fd_ = -1;
    char* mmap_ptr_ = nullptr;
    size_t mmap_len_ = 0;
    std::string filename_;
    uint32_t dim_ = 0;
    size_t num_vectors_ = 0;
    size_t vector_bytes_ = 0;
    void cleanup() {
        if (mmap_ptr_) munmap(mmap_ptr_, mmap_len_);
        if (fd_ != -1) close(fd_);
        mmap_ptr_ = nullptr; fd_ = -1; mmap_len_ = 0;
    }
};




// [NL size, NL max_degree, 
// NL node0.neighbor_count, NID node0.neighbor0, NID node0.neighbor1, ..., NID node0.neighborN,
// NL node1.neighbor_count, NID node1.neighbor0, ...]
// Node -> adjacency offset index for variable-degree graphs: 64-bit base every 4096 nodes + 32-bit offset relative to it.
// 4 B/node instead of 8 (1B-node KSG: 4 GB instead of 8 GB of RSS); values must be written in increasing index order via set().
class CompactOffsets {
public:
    static constexpr size_t kShift = 12;
    void resize(size_t n) { n_ = n; rel_.assign(n, 0); base_.assign((n >> kShift) + 1, 0); }
    void clear() { n_ = 0; rel_.clear(); rel_.shrink_to_fit(); base_.clear(); base_.shrink_to_fit(); }
    bool empty() const { return n_ == 0; }
    size_t size() const { return n_; }
    void set(size_t i, size_t v) {
        if ((i & ((size_t(1) << kShift) - 1)) == 0) base_[i >> kShift] = v;
        const size_t d = v - base_[i >> kShift];
        if (d > UINT32_MAX) throw std::runtime_error("CompactOffsets: block span exceeds 32 bits");
        rel_[i] = static_cast<uint32_t>(d);
    }
    size_t operator[](size_t i) const { return base_[i >> kShift] + rel_[i]; }
    size_t front() const { return (*this)[0]; }
    size_t back() const { return (*this)[n_ - 1]; }
private:
    size_t n_ = 0;
    std::vector<uint32_t> rel_;
    std::vector<size_t> base_;
};

template<typename NID, typename NL>
class GraphMMapRW {
public:
    // Header is [node_count, max_degree]. max_degree == -1 means KSG variable
    // records; a non-negative degree means padded fixed SSG records. This
    // format carries no builder/source provenance.
    struct NeighborView {
        const NID* data = nullptr;
        NL count = 0;
        const NID* begin() const { return data; }
        const NID* end() const { return data + count; }
        bool empty() const { return count == 0; }
        size_t size() const { return count; }
    };
    enum class Mode { ReadOnly, ReadWrite };
    GraphMMapRW(const GraphMMapRW&) = delete;
    GraphMMapRW& operator=(const GraphMMapRW&) = delete;
    GraphMMapRW(GraphMMapRW&& other) noexcept {
        move_assign(std::move(other));
    }
    GraphMMapRW& operator=(GraphMMapRW&& other) noexcept {
        cleanup();
        move_assign(std::move(other));
        return *this;
    }

    GraphMMapRW(const std::string& filename, NL size = 0, NL max_degree = 0, Mode mode = Mode::ReadOnly, size_t total_degree = 0)
        : filename_(filename), mode_(mode) {
        
        if (mode_ == Mode::ReadOnly) {
            fd_ = open(filename.c_str(), O_RDONLY);
            if (fd_ == -1) throw std::runtime_error("[GraphMMapRW]Failed to open file: " + filename);
            struct stat sb;
            if (fstat(fd_, &sb) == -1) { close(fd_); throw std::runtime_error("[GraphMMapRW]Failed to stat file: " + filename); }
            mmap_len_ = sb.st_size;
            mmap_ptr_ = (char*)mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (mmap_ptr_ == MAP_FAILED) { close(fd_); mmap_ptr_ = nullptr; throw std::runtime_error("[GraphMMapRW]Failed to mmap file: " + filename); }
            std::memcpy(&size_, mmap_ptr_, sizeof(NL));
            std::memcpy(&max_degree_, mmap_ptr_ + sizeof(NL), sizeof(NL));
            meta_data_size_ = sizeof(NL) + sizeof(NL);
            if(max_degree_ == -1) node_size_ = -1;
            else node_size_ = static_cast<size_t>(max_degree_ )* sizeof(NID) + sizeof(NL);
        } else {
            if (size == 0 || max_degree == 0) throw std::invalid_argument("[GraphMMapRW]Write mode must specify size/max_degree!");
            size_ = size;
            max_degree_ = max_degree;
            if(max_degree_ == -1) node_size_ = -1;
            else node_size_ = static_cast<size_t>(max_degree_ )* sizeof(NID) + sizeof(NL);
            meta_data_size_ = sizeof(NL) + sizeof(NL);
            if(max_degree_ == -1) file_bytes_ = total_degree * sizeof(NID) + sizeof(NL) * size_ + meta_data_size_;
            else file_bytes_ = meta_data_size_ + size_ * node_size_;
            fd_ = open(filename.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
            if (fd_ == -1) throw std::runtime_error("[GraphMMapRW]Failed to open file for writing: " + filename);
            if (ftruncate(fd_, file_bytes_) == -1) { close(fd_); throw std::runtime_error("[GraphMMapRW]Failed to resize file: " + filename); }
            mmap_ptr_ = (char*)mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
            if (mmap_ptr_ == MAP_FAILED) { close(fd_); throw std::runtime_error("[GraphMMapRW]Failed to mmap file for writing: " + filename); }
            std::memcpy(mmap_ptr_, &size_, sizeof(NL));
            std::memcpy(mmap_ptr_ + sizeof(NL), &max_degree_, sizeof(NL));
            mmap_len_ = file_bytes_;
        }
    }
    ~GraphMMapRW() { cleanup(); }


    std::vector<NID> get_neighbors(NL nodex) const {

        if (!mmap_ptr_) throw std::runtime_error("[get_neighbors]MMap not initialized");
        if (nodex >= size_) throw std::out_of_range("[get_neighbors]Node index out of range.");
        if(max_degree_ == -1 && node_offsets_.empty()) {
            throw std::runtime_error("[get_neighbors][max_degree=-1]Node offsets not initialized. Please call load_node_offset() first.");
        }
        if (max_degree_ == -1){
            if(graph_.empty()){
                size_t nodex_offset = node_offsets_[nodex];
                size_t neighbor_count = node_offsets_[static_cast<size_t>(nodex) + 1] - nodex_offset;
                size_t offset = meta_data_size_ + static_cast<size_t>(nodex) * sizeof(NL) + nodex_offset * sizeof(NID);
                NL stored_neighbor_count;
                std::memcpy(&stored_neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                if (static_cast<size_t>(stored_neighbor_count) != neighbor_count) {
                    throw std::runtime_error("[get_neighbors]Neighbor offset/count mismatch.");
                }
                std::vector<NID> neighbors(neighbor_count);
                std::memcpy(neighbors.data(), mmap_ptr_ + offset + sizeof(NL), neighbor_count * sizeof(NID));
                return neighbors;
            }
            size_t nodex_offset = node_offsets_[nodex];
            size_t neighbor_count = node_offsets_[nodex + 1] - nodex_offset;
            if (nodex_offset + neighbor_count > graph_.size()) throw std::out_of_range("[get_neighbors]Node offset out of range.");
            std::vector<NID> neighbors(neighbor_count);
            std::memcpy(neighbors.data(), graph_.data() + nodex_offset, neighbor_count * sizeof(NID));
            return neighbors;
        }
        size_t offset = meta_data_size_ + nodex * node_size_;
        NL neighbor_count;
        std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
        if (neighbor_count < 0 || neighbor_count > static_cast<NL>(max_degree_)) throw std::runtime_error("[get_neighbors]Invalid neighbor count: " + std::to_string(neighbor_count));
        size_t data_offset = offset + sizeof(NL);
        std::vector<NID> neighbors(neighbor_count);
        std::memcpy(neighbors.data(), mmap_ptr_ + data_offset, neighbor_count * sizeof(NID));
        return neighbors;
    }

    NeighborView get_neighbors_view(NL nodex) const {
        if (ssd_fd_ >= 0) return ssd_neighbors_view(nodex);
        if (compact_) {   // compressed heap adjacency: decoded into a thread-local buffer (valid until the next call on this thread)
            uint32_t c; const uint32_t* d = compact_->neighbors(static_cast<uint32_t>(nodex), c);
            return NeighborView{reinterpret_cast<const NID*>(d), static_cast<NL>(c)};
        }
        if (!mmap_ptr_) throw std::runtime_error("[get_neighbors_view]MMap not initialized");
        if (nodex >= size_) throw std::out_of_range("[get_neighbors_view]Node index out of range.");
        if (max_degree_ == -1) {
            if (node_offsets_.empty()) {
                throw std::runtime_error("[get_neighbors_view][max_degree=-1]Node offsets not initialized. Please call load_node_offset() or load_graph() first.");
            }
            size_t nodex_offset = node_offsets_[nodex];
            size_t neighbor_count = node_offsets_[nodex + 1] - nodex_offset;
            if (!graph_.empty()) {
                if (nodex_offset + neighbor_count > graph_.size()) throw std::out_of_range("[get_neighbors_view]Node offset out of graph cache.");
                return NeighborView{graph_.data() + nodex_offset, static_cast<NL>(neighbor_count)};
            }
            if (fixed_block_cache_enabled_ && fixed_block_cache_blocks_ > 0 && fixed_block_cache_nodes_ > 0) {
                return cached_variable_neighbors_view(nodex);
            }
            size_t offset = meta_data_size_ + nodex * sizeof(NL) + static_cast<size_t>(nodex_offset) * sizeof(NID);
            NL stored_neighbor_count;
            std::memcpy(&stored_neighbor_count, mmap_ptr_ + offset, sizeof(NL));
            if (stored_neighbor_count != neighbor_count) throw std::runtime_error("[get_neighbors_view]Neighbor offset/count mismatch.");
            return NeighborView{reinterpret_cast<const NID*>(mmap_ptr_ + offset + sizeof(NL)), static_cast<NL>(neighbor_count)};
        }

        if (fixed_block_cache_enabled_ && fixed_block_cache_blocks_ > 0 && fixed_block_cache_nodes_ > 0) {
            return cached_fixed_neighbors_view(nodex);
        }
        size_t offset = meta_data_size_ + static_cast<size_t>(nodex) * node_size_;
        NL neighbor_count;
        std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
        if (neighbor_count > static_cast<NL>(max_degree_)) throw std::runtime_error("[get_neighbors_view]Invalid neighbor count: " + std::to_string(neighbor_count));
        return NeighborView{reinterpret_cast<const NID*>(mmap_ptr_ + offset + sizeof(NL)), neighbor_count};
    }

    void discard_pages() const noexcept {
        if (resident_) return;  // anonymous copy: MADV_DONTNEED would zero it
        if (mmap_ptr_ && mmap_len_ > 0) (void)madvise(mmap_ptr_, mmap_len_, MADV_DONTNEED);
    }

    // so that the in-memory layer is not evicted by SSD-layer page-cache traffic
    // and is mapped with 2 MB pages. Uses parallel pread; returns seconds spent.
    double make_resident(bool hugepage, int threads) {
        if (mode_ != Mode::ReadOnly || resident_ || !mmap_ptr_ || mmap_len_ == 0) return 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        void* p = mmap(nullptr, mmap_len_, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) throw std::runtime_error("[GraphMMapRW]resident mmap failed: " + filename_);
#ifdef MADV_HUGEPAGE
        if (hugepage) (void)madvise(p, mmap_len_, MADV_HUGEPAGE);
#endif
        const size_t chunk = size_t{64} << 20;
        const size_t nchunks = (mmap_len_ + chunk - 1) / chunk;
        bool failed = false;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(std::max(1, threads))
        for (long long c = 0; c < static_cast<long long>(nchunks); ++c) {
            const size_t off = static_cast<size_t>(c) * chunk;
            const size_t len = std::min(chunk, mmap_len_ - off);
            size_t done = 0;
            while (done < len) {
                const ssize_t r = pread(fd_, static_cast<char*>(p) + off + done, len - done,
                                        static_cast<off_t>(off + done));
                if (r <= 0) { failed = true; break; }
                done += static_cast<size_t>(r);
            }
        }
        if (failed) { munmap(p, mmap_len_); throw std::runtime_error("[GraphMMapRW]resident pread failed: " + filename_); }
        munmap(mmap_ptr_, mmap_len_);
        mmap_ptr_ = static_cast<char*>(p);
        resident_ = true;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    bool resident() const { return resident_; }
    // low-memory variant: serve neighbour lists from a compressed heap copy (include/compact_adj.h); no node offsets are needed
    void set_compact(std::unique_ptr<ksg::CompactAdj> c) { compact_ = std::move(c); }
    // KSG-on-SSD variant (--mem-graph-ssd): neighbour lists are read from the graph file with O_DIRECT pread on every access
    // (the 4 KB pages that hold the list; no cache), only the node offsets stay in memory. Variable-degree graphs only.
    void set_ssd(const std::string& path) {
        if (max_degree_ != static_cast<NL>(-1) || node_offsets_.empty()) throw std::runtime_error("[set_ssd] needs a variable-degree graph with loaded offsets");
        ssd_fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (ssd_fd_ < 0) throw std::runtime_error("[set_ssd] cannot open with O_DIRECT: " + path);
        int fd = ::open(path.c_str(), O_RDONLY); if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); ::close(fd); }
        if (mmap_ptr_ && mmap_len_) madvise(mmap_ptr_, mmap_len_, MADV_DONTNEED);
    }
    uint64_t ssd_pages_read() const { return ssd_pages_.load(std::memory_order_relaxed); }
    uint64_t ssd_reads() const { return ssd_reads_.load(std::memory_order_relaxed); }
    NeighborView ssd_neighbors_view(NL nodex) const {
        thread_local void* buf = nullptr; thread_local size_t cap = 0;
        const size_t o0 = node_offsets_[nodex], cnt = node_offsets_[static_cast<size_t>(nodex) + 1] - o0;
        const size_t off = meta_data_size_ + static_cast<size_t>(nodex) * sizeof(NL) + o0 * sizeof(NID);   // count word
        const size_t a = off & ~size_t(4095), e = (off + sizeof(NL) + cnt * sizeof(NID) + 4095) & ~size_t(4095);
        if (e - a > cap) { free(buf); cap = e - a; if (posix_memalign(&buf, 4096, cap) != 0) throw std::runtime_error("[ssd] alloc"); }
        for (size_t done = 0; done < e - a;) {
            const ssize_t r = ::pread(ssd_fd_, static_cast<char*>(buf) + done, e - a - done, static_cast<off_t>(a + done));
            if (r <= 0) { if (a + done >= mmap_len_) break; throw std::runtime_error("[ssd] pread failed"); }
            done += static_cast<size_t>(r);
        }
        ssd_pages_.fetch_add((e - a) / 4096, std::memory_order_relaxed); ssd_reads_.fetch_add(1, std::memory_order_relaxed);
        const char* p = static_cast<const char*>(buf) + (off - a);
        return NeighborView{reinterpret_cast<const NID*>(p + sizeof(NL)), static_cast<NL>(cnt)};
    }
    const ksg::CompactAdj* compact() const { return compact_.get(); }

    void discard_node_range(size_t begin, size_t end) const noexcept {
        if (resident_ || !mmap_ptr_ || begin >= end || max_degree_ < 0) return;
        begin = std::min(begin, static_cast<size_t>(size_));
        end = std::min(end, static_cast<size_t>(size_));
        if (begin >= end) return;
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        const uintptr_t first = reinterpret_cast<uintptr_t>(mmap_ptr_ + meta_data_size_ + begin * node_size_);
        const uintptr_t last = reinterpret_cast<uintptr_t>(mmap_ptr_ + meta_data_size_ + end * node_size_);
        const uintptr_t aligned = first & ~(static_cast<uintptr_t>(page) - 1);
        (void)madvise(reinterpret_cast<void*>(aligned), last - aligned, MADV_DONTNEED);
    }

    void load_node_offset() {
        if (!mmap_ptr_) throw std::runtime_error("[load_node_offset]MMap not initialized");
        if (max_degree_ == -1) {
            size_t offset = meta_data_size_;
            node_offsets_.resize(size_+1);
            size_t current_node_offset = 0;
            for(size_t i = 0; i < size_; ++i) {
                node_offsets_.set(i, current_node_offset);
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                offset += sizeof(NL) + neighbor_count * sizeof(NID);
                current_node_offset += neighbor_count;
            }
            node_offsets_.set(size_, current_node_offset);
        }
        else{
            node_offsets_.resize(size_+1);
            size_t current_node_offset = 0;
            for (size_t i = 0; i < size_; ++i) {
                node_offsets_.set(i, current_node_offset);
                size_t offset = meta_data_size_ + i * node_size_;
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                offset += sizeof(NL) + neighbor_count * sizeof(NID);
                current_node_offset += neighbor_count;
            }
            node_offsets_.set(size_, current_node_offset);
        }
    }

    // Returns true when offsets were restored from a valid sidecar. On a miss,
    // scans the graph once and atomically creates the sidecar for future starts.
    bool load_node_offset(const std::string& cache_file) {
        if (max_degree_ != -1 || cache_file.empty()) {
            load_node_offset();
            return false;
        }
        if (load_node_offset_cache(cache_file)) return true;
        load_node_offset();
        save_node_offset_cache(cache_file);
        return false;
    }

    void load_graph(std::vector<std::vector<NID>>& graph) const {
        graph.resize(size_);
        if(max_degree_ == -1) {
            size_t offset = meta_data_size_;
            for(size_t i = 0; i < size_; ++i) {
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                if (neighbor_count < 0) throw std::runtime_error("[load_graph]Invalid neighbor count for node " + std::to_string(i) + ": " + std::to_string(neighbor_count));
                std::vector<NID> neighbors(neighbor_count);
                std::memcpy(neighbors.data(), mmap_ptr_ + offset + sizeof(NL), neighbor_count * sizeof(NID));
                graph[i] = std::move(neighbors);
                offset += sizeof(NL) + neighbor_count * sizeof(NID);
            }
        }
        else{
            for (size_t i = 0; i < size_; ++i) {
                size_t offset = meta_data_size_ + i * node_size_;
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                std::vector<NID> neighbors(neighbor_count);
                std::memcpy(neighbors.data(), mmap_ptr_ + offset + sizeof(NL), neighbor_count * sizeof(NID));
                graph[i] = std::move(neighbors);
            }
        }
        
    }

    void load_graph() const {
        if(max_degree_ == -1) {
            size_t offset = meta_data_size_;
            size_t total_neighbors = (mmap_len_ - meta_data_size_ - size_ * sizeof(NL)) / sizeof(NID);
            graph_.resize(total_neighbors);
            node_offsets_.resize(size_+1);
            size_t current_node_offset = 0;
            for(size_t i = 0; i < size_; ++i) {
                node_offsets_.set(i, current_node_offset);
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                std::memcpy(graph_.data() + current_node_offset, mmap_ptr_ + offset + sizeof(NL), neighbor_count * sizeof(NID));
                offset += sizeof(NL) + neighbor_count * sizeof(NID);
                current_node_offset += neighbor_count;
            }
            node_offsets_.set(size_, current_node_offset);
        }
        else{
            graph_.resize(max_degree_ * size_);
            node_offsets_.resize(size_+1);
            size_t current_node_offset = 0;
            for (size_t i = 0; i < size_; ++i) {
                node_offsets_.set(i, current_node_offset);
                size_t offset = meta_data_size_ + i * node_size_;
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                std::memcpy(graph_.data() + current_node_offset, mmap_ptr_ + offset + sizeof(NL), neighbor_count * sizeof(NID));
                current_node_offset += neighbor_count;
            }
            node_offsets_.set(size_, current_node_offset);
        }
        
    }

    void update_node(NL nodex, const std::vector<NID>& new_neighbors) {
        if (mode_ != Mode::ReadWrite) throw std::runtime_error("[update_node]MMap not initialized");
        if (!mmap_ptr_) throw std::runtime_error("[update_node]MMap not initialized");
        if (nodex >= size_) throw std::out_of_range("[update_node]Node index out of range.");
        if (max_degree_ == -1) throw std::runtime_error("[update_node]TODO:update neighbors when max_degree=-1.");
        try
        {
            size_t offset = meta_data_size_ + nodex * node_size_;
            NL neighbor_count = new_neighbors.size();
            if (neighbor_count > static_cast<NL>(max_degree_)) {
                throw std::out_of_range("[update_node]Neighbor count exceeds max_degree. count:" + std::to_string(neighbor_count) + ";max_degree:" + std::to_string(max_degree_));
            }
            if(offset > mmap_len_ || offset + sizeof(NL) + neighbor_count * sizeof(NID) > mmap_len_ )  throw std::out_of_range("[update_node]Offset out of mmap_len_.offset:" + std::to_string(offset) + ";mmap_len_:" + std::to_string(mmap_len_));
            
            std::memcpy(mmap_ptr_ + offset, &neighbor_count, sizeof(NL));
            char* neighbors_ptr = mmap_ptr_ + offset + sizeof(NL);
            std::memcpy(neighbors_ptr, new_neighbors.data(), neighbor_count * sizeof(NID));
            fill_fixed_degree_padding(neighbors_ptr, neighbor_count);
        }
        catch(const std::exception& e)
        {
            std::cerr << "[update_node]Error: " << e.what() << '\n';
            throw;
        }
    }


    void save_graph(const std::vector<std::vector<NID>>& graph) {
        if (mode_ != Mode::ReadWrite) throw std::runtime_error("[save_graph]MMap not initialized");
        if (!mmap_ptr_) throw std::runtime_error("[save_graph]MMap not initialized");
        if(max_degree_ == -1){
            size_t offset = meta_data_size_;
            for (NL nodex = 0; nodex < size_; ++nodex) {
                NL neighbor_count = graph[nodex].size();
                if (neighbor_count < 0) throw std::runtime_error("[save_graph]Invalid neighbor count for node " + std::to_string(nodex) + ": " + std::to_string(neighbor_count));
                std::memcpy(mmap_ptr_ + offset, &neighbor_count, sizeof(NL));
                std::memcpy(mmap_ptr_ + offset + sizeof(NL), graph[nodex].data(), neighbor_count * sizeof(NID));
                offset += sizeof(NL) + neighbor_count * sizeof(NID);
            }
            return;
        }
        for (NL nodex = 0; nodex < size_; ++nodex) {
            size_t offset = meta_data_size_ + nodex * node_size_;
            NL neighbor_count = graph[nodex].size();
            if (neighbor_count > static_cast<NL>(max_degree_)) {
                throw std::out_of_range("[save_graph]Neighbor count exceeds max_degree for node " + std::to_string(nodex));
            }
            std::memcpy(mmap_ptr_ + offset, &neighbor_count, sizeof(NL));
            char* neighbors_ptr = mmap_ptr_ + offset + sizeof(NL);
            std::memcpy(neighbors_ptr, graph[nodex].data(), neighbor_count * sizeof(NID));
            fill_fixed_degree_padding(neighbors_ptr, neighbor_count);
        }
    }

    void flush() {
        if (mmap_ptr_ && mode_ == Mode::ReadWrite) msync(mmap_ptr_, mmap_len_, MS_SYNC);
    }

    std::string filename() const { return filename_; }
    NL size() const { return size_; }
    NL max_degree() const { return max_degree_; }
    uint32_t meta_data_size() const { return meta_data_size_; }
    size_t file_size() const { return mmap_len_; }
    Mode mode() const { return mode_; }
    void enable_block_cache(size_t block_nodes, size_t block_count) const {
        fixed_block_cache_nodes_ = std::max<size_t>(1, block_nodes);
        fixed_block_cache_blocks_ = block_count;
        fixed_block_cache_enabled_ = block_count > 0;
        if (fixed_block_cache_enabled_) advise_random_read_access(fd_, mmap_ptr_, mmap_len_);
    }
    void enable_fixed_block_cache(size_t block_nodes, size_t block_count) const {
        enable_block_cache(block_nodes, block_count);
    }
    void advise_random() const { advise_random_read_access(fd_, mmap_ptr_, mmap_len_); }
    bool fixed_block_cache_enabled() const { return fixed_block_cache_enabled_; }
    size_t node_record_offset(size_t id) const {
        if (id >= static_cast<size_t>(size_)) throw std::out_of_range("node_record_offset id");
        if (max_degree_ >= 0) return meta_data_size_ + id * node_size_;
        if (node_offsets_.empty()) throw std::runtime_error("variable graph offsets are not loaded");
        return meta_data_size_ + id * sizeof(NL) + node_offsets_[id] * sizeof(NID);
    }
    size_t node_record_bytes(size_t id) const {
        if (id >= static_cast<size_t>(size_)) throw std::out_of_range("node_record_bytes id");
        if (max_degree_ >= 0) return node_size_;
        if (node_offsets_.empty()) throw std::runtime_error("variable graph offsets are not loaded");
        return sizeof(NL) + (node_offsets_[id + 1] - node_offsets_[id]) * sizeof(NID);
    }
    size_t total_neighbors() const {
        if (node_offsets_.empty()) throw std::runtime_error("[total_neighbors]Node offsets not initialized. Please call load_node_offset() first.");
        return node_offsets_.back();
    }
private:
    struct ThreadGraphBlock {
        const GraphMMapRW<NID, NL>* owner = nullptr;
        size_t block_id = std::numeric_limits<size_t>::max();
        size_t valid_nodes = 0;
        std::vector<char> bytes;
        bool valid = false;
    };
    struct ThreadGraphCache {
        size_t configured_blocks = 0;
        size_t configured_block_nodes = 0;
        size_t next_slot = 0;
        std::vector<ThreadGraphBlock> blocks;
    };

    int fd_ = -1;
    char* mmap_ptr_ = nullptr;
    size_t mmap_len_ = 0;
    size_t file_bytes_ = 0;
    std::string filename_;
    NL size_ = 0;
    int max_degree_ = 0;
    size_t node_size_ = 0;
    size_t meta_data_size_ = 0;
    Mode mode_ = Mode::ReadOnly;
    mutable std::vector<NID> graph_;
    mutable CompactOffsets node_offsets_;
    mutable bool fixed_block_cache_enabled_ = false;
    bool resident_ = false;
    std::unique_ptr<ksg::CompactAdj> compact_;
    int ssd_fd_ = -1; mutable std::atomic<uint64_t> ssd_pages_{0}, ssd_reads_{0};
    mutable size_t fixed_block_cache_nodes_ = 0;
    mutable size_t fixed_block_cache_blocks_ = 0;

    static constexpr char offset_cache_magic_[8] = {'K', 'S', 'G', 'O', 'F', 'F', '0', '1'};

    bool graph_stat(uint64_t& bytes, int64_t& mtime_seconds) const noexcept {
        struct stat sb;
        if (fstat(fd_, &sb) != 0) return false;
        bytes = static_cast<uint64_t>(sb.st_size);
        mtime_seconds = static_cast<int64_t>(sb.st_mtim.tv_sec);
        return true;
    }

    bool load_node_offset_cache(const std::string& cache_file) {
        std::ifstream input(cache_file, std::ios::binary);
        if (!input) return false;
        char magic[8];
        uint32_t version = 0;
        uint64_t graph_size = 0, graph_bytes = 0, offset_count = 0;
        int64_t graph_mtime = 0;
        input.read(magic, sizeof(magic));
        input.read(reinterpret_cast<char*>(&version), sizeof(version));
        input.read(reinterpret_cast<char*>(&graph_size), sizeof(graph_size));
        input.read(reinterpret_cast<char*>(&graph_bytes), sizeof(graph_bytes));
        input.read(reinterpret_cast<char*>(&graph_mtime), sizeof(graph_mtime));
        input.read(reinterpret_cast<char*>(&offset_count), sizeof(offset_count));
        uint64_t current_bytes = 0;
        int64_t current_mtime = 0;
        if (!input || std::memcmp(magic, offset_cache_magic_, sizeof(magic)) != 0 || version != 1 ||
            graph_size != static_cast<uint64_t>(size_) || offset_count != static_cast<uint64_t>(size_) + 1 ||
            !graph_stat(current_bytes, current_mtime) || graph_bytes != current_bytes || graph_mtime != current_mtime) {
            return false;
        }

        node_offsets_.resize(static_cast<size_t>(offset_count));
        {
            std::vector<uint64_t> chunk(1 << 20);
            for (size_t i0 = 0; i0 < node_offsets_.size(); i0 += chunk.size()) {
                const size_t c = std::min(chunk.size(), node_offsets_.size() - i0);
                input.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(c * sizeof(uint64_t)));
                if (!input) break;
                for (size_t k = 0; k < c; ++k) node_offsets_.set(i0 + k, static_cast<size_t>(chunk[k]));
            }
        }
        const size_t expected_edges =
            (mmap_len_ - meta_data_size_ - static_cast<size_t>(size_) * sizeof(NL)) / sizeof(NID);
        if (!input || node_offsets_.front() != 0 || node_offsets_.back() != expected_edges) {
            node_offsets_.clear();
            return false;
        }
        return true;
    }

    void save_node_offset_cache(const std::string& cache_file) const noexcept {
        if (node_offsets_.empty()) return;
        try {
            uint64_t graph_bytes = 0;
            int64_t graph_mtime = 0;
            if (!graph_stat(graph_bytes, graph_mtime)) return;
            const std::string temporary = cache_file + ".tmp." + std::to_string(static_cast<long long>(getpid()));
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) return;
            const uint32_t version = 1;
            const uint64_t graph_size = static_cast<uint64_t>(size_);
            const uint64_t offset_count = static_cast<uint64_t>(node_offsets_.size());
            output.write(offset_cache_magic_, sizeof(offset_cache_magic_));
            output.write(reinterpret_cast<const char*>(&version), sizeof(version));
            output.write(reinterpret_cast<const char*>(&graph_size), sizeof(graph_size));
            output.write(reinterpret_cast<const char*>(&graph_bytes), sizeof(graph_bytes));
            output.write(reinterpret_cast<const char*>(&graph_mtime), sizeof(graph_mtime));
            output.write(reinterpret_cast<const char*>(&offset_count), sizeof(offset_count));
            for (size_t i = 0; i < node_offsets_.size(); ++i) {
                const uint64_t stored = static_cast<uint64_t>(node_offsets_[i]);
                output.write(reinterpret_cast<const char*>(&stored), sizeof(stored));
            }
            output.close();
            if (output) {
                if (std::rename(temporary.c_str(), cache_file.c_str()) != 0) std::remove(temporary.c_str());
            } else {
                std::remove(temporary.c_str());
            }
        } catch (...) {
            // Cache creation is an optimization; graph loading remains valid.
        }
    }

    void fill_fixed_degree_padding(char* neighbors_ptr, NL neighbor_count) {
        if (max_degree_ <= 0) return;
        const size_t count = static_cast<size_t>(neighbor_count);
        const size_t degree = static_cast<size_t>(max_degree_);
        if (count >= degree) return;
        NID* padding_begin = reinterpret_cast<NID*>(neighbors_ptr) + count;
        std::fill_n(padding_begin, degree - count, static_cast<NID>(-1));
    }

    bool read_exact(void* dst, size_t bytes, size_t offset) const noexcept {
        char* out = static_cast<char*>(dst);
        size_t done = 0;
        while (done < bytes) {
            ssize_t ret = pread(fd_, out + done, bytes - done, static_cast<off_t>(offset + done));
            if (ret <= 0) return false;
            done += static_cast<size_t>(ret);
        }
        return true;
    }

    NeighborView cached_variable_neighbors_view(NL nodex) const {
        static thread_local std::unordered_map<const GraphMMapRW<NID, NL>*, ThreadGraphCache> thread_caches;
        ThreadGraphCache& cache = thread_caches[this];
        if (cache.configured_blocks != fixed_block_cache_blocks_ || cache.configured_block_nodes != fixed_block_cache_nodes_) {
            cache.configured_blocks = fixed_block_cache_blocks_;
            cache.configured_block_nodes = fixed_block_cache_nodes_;
            cache.next_slot = 0;
            cache.blocks.clear();
            cache.blocks.resize(fixed_block_cache_blocks_);
        }
        if (cache.blocks.empty()) {
            size_t nodex_offset = node_offsets_[nodex];
            size_t offset = meta_data_size_ + static_cast<size_t>(nodex) * sizeof(NL) + nodex_offset * sizeof(NID);
            NL neighbor_count;
            std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
            return NeighborView{reinterpret_cast<const NID*>(mmap_ptr_ + offset + sizeof(NL)), neighbor_count};
        }

        const size_t block_id = static_cast<size_t>(nodex) / fixed_block_cache_nodes_;
        ThreadGraphBlock* found = nullptr;
        for (auto& block : cache.blocks) {
            if (block.valid && block.owner == this && block.block_id == block_id) {
                found = &block;
                break;
            }
        }
        if (!found) {
            ThreadGraphBlock& block = cache.blocks[cache.next_slot++ % cache.blocks.size()];
            const size_t start_node = block_id * fixed_block_cache_nodes_;
            const size_t end_node = std::min(start_node + fixed_block_cache_nodes_, static_cast<size_t>(size_));
            const size_t valid_nodes = end_node - start_node;
            const size_t start_edge = node_offsets_[start_node];
            const size_t end_edge = node_offsets_[end_node];
            const size_t bytes = valid_nodes * sizeof(NL) + (end_edge - start_edge) * sizeof(NID);
            const size_t offset = meta_data_size_ + start_node * sizeof(NL) + start_edge * sizeof(NID);
            block.bytes.resize(bytes);
            if (!read_exact(block.bytes.data(), bytes, offset)) {
                block.valid = false;
                size_t nodex_offset = node_offsets_[nodex];
                size_t mmap_offset = meta_data_size_ + static_cast<size_t>(nodex) * sizeof(NL) + nodex_offset * sizeof(NID);
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + mmap_offset, sizeof(NL));
                return NeighborView{reinterpret_cast<const NID*>(mmap_ptr_ + mmap_offset + sizeof(NL)), neighbor_count};
            }
            block.owner = this;
            block.block_id = block_id;
            block.valid_nodes = valid_nodes;
            block.valid = true;
            found = &block;
        }

        const size_t start_node = block_id * fixed_block_cache_nodes_;
        const size_t node_in_block = static_cast<size_t>(nodex) - start_node;
        const size_t local_edge = node_offsets_[nodex] - node_offsets_[start_node];
        const size_t node_offset = node_in_block * sizeof(NL) + local_edge * sizeof(NID);
        NL neighbor_count;
        std::memcpy(&neighbor_count, found->bytes.data() + node_offset, sizeof(NL));
        const size_t expected_count = node_offsets_[static_cast<size_t>(nodex) + 1] - node_offsets_[nodex];
        if (static_cast<size_t>(neighbor_count) != expected_count) {
            throw std::runtime_error("[cached_variable_neighbors_view]Neighbor offset/count mismatch.");
        }
        return NeighborView{reinterpret_cast<const NID*>(found->bytes.data() + node_offset + sizeof(NL)), neighbor_count};
    }

    NeighborView cached_fixed_neighbors_view(NL nodex) const {
        static thread_local std::unordered_map<const GraphMMapRW<NID, NL>*, ThreadGraphCache> thread_caches;
        ThreadGraphCache& cache = thread_caches[this];
        if (cache.configured_blocks != fixed_block_cache_blocks_ || cache.configured_block_nodes != fixed_block_cache_nodes_) {
            cache.configured_blocks = fixed_block_cache_blocks_;
            cache.configured_block_nodes = fixed_block_cache_nodes_;
            cache.next_slot = 0;
            cache.blocks.clear();
            cache.blocks.resize(fixed_block_cache_blocks_);
        }
        if (cache.blocks.empty()) {
            size_t offset = meta_data_size_ + static_cast<size_t>(nodex) * node_size_;
            NL neighbor_count;
            std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
            return NeighborView{reinterpret_cast<const NID*>(mmap_ptr_ + offset + sizeof(NL)), neighbor_count};
        }

        const size_t block_id = static_cast<size_t>(nodex) / fixed_block_cache_nodes_;
        ThreadGraphBlock* found = nullptr;
        for (auto& block : cache.blocks) {
            if (block.valid && block.owner == this && block.block_id == block_id) {
                found = &block;
                break;
            }
        }
        if (!found) {
            ThreadGraphBlock& block = cache.blocks[cache.next_slot++ % cache.blocks.size()];
            const size_t start_node = block_id * fixed_block_cache_nodes_;
            const size_t valid_nodes = std::min(fixed_block_cache_nodes_, static_cast<size_t>(size_) - start_node);
            const size_t bytes = valid_nodes * node_size_;
            block.bytes.resize(bytes);
            const size_t offset = meta_data_size_ + start_node * node_size_;
            if (!read_exact(block.bytes.data(), bytes, offset)) {
                block.valid = false;
                size_t mmap_offset = meta_data_size_ + static_cast<size_t>(nodex) * node_size_;
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + mmap_offset, sizeof(NL));
                return NeighborView{reinterpret_cast<const NID*>(mmap_ptr_ + mmap_offset + sizeof(NL)), neighbor_count};
            }
            block.owner = this;
            block.block_id = block_id;
            block.valid_nodes = valid_nodes;
            block.valid = true;
            found = &block;
        }

        const size_t node_in_block = static_cast<size_t>(nodex) - block_id * fixed_block_cache_nodes_;
        const size_t node_offset = node_in_block * node_size_;
        NL neighbor_count;
        std::memcpy(&neighbor_count, found->bytes.data() + node_offset, sizeof(NL));
        if (neighbor_count > static_cast<NL>(max_degree_)) throw std::runtime_error("[cached_fixed_neighbors_view]Invalid neighbor count: " + std::to_string(neighbor_count));
        return NeighborView{reinterpret_cast<const NID*>(found->bytes.data() + node_offset + sizeof(NL)), neighbor_count};
    }

    void cleanup() {
        if (mmap_ptr_ && mode_ == Mode::ReadWrite) msync(mmap_ptr_, mmap_len_, MS_SYNC);
        if (mmap_ptr_) munmap(mmap_ptr_, mmap_len_);
        if (fd_ != -1) close(fd_);
        mmap_ptr_ = nullptr; fd_ = -1; mmap_len_ = 0; file_bytes_ = 0;
    }
    void move_assign(GraphMMapRW&& other) {
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        file_bytes_ = other.file_bytes_;
        filename_ = std::move(other.filename_);
        size_ = other.size_;
        max_degree_ = other.max_degree_;
        node_size_ = other.node_size_;
        meta_data_size_ = other.meta_data_size_;
        mode_ = other.mode_;
        graph_ = std::move(other.graph_);
        node_offsets_ = std::move(other.node_offsets_);
        fixed_block_cache_enabled_ = other.fixed_block_cache_enabled_;
        fixed_block_cache_nodes_ = other.fixed_block_cache_nodes_;
        fixed_block_cache_blocks_ = other.fixed_block_cache_blocks_;
        resident_ = other.resident_; other.resident_ = false;
    }
};













template<typename NID, typename NL>
class GraphMMapIO {
public:
    GraphMMapIO(const GraphMMapIO&) = delete;
    GraphMMapIO& operator=(const GraphMMapIO&) = delete;
    GraphMMapIO(GraphMMapIO&& other) noexcept {
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        filename_ = std::move(other.filename_);
        size_ = other.size_;
        max_degree_ = other.max_degree_;
        meta_data_size_ = other.meta_data_size_;
    }
    GraphMMapIO& operator=(GraphMMapIO&& other) noexcept {
        cleanup();
        fd_ = other.fd_; other.fd_ = -1;
        mmap_ptr_ = other.mmap_ptr_; other.mmap_ptr_ = nullptr;
        mmap_len_ = other.mmap_len_;
        filename_ = std::move(other.filename_);
        size_ = other.size_;
        max_degree_ = other.max_degree_;
        meta_data_size_ = other.meta_data_size_;
        return *this;
    }

    explicit GraphMMapIO(const std::string& filename) : filename_(filename) {
        fd_ = open(filename.c_str(), O_RDONLY);
        if (fd_ == -1) throw std::runtime_error("[GraphMMapIO]Failed to open file: " + filename);
        struct stat sb;
        if (fstat(fd_, &sb) == -1) { close(fd_); throw std::runtime_error("[GraphMMapIO]Failed to stat file: " + filename); }
        mmap_len_ = sb.st_size;
        mmap_ptr_ = static_cast<char*>(mmap(nullptr, mmap_len_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mmap_ptr_ == MAP_FAILED) { close(fd_); mmap_ptr_ = nullptr; throw std::runtime_error("[GraphMMapIO]Failed to mmap file: " + filename); }
        std::memcpy(&size_, mmap_ptr_, sizeof(NL));
        std::memcpy(&max_degree_, mmap_ptr_ + sizeof(NL), sizeof(NL));
        meta_data_size_ = sizeof(NL) + sizeof(NL);
        node_size_ = static_cast<uint32_t>(max_degree_ * sizeof(NID) + sizeof(NL));
    }
    ~GraphMMapIO() { cleanup(); }

    std::vector<NID> get_neighbors(NL nodex) const {
        if (!mmap_ptr_) throw std::runtime_error("[get_neighbors]MMap not initialized");
        if (nodex >= size_) throw std::out_of_range("[get_neighbors]Node index out of range.");
        if(max_degree_ == -1) throw std::runtime_error("[get_neighbors]Graph max_degree = -1, this is an indefinite-length graph, you must load the entire graph once.");
        size_t offset = meta_data_size_ + nodex * node_size_;
        NL neighbor_count;
        std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
        if (neighbor_count < 0 || neighbor_count > static_cast<NL>(max_degree_)) throw std::runtime_error("[get_neighbors]Invalid neighbor count: " + std::to_string(neighbor_count));
        size_t data_offset = offset + sizeof(NL);
        std::vector<NID> neighbors(max_degree_);
        std::memcpy(neighbors.data(), mmap_ptr_ + data_offset, max_degree_ * sizeof(NID));
        std::vector<NID> valid_neighbors;
        for (NID neighbor : neighbors) {
            if (neighbor != static_cast<NID>(-1)) valid_neighbors.push_back(neighbor);
        }
        return valid_neighbors;
    }

    void load_graph(std::vector<std::vector<NID>>& graph) const {
        graph.resize(size_);
        if(max_degree_ == -1){
            size_t offset = meta_data_size_;
            for(size_t i = 0; i < size_; ++i) {
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                if (neighbor_count < 0) throw std::runtime_error("[load_graph]Invalid neighbor count for node " + std::to_string(i) + ": " + std::to_string(neighbor_count));
                std::vector<NID> neighbors(neighbor_count);
                std::memcpy(neighbors.data(), mmap_ptr_ + offset + sizeof(NL), neighbor_count * sizeof(NID));
                graph[i] = std::move(neighbors);
                offset += sizeof(NL) + neighbor_count * sizeof(NID);
            }
        }
        else{
            for (size_t i = 0; i < size_; ++i) {
                size_t offset = meta_data_size_ + i * node_size_;
                NL neighbor_count;
                std::memcpy(&neighbor_count, mmap_ptr_ + offset, sizeof(NL));
                std::vector<NID> neighbors(max_degree_);
                std::memcpy(neighbors.data(), mmap_ptr_ + offset + sizeof(NL), max_degree_ * sizeof(NID));
                graph[i].assign(neighbors.begin(), neighbors.begin() + neighbor_count);
            }
        }
    }

    static void update_graph(const std::string& graph_file, NL nodex, const std::vector<NID>& new_neighbors, NL size, NL file_max_degree) {
        std::fstream ofs(graph_file, std::ios::in | std::ios::out | std::ios::binary);
        if (!ofs.is_open()) throw std::runtime_error("Failed to open graph file for updating: " + graph_file);
        if (nodex >= size) throw std::out_of_range("Node index out of range.");
        size_t offset = sizeof(NL) + sizeof(NL) + nodex * (file_max_degree * sizeof(NID) + sizeof(NL));
        ofs.seekp(offset, std::ios::beg);
        NL neighbor_count = new_neighbors.size();
        ofs.write(reinterpret_cast<const char*>(&neighbor_count), sizeof(NL));
        ofs.write(reinterpret_cast<const char*>(new_neighbors.data()), neighbor_count * sizeof(NID));
        if (neighbor_count < file_max_degree) {
            std::vector<NID> padding(file_max_degree - neighbor_count, static_cast<NID>(-1));
            ofs.write(reinterpret_cast<const char*>(padding.data()), (file_max_degree - neighbor_count) * sizeof(NID));
        }
    }

    static void save_graph(const std::vector<std::vector<NID>>& graph, const std::string& graph_file, NL size, NL max_degree) {
        std::ofstream ofs(graph_file, std::ios::binary);
        if (!ofs.is_open()) throw std::runtime_error("Failed to open graph file for writing: " + graph_file);
        ofs.write(reinterpret_cast<const char*>(&size), sizeof(NL));
        ofs.write(reinterpret_cast<const char*>(&max_degree), sizeof(NL));
        for (NL i = 0; i < size; ++i) {
            NL neighbor_count = graph[i].size();
            ofs.write(reinterpret_cast<const char*>(&neighbor_count), sizeof(NL));
            ofs.write(reinterpret_cast<const char*>(graph[i].data()), neighbor_count * sizeof(NID));
            if (neighbor_count < max_degree) {
                std::vector<NID> padding(max_degree - neighbor_count, static_cast<NID>(-1));
                ofs.write(reinterpret_cast<const char*>(padding.data()), (max_degree - neighbor_count) * sizeof(NID));
            }
        }
        if (!ofs) throw std::runtime_error("Failed to write graph data to file: " + graph_file);
    }

    std::string filename() const { return filename_; }
    NL size() const { return size_; }
    NL max_degree() const { return max_degree_; }
    uint32_t meta_data_size() const { return meta_data_size_; }
    size_t mmap_len() const { return mmap_len_; }

private:
    int fd_ = -1;
    char* mmap_ptr_ = nullptr;
    size_t mmap_len_ = 0;
    std::string filename_;
    NL size_ = 0;
    NL max_degree_ = 0;
    uint32_t meta_data_size_ = 0;
    uint32_t node_size_ = 0;
    void cleanup() {
        if (mmap_ptr_) munmap(mmap_ptr_, mmap_len_);
        if (fd_ != -1) close(fd_);
        mmap_ptr_ = nullptr; fd_ = -1; mmap_len_ = 0;
    }
};





template<typename NID, typename NL>
class GraphMMapWriter {
public:
    GraphMMapWriter(const std::string& filename, NL size, NL max_degree)
        : filename_(filename), size_(size), max_degree_(max_degree) {
        file_bytes_ = sizeof(NL) + sizeof(NL) + size_ * (sizeof(NL) + max_degree_ * sizeof(NID));
        fd_ = open(filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (fd_ == -1) throw std::runtime_error("Failed to open graph file for writing: " + filename);
        if (ftruncate(fd_, file_bytes_) == -1) { close(fd_); throw std::runtime_error("Failed to resize file: " + filename); }
        mmap_ptr_ = (char*)mmap(nullptr, file_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mmap_ptr_ == MAP_FAILED) { close(fd_); throw std::runtime_error("Failed to mmap file for writing: " + filename); }
        std::memcpy(mmap_ptr_, &size_, sizeof(NL));
        std::memcpy(mmap_ptr_ + sizeof(NL), &max_degree_, sizeof(NL));
        meta_data_size_ = sizeof(NL) + sizeof(NL);
    }
    ~GraphMMapWriter() { cleanup(); }

    void update_node(NL nodex, const std::vector<NID>& new_neighbors) {
        if (!mmap_ptr_) throw std::runtime_error("[update_node]MMap not initialized");
        if (nodex >= size_) throw std::out_of_range("Node index out of range.");
        size_t offset = meta_data_size_ + nodex * (max_degree_ * sizeof(NID) + sizeof(NL));
        NL neighbor_count = new_neighbors.size();
        std::memcpy(mmap_ptr_ + offset, &neighbor_count, sizeof(NL));
        std::memcpy(mmap_ptr_ + offset + sizeof(NL), new_neighbors.data(), neighbor_count * sizeof(NID));
        if (neighbor_count < max_degree_) {
            std::vector<NID> padding(max_degree_ - neighbor_count, static_cast<NID>(-1));
            std::memcpy(mmap_ptr_ + offset + sizeof(NL) + neighbor_count * sizeof(NID), padding.data(), (max_degree_ - neighbor_count) * sizeof(NID));
        }
    }

    void save_graph(const std::vector<std::vector<NID>>& graph) {
        for (NL i = 0; i < size_; ++i) {
            update_node(i, graph[i]);
        }
    }

    void flush() {
        if (mmap_ptr_) msync(mmap_ptr_, file_bytes_, MS_SYNC);
    }

private:
    int fd_ = -1;
    char* mmap_ptr_ = nullptr;
    size_t file_bytes_ = 0;
    NL size_ = 0;
    NL max_degree_ = 0;
    uint32_t meta_data_size_ = 0;
    std::string filename_;
    void cleanup() {
        if (mmap_ptr_) msync(mmap_ptr_, file_bytes_, MS_SYNC);
        if (mmap_ptr_) munmap(mmap_ptr_, file_bytes_);
        if (fd_ != -1) close(fd_);
        mmap_ptr_ = nullptr; fd_ = -1; file_bytes_ = 0;
    }
};
