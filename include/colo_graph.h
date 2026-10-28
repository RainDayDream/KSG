#pragma once
// Co-located SSG: each node record holds the node's exact vector AND its neighbour list, so expanding a node
// costs exactly one 4 KB page and yields the node's exact distance for free (DiskANN-style layout).
//   sector 0 : header [u32 n][u32 R][u32 rec_len][u32 per_sector][u32 dim][u32 vbytes (0/1 = uint8, 2 = fp16)]
//   sector 1+: records packed per_sector per 4 KB sector: [vec dim*vbytes bytes][u32 count][u32 ids x R][u32 orig id]?, zero padded
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ksg {

class ColoGraph {
public:
    explicit ColoGraph(const std::string& path, bool random_advice = true) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open colo graph: " + path);
        struct stat st; fstat(fd_, &st); bytes_ = static_cast<size_t>(st.st_size);
        map_ = static_cast<const uint8_t*>(mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0));
        if (map_ == MAP_FAILED) throw std::runtime_error("mmap colo graph failed");
        if (random_advice) madvise(const_cast<uint8_t*>(map_), bytes_, MADV_RANDOM);
        std::memcpy(&n_, map_, 4); std::memcpy(&R_, map_ + 4, 4); std::memcpy(&rec_len_, map_ + 8, 4);
        std::memcpy(&per_sector_, map_ + 12, 4); std::memcpy(&dim_, map_ + 16, 4); std::memcpy(&vb_, map_ + 20, 4);
        if (vb_ == 0) vb_ = 1;
        vbytes_ = dim_ * vb_;
        has_orig_ = rec_len_ == vbytes_ + 8 + 4u * R_;   // optional trailing u32: original node id (permuted layouts)
        if ((rec_len_ != vbytes_ + 4 + 4u * R_ && !has_orig_) || per_sector_ == 0 || per_sector_ * rec_len_ > 4096)
            throw std::runtime_error("bad colo graph header");
    }
    ~ColoGraph() { if (map_ && map_ != MAP_FAILED) munmap(const_cast<uint8_t*>(map_), bytes_); if (fd_ >= 0) ::close(fd_); if (dfd_ >= 0) ::close(dfd_); }
    uint32_t size() const { return n_; }
    uint32_t degree() const { return R_; }
    uint32_t dim() const { return dim_; }
    bool fp16() const { return vb_ == 2; }
    uint32_t per_sector() const { return per_sector_; }
    // O_DIRECT descriptor for explicit I/O (opened lazily)
    int direct_fd(const std::string& path) {
        if (dfd_ < 0) { dfd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT); if (dfd_ < 0) throw std::runtime_error("cannot open O_DIRECT: " + path); }
        return dfd_;
    }
    const std::string& path() const { return path_; }
    bool has_orig() const { return has_orig_; }
    uint32_t rec_len() const { return rec_len_; }
    inline uint32_t orig_id(const uint8_t* rec) const { uint32_t o; std::memcpy(&o, rec + vbytes_ + 4 + 4ull * R_, 4); return o; }
    uint32_t record_offset(uint32_t id) const { return (id % per_sector_) * rec_len_; }
    uint64_t sector_of(uint32_t id) const { return 1 + id / per_sector_; }
    const uint8_t* record(uint32_t id) const { return map_ + sector_of(id) * 4096ull + (id % per_sector_) * static_cast<uint64_t>(rec_len_); }
    const uint8_t* vector(uint32_t id) const { return record(id); }
    // neighbours of the record at `rec`
    static inline uint32_t count(const uint8_t* rec, uint32_t vbytes) { uint32_t c; std::memcpy(&c, rec + vbytes, 4); return c; }
    inline const uint32_t* neighbors(const uint8_t* rec) const { return reinterpret_cast<const uint32_t*>(rec + vbytes_ + 4); }
    inline uint32_t neighbor_count(const uint8_t* rec) const { return count(rec, vbytes_); }
private:
    std::string path_; int dfd_ = -1;
    int fd_ = -1; const uint8_t* map_ = nullptr; size_t bytes_ = 0;
    uint32_t n_ = 0, R_ = 0, rec_len_ = 0, per_sector_ = 0, dim_ = 0, vb_ = 1, vbytes_ = 0; bool has_orig_ = false;
};

}  // namespace ksg
