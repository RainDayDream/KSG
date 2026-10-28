#pragma once
// Minimal io_uring wrapper (raw syscalls; liburing is not installed on the target machine).
// One ring per thread. read_sectors() submits n 4 KB O_DIRECT reads at once and waits for all of them,
// so a search thread keeps `beam` reads in flight instead of blocking on one page fault at a time.
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <linux/io_uring.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace ksg {

class UringReader {
public:
    explicit UringReader(unsigned entries = 64) {
        io_uring_params p; std::memset(&p, 0, sizeof(p));
        fd_ = static_cast<int>(::syscall(__NR_io_uring_setup, entries, &p));
        if (fd_ < 0) throw std::runtime_error(std::string("io_uring_setup failed: ") + std::strerror(errno));
        size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        const bool single = p.features & IORING_FEAT_SINGLE_MMAP;
        if (single) { if (cq_sz > sq_sz) sq_sz = cq_sz; cq_sz = sq_sz; }
        sq_ptr_ = ::mmap(nullptr, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQ_RING);
        if (sq_ptr_ == MAP_FAILED) throw std::runtime_error("mmap sq ring failed");
        cq_ptr_ = single ? sq_ptr_ : ::mmap(nullptr, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_CQ_RING);
        if (cq_ptr_ == MAP_FAILED) throw std::runtime_error("mmap cq ring failed");
        sqes_ = static_cast<io_uring_sqe*>(::mmap(nullptr, p.sq_entries * sizeof(io_uring_sqe), PROT_READ | PROT_WRITE,
                                                  MAP_SHARED | MAP_POPULATE, fd_, IORING_OFF_SQES));
        if (sqes_ == MAP_FAILED) throw std::runtime_error("mmap sqes failed");
        auto* sq = static_cast<uint8_t*>(sq_ptr_); auto* cq = static_cast<uint8_t*>(cq_ptr_);
        sq_head_ = reinterpret_cast<unsigned*>(sq + p.sq_off.head); sq_tail_ = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
        sq_mask_ = *reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask); sq_array_ = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
        cq_head_ = reinterpret_cast<unsigned*>(cq + p.cq_off.head); cq_tail_ = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
        cq_mask_ = *reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask); cqes_ = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
        slots_ = p.sq_entries < p.cq_entries ? p.sq_entries : p.cq_entries;
        if (::posix_memalign(reinterpret_cast<void**>(&bufs_), 4096, static_cast<size_t>(slots_) * 4096) != 0) throw std::runtime_error("posix_memalign failed");
    }
    ~UringReader() { if (bufs_) std::free(bufs_); if (fd_ >= 0) ::close(fd_); }
    UringReader(const UringReader&) = delete; UringReader& operator=(const UringReader&) = delete;
    unsigned capacity() const { return slots_; }

    // Read n sectors (n <= capacity) of `file_fd` (opened O_DIRECT). out[i] points to the 4 KB buffer of sectors[i]; valid until the next call.
    void read_sectors(int file_fd, const uint64_t* sectors, size_t n, const uint8_t** out) {
        if (n == 0) return;
        if (n > slots_) throw std::runtime_error("too many sectors in one batch");
        unsigned tail = *sq_tail_;
        for (size_t i = 0; i < n; ++i) {
            const unsigned idx = tail & sq_mask_;
            io_uring_sqe* sqe = &sqes_[idx];
            std::memset(sqe, 0, sizeof(*sqe));
            sqe->opcode = IORING_OP_READ; sqe->fd = file_fd; sqe->off = sectors[i] * 4096ull;
            sqe->addr = reinterpret_cast<uint64_t>(bufs_ + i * 4096); sqe->len = 4096; sqe->user_data = i;
            sq_array_[idx] = idx; ++tail;
            out[i] = bufs_ + i * 4096;
        }
        __atomic_store_n(sq_tail_, tail, __ATOMIC_RELEASE);
        unsigned to_submit = static_cast<unsigned>(n), done = 0;
        while (done < n) {
            const long r = ::syscall(__NR_io_uring_enter, fd_, to_submit, static_cast<unsigned>(n - done), IORING_ENTER_GETEVENTS, nullptr, 0);
            if (r < 0) { if (errno == EINTR) continue; throw std::runtime_error(std::string("io_uring_enter failed: ") + std::strerror(errno)); }
            to_submit = 0;   // the kernel consumed what it could; re-submission of the remainder is handled below
            unsigned head = *cq_head_;
            const unsigned ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
            while (head != ctail) {
                const io_uring_cqe* cqe = &cqes_[head & cq_mask_];
                if (cqe->res != 4096) throw std::runtime_error("io_uring short/failed read: res=" + std::to_string(cqe->res));
                ++head; ++done;
            }
            __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
            if (done < n && r == 0 && to_submit == 0) {
                // nothing completed yet (spurious wake-up): loop and wait again; SQEs were already submitted
            }
        }
    }
    // ---- asynchronous interface (pipelined search): one 4 KB buffer slot per in-flight read ----
    // submit() queues a read and returns false when no slot is free; flush() submits what was queued; reap() takes one
    // completion (waiting for it if `wait`), giving back the caller's tag and the buffer, which stays valid until release(slot).
    unsigned inflight() const { return inflight_; }
    bool submit(int file_fd, uint64_t sector, uint64_t tag) {
        if (free_.empty()) { if (!free_init_) { for (unsigned i = slots_; i-- > 0;) free_.push_back(i); free_init_ = true; } if (free_.empty()) return false; }
        const unsigned slot = free_.back(); free_.pop_back();
        unsigned tail = *sq_tail_; const unsigned idx = tail & sq_mask_;
        io_uring_sqe* sqe = &sqes_[idx]; std::memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ; sqe->fd = file_fd; sqe->off = sector * 4096ull;
        sqe->addr = reinterpret_cast<uint64_t>(bufs_ + static_cast<size_t>(slot) * 4096); sqe->len = 4096; sqe->user_data = slot;
        sq_array_[idx] = idx; __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
        tags_[slot] = tag; ++queued_; ++inflight_;
        return true;
    }
    void flush() { enter(0); }
    bool reap(bool wait, uint64_t& tag, const uint8_t*& buf, unsigned& slot) {
        for (;;) {
            unsigned head = *cq_head_; const unsigned ctail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
            if (head != ctail) {
                const io_uring_cqe* cqe = &cqes_[head & cq_mask_];
                if (cqe->res != 4096) throw std::runtime_error("io_uring short/failed read: res=" + std::to_string(cqe->res));
                slot = static_cast<unsigned>(cqe->user_data); tag = tags_[slot]; buf = bufs_ + static_cast<size_t>(slot) * 4096;
                __atomic_store_n(cq_head_, head + 1, __ATOMIC_RELEASE); --inflight_;
                return true;
            }
            if (!wait || inflight_ == 0) { if (queued_) enter(0); return false; }
            enter(1);
        }
    }
    void release(unsigned slot) { free_.push_back(slot); }
    // block until at least k completions are ready (k is clamped to the reads in flight)
    void wait_for(unsigned k) {
        if (k > inflight_) k = inflight_;
        for (;;) {
            const unsigned ready = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE) - *cq_head_;
            if (ready >= k) { if (queued_) enter(0); return; }
            enter(k);
        }
    }
private:
    void enter(unsigned min_complete) {
        for (;;) {
            const long r = ::syscall(__NR_io_uring_enter, fd_, queued_, min_complete, min_complete ? IORING_ENTER_GETEVENTS : 0u, nullptr, 0);
            if (r < 0) { if (errno == EINTR) continue; throw std::runtime_error(std::string("io_uring_enter failed: ") + std::strerror(errno)); }
            queued_ -= static_cast<unsigned>(r) < queued_ ? static_cast<unsigned>(r) : queued_;
            return;
        }
    }
    std::vector<unsigned> free_; bool free_init_ = false; uint64_t tags_[4096] = {0}; unsigned queued_ = 0, inflight_ = 0;
    int fd_ = -1; void* sq_ptr_ = nullptr; void* cq_ptr_ = nullptr; io_uring_sqe* sqes_ = nullptr;
    unsigned *sq_head_ = nullptr, *sq_tail_ = nullptr, *sq_array_ = nullptr, *cq_head_ = nullptr, *cq_tail_ = nullptr;
    unsigned sq_mask_ = 0, cq_mask_ = 0, slots_ = 0; io_uring_cqe* cqes_ = nullptr; uint8_t* bufs_ = nullptr;
};

}  // namespace ksg
