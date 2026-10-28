// pack_sectors: page-locality ordering for the co-located SSG. Groups each node with its nearest still-unplaced neighbours so that
// one 4 KB sector (P records) holds mutual near neighbours; reading a sector then serves several useful nodes at once.
// Greedy: for every seed (in id order) that is not placed yet: take the seed, then its unplaced out-neighbours in stored order
// (nearest first), then unplaced 2-hop neighbours, until the sector is full. Groups that stay smaller than P are collected and
// packed together at the end (no locality there).
// Output: new2old (u32 x n, sector-major) and old2new (u32 x n).
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o pack_sectors tools/pack_sectors.cpp
// Usage: pack_sectors graph.ssg_graph per_sector out_prefix [min_group] [mode] [refine_rounds] [base.bin]
//   mode 1 = nearest-first greedy, 2 = connectivity greedy, 4 = spatial kNN greedy (needs base.bin: approximate 16-NN of every
//   node from its 2-hop graph neighbourhood by exact distance, then seed + its unplaced nearest points fill the sector); refine_rounds > 0 then improves the layout by pairwise swaps:
//   objective = number of same-page pairs (x, y) with y among the first KW out-neighbours of x (either direction); a node v on
//   page A is tried against every node w of the page B that holds most of v's neighbours, and the best swap is applied when it
//   raises the objective. Pages are try-locked, so swaps run in parallel.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <omp.h>
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s graph.ssg_graph per_sector out_prefix [min_group]\n", argv[0]); return 2; }
    int fd = ::open(argv[1], O_RDONLY); struct stat st; fstat(fd, &st);
    const uint8_t* g = static_cast<const uint8_t*>(mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    uint32_t n, R; std::memcpy(&n, g, 4); std::memcpy(&R, g + 4, 4);
    const uint32_t P = std::atoi(argv[2]); const std::string out = argv[3]; const uint32_t minG = argc > 4 ? std::atoi(argv[4]) : P; const int mode = argc > 5 ? std::atoi(argv[5]) : 1;
    auto rec = [&](uint32_t i) { return reinterpret_cast<const uint32_t*>(g + 8 + size_t(i) * 4ull * (R + 1)); };
    // Parallel greedy: threads take seeds from disjoint chunks and claim nodes with an atomic exchange on placed[];
    // a node can only end up in one group. Each thread appends its groups to a private list; lists are concatenated at the end.
    std::vector<uint8_t> placed(n, 0);
    // mode 4: approximate spatial kNN (16) of every node from its 2-hop neighbourhood (first 8 neighbours' lists), exact L2
    const uint32_t KN = 16; std::vector<uint32_t> knn;
    if (mode == 4) {
        if (argc < 8) { std::fprintf(stderr, "mode 4 needs base.bin\n"); return 2; }
        int bfd = ::open(argv[7], O_RDONLY); struct stat bst; fstat(bfd, &bst);
        const uint8_t* bm = static_cast<const uint8_t*>(mmap(nullptr, bst.st_size, PROT_READ, MAP_PRIVATE, bfd, 0));
        uint32_t bn, bd; std::memcpy(&bn, bm, 4); std::memcpy(&bd, bm + 4, 4);
        if (bn != n) { std::fprintf(stderr, "base n mismatch\n"); return 2; }
        const uint8_t* B = bm + 8;
        knn.assign(size_t(n) * KN, UINT32_MAX);
        #pragma omp parallel
        {
            std::vector<uint32_t> cand; std::vector<std::pair<uint32_t, uint32_t>> sc;
            #pragma omp for schedule(dynamic, 4096)
            for (uint64_t i64 = 0; i64 < n; ++i64) {
                const uint32_t i = uint32_t(i64); const uint32_t* r = rec(i); const uint32_t c = std::min(r[0], R);
                cand.clear();
                for (uint32_t x = 0; x < c; ++x) { const uint32_t v = r[1 + x]; if (v < n) cand.push_back(v); }
                for (uint32_t x = 0; x < c && x < 8; ++x) { const uint32_t v = r[1 + x]; if (v >= n) continue; const uint32_t* r2 = rec(v); const uint32_t c2 = std::min(r2[0], R);
                    for (uint32_t y = 0; y < c2; ++y) { const uint32_t w = r2[1 + y]; if (w < n && w != i) cand.push_back(w); } }
                std::sort(cand.begin(), cand.end()); cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
                sc.clear(); const uint8_t* xi = B + size_t(i) * bd;
                for (uint32_t v : cand) { const uint8_t* xv = B + size_t(v) * bd; uint32_t d2 = 0; for (uint32_t j = 0; j < bd; ++j) { const int t = int(xi[j]) - int(xv[j]); d2 += uint32_t(t * t); } sc.emplace_back(d2, v); }
                const size_t k = std::min<size_t>(KN, sc.size());
                std::partial_sort(sc.begin(), sc.begin() + k, sc.end());
                for (size_t t = 0; t < k; ++t) knn[size_t(i) * KN + t] = sc[t].second;
            }
        }
        std::fprintf(stderr, "mode 4: approximate %u-NN computed\n", KN);
    }
    auto claim = [&](uint32_t v) { return __atomic_exchange_n(&placed[v], uint8_t(1), __ATOMIC_ACQ_REL) == 0; };
    const int nt = omp_get_max_threads();
    std::vector<std::vector<uint32_t>> tl_full(nt), tl_left(nt);
    uint64_t full = 0, small = 0;
    #pragma omp parallel reduction(+:full,small)
    {
        const int t = omp_get_thread_num();
        std::vector<uint32_t> grp; grp.reserve(P);
        std::vector<uint32_t> cand;
        #pragma omp for schedule(dynamic, 4096)
        for (uint64_t s64 = 0; s64 < n; ++s64) {
            const uint32_t s = static_cast<uint32_t>(s64);
            if (placed[s] || !claim(s)) continue;
            grp.clear(); grp.push_back(s);
            const uint32_t* r = rec(s); const uint32_t c = std::min(r[0], R);
            if (mode == 4) {   // seed's nearest unplaced points first, then their nearest points
                for (uint32_t t = 0; t < KN && grp.size() < P; ++t) { const uint32_t v = knn[size_t(s) * KN + t]; if (v < n && !placed[v] && claim(v)) grp.push_back(v); }
                for (size_t gi = 1; gi < grp.size() && grp.size() < P; ++gi)
                    for (uint32_t t = 0; t < KN && grp.size() < P; ++t) { const uint32_t v = knn[size_t(grp[gi]) * KN + t]; if (v < n && !placed[v] && claim(v)) grp.push_back(v); }
            } else if (mode == 2) {
                cand.clear();
                for (uint32_t x = 0; x < c && x < 32; ++x) { const uint32_t v = r[1 + x]; if (v < n && !placed[v]) cand.push_back(v); }
                for (uint32_t x = 0; x < c && x < 8; ++x) { const uint32_t v = r[1 + x]; if (v >= n) continue; const uint32_t* r2 = rec(v); const uint32_t c2 = std::min(r2[0], R);
                    for (uint32_t y = 0; y < c2 && y < 8; ++y) { const uint32_t w = r2[1 + y]; if (w < n && !placed[w] && std::find(cand.begin(), cand.end(), w) == cand.end()) cand.push_back(w); } }
                auto links = [&](uint32_t a2, uint32_t b2) { const uint32_t* ra = rec(a2); const uint32_t ca = std::min(ra[0], std::min(R, 16u)); for (uint32_t x = 0; x < ca; ++x) if (ra[1 + x] == b2) return 1; return 0; };
                while (grp.size() < P && !cand.empty()) {
                    int best = -1, bs = -1;
                    for (size_t i = 0; i < cand.size(); ++i) { if (placed[cand[i]]) continue; int sc = 0; for (uint32_t g2 : grp) sc += links(g2, cand[i]) + links(cand[i], g2); if (sc > bs) { bs = sc; best = int(i); } }
                    if (best < 0) break;
                    const uint32_t v = cand[best]; cand.erase(cand.begin() + best);
                    if (claim(v)) grp.push_back(v);
                }
            } else {
                for (uint32_t x = 0; x < c && grp.size() < P; ++x) { const uint32_t v = r[1 + x]; if (v < n && !placed[v] && claim(v)) grp.push_back(v); }
                for (uint32_t x = 0; x < c && x < 16 && grp.size() < P; ++x) {
                    const uint32_t v = r[1 + x]; if (v >= n) continue; const uint32_t* r2 = rec(v); const uint32_t c2 = std::min(r2[0], R);
                    for (uint32_t y = 0; y < c2 && y < 16 && grp.size() < P; ++y) { const uint32_t w = r2[1 + y]; if (w < n && !placed[w] && claim(w)) grp.push_back(w); }
                }
            }
            if (grp.size() >= minG) { ++full; tl_full[t].insert(tl_full[t].end(), grp.begin(), grp.end()); }
            else { ++small; tl_left[t].insert(tl_left[t].end(), grp.begin(), grp.end()); }
        }
    }
    std::vector<uint32_t> new2old; new2old.reserve(n);
    for (auto& v : tl_full) { new2old.insert(new2old.end(), v.begin(), v.end()); std::vector<uint32_t>().swap(v); }
    std::vector<uint32_t> leftover;
    for (auto& v : tl_left) { leftover.insert(leftover.end(), v.begin(), v.end()); std::vector<uint32_t>().swap(v); }
    const size_t local_nodes = new2old.size();
    for (uint32_t v : leftover) new2old.push_back(v);
    // compact: drop padding slots so that the file stays n records (padding only inside full groups is impossible: groups == P)
    std::vector<uint32_t> compact; compact.reserve(n);
    for (uint32_t v : new2old) if (v != UINT32_MAX) compact.push_back(v);
    if (compact.size() != n) { std::fprintf(stderr, "internal error: %zu != %u\n", compact.size(), n); return 1; }
    std::vector<uint32_t> old2new(n);
    for (uint32_t i = 0; i < n; ++i) old2new[compact[i]] = i;
    const int rounds = argc > 6 ? std::atoi(argv[6]) : 0;
    if (rounds > 0) {
        const uint32_t KW = 32;
        const uint64_t nb = (uint64_t(n) + P - 1) / P;
        std::vector<uint8_t> lock(nb, 0);
        auto try_lock = [&](uint64_t b) { return __atomic_exchange_n(&lock[b], uint8_t(1), __ATOMIC_ACQUIRE) == 0; };
        auto unlock = [&](uint64_t b) { __atomic_store_n(&lock[b], uint8_t(0), __ATOMIC_RELEASE); };
        auto posof = [&](uint32_t v) { return __atomic_load_n(&old2new[v], __ATOMIC_RELAXED); };
        auto link = [&](uint32_t a2, uint32_t b2) { const uint32_t* ra = rec(a2); const uint32_t ca = std::min(ra[0], std::min(R, KW));
            for (uint32_t x = 0; x < ca; ++x) if (ra[1 + x] == b2) return 1; return 0; };
        // links between x and the members of page b, skipping `skip1`/`skip2` (x itself is never counted)
        auto f = [&](uint32_t x, uint64_t b, uint32_t skip1, uint32_t skip2) {
            int c = 0; const uint64_t lo = b * P, hi = std::min<uint64_t>(lo + P, n);
            for (uint64_t sl = lo; sl < hi; ++sl) { const uint32_t y = compact[sl]; if (y == x || y == skip1 || y == skip2) continue; c += link(x, y) + link(y, x); }
            return c; };
        for (int it = 0; it < rounds; ++it) {
            uint64_t swaps = 0, gain = 0;
            #pragma omp parallel reduction(+:swaps,gain)
            {
                std::vector<std::pair<uint64_t, int>> cnt;
                #pragma omp for schedule(dynamic, 8192)
                for (uint64_t v64 = 0; v64 < n; ++v64) {
                    const uint32_t v = static_cast<uint32_t>(v64);
                    const uint64_t A = posof(v) / P;
                    const uint32_t* r = rec(v); const uint32_t c = std::min(r[0], std::min(R, KW));
                    cnt.clear();
                    for (uint32_t x = 0; x < c; ++x) { const uint32_t y = r[1 + x]; if (y >= n) continue; const uint64_t b = posof(y) / P; if (b == A) continue;
                        auto itb = std::find_if(cnt.begin(), cnt.end(), [&](const std::pair<uint64_t, int>& e) { return e.first == b; });
                        if (itb == cnt.end()) cnt.emplace_back(b, 1); else ++itb->second; }
                    if (cnt.empty()) continue;
                    const uint64_t B = std::max_element(cnt.begin(), cnt.end(), [](const auto& x, const auto& y) { return x.second < y.second; })->first;
                    if (!try_lock(std::min(A, B))) continue;
                    if (!try_lock(std::max(A, B))) { unlock(std::min(A, B)); continue; }
                    if (posof(v) / P == A) {   // still on A after locking
                        const int fvA = f(v, A, v, v);
                        int best = 0; uint64_t bslot = 0;
                        const uint64_t lo = B * P, hi = std::min<uint64_t>(lo + P, n);
                        for (uint64_t sl = lo; sl < hi; ++sl) {
                            const uint32_t w = compact[sl];
                            const int d = f(v, B, w, w) - fvA + f(w, A, v, v) - f(w, B, w, w);
                            if (d > best) { best = d; bslot = sl; }
                        }
                        if (best > 0) {
                            const uint64_t sv = posof(v); const uint32_t w = compact[bslot];
                            compact[sv] = w; compact[bslot] = v;
                            __atomic_store_n(&old2new[v], uint32_t(bslot), __ATOMIC_RELAXED); __atomic_store_n(&old2new[w], uint32_t(sv), __ATOMIC_RELAXED);
                            ++swaps; gain += best;
                        }
                    }
                    unlock(std::max(A, B)); unlock(std::min(A, B));
                }
            }
            uint64_t same8 = 0, tot8 = 0;
            for (uint32_t i = 0; i < n; i += 97) { const uint32_t si = old2new[i] / P; const uint32_t* r = rec(i); const uint32_t c = std::min(r[0], R);
                for (uint32_t x = 0; x < c && x < 8; ++x) { const uint32_t v = r[1 + x]; if (v >= n) continue; tot8++; same8 += old2new[v] / P == si; } }
            std::fprintf(stderr, "refine round %d: %llu swaps, objective +%llu, first-8 same sector %.3f\n", it + 1,
                         (unsigned long long)swaps, (unsigned long long)gain, double(same8) / tot8);
        }
    }
    FILE* f = std::fopen((out + ".new2old").c_str(), "wb"); std::fwrite(compact.data(), 4, n, f); std::fclose(f);
    f = std::fopen((out + ".old2new").c_str(), "wb"); std::fwrite(old2new.data(), 4, n, f); std::fclose(f);
    {   // locality metric: share of each node's first-k neighbours stored in the same sector
        uint64_t same8 = 0, same16 = 0, tot8 = 0, tot16 = 0;
        for (uint32_t i = 0; i < n; i += 97) { const uint32_t si = old2new[i] / P; const uint32_t* r = rec(i); const uint32_t c = std::min(r[0], R);
            for (uint32_t x = 0; x < c && x < 16; ++x) { const uint32_t v = r[1 + x]; if (v >= n) continue; const bool same = old2new[v] / P == si; if (x < 8) { tot8++; same8 += same; } tot16++; same16 += same; } }
        std::fprintf(stderr, "locality: first-8 neighbours in same sector %.3f, first-16 %.3f\n", double(same8) / tot8, double(same16) / tot16);
    }
    std::fprintf(stderr, "n=%u P=%u: %llu local sectors (%.1f%% of nodes), %llu small groups -> %zu leftover nodes packed without locality\n",
                 n, P, (unsigned long long)full, 100.0 * (local_nodes) / n, (unsigned long long)small, leftover.size());
    return 0;
}
