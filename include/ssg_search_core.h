#pragma once
// Core of the search-based SSG refinement (Vamana-style candidate generation + robust prune), shared by
// tools/build_ssg_search.cpp (whole graph in memory) and tools/build_ssg_shard.cpp (one shard of a partitioned build).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <immintrin.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <omp.h>

static const uint8_t* map_ro(const std::string& path, size_t& bytes) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::perror(path.c_str()); std::exit(1); }
    struct stat st; fstat(fd, &st); bytes = st.st_size;
    void* m = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { std::perror("mmap"); std::exit(1); }
    return static_cast<const uint8_t*>(m);
}
struct Data {
    uint32_t n, d; const uint8_t* v = nullptr; std::vector<int8_t> s8; std::vector<int32_t> sq, sum;
    explicit Data(const std::string& path) {
        size_t b; const uint8_t* m = map_ro(path, b);
        std::memcpy(&n, m, 4); std::memcpy(&d, m + 4, 4); v = m + 8;
        if (d != 128) { std::fprintf(stderr, "dim must be 128\n"); std::exit(1); }
        s8.resize(size_t(n) * d); sq.resize(n); sum.resize(n);
        #pragma omp parallel for
        for (uint32_t i = 0; i < n; ++i) {
            int q = 0, s = 0;
            for (uint32_t j = 0; j < d; ++j) { int x = v[size_t(i) * d + j]; s8[size_t(i) * d + j] = int8_t(x - 128); q += x * x; s += x; }
            sq[i] = q; sum[i] = s;
        }
    }
    // build from an owned, already-gathered buffer (n x 128 uint8, contiguous)
    Data(std::vector<uint8_t>&& own, uint32_t n_) : n(n_), d(128), owned(std::move(own)) {
        v = owned.data();
        s8.resize(size_t(n) * d); sq.resize(n); sum.resize(n);
        #pragma omp parallel for
        for (uint32_t i = 0; i < n; ++i) {
            int q = 0, s = 0;
            for (uint32_t j = 0; j < d; ++j) { int x = v[size_t(i) * d + j]; s8[size_t(i) * d + j] = int8_t(x - 128); q += x * x; s += x; }
            sq[i] = q; sum[i] = s;
        }
    }
    std::vector<uint8_t> owned;
    const uint8_t* vec(uint32_t i) const { return v + size_t(i) * d; }
    // squared L2 between node i (u8 side) and node j (s8 side): ||a||^2+||b||^2-2a.b, a.b = dpbusd(a,b-128)+128*sum(a)
    inline int32_t dist(uint32_t i, uint32_t j) const {
        const uint8_t* a = vec(i); const int8_t* b = &s8[size_t(j) * 128];
        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_loadu_si512(a), _mm512_loadu_si512(b));
        acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(a + 64), _mm512_loadu_si512(b + 64));
        return sq[i] + sq[j] - 2 * (_mm512_reduce_add_epi32(acc) + 128 * sum[i]);
    }
};

// float base of any dimension (e.g. GIST1M, d = 960): same interface as Data; distances are squared L2 scaled by `scale`
// and rounded to int32 (GIST values lie in [0, 1], so d^2 <= 960 and scale 1e6 keeps full ordering resolution).
struct DataF {
    uint32_t n, d; const float* v = nullptr; std::vector<float> nrm; float scale = 1e6f;
    explicit DataF(const std::string& path, float sc = 1e6f) : scale(sc) {
        size_t b; const uint8_t* m = map_ro(path, b);
        std::memcpy(&n, m, 4); std::memcpy(&d, m + 4, 4); v = reinterpret_cast<const float*>(m + 8);
        if (b != 8 + size_t(n) * d * 4) { std::fprintf(stderr, "not a float32 base file\n"); std::exit(1); }
        nrm.resize(n);
        #pragma omp parallel for
        for (uint32_t i = 0; i < n; ++i) { double q = 0; for (uint32_t j = 0; j < d; ++j) q += double(v[size_t(i) * d + j]) * v[size_t(i) * d + j]; nrm[i] = float(q); }
        // large-magnitude data (e.g. SIFT1M stored as float, values 0..255): lower the scale so distances do not saturate at 2.1e9
        // (bound ||a-b||^2 <= 2(|a|^2+|b|^2) by the largest norm); unit-scale data (GIST, DEEP) keeps 1e6 and identical results
        const double mx = n ? *std::max_element(nrm.begin(), nrm.end()) : 0.0;
        if (4.0 * mx * scale > 2.0e9) { scale = float(2.0e9 / (4.0 * mx)); std::fprintf(stderr, "DataF: distance scale lowered to %g\n", double(scale)); }
    }
    const float* vec(uint32_t i) const { return v + size_t(i) * d; }
    inline int32_t dist(uint32_t i, uint32_t j) const {
        const float* a = vec(i); const float* b = vec(j); __m512 acc = _mm512_setzero_ps(); uint32_t k = 0;
        for (; k + 16 <= d; k += 16) { const __m512 t = _mm512_sub_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k)); acc = _mm512_fmadd_ps(t, t, acc); }
        float s2 = _mm512_reduce_add_ps(acc); for (; k < d; ++k) { const float t = a[k] - b[k]; s2 += t * t; }
        const double r = double(s2) * scale; return r > 2.1e9 ? 2100000000 : int32_t(r + 0.5);
    }
};

struct Graph {
    uint32_t n, R; std::vector<uint32_t> a;   // a[i*(R+1)] = count, then R ids
    Graph(uint32_t n_, uint32_t R_) : n(n_), R(R_), a(size_t(n_) * (R_ + 1), 0) {}
    uint32_t cnt(uint32_t i) const { return a[size_t(i) * (R + 1)]; }
    const uint32_t* nb(uint32_t i) const { return &a[size_t(i) * (R + 1) + 1]; }
    uint32_t* rec(uint32_t i) { return &a[size_t(i) * (R + 1)]; }
};
static Graph load_graph(const std::string& path) {
    size_t b; const uint8_t* m = map_ro(path, b);
    uint32_t n, R; std::memcpy(&n, m, 4); std::memcpy(&R, m + 4, 4);
    Graph g(n, R); std::memcpy(g.a.data(), m + 8, g.a.size() * 4); return g;
}
static void save_graph(const Graph& g, const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "wb"); uint32_t h[2] = {g.n, g.R};
    std::fwrite(h, 4, 2, f); std::fwrite(g.a.data(), 4, g.a.size(), f); std::fclose(f);
}

struct Opt { bool save_rounds = true; uint32_t R = 96, Lc = 128, C = 512, rounds = 2, entries = 8, rev = 32; float alpha = 1.2f; bool backlink = true, fill = true; };

// robust prune: `cand` = (dist, id) sorted ascending, unique, self removed
template <class DT>
static void robust_prune(const DT& D, std::vector<std::pair<int32_t, uint32_t>>& cand, uint32_t C, const Opt& o, std::vector<uint32_t>& out) {
    if (cand.size() > C) cand.resize(C);
    out.clear();
    const double a2 = double(o.alpha) * o.alpha;
    for (size_t x = 0; x < cand.size() && out.size() < o.R; ++x) {
        bool occ = false;
        for (uint32_t s : out)
            if (a2 * double(D.dist(s, cand[x].second)) <= double(cand[x].first)) { occ = true; break; }
        if (!occ) out.push_back(cand[x].second);
    }
    if (o.fill) for (size_t x = 0; x < cand.size() && out.size() < o.R; ++x)
        if (std::find(out.begin(), out.end(), cand[x].second) == out.end()) out.push_back(cand[x].second);
}

struct Searcher {
    std::vector<uint32_t> stamp; uint32_t epoch = 0;
    std::vector<std::pair<int32_t, uint32_t>> beam, visited; std::vector<uint8_t> done;
    explicit Searcher(uint32_t n) : stamp(n, 0) {}
    // best-first search for query node p; fills `visited` with every scored node
    template <class DT>
    void run(const DT& D, const Graph& G, uint32_t p, uint32_t L, uint32_t entries) {
        if (++epoch == 0) { std::fill(stamp.begin(), stamp.end(), 0); epoch = 1; }
        beam.clear(); done.clear(); visited.clear();
        stamp[p] = epoch;
        auto add = [&](uint32_t id) {
            if (stamp[id] == epoch) return;
            stamp[id] = epoch;
            const int32_t dd = D.dist(p, id);
            visited.emplace_back(dd, id);
            if (beam.size() >= L && dd >= beam.back().first) return;
            auto pos = std::lower_bound(beam.begin(), beam.end(), std::make_pair(dd, id));
            const size_t idx = pos - beam.begin();
            beam.insert(pos, {dd, id}); done.insert(done.begin() + idx, 0);
            if (beam.size() > L) { beam.pop_back(); done.pop_back(); }
        };
        const uint32_t c0 = std::min(G.cnt(p), entries); const uint32_t* nb0 = G.nb(p);
        for (uint32_t x = 0; x < c0; ++x) add(nb0[x]);
        size_t i = 0;
        while (true) {
            while (i < beam.size() && done[i]) ++i;
            if (i >= beam.size()) break;
            done[i] = 1;
            const uint32_t u = beam[i].second; const uint32_t c = G.cnt(u); const uint32_t* nb = G.nb(u);
            for (uint32_t x = 0; x < c; ++x) add(nb[x]);
            i = 0;   // beam changed by insertions: restart scan from the best unexpanded
        }
    }
};

template <class DT>
static Graph one_round(const DT& D, const Graph& G, const Opt& o, const Graph* rev) {
    const uint32_t n = D.n;
    Graph out(n, o.R);
    #pragma omp parallel
    {
        Searcher S(n); std::vector<uint32_t> sel;
        std::vector<std::pair<int32_t, uint32_t>> cand;
        #pragma omp for schedule(dynamic, 64)
        for (uint32_t p = 0; p < n; ++p) {
            S.run(D, G, p, o.Lc, o.entries);
            cand = S.visited;
            // current + reverse neighbours not already scored
            auto addc = [&](uint32_t id) { if (id != p && S.stamp[id] != S.epoch) { S.stamp[id] = S.epoch; cand.emplace_back(D.dist(p, id), id); } };
            for (uint32_t x = 0; x < G.cnt(p); ++x) addc(G.nb(p)[x]);
            if (rev) for (uint32_t x = 0; x < rev->cnt(p); ++x) addc(rev->nb(p)[x]);
            std::sort(cand.begin(), cand.end());
            robust_prune(D, cand, o.C, o, sel);
            uint32_t* r = out.rec(p); r[0] = sel.size(); std::copy(sel.begin(), sel.end(), r + 1);
        }
    }
    return out;
}
static Graph reverse_graph(const Graph& G, uint32_t cap) {
    Graph r(G.n, cap);
    std::vector<std::atomic<uint32_t>> cnt(G.n);
    for (auto& c : cnt) c.store(0, std::memory_order_relaxed);
    #pragma omp parallel for schedule(static, 4096)
    for (uint32_t u = 0; u < G.n; ++u)
        for (uint32_t x = 0; x < G.cnt(u); ++x) {
            const uint32_t v = G.nb(u)[x]; const uint32_t slot = cnt[v].fetch_add(1, std::memory_order_relaxed);
            if (slot < cap) r.a[size_t(v) * (cap + 1) + 1 + slot] = u;
        }
    for (uint32_t v = 0; v < G.n; ++v) r.a[size_t(v) * (cap + 1)] = std::min<uint32_t>(cnt[v].load(), cap);
    return r;
}

