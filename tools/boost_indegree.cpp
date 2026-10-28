// boost_indegree: raise the in-degree of under-referenced nodes ("anti-hubs") of a fixed-degree graph in memory.
// For every node p with in-degree < T, try its nearest out-neighbours s (p's neighbours are near p, so s->p is a short edge):
// put p into one of s's tail slots (slot index >= protect) that currently points to a node whose in-degree is still > T_donor.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o boost_indegree tools/boost_indegree.cpp
// Usage: boost_indegree --base b.bin --in g.ssg_graph --out o.ssg_graph [--T 64] [--donor 72] [--src 48] [--protect 16]
//                       [--float 1] [--hop2 K: also try the out-neighbours of p's first K out-neighbours as sources, nearest first]
//                       [--passes P: repeat over the nodes still below T] [--batch B: gather sources of B nodes in parallel]   (defaults: uint8, hop2 0, passes 1 = original behaviour)
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
#include "../include/ssg_search_core.h"

static void indeg_stats(const char* tag, const Graph& G, const std::vector<uint32_t>& ind) {
    std::vector<uint32_t> v(ind); std::sort(v.begin(), v.end());
    auto q = [&](double x) { return v[std::min<size_t>(v.size() - 1, size_t(x * v.size()))]; };
    uint64_t lt16 = 0, lt32 = 0, lt48 = 0; for (uint32_t x : ind) { lt16 += x < 16; lt32 += x < 32; lt48 += x < 48; }
    std::fprintf(stderr, "[%s] in-degree: min=%u p1=%u p5=%u p10=%u median=%u p90=%u max=%u ; share <16: %.3f%%  <32: %.3f%%  <48: %.3f%%\n",
                 tag, v.front(), q(.01), q(.05), q(.10), q(.5), q(.9), v.back(), 100.0 * lt16 / ind.size(), 100.0 * lt32 / ind.size(), 100.0 * lt48 / ind.size());
}
template <class DT>
static void boost(const DT& D, Graph& G, const std::vector<uint8_t>& lab, std::vector<uint32_t>& ind,
                  uint32_t T, uint32_t donor, uint32_t nsrc, uint32_t protect, uint32_t hop2, uint32_t passes, uint32_t batch) {
    const uint32_t n = G.n;
    std::vector<uint32_t> low; for (uint32_t p = 0; p < n; ++p) if (ind[p] < T) low.push_back(p);
    std::fprintf(stderr, "T=%u donor=%u src=%u protect=%u hop2=%u passes=%u : %zu nodes below T\n", T, donor, nsrc, protect, hop2, passes, low.size());
    std::vector<uint32_t> mark(batch ? 0 : n, 0); uint32_t ep = 0;
    // --batch B: the source lists (distances, 2-hop expansion: the expensive, read-only part) of B low nodes are gathered in parallel
    // from the graph as it is at the start of the batch; the edge transfers stay sequential in the same order. Not byte-identical
    // to the sequential mode (sources do not see transfers made earlier in the same batch).
    auto gather = [&](uint32_t p, std::vector<std::pair<int32_t, uint32_t>>& src) {
        src.clear(); std::vector<uint32_t> ids;
        for (uint32_t x = 0; x < std::min(G.cnt(p), nsrc); ++x) ids.push_back(G.nb(p)[x]);
        const size_t n1 = ids.size();
        for (size_t x = 0; x < n1; ++x) src.emplace_back(D.dist(p, ids[x]), ids[x]);
        if (hop2) {
            auto t = src; std::sort(t.begin(), t.end()); const size_t k1 = std::min<size_t>(hop2, t.size());
            for (size_t y = 0; y < k1; ++y) { const uint32_t u = t[y].second; for (uint32_t x = 0; x < std::min(G.cnt(u), nsrc); ++x) ids.push_back(G.nb(u)[x]); }
            std::vector<uint32_t> first(ids.begin(), ids.begin() + n1), more(ids.begin() + n1, ids.end());
            std::sort(first.begin(), first.end()); std::sort(more.begin(), more.end()); more.erase(std::unique(more.begin(), more.end()), more.end());
            for (uint32_t w : more) if (w != p && !std::binary_search(first.begin(), first.end(), w)) src.emplace_back(D.dist(p, w), w);
        }
        std::sort(src.begin(), src.end());
    };
    std::vector<std::vector<std::pair<int32_t, uint32_t>>> pre;
    for (uint32_t pass = 0; pass < passes; ++pass) {
    uint64_t added = 0, failed_nodes = 0; std::vector<uint32_t> still;
    for (size_t li = 0; li < low.size(); ++li) {
        const uint32_t p = low[li];
        if (batch && li % batch == 0) {
            const size_t e = std::min(low.size(), li + batch); pre.resize(e - li);
            #pragma omp parallel for schedule(dynamic, 256)
            for (size_t k = li; k < e; ++k) gather(low[k], pre[k - li]);
        }
        // p's nearest out-neighbours, by distance to p (plus, with hop2, their out-neighbours)
        std::vector<std::pair<int32_t, uint32_t>> src;
        if (batch) src.swap(pre[li % batch]); else {
        for (uint32_t x = 0; x < std::min(G.cnt(p), nsrc); ++x) src.emplace_back(D.dist(p, G.nb(p)[x]), G.nb(p)[x]);
        if (hop2) {
            ++ep; mark[p] = ep; for (auto& e : src) mark[e.second] = ep;
            const size_t k1 = std::min<size_t>(hop2, src.size()); std::vector<uint32_t> first;
            { auto t = src; std::sort(t.begin(), t.end()); for (size_t x = 0; x < k1; ++x) first.push_back(t[x].second); }
            for (uint32_t u : first) for (uint32_t x = 0; x < std::min(G.cnt(u), nsrc); ++x) {
                const uint32_t w = G.nb(u)[x]; if (mark[w] == ep) continue; mark[w] = ep; src.emplace_back(D.dist(p, w), w); }
        }
        std::sort(src.begin(), src.end());
        }
        for (auto& sp : src) {
            if (ind[p] >= T) break;
            uint32_t s = sp.second; if (!lab.empty() && lab[s] != lab[p]) continue;
            uint32_t* rec = G.rec(s); uint32_t c = rec[0];
            bool has = false; for (uint32_t x = 0; x < c; ++x) if (rec[1 + x] == p) { has = true; break; }
            if (has) continue;
            // donor slot: scan s's tail from the end for a target with spare in-degree
            int slot = -1;
            for (int x = int(c) - 1; x >= int(protect); --x) if (ind[rec[1 + x]] > donor) { slot = x; break; }
            if (slot < 0) continue;
            ind[rec[1 + slot]]--; rec[1 + slot] = p; ind[p]++; ++added;
        }
        if (ind[p] < T) { ++failed_nodes; still.push_back(p); }
    }
    std::fprintf(stderr, "edges replaced=%llu ; nodes still below T=%llu\n", (unsigned long long)added, (unsigned long long)failed_nodes);
    low.swap(still); if (low.empty() || added == 0) break;
    }
}
int main(int argc, char** argv) {
    std::string base, in, out, labels_file; uint32_t T = 64, donor = 72, nsrc = 48, protect = 16, hop2 = 0, passes = 1, batch = 0; bool flt = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--base") base = v; else if (a == "--in") in = v; else if (a == "--out") out = v;
        else if (a == "--T") T = std::stoul(v); else if (a == "--donor") donor = std::stoul(v);
        else if (a == "--labels") labels_file = v; else if (a == "--src") nsrc = std::stoul(v); else if (a == "--protect") protect = std::stoul(v);
        else if (a == "--float") flt = v != "0"; else if (a == "--hop2") hop2 = std::stoul(v); else if (a == "--passes") passes = std::stoul(v);
        else if (a == "--batch") batch = std::stoul(v);
    }
    Graph G = load_graph(in);
    std::vector<uint8_t> lab;   // optional: only sources in the same primary shard may receive the new edge
    if (!labels_file.empty()) { size_t lb; const uint8_t* lm = map_ro(labels_file, lb); lab.assign(lm, lm + lb); }
    const uint32_t n = G.n;
    std::vector<uint32_t> ind(n, 0);
    for (uint32_t u = 0; u < n; ++u) for (uint32_t x = 0; x < G.cnt(u); ++x) ind[G.nb(u)[x]]++;
    indeg_stats("before", G, ind);
    if (flt) { DataF D(base); boost(D, G, lab, ind, T, donor, nsrc, protect, hop2, passes, batch); }
    else { Data D(base); boost(D, G, lab, ind, T, donor, nsrc, protect, hop2, passes, batch); }
    indeg_stats("after ", G, ind);
    save_graph(G, out);
}
