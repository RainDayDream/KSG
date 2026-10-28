// build_ssg_search: refine a fixed-degree graph with SEARCH-BASED candidate generation (Vamana-style), in memory.
//   Each round, for every node p:
//     1. greedy best-first search (list size Lc, exact distance) on the current graph, started from p's own best
//        `entries` neighbours; every node whose distance was computed is a candidate;
//     2. candidates = best C visited  U  current neighbours  U (optional) capped reverse neighbours;
//     3. Vamana robust-prune(alpha) to degree R (+ optional fill with nearest remaining).
//   Optional back-link pass: build the reverse graph of the result and re-prune with it.
// Scale: needs base (uint8) + 2 graphs + per-thread visited arrays in RAM -> intended for <= ~30M points.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 -o build_ssg_search tools/build_ssg_search.cpp
// Usage: build_ssg_search --base b.bin --init g.ssg_graph --out o.ssg_graph [--R 96] [--Lc 128] [--C 512] [--alpha 1.2]
//                         [--rounds 2] [--entries 8] [--backlink 1] [--rev 32] [--fill 1] [--float 1]
//   --float 1: float32 base of any dimension (DataF), otherwise uint8 d=128 (Data).
#include "../include/ssg_search_core.h"
template <class DT> static int build(const DT& D, const std::string& init, const std::string& out, const Opt& o);

int main(int argc, char** argv) {
    std::string base, init, out; Opt o; bool flt = false;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i], v = argv[i + 1];
        if (a == "--base") base = v; else if (a == "--init") init = v; else if (a == "--out") out = v;
        else if (a == "--R") o.R = std::stoul(v); else if (a == "--Lc") o.Lc = std::stoul(v); else if (a == "--C") o.C = std::stoul(v);
        else if (a == "--alpha") o.alpha = std::stof(v); else if (a == "--rounds") o.rounds = std::stoul(v);
        else if (a == "--entries") o.entries = std::stoul(v); else if (a == "--backlink") o.backlink = v != "0";
        else if (a == "--rev") o.rev = std::stoul(v); else if (a == "--save_rounds") o.save_rounds = v != "0"; else if (a == "--fill") o.fill = v != "0";
        else if (a == "--float") flt = v != "0";
    }
    if (flt) { DataF D(base); return build(D, init, out, o); }
    Data D(base); return build(D, init, out, o);
}
template <class DT>
static int build(const DT& D, const std::string& init, const std::string& out, const Opt& o) {
    Graph G = load_graph(init);
    std::fprintf(stderr, "n=%u init R=%u ; R=%u Lc=%u C=%u alpha=%.2f rounds=%u entries=%u backlink=%d\n", D.n, G.R, o.R, o.Lc, o.C, o.alpha, o.rounds, o.entries, int(o.backlink));
    for (uint32_t r = 0; r < o.rounds; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        Graph rev = reverse_graph(G, o.rev);
        Graph N = one_round(D, G, o, &rev);
        if (o.backlink) { Graph rev2 = reverse_graph(N, o.rev); N = one_round(D, N, [&] { Opt p = o; p.Lc = 1; return p; }(), &rev2); }
        G = std::move(N);
        std::fprintf(stderr, "round %u done in %.1f s\n", r + 1, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        if (o.save_rounds && r + 1 < o.rounds) save_graph(G, out + ".r" + std::to_string(r + 1));
    }
    save_graph(G, out);
    return 0;
}
