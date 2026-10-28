// ksg_search.cpp — KSG two-layer search (canonical implementation).
//
// Pipeline per query:
//   IVFPQ starters (or starter cache)  ->  KSG / KSG sparse graph, in memory, top-L1
//   -> first min(L1, L2) candidates seed the SSG dense graph on SSD, top-L2
//   -> optional exact rerank of the first rerank_k candidates (dense_ivfpq mode)
//
// Distances:
//   exact       : squared L2 on base vectors (mmap of base.bin)
//   dense_ivfpq : IVFPQ reconstruction distance from a resident/mmap sidecar for
//                 traversal; the closest `dense_refine_neighbors` neighbours of each
//                 expanded node are re-scored exactly; final exact rerank.
//
// This file replaces the historical 7.8k-line search.cpp. It keeps only the
// canonical path and its ablations (ksg_only / ssg_only / starter_only); the
// legacy variants are preserved on git branch archive/pre-cleanup-20261008.
// Behaviour of the canonical path is identical to search.cpp at that commit

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include <cblas.h>
#include <omp.h>
#include <sys/stat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

#include "IO_utils.h"
#include "dense_adc.h"
#include "sq4.h"
#include "sqf.h"
#include "pqc.h"
#include "colo_graph.h"
#include "uring_reader.h"
#include "dense_ivfpq.h"
#include "distance.h"
#include "types.h"
#include "utils.h"
#include "visited_set.h"

extern "C" {
int openblas_get_num_threads(void);
void openblas_set_num_threads(int);
}

// ============================================================================
// Parameters
// ============================================================================
struct Params {
    // data
    std::string data_type, vec_file, query_file, gt_file, result_file, log_file = "ksg_search.log";
    // index
    std::string mem_graph_file, disk_graph_file, mem_graph_offsets_file;
    std::string pq_index_file, starter_cache_file, dense_ivfpq_file;
    // search
    std::string mode = "two_layer";  // two_layer | ksg_only | ssg_only | starter_only
    int K = 0, L1 = 0, L2 = 0, beam = 1, starter_count = 0;
    bool random_starter = false;
    int prefetch = 0;
    int mem_neighbor_limit = 0, disk_neighbor_limit = 0;
    std::string search_distance = "exact";  // exact | dense_ivfpq
    int dense_refine_neighbors = 0;
    bool dense_refine_push_all = true;      // --dense-refine-push all|refined
    bool exact_read_order_id = false;       // --exact-read-order candidate|id
    std::string rerank_read_order = "candidate";  // candidate | id
    size_t rerank_k = 0;
    // system
    bool mmap_random = true;          // --mmap-advice random|normal
    bool mem_graph_resident = false;  // copy KSG into anonymous (THP) memory
    bool dense_resident = false;      // copy dense sidecar into memory
    bool adc_prefetch = true;         // --no-adc-prefetch disables: prefetch PQ codes/beta/list ids of all neighbours before scoring
    bool adc_gather = true;           // --no-adc-gather disables; AVX-512 gather: AVX-512 gather for the 16 LUT lookups (changes float summation order)
    bool trace_found = false;         // --trace-found: when are the true neighbours discovered (per layer)
    int colo_io = 0;                  // --colo-io mmap|uring|pread
    bool sq_only = false;             // --sq-only (needs --sq4-file)
    std::string colo_perm;            // --colo-perm prefix: page-locality order (prefix.new2old / prefix.old2new)
    int colo_free = 0;                // --colo-free n: free expansions per sector read
    std::string colo_sq_results;      // --colo-sq-results perm_prefix: EXPERIMENT, unread candidates (SQ distance, ids via the new2old map) also enter the final top-K
    int colo_pipe_batch = 1;          // --colo-pipe-batch k: with --colo-pipe, wait until k reads have completed before processing (1 = each as it arrives)
    bool colo_pipe = false;           // --colo-pipe: pipelined SSG reads (io_uring): keep `beam` reads in flight, process each page as it completes
    bool colo_ssg_off = false;        // --colo-ssg-off: ABLATION, pages are read for exact distances only; their SSG neighbour lists are ignored
    int colo_mem_L = 0;               // --colo-mem-L m: hybrid search - before every SSD round, expand candidates in memory over the KSG (SQ scores) while they rank within the best m (0 = off)
    std::string colo_exact_frontier;  // --colo-exact-frontier perm_prefix: DIAGNOSTIC, score neighbours (SSG frontier and in-memory KSG phase) with exact distances (base vectors) instead of SQ
    int colo_cache = 0;               // --colo-cache n: per-query sector cache
    std::string entry_file;           // --entry-file f: online starters from an in-memory centroid entry index
    std::string dump_expanded;        // --dump-expanded f: binary dump of the SSG nodes expanded by every query (analysis)
    std::string latency_dump;         // --latency-dump f: per-query latency (float32 us, query order, no header; same format as BL7_LAT_DUMP)
    std::string colo_file;            // --colo-file f: co-located SSG (see include/colo_graph.h); final result = exact distances of expanded nodes
    std::string sq4_file;             // --sq4-file f (+ --sq4-resident): 4-bit sidecar for in-memory refine
    std::string sqf_file;             // --sqf-file f: affine SQ sidecar for float data of any dimension (include/sqf.h)
    bool mem_graph_ssd = false;       // --mem-graph-ssd: KSG lists read from SSD (O_DIRECT pread per access); only offsets in memory
    std::string mem_graph_compact;    // --mem-graph-compact f: compressed heap KSG adjacency (low-memory variant, include/compact_adj.h)
    std::string pqc_file;             // --pqc-file f: heap-resident PQ codes in place of the SQ codes (low-memory variant, include/pqc.h)
    bool sq4_resident = false;
    bool sq4_heap = false;            // --sq4-heap: SQ codes copied into anonymous memory (explicit, cgroup-safe)
    float sq4_mix = 1.f;
    bool io_stats = false;            // --io-stats: per-query SSD read accounting
    float gate_k = 0.f;               // --refine-gate-k g: SSG refine only if ADC dist <= g * K-th best (0 = off)
    bool rerank_skip_known = false;   // --rerank-skip-known: skip exact reads for ids already refined (identical results)
    float stop_eps = 0.f;             // --ssg_graph-stop-eps e: SSG early stop (see LayerOptions); 0 = off
    float refine_gate = 1.1f;         // --refine-gate m: skip exact refine of neighbours with ADC dist > m * worst(top-L); 0 = off
    bool dense_adc = true;            // table-based (ADC) dense distance; --no-dense-adc disables
    bool reuse_bitsets = true;        // --legacy-bitset-alloc disables
};

static void usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --data_type uint8|float --vec base.bin --query query.bin --query_groundtruth_file gt.bin\n"
        "  -K k -L1 l1 -L2 l2 [-B beam] --mem_graph KSG.graph --disk_graph SSG.graph\n"
        "  [--pq_index_file ivfpq] [--starter-cache-file f] [--starter-count n] [--random]\n"
        "  [--mode two_layer|ksg_only|ssg_only|starter_only] (legacy: --single, --starter-only)\n"
        "  [--search-distance exact|dense_ivfpq] [--dense-ivfpq-file f] [--dense-ivfpq-resident] [--no-dense-adc]\n"
        "  [--dense-refine-neighbors n] [--dense-refine-push all|refined] [--rerank-k n]\n"
        "  [--exact-read-order candidate|id] [--rerank-read-order candidate|id]\n"
        "  [--prefetch d] [--mem-neighbor-limit n] [--disk-neighbor-limit n]\n"
        "  [--mem-graph-offsets-file f] [--mem-graph-resident] [--no-resident-hugepage]\n"
        "  [--mmap-advice random|normal] [--visited-set auto|bitmap|hash] [--legacy-bitset-alloc]\n"
        "  [--ssg_graph-stop-eps e] [--refine-gate-k g] [--rerank-skip-known] [--colo-file f] [--sq-only] [--colo-perm prefix] [--colo-free n] [--colo-cache n] [--entry-file f] [--colo-io mmap|uring|pread] [--sq4-file f] [--sq4-resident] [--sq4-heap] [--pqc-file f] [--mem-graph-compact f] [--mem-graph-ssd] [--io-stats] [--trace-found] [--refine-gate m] [--no-adc-prefetch] [--no-adc-gather] [--result_file f] [--log_file f]\n"
        "Accepted no-op legacy flags: --indisk --heap-search\n", prog);
}

static bool g_resident_hugepage = true;

static bool parse(int argc, char** argv, Params& p) {
    bool single = false, starter_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--data_type") p.data_type = next();
        else if (a == "--vec") p.vec_file = next();
        else if (a == "--query") p.query_file = next();
        else if (a == "--query_groundtruth_file") p.gt_file = next();
        else if (a == "--result_file") p.result_file = next();
        else if (a == "--log_file") p.log_file = next();
        else if (a == "--mem_graph" || a == "-G1") p.mem_graph_file = next();
        else if (a == "--disk_graph" || a == "-G2") p.disk_graph_file = next();
        else if (a == "--mem-graph-offsets-file") p.mem_graph_offsets_file = next();
        else if (a == "--pq_index_file" || a == "-P") p.pq_index_file = next();
        else if (a == "--starter-cache-file") p.starter_cache_file = next();
        else if (a == "--dense-ivfpq-file") p.dense_ivfpq_file = next();
        else if (a == "--mode") p.mode = next();
        else if (a == "--single") single = true;
        else if (a == "--starter-only") starter_only = true;
        else if (a == "-K") p.K = std::stoi(next());
        else if (a == "-L1") p.L1 = std::stoi(next());
        else if (a == "-L2") p.L2 = std::stoi(next());
        else if (a == "-B" || a == "--beam") p.beam = std::stoi(next());
        else if (a == "--starter-count" || a == "--starter_count") p.starter_count = std::stoi(next());
        else if (a == "--random") p.random_starter = true;
        else if (a == "--prefetch") p.prefetch = std::stoi(next());
        else if (a == "--mem-neighbor-limit") p.mem_neighbor_limit = std::stoi(next());
        else if (a == "--disk-neighbor-limit") p.disk_neighbor_limit = std::stoi(next());
        else if (a == "--search-distance") p.search_distance = next();
        else if (a == "--dense-refine-neighbors") p.dense_refine_neighbors = std::stoi(next());
        else if (a == "--dense-refine-push") p.dense_refine_push_all = next() == "all";
        else if (a == "--exact-read-order") p.exact_read_order_id = next() == "id";
        else if (a == "--rerank-read-order") p.rerank_read_order = next();
        else if (a == "--rerank-k") p.rerank_k = static_cast<size_t>(std::stoull(next()));
        else if (a == "--dense-ivfpq-resident") p.dense_resident = true;
        else if (a == "--dense-adc") p.dense_adc = true;
        else if (a == "--no-dense-adc") p.dense_adc = false;
        else if (a == "--io-stats") p.io_stats = true;
        else if (a == "--sq4-file") p.sq4_file = next();
        else if (a == "--sqf-file") p.sqf_file = next();
        else if (a == "--pqc-file") p.pqc_file = next();
        else if (a == "--mem-graph-compact") p.mem_graph_compact = next();
        else if (a == "--mem-graph-ssd") p.mem_graph_ssd = true;
        else if (a == "--colo-file") p.colo_file = next();
        else if (a == "--dump-expanded") p.dump_expanded = next();
        else if (a == "--latency-dump") p.latency_dump = next();
        else if (a == "--sq-only") p.sq_only = true;
        else if (a == "--colo-perm") p.colo_perm = next();
        else if (a == "--colo-free") p.colo_free = std::stoi(next());
        else if (a == "--colo-mem-L") p.colo_mem_L = std::stoi(next());
        else if (a == "--colo-ssg-off") p.colo_ssg_off = true;
        else if (a == "--colo-pipe") p.colo_pipe = true;
        else if (a == "--colo-pipe-batch") p.colo_pipe_batch = std::max(1, std::stoi(next()));
        else if (a == "--colo-sq-results") p.colo_sq_results = next();
        else if (a == "--colo-exact-frontier") p.colo_exact_frontier = next();
        else if (a == "--colo-cache") p.colo_cache = std::stoi(next());
        else if (a == "--entry-file") p.entry_file = next();
        else if (a == "--colo-io") { const std::string v = next(); p.colo_io = v == "uring" ? 1 : v == "pread" ? 2 : 0; }
        else if (a == "--sq4-resident") p.sq4_resident = true;
        else if (a == "--sq4-heap") p.sq4_heap = true;
        else if (a == "--sq4-mix") p.sq4_mix = std::stof(next());
        else if (a == "--ssg_graph-stop-eps") p.stop_eps = std::stof(next());
        else if (a == "--refine-gate-k") p.gate_k = std::stof(next());
        else if (a == "--rerank-skip-known") p.rerank_skip_known = true;
        else if (a == "--trace-found") p.trace_found = true;
        else if (a == "--adc-gather") p.adc_gather = true;
        else if (a == "--no-adc-gather") p.adc_gather = false;
        else if (a == "--adc-prefetch") p.adc_prefetch = true;
        else if (a == "--no-adc-prefetch") p.adc_prefetch = false;
        else if (a == "--refine-gate") p.refine_gate = std::stof(next());
        else if (a == "--mem-graph-resident") p.mem_graph_resident = true;
        else if (a == "--no-resident-hugepage") g_resident_hugepage = false;
        else if (a == "--mmap-advice") {
            const std::string v = next();
            if (v != "random" && v != "normal") throw std::invalid_argument("--mmap-advice random|normal");
            p.mmap_random = v == "random";
        } else if (a == "--visited-set") {
            const std::string v = next();
            if (v == "auto") g_visited_mode = VisitedMode::Auto;
            else if (v == "bitmap") g_visited_mode = VisitedMode::Bitmap;
            else if (v == "hash") g_visited_mode = VisitedMode::Hash;
            else throw std::invalid_argument("--visited-set auto|bitmap|hash");
        } else if (a == "--legacy-bitset-alloc") p.reuse_bitsets = false;
        else if (a == "--indisk" || a == "--heap-search" || a == "--heap_search") { /* historical, always on */ }
        else { std::fprintf(stderr, "Unknown argument: %s\n", a.c_str()); return false; }
    }
    if (single && starter_only) { std::fprintf(stderr, "--single and --starter-only are exclusive\n"); return false; }
    if (starter_only) p.mode = "starter_only";
    if (single) p.mode = p.mem_graph_file.empty() ? "ssg_only" : "ksg_only";
    const bool known_mode = p.mode == "two_layer" || p.mode == "ksg_only" || p.mode == "ssg_only" || p.mode == "starter_only";
    if (!known_mode) { std::fprintf(stderr, "unknown --mode %s\n", p.mode.c_str()); return false; }
    if (p.data_type != "uint8" && p.data_type != "float") { std::fprintf(stderr, "--data_type uint8|float required\n"); return false; }
    if (p.vec_file.empty() || p.query_file.empty() || p.gt_file.empty()) { std::fprintf(stderr, "--vec, --query, --query_groundtruth_file required\n"); return false; }
    if (p.K <= 0 || p.beam <= 0) { std::fprintf(stderr, "-K and -B must be positive\n"); return false; }
    if (p.search_distance != "exact" && p.search_distance != "dense_ivfpq") { std::fprintf(stderr, "--search-distance exact|dense_ivfpq\n"); return false; }
    if (p.search_distance == "dense_ivfpq" && p.dense_ivfpq_file.empty()) { std::fprintf(stderr, "dense_ivfpq needs --dense-ivfpq-file\n"); return false; }
    if (p.rerank_read_order != "candidate" && p.rerank_read_order != "id") { std::fprintf(stderr, "--rerank-read-order candidate|id\n"); return false; }
    if (p.mode == "two_layer" && (p.mem_graph_file.empty() || (p.disk_graph_file.empty() && p.colo_file.empty()) || p.L1 <= 0 || p.L2 <= 0)) {
        std::fprintf(stderr, "two_layer needs --mem_graph, --disk_graph (or --colo-file), -L1, -L2\n"); return false;
    }
    if (p.mode == "ksg_only" && (p.mem_graph_file.empty() || p.L1 <= 0)) { std::fprintf(stderr, "ksg_only needs --mem_graph and -L1\n"); return false; }
    if (p.mode == "ssg_only" && (p.disk_graph_file.empty() || p.L2 <= 0)) { std::fprintf(stderr, "ssg_only needs --disk_graph and -L2\n"); return false; }
    if (p.L1 <= 0 && p.starter_count <= 0) { std::fprintf(stderr, "need -L1 (starters = L1/5) or --starter-count\n"); return false; }
    return true;
}

static int effective_starter_count(const Params& p) {
    return p.starter_count > 0 ? p.starter_count : std::max(1, p.L1 / 5);
}

// ============================================================================
// Distances
// ============================================================================
static std::unique_ptr<ksg::DenseIVFPQReader> g_dense;
static std::unique_ptr<ksg::DenseADC> g_adc;
static int g_colo_io = 0;   // 0 = mmap, 1 = io_uring O_DIRECT (batch per beam), 2 = synchronous pread O_DIRECT
static std::unique_ptr<ksg::ColoGraph> g_colo;   // --colo-file: co-located SSG (vector + neighbours in one record)
static thread_local std::vector<Neighbor> t_colo_exact;   // exact distances of the nodes expanded in the current query
static bool g_dump_kind = false;                         // set when --dump-expanded is given
static thread_local std::vector<uint8_t> t_colo_kind;    // per t_colo_exact entry: 0 = expansion with a new sector read, 2 = expansion served without a new read (cache / same round), 1 = free in-sector expansion (--dump-expanded only)
static float g_sq4_mix = 1.f;
static bool g_sq_only = false;   // --sq-only: every neighbour is scored with the in-memory SQ code (no dense ADC, no exact refine)
// --entry-file: in-memory entry structure (k-means centroids + representative node ids); starters are computed ONLINE per query
struct EntryIndex {
    uint32_t C = 0, d = 0, r = 0; std::vector<float> cent; std::vector<float> cnorm; std::vector<uint32_t> reps;
    void load(const std::string& path) {
        std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot open entry file " + path);
        in.read(reinterpret_cast<char*>(&C), 4); in.read(reinterpret_cast<char*>(&d), 4);
        cent.resize(size_t(C) * d); in.read(reinterpret_cast<char*>(cent.data()), cent.size() * 4);
        in.read(reinterpret_cast<char*>(&r), 4); reps.resize(size_t(C) * r); in.read(reinterpret_cast<char*>(reps.data()), reps.size() * 4);
        if (!in) throw std::runtime_error("bad entry file " + path);
        cnorm.resize(C); for (uint32_t c = 0; c < C; ++c) { float s = 0; for (uint32_t j = 0; j < d; ++j) s += cent[size_t(c) * d + j] * cent[size_t(c) * d + j]; cnorm[c] = s; }
    }
    template <class T> void starters(const T* q, size_t want, std::vector<GID_Type>& out) const {
        thread_local std::vector<std::pair<float, uint32_t>> sc; thread_local std::vector<float> qf;
        qf.assign(q, q + d); sc.resize(C);
        for (uint32_t c = 0; c < C; ++c) {   // ||c||^2 - 2 q.c  (||q||^2 is constant)
            const float* v = &cent[size_t(c) * d]; float dot = 0;
            for (uint32_t j = 0; j < d; ++j) dot += qf[j] * v[j];
            sc[c] = {cnorm[c] - 2 * dot, c};
        }
        const size_t nc = std::min<size_t>(C, (want + r - 1) / r);
        std::partial_sort(sc.begin(), sc.begin() + nc, sc.end());
        out.clear();
        for (size_t i = 0; i < nc; ++i) for (uint32_t k = 0; k < r && out.size() < want; ++k) { const uint32_t id = reps[size_t(sc[i].second) * r + k]; if (id != UINT32_MAX) out.push_back(id); }
    }
};
static EntryIndex g_entry; static bool g_has_entry = false;   // --sq4-mix a: refine distance = a*sq4 + (1-a)*adc
static std::unique_ptr<ksg::SQ4> g_sq4;   // --sq4-file: in-memory 4-bit refine (replaces exact refine reads)  // --dense-adc: table-based dense distance

template <typename T>
inline float exact_distance(const T* q, const T* v, size_t dim) { return squared_l2_distance_avx512<T>(q, v, dim); }
template <typename T>
inline float exact_distance(const T* q, const VecMMapReader<T>& vecs, GID_Type id, size_t dim) {
    return squared_l2_distance_avx512<T>(q, vecs.vector_data_unchecked(id), dim);
}

// ---- data-type dispatch for the SQ sidecar and the co-located record vectors (uint8 d=128: SQ4/SQ5 + u8 records;
//      float any d: SQF + fp16 records) ----
static std::unique_ptr<ksg::SQF> g_sqf;   // --sqf-file: affine SQ for float vectors of any dimension
static std::unique_ptr<ksg::PQC> g_pqc;   // --pqc-file: PQ codes replace the SQ codes (takes precedence)
static const void* g_exact_base = nullptr;               // --colo-exact-frontier: base vectors (original ids), uint8 or float
template <class T> inline float exact_frontier_dist(const T* q, uint32_t orig, size_t dim) {   // diagnostic
    if constexpr (std::is_same<T, uint8_t>::value) return exact_distance(q, static_cast<const uint8_t*>(g_exact_base) + size_t(orig) * dim, dim);
    else { const float* x = static_cast<const float*>(g_exact_base) + size_t(orig) * dim; float s = 0; for (size_t j = 0; j < dim; ++j) { const float t = float(q[j]) - x[j]; s += t * t; } return s; }
}
static std::vector<uint32_t> g_exact_n2o;               // layout id -> original id for the exact frontier
static size_t g_exact_dim = 0;
template <class T> inline float sq_dist_t(const T* q, uint64_t id) {
    if (g_exact_base) return exact_frontier_dist<T>(q, g_exact_n2o[id], g_exact_dim);   // DIAGNOSTIC (--colo-exact-frontier): every SQ score exact
    if (g_pqc) return g_pqc->distance(q, id);
    if constexpr (std::is_same<T, uint8_t>::value) return static_cast<float>(g_sq4->distance(q, id));
    else return g_sqf->distance(q, id);
}
template <class T> inline void sq_prefetch_t(uint64_t id) {
    if (g_pqc) g_pqc->prefetch(id);
    else if constexpr (std::is_same<T, uint8_t>::value) g_sq4->prefetch(id); else g_sqf->prefetch(id);
}
static inline float half_to_float(uint16_t h) {   // IEEE fp16 -> fp32 without F16C (tail elements only)
    const uint32_t sgn = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF; uint32_t f;
    if (e == 0) { if (m == 0) f = sgn; else { int k = -1; uint32_t mm = m; do { ++k; mm <<= 1; } while (!(mm & 0x400)); f = sgn | ((127 - 15 - k) << 23) | ((mm & 0x3FF) << 13); } }
    else if (e == 31) f = sgn | 0x7F800000 | (m << 13);
    else f = sgn | ((e + 112) << 23) | (m << 13);
    float r; std::memcpy(&r, &f, 4); return r;
}
template <class T> inline float rec_exact(const T* q, const uint8_t* rec, size_t dim) {
    if constexpr (std::is_same<T, uint8_t>::value) return exact_distance(q, rec, dim);
    else {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(rec); __m512 acc = _mm512_setzero_ps(); size_t j = 0;
        for (; j + 16 <= dim; j += 16) {
            const __m512 t = _mm512_sub_ps(_mm512_loadu_ps(q + j), _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(h + j))));
            acc = _mm512_fmadd_ps(t, t, acc);
        }
        float s2 = _mm512_reduce_add_ps(acc);
        for (; j < dim; ++j) { const float t = q[j] - half_to_float(h[j]); s2 += t * t; }
        return s2;
    }
}
template <class T> inline float exact_frontier_dist(const T* q, uint32_t orig, size_t dim);

// ---------------------------------------------------------------------------
// [trace] --trace-found: for each query, when (in expansions of the current layer) every ground-truth
// top-100 neighbour was first *discovered* (distance computed / seeded). Statistics only.
// ---------------------------------------------------------------------------
static bool g_trace = false;
static const int32_t kTH[] = {0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000};
static constexpr int kNTH = sizeof(kTH) / sizeof(kTH[0]);
// [0]=top-100, [1]=top-10 ; ksg_cum: found by KSG within e KSG expansions ; all_cum: found by KSG or by SSG within e SSG expansions
static std::vector<uint32_t> g_indeg;   // in-degree of every SSG node (only with --trace-found), capped by disk_neighbor_limit
static const uint32_t kIndegEdges[] = {0, 8, 16, 32, 48, 64, 96, 128, 192};   // bucket lower bounds
static constexpr int kNIB = sizeof(kIndegEdges) / sizeof(kIndegEdges[0]);
// discovery class: 0 = KSG phase, 1 = SSG <=10, 2 = <=50, 3 = <=200, 4 = later or never
static std::atomic<uint64_t> g_tr_cls_indeg[5][kNIB], g_tr_cls_rank[5][3];
static std::atomic<uint64_t> g_tr_ksg_cum[2][kNTH], g_tr_all_cum[2][kNTH], g_tr_total[2], g_tr_ksg_exp, g_tr_ssg_exp, g_tr_q;
struct FoundTrace {
    std::vector<GID_Type> gt;      // sorted ground-truth ids (top-100)
    std::vector<int32_t> rk;       // true rank of gt[i]
    std::vector<int32_t> at;       // expansion count at discovery, -1 = not found
    int32_t nexp = 0;
    void begin(const std::vector<GID_Type>& g, size_t k) {
        std::vector<std::pair<GID_Type, int32_t>> v;
        for (size_t i = 0; i < std::min(k, g.size()); ++i) v.emplace_back(g[i], static_cast<int32_t>(i));
        std::sort(v.begin(), v.end());
        gt.clear(); rk.clear();
        for (auto& x : v) { gt.push_back(x.first); rk.push_back(x.second); }
        at.assign(gt.size(), -1); nexp = 0;
    }
    inline void note(GID_Type id) {
        auto it = std::lower_bound(gt.begin(), gt.end(), id);
        if (it != gt.end() && *it == id) { int32_t& a = at[it - gt.begin()]; if (a < 0) a = nexp; }
    }
};

// ---------------------------------------------------------------------------
// [B1] Optional per-query I/O accounting (--io-stats). Counts what an SSD-resident
// implementation would have to read: SSG records (4 KB pages they touch) and exact
// base vectors (refine / rerank / exact traversal), deduplicated per query.
// Disabled by default; the counting branch is a single predictable test.
// ---------------------------------------------------------------------------
static bool g_io_stats = false;
static bool g_track_exact = false;   // --rerank-skip-known: remember ids whose distance is already exact
static VisitedSet& exact_set() { thread_local VisitedSet t; thread_local bool init = false; if (!init) { t.reserve_hash(); init = true; } return t; }
struct IoStats {
    uint64_t ksg_expand = 0, ssg_expand = 0, ssg_pages = 0;
    uint64_t vec_reads = 0, vec_pages = 0, rerank_reads = 0;
    uint64_t ksg_vec = 0, ssg_vec = 0, rerank_skipped = 0;   // refine reads split by layer; rerank reads avoided
};
struct IoTracker {
    IoStats q;            // current query
    VisitedSet ssg_seen;  // page ids touched in this query
    VisitedSet vec_seen;
    bool in_ssg = false;   // layer currently searching (set by graph_search)
    bool init = false;
    void begin() {
        if (!init) { ssg_seen.reserve_hash(); vec_seen.reserve_hash(); init = true; }
        ssg_seen.clear(); vec_seen.clear(); q = IoStats{};
    }
};
static IoTracker& io_tracker() { thread_local IoTracker t; return t; }
static size_t g_ssg_record_bytes = 0, g_ssg_meta_bytes = 0, g_vec_bytes = 0;
inline void io_note_ssg(GID_Type id) {
    IoTracker& t = io_tracker();
    ++t.q.ssg_expand;
    if (g_colo) { if (t.ssg_seen.insert(static_cast<uint32_t>(g_colo->sector_of(id)))) ++t.q.ssg_pages; return; }
    const uint64_t off = g_ssg_meta_bytes + static_cast<uint64_t>(id) * g_ssg_record_bytes;
    for (uint64_t pg = off >> 12; pg <= (off + g_ssg_record_bytes - 1) >> 12; ++pg)
        if (t.ssg_seen.insert(static_cast<uint32_t>(pg))) ++t.q.ssg_pages;
}
inline void io_note_vec(GID_Type id, bool rerank) {
    IoTracker& t = io_tracker();
    if (rerank) ++t.q.rerank_reads; else { ++t.q.vec_reads; if (t.in_ssg) ++t.q.ssg_vec; else ++t.q.ksg_vec; }
    const uint64_t off = 8 + static_cast<uint64_t>(id) * g_vec_bytes;
    for (uint64_t pg = off >> 12; pg <= (off + g_vec_bytes - 1) >> 12; ++pg)
        if (t.vec_seen.insert(static_cast<uint32_t>(pg))) ++t.q.vec_pages;
}

// Traversal distance: dense IVFPQ approximation when enabled, exact otherwise.
template <typename T>
inline float traversal_distance(const T* q, const VecMMapReader<T>& vecs, GID_Type id, size_t dim) {
    if (id >= vecs.num_vectors()) return std::numeric_limits<float>::infinity();
    if (g_adc) return g_adc->distance(id);
    if (g_dense) return g_dense->distance(q, id);
    if (g_io_stats) io_note_vec(id, false);
    return exact_distance(q, vecs, id, dim);
}

// ============================================================================
// Graph search (shared by the KSG and SSG layers)
// ============================================================================
struct LayerOptions {
    int prefetch = 0;
    int neighbor_limit = 0;
    int dense_refine_neighbors = 0;
    bool dense_refine_push_all = true;
    bool exact_read_order_id = false;
    bool reuse_bitsets = true;
    bool is_ssg = false;   // layer identity, only used for I/O accounting
    float refine_gate = 0.f;
    bool adc_prefetch = false;
    const ksg::ColoGraph* colo = nullptr;   // SSG layer only: read neighbours AND the exact vector from one co-located record
    float gate_k = 0.f;     // >0 (SSG): refine only neighbours whose ADC distance <= gate_k * (K-th best so far)
    float stop_eps = 0.f;   // >0: stop when the best unexpanded candidate is farther than (1+eps)^2 * (K-th best); K = stop_k
    size_t stop_k = 100;
};

static VisitedSet& thread_set(int slot, size_t n) {
    thread_local VisitedSet sets[2];
    thread_local size_t caps[2] = {0, 0};
    if (caps[slot] < n) { sets[slot].reserve(n); caps[slot] = n; }
    else sets[slot].clear();
    return sets[slot];
}

// Best-first beam search. `candidates` holds the seeds on entry and the sorted
// top-L result on exit. `visited` marks nodes whose distance was computed in
// this layer; nodes are expanded at most once.
template <typename T>
void graph_search(const std::vector<T>& query, const VecMMapReader<T>& vecs,
                  const GraphMMapRW<GID_Type, GL_Type>& graph, int L, int beam,
                  std::vector<Neighbor>& candidates, VisitedSet& visited, const LayerOptions& opt, FoundTrace* tr = nullptr) {
    if (L <= 0) { candidates.clear(); return; }
    if (g_io_stats) io_tracker().in_ssg = opt.is_ssg;
    const size_t n_graph = graph.size();
    const size_t n_vec = vecs.num_vectors();
    const size_t dim = query.size();
    const size_t cap = static_cast<size_t>(L);

    auto vis = [&](GID_Type id) { const bool nw = visited.insert(id); if (nw && tr) tr->note(id); return nw; };
    VisitedSet local_expanded, local_seeded;
    VisitedSet* expanded = nullptr;
    VisitedSet* seeded = nullptr;
    if (opt.reuse_bitsets) {
        expanded = &thread_set(0, n_vec);
        seeded = &thread_set(1, n_vec);
    } else {
        local_expanded.reserve(n_vec); local_seeded.reserve(n_vec);
        expanded = &local_expanded; seeded = &local_seeded;
    }

    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() > cap) candidates.resize(cap);

    dary_max_heap<4> top;      // current best L (worst on top)
    dary_max_heap<4> topk;     // current best K, only used with stop_eps > 0
    const float stop_factor = (1.f + opt.stop_eps) * (1.f + opt.stop_eps);
    auto topk_add = [&](const Neighbor& nb) { if ((opt.stop_eps > 0.f || opt.gate_k > 0.f) && (topk.size() < opt.stop_k || nb < topk.top())) { topk.push(nb); if (topk.size() > opt.stop_k) topk.pop(); } };
    dary_min_heap<4> frontier; // unexpanded candidates (best on top)
    top.reserve(cap + 1);
    frontier.reserve(cap * 2);
    for (const auto& c : candidates) {
        if (c.id >= n_graph || c.id >= n_vec) continue;
        if (!seeded->insert(c.id)) continue;
        vis(c.id);
        top.push(c);
        topk_add(c);
        frontier.push(c);
        if (top.size() > cap) top.pop();
    }

    auto push = [&](const Neighbor& nb) {
        if (top.size() < cap || nb < top.top()) {
            top.push(nb);
            topk_add(nb);
            frontier.push(nb);
            if (top.size() > cap) top.pop();
        }
    };

    std::vector<Neighbor> expand;
    expand.reserve(static_cast<size_t>(beam));
    std::vector<Neighbor> approx;
    std::vector<size_t> order;

    while (true) {
        expand.clear();
        while (!frontier.empty() && expand.size() < static_cast<size_t>(beam)) {
            const Neighbor c = frontier.top();
            if (top.size() >= cap && top.top() < c) break;  // cannot improve top-L
            if (opt.stop_eps > 0.f && topk.size() >= opt.stop_k && c.distance > stop_factor * topk.top().distance) break;  // beyond the K-th best by margin
            frontier.pop();
            if (!expanded->insert(c.id)) continue;
            expand.push_back(c);
        }
        if (expand.empty()) break;
        // Fixed-size records with close ids are close on disk.
        if (expand.size() > 1)
            std::sort(expand.begin(), expand.end(), [](const Neighbor& a, const Neighbor& b) { return a.id < b.id; });

        thread_local std::vector<uint64_t> colo_sectors;
        thread_local std::vector<const uint8_t*> colo_bufs;
        if constexpr (std::is_same<T, uint8_t>::value) {
            if (opt.colo && g_colo_io != 0) {
                colo_sectors.clear();
                for (const auto& nd : expand) colo_sectors.push_back(nd.id < n_graph ? opt.colo->sector_of(nd.id) : 1);
                colo_bufs.resize(colo_sectors.size());
                static thread_local ksg::UringReader ring(64);
                const int dfd = const_cast<ksg::ColoGraph*>(opt.colo)->direct_fd(opt.colo->path());
                if (g_colo_io == 1) ring.read_sectors(dfd, colo_sectors.data(), colo_sectors.size(), colo_bufs.data());
                else {
                    thread_local uint8_t* sbuf = nullptr;
                    if (!sbuf && posix_memalign(reinterpret_cast<void**>(&sbuf), 4096, 64 * 4096) != 0) throw std::runtime_error("posix_memalign");
                    for (size_t i = 0; i < colo_sectors.size(); ++i) {
                        if (::pread(dfd, sbuf + i * 4096, 4096, static_cast<off_t>(colo_sectors[i] * 4096)) != 4096) throw std::runtime_error("pread failed");
                        colo_bufs[i] = sbuf + i * 4096;
                    }
                }
            }
        }
        for (size_t ei = 0; ei < expand.size(); ++ei) {
            const auto& node = expand[ei];
            if (node.id >= n_graph) continue;
            if (tr) ++tr->nexp;
            if (g_io_stats) { if (opt.is_ssg) io_note_ssg(node.id); else ++io_tracker().q.ksg_expand; }
            const GID_Type* nbrs = nullptr;
            size_t count = 0;
            if constexpr (std::is_same<T, uint8_t>::value) {
                if (opt.colo) {
                    const uint8_t* rec = g_colo_io != 0 ? colo_bufs[ei] + opt.colo->record_offset(node.id) : opt.colo->record(node.id);
                    nbrs = opt.colo->neighbors(rec);
                    count = opt.colo->neighbor_count(rec);
                    t_colo_exact.emplace_back(node.id, exact_distance(query.data(), rec, dim));   // free: same page as the neighbour list
                }
            }
            if (!nbrs) {
                const auto view = graph.get_neighbors_view(node.id);
                nbrs = view.data;
                count = view.size();
            }
            if (opt.neighbor_limit > 0) count = std::min(count, static_cast<size_t>(opt.neighbor_limit));

            if ((g_dense || g_sq_only) && opt.dense_refine_neighbors > 0) {
                // 1) approximate distance for every new neighbour
                approx.clear();
                bool sq_scored = false;
                if (g_sq_only) {
                    thread_local std::vector<GID_Type> fresh2;
                    fresh2.clear();
                    for (size_t i = 0; i < count; ++i) { const GID_Type nb = nbrs[i]; if (nb >= n_vec || !vis(nb)) continue; fresh2.push_back(nb); }
                    for (size_t i = 0; i < fresh2.size() && i < 8; ++i) sq_prefetch_t<T>(fresh2[i]);
                    for (size_t i = 0; i < fresh2.size(); ++i) {
                        if (i + 8 < fresh2.size()) sq_prefetch_t<T>(fresh2[i + 8]);
                        approx.emplace_back(fresh2[i], static_cast<DisType>(sq_dist_t<T>(query.data(), fresh2[i])));
                    }
                    sq_scored = true;
                }
                if (sq_scored) {}
                else if (opt.adc_prefetch && g_adc) {
                    // batched scoring: collect unvisited ids first, then score with software prefetch
                    thread_local std::vector<GID_Type> fresh;
                    thread_local std::vector<float> fd;
                    fresh.clear();
                    for (size_t i = 0; i < count; ++i) {
                        const GID_Type nb = nbrs[i];
                        if (nb >= n_vec || !vis(nb)) continue;
                        fresh.push_back(nb);
                    }
                    fd.resize(fresh.size());
                    g_adc->distance_many(fresh.data(), fresh.size(), fd.data());
                    for (size_t i = 0; i < fresh.size(); ++i) approx.emplace_back(fresh[i], fd[i]);
                }
                else for (size_t i = 0; i < count; ++i) {
                    const GID_Type nb = nbrs[i];
                    if (nb >= n_vec || !vis(nb)) continue;
                    approx.emplace_back(nb, traversal_distance(query.data(), vecs, nb, dim));
                }
                if (approx.empty()) continue;
                // 2) exact re-score of the closest `refine` ones
                size_t refine = std::min(approx.size(), static_cast<size_t>(opt.dense_refine_neighbors));
                if (refine < approx.size()) std::nth_element(approx.begin(), approx.begin() + refine, approx.end());
                if (opt.gate_k > 0.f && topk.size() >= opt.stop_k) {
                    const float thr = topk.top().distance * opt.gate_k;
                    refine = std::partition(approx.begin(), approx.begin() + refine,
                                            [thr](const Neighbor& n) { return n.distance <= thr; }) - approx.begin();
                    if (refine == 0) continue;
                }
                if (opt.refine_gate > 0.f && top.size() >= cap) {
                    // Neighbours whose ADC distance is far beyond the current worst of top-L cannot enter it.
                    const float thr = top.top().distance * opt.refine_gate;
                    refine = std::partition(approx.begin(), approx.begin() + refine,
                                            [thr](const Neighbor& n) { return n.distance <= thr; }) - approx.begin();
                    if (refine == 0) continue;
                }
                order.resize(refine);
                std::iota(order.begin(), order.end(), size_t{0});
                if (opt.exact_read_order_id)
                    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return approx[a].id < approx[b].id; });
                if (g_sq_only) {
                    const size_t n_push_sq = opt.dense_refine_push_all ? approx.size() : refine;
                    for (size_t i = 0; i < n_push_sq; ++i) push(approx[i]);
                    continue;
                }
                if constexpr (std::is_same<T, uint8_t>::value) {
                    if (g_sq4) {
                        // refine on the in-memory SQ4 code: no SSD read; exact vectors are only touched by the final rerank
                        for (size_t idx : order) g_sq4->prefetch(approx[idx].id);
                        for (size_t idx : order) approx[idx].distance = g_sq4_mix >= 1.f ? static_cast<DisType>(g_sq4->distance(query.data(), approx[idx].id))
                                                                : static_cast<DisType>(g_sq4_mix * static_cast<float>(g_sq4->distance(query.data(), approx[idx].id)) + (1.f - g_sq4_mix) * approx[idx].distance);
                        const size_t n_push_sq = opt.dense_refine_push_all ? approx.size() : refine;
                        for (size_t i = 0; i < n_push_sq; ++i) push(approx[i]);
                        continue;
                    }
                }
                if (g_io_stats) for (size_t idx : order) io_note_vec(approx[idx].id, false);
                for (size_t idx : order) vecs.prefetch_vector(approx[idx].id);
                size_t r = 0;
                for (; r + 1 < refine; r += 2) {
                    DisType d0, d1;
                    squared_l2_batch2<T>(query.data(), vecs.vector_data_unchecked(approx[order[r]].id),
                                         vecs.vector_data_unchecked(approx[order[r + 1]].id), dim, d0, d1);
                    approx[order[r]].distance = d0;
                    approx[order[r + 1]].distance = d1;
                }
                if (r < refine) approx[order[r]].distance = exact_distance(query.data(), vecs, approx[order[r]].id, dim);
                if (g_track_exact) for (size_t idx : order) exact_set().insert(approx[idx].id);
                const size_t n_push = opt.dense_refine_push_all ? approx.size() : refine;
                for (size_t i = 0; i < n_push; ++i) push(approx[i]);
            } else if (opt.prefetch > 0) {
                const size_t pf = static_cast<size_t>(opt.prefetch);
                for (size_t i = 0; i < count; ++i) {
                    if (i + pf < count && nbrs[i + pf] < n_vec) vecs.prefetch_vector(nbrs[i + pf]);
                    const GID_Type nb = nbrs[i];
                    if (nb >= n_vec || !vis(nb)) continue;
                    push(Neighbor(nb, traversal_distance(query.data(), vecs, nb, dim)));
                }
            } else {
                size_t i = 0;
                for (; i + 1 < count; i += 2) {
                    const GID_Type n0 = nbrs[i], n1 = nbrs[i + 1];
                    const bool v0 = n0 < n_vec && vis(n0);
                    const bool v1 = n1 < n_vec && vis(n1);
                    if (v0 && v1 && !g_dense) {
                        if (g_io_stats) { io_note_vec(n0, false); io_note_vec(n1, false); }
                        DisType d0, d1;
                        squared_l2_batch2<T>(query.data(), vecs.vector_data_unchecked(n0),
                                             vecs.vector_data_unchecked(n1), dim, d0, d1);
                        push(Neighbor(n0, d0));
                        push(Neighbor(n1, d1));
                    } else {
                        if (v0) push(Neighbor(n0, traversal_distance(query.data(), vecs, n0, dim)));
                        if (v1) push(Neighbor(n1, traversal_distance(query.data(), vecs, n1, dim)));
                    }
                }
                for (; i < count; ++i) {
                    const GID_Type nb = nbrs[i];
                    if (nb >= n_vec || !vis(nb)) continue;
                    push(Neighbor(nb, traversal_distance(query.data(), vecs, nb, dim)));
                }
            }
        }
    }

    candidates.clear();
    candidates.reserve(top.size());
    while (!top.empty()) { candidates.push_back(top.top()); top.pop(); }
    std::sort(candidates.begin(), candidates.end());
}

// ---------------------------------------------------------------------------
// SSG search over the CO-LOCATED layout with SQ-only neighbour scoring (memory-budget configuration).
// One expansion = one 4 KB sector read (io_uring batch per beam, or mmap); the sector also carries the exact vectors of the other
// records stored with it. With a page-locality ordering (--colo-perm) those are near neighbours; --colo-free lets the search expand
// them "for free" (no extra I/O) when their exact distance would enter the current top-L.
// Ids inside this function are layout ids (new ids when a permutation is used); SQ lookups and results are mapped back.
// ---------------------------------------------------------------------------
static std::vector<uint32_t> g_perm_n2o, g_perm_o2n;   // --colo-perm prefix
static int g_colo_free = 0;                              // --colo-free n: max free expansions per sector read (0 = off)
static std::vector<uint32_t> g_sqres_n2o;                // --colo-sq-results: layout id -> original id
static int g_colo_pipe_batch = 1;                        // --colo-pipe-batch (see Params)
static bool g_colo_pipe = false;                         // --colo-pipe (see Params)
static bool g_colo_ssg_off = false;                      // --colo-ssg-off (see Params)
static int g_colo_mem_L = 0;                             // --colo-mem-L (see Params)
static const GraphMMapRW<GID_Type, GL_Type>* g_ksg = nullptr;   // KSG for the hybrid in-memory phase (layout ids)
static int g_colo_cache = 0;                             // --colo-cache n: per-query cache of the last n sectors read (0 = off)
struct ColoStats { uint64_t sectors = 0, expand = 0, free_expand = 0, cache_hits = 0, mem_expand = 0; };
static thread_local ColoStats t_colo_stats;
template <typename T>
static void graph_search_colo_sq(const std::vector<T>& query, const ksg::ColoGraph& colo, int L, int beam,
                                 std::vector<Neighbor>& seeds, VisitedSet& visited, const LayerOptions& opt) {
    const size_t cap = static_cast<size_t>(L), n = colo.size(), dim = query.size();
    const bool perm = !g_perm_n2o.empty();
    auto sqid = [&](uint32_t v) -> uint64_t { return perm ? g_perm_n2o[v] : v; };
    VisitedSet& expanded = thread_set(0, n);
    dary_max_heap<4> top; dary_min_heap<4> frontier; top.reserve(cap + 1); frontier.reserve(cap * 2);
    // hybrid (--colo-mem-L m): every candidate also enters an in-memory frontier; mtop keeps the best m distances seen
    const size_t mcap = static_cast<size_t>(g_colo_mem_L);
    dary_min_heap<4> mfront; dary_max_heap<4> mtop;
    VisitedSet* mexp = mcap ? &thread_set(1, n) : nullptr;
    auto push = [&](const Neighbor& nb) {
        if (top.size() < cap || nb < top.top()) { top.push(nb); frontier.push(nb); if (top.size() > cap) top.pop(); }
        if (mcap && (mtop.size() < mcap || nb < mtop.top())) { mtop.push(nb); mfront.push(nb); if (mtop.size() > mcap) mtop.pop(); }
    };
    // --ssg_graph-stop-eps e: stop once the best unexpanded candidate (SQ distance) is beyond (1+e)^2 x the K-th best EXACT distance seen
    dary_max_heap<4> exk; const float stop_factor = (1.f + opt.stop_eps) * (1.f + opt.stop_eps);
    auto note_exact = [&](uint32_t id, float d) {
        if (opt.stop_eps > 0.f && (exk.size() < opt.stop_k || d < exk.top().distance)) { exk.push(Neighbor(id, d)); if (exk.size() > opt.stop_k) exk.pop(); }
    };
    for (const auto& c0 : seeds) {   // seeds arrive with original ids
        const uint32_t v = perm ? g_perm_o2n[c0.id] : c0.id;
        if (v >= n || !visited.insert(v)) continue;
        push(Neighbor(v, c0.distance));
    }
    thread_local std::vector<Neighbor> expand_list; thread_local std::vector<uint64_t> secs; thread_local std::vector<const uint8_t*> bufs;
    thread_local std::vector<Neighbor> approx; thread_local std::vector<uint32_t> fresh;
    static thread_local ksg::UringReader ring(64);
    thread_local uint8_t* sbuf = nullptr;
    thread_local std::vector<uint8_t> qcache_buf; thread_local std::vector<uint64_t> qcache_sec;
    size_t qcache_n = 0, qcache_next = 0;
    if (g_colo_cache > 0 && qcache_buf.size() != static_cast<size_t>(g_colo_cache) * 4096) { qcache_buf.assign(static_cast<size_t>(g_colo_cache) * 4096, 0); qcache_sec.assign(g_colo_cache, 0); }
    // sector -> cache slot: open addressing (8x the slots), per-query epoch instead of clearing, tombstone on eviction.
    // Same contents and ring eviction as the former linear scan, so results are identical; lookups are O(1).
    struct SecIndex { std::vector<uint64_t> key; std::vector<uint32_t> val, stamp; uint32_t epoch = 0; size_t mask = 0, used = 0; };
    thread_local SecIndex qidx;
    constexpr uint32_t kTomb = UINT32_MAX;
    if (g_colo_cache > 0) {
        size_t sz = 16; while (sz < static_cast<size_t>(g_colo_cache) * 8) sz <<= 1;
        if (qidx.key.size() != sz) { qidx.key.assign(sz, 0); qidx.val.assign(sz, 0); qidx.stamp.assign(sz, 0); qidx.epoch = 0; qidx.mask = sz - 1; }
        if (++qidx.epoch == 0) { std::fill(qidx.stamp.begin(), qidx.stamp.end(), 0); qidx.epoch = 1; }
        qidx.used = 0;
    }
    auto qhash = [&](uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; return static_cast<size_t>(k) & qidx.mask; };
    auto qfind = [&](uint64_t sct) -> const uint8_t* {
        if (g_colo_cache <= 0) return nullptr;
        for (size_t h = qhash(sct); qidx.stamp[h] == qidx.epoch; h = (h + 1) & qidx.mask)
            if (qidx.key[h] == sct) return qidx.val[h] == kTomb ? nullptr : qcache_buf.data() + static_cast<size_t>(qidx.val[h]) * 4096;
        return nullptr;
    };
    auto qput = [&](uint64_t sct, const uint8_t* b) {
        if (g_colo_cache <= 0) return;
        const size_t slot = qcache_next++ % static_cast<size_t>(g_colo_cache);
        if (qcache_n == static_cast<size_t>(g_colo_cache)) {   // evict the sector held by this slot
            const uint64_t old = qcache_sec[slot];
            for (size_t h = qhash(old); qidx.stamp[h] == qidx.epoch; h = (h + 1) & qidx.mask)
                if (qidx.key[h] == old && qidx.val[h] == slot) { qidx.val[h] = kTomb; break; }
        }
        std::memcpy(qcache_buf.data() + slot * 4096, b, 4096); qcache_sec[slot] = sct;
        if (qcache_n < static_cast<size_t>(g_colo_cache)) ++qcache_n;
        auto ins = [&](uint64_t k, uint32_t v) {   // first tombstone or empty entry of the probe chain
            size_t h = qhash(k);
            while (qidx.stamp[h] == qidx.epoch && qidx.val[h] != kTomb) h = (h + 1) & qidx.mask;
            if (qidx.stamp[h] != qidx.epoch) ++qidx.used;
            qidx.stamp[h] = qidx.epoch; qidx.key[h] = k; qidx.val[h] = v;
        };
        if (qidx.used + 1 > (qidx.mask + 1) / 2) {   // too many tombstones: rebuild from the live slots
            if (++qidx.epoch == 0) { std::fill(qidx.stamp.begin(), qidx.stamp.end(), 0); qidx.epoch = 1; }
            qidx.used = 0;
            for (size_t i = 0; i < qcache_n; ++i) if (i != slot) ins(qcache_sec[i], static_cast<uint32_t>(i));
        }
        ins(sct, static_cast<uint32_t>(slot));
    };
    const int dfd = g_colo_io != 0 ? const_cast<ksg::ColoGraph&>(colo).direct_fd(colo.path()) : -1;
    // score the neighbours of one record and push the best ones (same rule as the SQ-only path of graph_search)
    auto relax = [&](const uint8_t* rec) {
        if (g_colo_ssg_off) return;   // ablation: SSG neighbours not used (expansion only through the KSG)
        const uint32_t* nb = colo.neighbors(rec);
        size_t cnt = colo.neighbor_count(rec);
        if (opt.neighbor_limit > 0) cnt = std::min(cnt, static_cast<size_t>(opt.neighbor_limit));
        fresh.clear();
        for (size_t i = 0; i < cnt; ++i) { const uint32_t v = nb[i]; if (v < n && visited.insert(v)) fresh.push_back(v); }
        approx.clear();
        for (size_t i = 0; i < fresh.size() && i < 8; ++i) sq_prefetch_t<T>(sqid(fresh[i]));
        for (size_t i = 0; i < fresh.size(); ++i) {
            if (i + 8 < fresh.size()) sq_prefetch_t<T>(sqid(fresh[i + 8]));
            approx.emplace_back(fresh[i], g_exact_base ? exact_frontier_dist<T>(query.data(), g_exact_n2o[fresh[i]], dim)
                                                       : static_cast<DisType>(sq_dist_t<T>(query.data(), sqid(fresh[i]))));
        }
        if (approx.empty()) return;
        size_t keep = std::min(approx.size(), static_cast<size_t>(opt.dense_refine_neighbors));
        if (keep < approx.size()) std::nth_element(approx.begin(), approx.begin() + keep, approx.end());
        if (opt.refine_gate > 0.f && top.size() >= cap) {
            const float thr = top.top().distance * opt.refine_gate;
            keep = std::partition(approx.begin(), approx.begin() + keep, [thr](const Neighbor& x) { return x.distance <= thr; }) - approx.begin();
        }
        for (size_t i = 0; i < keep; ++i) push(approx[i]);
    };
    // in-memory phase: best-first over the KSG with SQ scores, restricted to the best mcap candidates (no I/O)
    thread_local std::vector<Neighbor> mapprox;
    auto memory_phase = [&]() {
        while (!mfront.empty()) {
            const Neighbor c = mfront.top();
            if (mtop.size() >= mcap && mtop.top() < c) break;
            mfront.pop();
            if (!mexp->insert(c.id)) continue;
            ++t_colo_stats.mem_expand;
            const auto nv = g_ksg->get_neighbors_view(static_cast<GL_Type>(c.id));
            mapprox.clear();
            for (size_t i = 0; i < static_cast<size_t>(nv.count); ++i) { const uint32_t v = nv.data[i]; if (v < n && visited.insert(v)) sq_prefetch_t<T>(sqid(v)); else continue; mapprox.emplace_back(v, 0); }
            for (auto& a2 : mapprox) a2.distance = g_exact_base ? static_cast<DisType>(exact_frontier_dist<T>(query.data(), g_exact_n2o[a2.id], dim))
                                                                : static_cast<DisType>(sq_dist_t<T>(query.data(), sqid(a2.id)));
            for (const auto& a2 : mapprox) push(a2);
        }
    };
    if (g_colo_pipe) {
        // Pipelined variant: up to `beam` sector reads stay in flight; every page is processed as soon as its read completes,
        // then the in-memory phase runs and the freed slots are refilled with the best candidates (same selection and stop rules).
        auto cache_lookup = [&](uint64_t sct) -> const uint8_t* {
            return qfind(sct);
        };
        auto cache_insert = [&](uint64_t sct, const uint8_t* b) {
            if (g_colo_cache <= 0) return;
            qput(sct, b);
        };
        auto process = [&](uint32_t u, const uint8_t* sec, uint64_t sct, uint8_t kind) {
            const uint8_t* rec = sec + colo.record_offset(u);
            ++t_colo_stats.expand;
            t_colo_exact.emplace_back(colo.has_orig() ? colo.orig_id(rec) : u, rec_exact<T>(query.data(), rec, dim));
            note_exact(u, t_colo_exact.back().distance);
            if (g_dump_kind) t_colo_kind.push_back(kind);
            relax(rec);
            if (g_colo_free > 0) {
                const uint32_t first = static_cast<uint32_t>((sct - 1) * colo.per_sector());
                int used = 0;
                for (uint32_t k = 0; k < colo.per_sector() && used < g_colo_free; ++k) {
                    const uint32_t v = first + k;
                    if (v >= n || v == u || expanded.contains(v)) continue;
                    const uint8_t* rv = sec + static_cast<size_t>(k) * colo.rec_len();
                    const float dv = rec_exact<T>(query.data(), rv, dim);
                    if (top.size() >= cap && !(Neighbor(v, dv) < top.top())) continue;
                    expanded.insert(v); visited.insert(v);
                    if (top.size() < cap || Neighbor(v, dv) < top.top()) { top.push(Neighbor(v, dv)); if (top.size() > cap) top.pop(); }
                    t_colo_exact.emplace_back(colo.has_orig() ? colo.orig_id(rv) : v, dv); note_exact(v, dv); if (g_dump_kind) t_colo_kind.push_back(1);
                    ++t_colo_stats.free_expand; ++used;
                    relax(rv);
                }
            }
        };
        struct Pend { uint64_t sct; std::vector<uint32_t> nodes; };
        thread_local std::vector<Pend> pend; pend.clear();
        for (;;) {
            if (mcap) memory_phase();
            while (ring.inflight() < static_cast<unsigned>(beam) && !frontier.empty()) {
                const Neighbor c = frontier.top();
                if (top.size() >= cap && top.top() < c) break;
                if (opt.stop_eps > 0.f && exk.size() >= opt.stop_k && c.distance > stop_factor * exk.top().distance) break;
                frontier.pop();
                if (!expanded.insert(c.id)) continue;
                const uint64_t sct = colo.sector_of(c.id);
                if (const uint8_t* hb = cache_lookup(sct)) { ++t_colo_stats.cache_hits; process(c.id, hb, sct, 2); continue; }
                bool attached = false;
                for (auto& pe : pend) if (pe.sct == sct) { pe.nodes.push_back(c.id); attached = true; break; }
                if (attached) continue;
                pend.push_back(Pend{sct, {c.id}});
                if (!ring.submit(dfd, sct, sct)) throw std::runtime_error("io_uring: no free slot");
                ++t_colo_stats.sectors;
            }
            if (ring.inflight() == 0) break;
            ring.flush();
            ring.wait_for(static_cast<unsigned>(g_colo_pipe_batch));
            uint64_t tag; const uint8_t* buf; unsigned slot;
            while (ring.reap(false, tag, buf, slot)) {
                size_t pi = 0; while (pi < pend.size() && pend[pi].sct != tag) ++pi;
                if (pi == pend.size()) throw std::runtime_error("io_uring: unexpected completion");
                std::vector<uint32_t> nodes; nodes.swap(pend[pi].nodes); pend[pi] = std::move(pend.back()); pend.pop_back();
                for (size_t k = 0; k < nodes.size(); ++k) process(nodes[k], buf, tag, k == 0 ? 0 : 2);
                cache_insert(tag, buf); ring.release(slot);
            }
        }
        if (!g_sqres_n2o.empty())
            for (const auto& c : top.data) if (!expanded.contains(c.id)) t_colo_exact.emplace_back(g_sqres_n2o[c.id], c.distance);
        return;
    }
    while (true) {
        if (mcap) memory_phase();
        expand_list.clear();
        while (!frontier.empty() && expand_list.size() < static_cast<size_t>(beam)) {
            const Neighbor c = frontier.top();
            if (top.size() >= cap && top.top() < c) break;
            if (opt.stop_eps > 0.f && exk.size() >= opt.stop_k && c.distance > stop_factor * exk.top().distance) break;
            frontier.pop();
            if (!expanded.insert(c.id)) continue;
            expand_list.push_back(c);
        }
        if (expand_list.empty()) break;
        secs.clear();
        for (const auto& e : expand_list) secs.push_back(colo.sector_of(e.id));
        bufs.resize(secs.size());
        // per-query sector cache: sectors already read in this query are served from memory
        thread_local std::vector<uint64_t> miss_secs; thread_local std::vector<size_t> miss_idx; thread_local std::vector<const uint8_t*> miss_bufs;
        miss_secs.clear(); miss_idx.clear();
        thread_local std::vector<uint8_t> newread; newread.assign(secs.size(), 0);
        for (size_t i = 0; i < secs.size(); ++i) {
            const uint8_t* hit = nullptr;
            hit = qfind(secs[i]);
            if (hit) { bufs[i] = hit; ++t_colo_stats.cache_hits; }
            else {
                bool dup = false;
                for (size_t j = 0; j < miss_secs.size(); ++j) if (miss_secs[j] == secs[i]) { dup = true; miss_idx.push_back(j); break; }
                if (!dup) { miss_idx.push_back(miss_secs.size()); miss_secs.push_back(secs[i]); newread[i] = 1; }
                bufs[i] = nullptr;
            }
        }
        miss_bufs.resize(miss_secs.size());
        if (g_colo_io == 1) ring.read_sectors(dfd, miss_secs.data(), miss_secs.size(), miss_bufs.data());
        else if (g_colo_io == 2) {
            if (!sbuf && posix_memalign(reinterpret_cast<void**>(&sbuf), 4096, 64 * 4096) != 0) throw std::runtime_error("posix_memalign");
            for (size_t i = 0; i < miss_secs.size(); ++i) { if (::pread(dfd, sbuf + i * 4096, 4096, static_cast<off_t>(miss_secs[i] * 4096)) != 4096) throw std::runtime_error("pread"); miss_bufs[i] = sbuf + i * 4096; }
        } else for (size_t i = 0; i < miss_secs.size(); ++i) miss_bufs[i] = colo.record(static_cast<uint32_t>((miss_secs[i] - 1) * colo.per_sector()));
        t_colo_stats.sectors += miss_secs.size();
        { size_t mi = 0; for (size_t i = 0; i < secs.size(); ++i) if (!bufs[i]) bufs[i] = miss_bufs[miss_idx[mi++]]; }
        for (size_t ei = 0; ei < expand_list.size(); ++ei) {
            const uint32_t u = expand_list[ei].id;
            const uint8_t* sec = bufs[ei];
            const uint8_t* rec = sec + colo.record_offset(u);
            ++t_colo_stats.expand;
            t_colo_exact.emplace_back(colo.has_orig() ? colo.orig_id(rec) : u, rec_exact<T>(query.data(), rec, dim));
            note_exact(u, t_colo_exact.back().distance);
            if (g_dump_kind) t_colo_kind.push_back(newread[ei] ? 0 : 2);
            relax(rec);
            if (g_colo_free > 0) {
                // the other records of this sector came with the same read
                const uint32_t first = static_cast<uint32_t>((secs[ei] - 1) * colo.per_sector());
                int used = 0;
                for (uint32_t k = 0; k < colo.per_sector() && used < g_colo_free; ++k) {
                    const uint32_t v = first + k;
                    if (v >= n || v == u || expanded.contains(v)) continue;
                    const uint8_t* rv = sec + static_cast<size_t>(k) * colo.rec_len();
                    const float dv = rec_exact<T>(query.data(), rv, dim);
                    if (top.size() >= cap && !(Neighbor(v, dv) < top.top())) continue;
                    expanded.insert(v); visited.insert(v);
                    if (top.size() < cap || Neighbor(v, dv) < top.top()) { top.push(Neighbor(v, dv)); if (top.size() > cap) top.pop(); }
                    t_colo_exact.emplace_back(colo.has_orig() ? colo.orig_id(rv) : v, dv); note_exact(v, dv); if (g_dump_kind) t_colo_kind.push_back(1);
                    ++t_colo_stats.free_expand; ++used;
                    relax(rv);
                }
            }
        }
        // remember the sectors read in this round (after use, so cache slots referenced above are not overwritten mid-round)
        if (g_colo_cache > 0) for (size_t j = 0; j < miss_secs.size(); ++j) {
            qput(miss_secs[j], miss_bufs[j]);
        }
    }
    if (!g_sqres_n2o.empty())   // unread candidates enter the result list with their SQ distance (original id via the map)
        for (const auto& c : top.data) if (!expanded.contains(c.id)) t_colo_exact.emplace_back(g_sqres_n2o[c.id], c.distance);
}

// Exact rerank of the first `k` candidates; the result keeps only those k.
template <typename T>
void rerank_exact(const std::vector<T>& query, const VecMMapReader<T>& vecs, size_t k, bool by_id,
                  std::vector<Neighbor>& candidates) {
    if (!k || candidates.empty()) return;
    const size_t count_all = std::min(k, candidates.size());
    size_t count = count_all;
    thread_local std::vector<GID_Type> ids;
    thread_local std::vector<Neighbor> exact;
    ids.clear();
    thread_local std::vector<Neighbor> known;
    known.clear();
    for (size_t i = 0; i < count; ++i) {
        if (g_track_exact && exact_set().contains(candidates[i].id)) { known.push_back(candidates[i]); continue; }   // distance already exact
        ids.push_back(candidates[i].id);
    }
    if (g_io_stats) io_tracker().q.rerank_skipped += known.size();
    if (by_id) std::sort(ids.begin(), ids.end());
    if (g_io_stats) for (GID_Type id : ids) io_note_vec(id, true);
    for (GID_Type id : ids) vecs.prefetch_vector(id);
    exact.clear();
    count = ids.size();
    for (const auto& kn : known) exact.push_back(kn);
    size_t i = 0;
    for (; i + 1 < count; i += 2) {
        DisType d0, d1;
        squared_l2_batch2<T>(query.data(), vecs.vector_data_unchecked(ids[i]), vecs.vector_data_unchecked(ids[i + 1]),
                             query.size(), d0, d1);
        exact.emplace_back(ids[i], d0);
        exact.emplace_back(ids[i + 1], d1);
    }
    if (i < count) exact.emplace_back(ids[i], exact_distance(query.data(), vecs, ids[i], query.size()));
    std::sort(exact.begin(), exact.end());
    candidates.swap(exact);
}

// ============================================================================
// Starters
// ============================================================================
struct StarterCacheHeader {
    char magic[8] = {'K', 'S', 'G', 'S', 'T', 'R', '0', '1'};
    uint32_t version = 1;
    uint32_t id_bytes = sizeof(GID_Type);
    uint64_t query_count = 0;
    uint64_t vector_count = 0;
    uint32_t starter_count = 0;
    uint32_t reserved = 0;
};

static bool load_starter_cache(const std::string& path, size_t nq, int sl, size_t nv,
                               std::vector<std::vector<GID_Type>>& out) {
    if (path.empty()) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    StarterCacheHeader h{};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    const char magic[8] = {'K', 'S', 'G', 'S', 'T', 'R', '0', '1'};
    if (!in || std::memcmp(h.magic, magic, 8) != 0 || h.version != 1 || h.id_bytes != sizeof(GID_Type) ||
        h.query_count != nq || h.starter_count != static_cast<uint32_t>(sl) || h.vector_count != nv) {
        g_logger->warn("Ignoring incompatible starter cache: {}", path);
        return false;
    }
    out.assign(nq, {});
    for (size_t i = 0; i < nq; ++i) {
        uint32_t c = 0;
        in.read(reinterpret_cast<char*>(&c), sizeof(c));
        if (!in || c > static_cast<uint32_t>(sl)) { out.clear(); g_logger->warn("Truncated starter cache: {}", path); return false; }
        out[i].resize(c);
        if (c) in.read(reinterpret_cast<char*>(out[i].data()), c * sizeof(GID_Type));
        if (!in) { out.clear(); g_logger->warn("Truncated starter cache: {}", path); return false; }
    }
    g_logger->info("Loaded starter cache {} (queries={}, starters={})", path, nq, sl);
    return true;
}

static void save_starter_cache(const std::string& path, size_t nv, int sl, const std::vector<std::vector<GID_Type>>& s) {
    if (path.empty()) return;
    const std::string tmp = path + ".building";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) { g_logger->warn("Cannot create starter cache {}", tmp); return; }
    StarterCacheHeader h{};
    h.query_count = s.size(); h.vector_count = nv; h.starter_count = static_cast<uint32_t>(sl);
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
    for (const auto& ids : s) {
        const uint32_t c = static_cast<uint32_t>(std::min(ids.size(), static_cast<size_t>(sl)));
        out.write(reinterpret_cast<const char*>(&c), sizeof(c));
        if (c) out.write(reinterpret_cast<const char*>(ids.data()), c * sizeof(GID_Type));
    }
    out.close();
    if (!out || std::rename(tmp.c_str(), path.c_str()) != 0) { g_logger->warn("Failed to write starter cache {}", path); return; }
    g_logger->info("Saved starter cache {} (queries={}, starters={})", path, s.size(), sl);
}

template <typename T>
static std::vector<std::vector<GID_Type>> ivfpq_starters(const std::string& file, const std::vector<std::vector<T>>& queries, int sl) {
    if (file.empty()) throw std::runtime_error("no starter cache hit and no --pq_index_file");
    g_logger->info("Loading PQ index from {}", file);
    std::unique_ptr<faiss::Index> base(faiss::read_index(file.c_str()));
    auto* pq = dynamic_cast<faiss::IndexIVFPQ*>(base.get());
    if (!pq) throw std::runtime_error("index is not IVFPQ");
    const size_t dim = queries[0].size();
    if (static_cast<size_t>(pq->d) != dim || pq->ntotal == 0 || !pq->is_trained || pq->metric_type != faiss::METRIC_L2)
        throw std::runtime_error("PQ index incompatible with queries");
    std::vector<float> flat(queries.size() * dim);
    for (size_t i = 0; i < queries.size(); ++i)
        for (size_t j = 0; j < dim; ++j) flat[i * dim + j] = static_cast<float>(queries[i][j]);
    const int nq = static_cast<int>(queries.size());
    std::vector<float> dist(static_cast<size_t>(nq) * sl);
    std::vector<faiss::Index::idx_t> idx(static_cast<size_t>(nq) * sl);
    const int blas = openblas_get_num_threads();
    openblas_set_num_threads(1);
    pq->search(nq, flat.data(), sl, dist.data(), idx.data());
    openblas_set_num_threads(blas);
    std::vector<std::vector<GID_Type>> out(queries.size(), std::vector<GID_Type>(sl));
    for (int i = 0; i < nq; ++i)
        for (int j = 0; j < sl; ++j) out[i][j] = static_cast<GID_Type>(idx[static_cast<size_t>(i) * sl + j]);
    return out;
}

template <typename T>
static std::vector<std::vector<GID_Type>> build_starters(const Params& p, const std::vector<std::vector<T>>& queries,
                                                         int sl, size_t nv) {
    std::vector<std::vector<GID_Type>> s;
    if (load_starter_cache(p.starter_cache_file, queries.size(), sl, nv, s)) return s;
    if (p.random_starter) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<GID_Type> d(0, static_cast<GID_Type>(nv - 1));
        s.assign(queries.size(), {});
        for (auto& v : s) {
            std::unordered_set<GID_Type> u;
            while (u.size() < static_cast<size_t>(sl)) u.insert(d(rng));
            v.assign(u.begin(), u.end());
        }
    } else {
        s = ivfpq_starters(p.pq_index_file, queries, sl);
    }
    // Never replace an existing cache file (it may belong to another starter
    // count / query set); only create missing ones.
    struct stat st{};
    if (!p.starter_cache_file.empty() && stat(p.starter_cache_file.c_str(), &st) == 0)
        g_logger->warn("Starter cache {} exists but does not match; not overwriting it", p.starter_cache_file);
    else
        save_starter_cache(p.starter_cache_file, nv, sl, s);
    return s;
}

template <typename T>
static void materialize_starters(const std::vector<T>& q, const VecMMapReader<T>& vecs,
                                 const std::vector<GID_Type>& ids, std::vector<Neighbor>& out) {
    out.clear();
    for (GID_Type id : ids) {
        if (id >= vecs.num_vectors()) continue;
        float dd;
        if (g_sq_only) { out.emplace_back(id, sq_dist_t<T>(q.data(), id)); continue; }   // in-memory SQ code: no base.bin read
        if (g_io_stats && !g_dense && !g_adc) io_note_vec(id, false);
        dd = g_adc ? g_adc->distance(id) : g_dense ? g_dense->distance(q.data(), id) : exact_distance(q.data(), vecs, id, q.size());
        out.emplace_back(id, dd);
    }
}

// ============================================================================
// Evaluation / output
// ============================================================================
// Historical definition kept for comparability: hits / min(K, |result|).
static float recall_at_k(const std::vector<Neighbor>& sorted, const std::vector<GID_Type>& gt, int K) {
    std::unordered_set<GID_Type> g(gt.begin(), gt.begin() + std::min<size_t>(K, gt.size()));
    const int k = std::min<int>(K, static_cast<int>(sorted.size()));
    if (k == 0) return 0.0f;
    int hit = 0;
    for (int j = 0; j < k; ++j) hit += g.count(sorted[j].id) ? 1 : 0;
    return static_cast<float>(hit) / k;
}

static void save_results(const std::string& file, const std::vector<std::vector<Neighbor>>& res) {
    if (file.empty()) return;
    std::ofstream out(file, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open result file " + file);
    for (const auto& r : res) {
        const int c = static_cast<int>(r.size());
        out.write(reinterpret_cast<const char*>(&c), sizeof(c));
        for (const auto& n : r) out.write(reinterpret_cast<const char*>(&n.id), sizeof(GID_Type));
    }
    out.flush();
    if (!out) throw std::runtime_error("failed writing result file " + file);
    g_logger->info("Results saved to {}", file);
}

// ============================================================================
// Driver
// ============================================================================
static std::unique_ptr<GraphMMapRW<GID_Type, GL_Type>> open_graph(const std::string& file, const std::string& label,
                                                                 const Params& p, bool resident, const std::string& offsets,
                                                                 const std::string& compact = "") {
    if (file.empty()) return nullptr;
    std::unique_ptr<GraphMMapRW<GID_Type, GL_Type>> g(new GraphMMapRW<GID_Type, GL_Type>(file));
    if (!compact.empty()) {   // header from the graph file only; lists from the compressed heap copy
        const auto t0 = std::chrono::steady_clock::now();
        std::unique_ptr<ksg::CompactAdj> c(new ksg::CompactAdj(compact));
        if (c->size() != g->size()) throw std::runtime_error("compact KSG node count does not match " + file);
        g_logger->info("{} graph compact (heap): {:.3f} GiB in {:.2f} s ({})", label, c->bytes() / 1073741824.0,
                       std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), compact);
        g->set_compact(std::move(c));
        g_logger->info("{} graph {}: nodes={}", label, file, g->size());
        return g;
    }
    if (resident) {
        const double sec = g->make_resident(g_resident_hugepage, omp_get_max_threads());
        g_logger->info("{} graph resident in anonymous memory ({:.2f} GiB, hugepage={}) in {:.2f} s", label,
                       g->file_size() / 1073741824.0, g_resident_hugepage, sec);
    }
    if (g->max_degree() == static_cast<GL_Type>(-1)) {
        const auto t0 = std::chrono::steady_clock::now();
        const bool hit = offsets.empty() ? (g->load_node_offset(), false) : g->load_node_offset(offsets);
        g_logger->info("Loaded {} graph offsets in {:.2f} ms (source={})", label,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
                       hit ? "sidecar" : "graph-scan");
    }
    if (p.mmap_random) g->advise_random();
    g_logger->info("{} graph {}: nodes={}, max_degree={}", label, file, g->size(), static_cast<int64_t>(static_cast<int32_t>(g->max_degree())));
    return g;
}

template <typename T>
int run(const Params& p) {
    VecMMapReader<T> vecs(p.vec_file);
    VecMMapReader<T> query_reader(p.query_file);
    VecMMapReader<GID_Type> gt_reader(p.gt_file);
    if (query_reader.dim() != vecs.dim()) throw std::runtime_error("query/base dimension mismatch");
    if (p.mmap_random) vecs.advise_random();
    const auto queries = query_reader.read_all_vectors();
    const auto gt = gt_reader.read_all_vectors();
    if (queries.empty() || gt.size() != queries.size())
        throw std::runtime_error("query count (" + std::to_string(queries.size()) + ") != groundtruth count (" + std::to_string(gt.size()) + ")");
    const size_t nv = vecs.num_vectors();
    const size_t nq = queries.size();
    g_logger->info("base {} ({} x {}), queries {}, mode={}, distance={}", p.vec_file, nv, vecs.dim(), nq, p.mode, p.search_distance);

    const bool need_ksg = p.mode == "two_layer" || p.mode == "ksg_only";
    // with a co-located layout the SSD graph is read from --colo-file only; --disk_graph is optional (opened when given, header only)
    const bool need_ssg = (p.mode == "two_layer" && !p.disk_graph_file.empty()) || p.mode == "ssg_only";
    auto ksg_graph = need_ksg ? open_graph(p.mem_graph_file, "KSG", p, p.mem_graph_resident, p.mem_graph_offsets_file, p.mem_graph_compact) : nullptr;
    if (p.mem_graph_ssd) {
        if (!ksg_graph || p.mem_graph_resident || !p.mem_graph_compact.empty()) throw std::runtime_error("--mem-graph-ssd needs --mem_graph without --mem-graph-resident / --mem-graph-compact");
        ksg_graph->set_ssd(p.mem_graph_file);
        g_logger->info("KSG graph on SSD: lists read with O_DIRECT on every access (offsets in memory)");
    }
    auto ssg_graph = need_ssg ? open_graph(p.disk_graph_file, "SSG", p, false, "") : nullptr;
    if (p.colo_mem_L > 0) { if (!ksg_graph) throw std::runtime_error("--colo-mem-L needs the KSG (two_layer mode)"); g_ksg = ksg_graph.get(); }

    const int K = p.K, sl = effective_starter_count(p), beam = p.beam;
    const int L1 = p.L1, L2 = p.L2;
    g_logger->info("Starting graph search with parameters: topK={}, candidate_l1={}, candidate_l2={}, starter_count={}, beam={}, "
                   "prefetch={}, mem_neighbor_limit={}, disk_neighbor_limit={}, dense_refine_neighbors={}, dense_refine_push={}, "
                   "rerank_k={}, exact_read_order={}, rerank_read_order={}, mmap_advice={}, visited_set={}, mem_graph_resident={}",
                   K, L1, L2, sl, beam, p.prefetch, p.mem_neighbor_limit, p.disk_neighbor_limit, p.dense_refine_neighbors,
                   p.dense_refine_push_all ? "all" : "refined", p.rerank_k, p.exact_read_order_id ? "id" : "candidate",
                   p.rerank_read_order, p.mmap_random ? "random" : "normal",
                   g_visited_mode == VisitedMode::Auto ? "auto" : (g_visited_mode == VisitedMode::Hash ? "hash" : "bitmap"),
                   p.mem_graph_resident);

    // Entry index is loaded before the timing window (like any index); its per-query starter search runs INSIDE the window.
    if (!p.entry_file.empty()) { g_entry.load(p.entry_file); g_has_entry = true; g_logger->info("entry index: {} centroids x {} reps (online starters)", g_entry.C, g_entry.r); }

    LayerOptions ksg_opt{p.prefetch, p.mem_neighbor_limit, p.dense_refine_neighbors, p.dense_refine_push_all, p.exact_read_order_id, p.reuse_bitsets};
    LayerOptions ssg_opt = ksg_opt;
    ssg_opt.neighbor_limit = p.disk_neighbor_limit;
    ssg_opt.is_ssg = true;
    ksg_opt.refine_gate = ssg_opt.refine_gate = p.refine_gate;
    ksg_opt.adc_prefetch = ssg_opt.adc_prefetch = p.adc_prefetch;
    ssg_opt.gate_k = p.gate_k;
    g_track_exact = p.rerank_skip_known;
    g_sq4_mix = p.sq4_mix;
    g_colo_io = p.colo_io;
    if (!p.colo_file.empty()) { g_colo.reset(new ksg::ColoGraph(p.colo_file)); ssg_opt.colo = g_colo.get(); g_logger->info("co-located SSG: {} nodes, degree {}, {} per sector", g_colo->size(), g_colo->degree(), g_colo->per_sector()); }
    if (!p.sq4_file.empty()) { g_sq4.reset(new ksg::SQ4(p.sq4_file, p.sq4_resident, p.sq4_heap)); g_logger->info("SQ4 sidecar loaded: {} vectors", g_sq4->size()); }
    if (!p.sqf_file.empty()) { g_sqf.reset(new ksg::SQF(p.sqf_file, p.sq4_resident)); g_logger->info("SQF sidecar loaded: {} vectors, d={}", g_sqf->size(), g_sqf->dim()); }
    if (!p.pqc_file.empty()) { g_pqc.reset(new ksg::PQC(p.pqc_file)); g_logger->info("PQC sidecar loaded: {} vectors, {} B/vector (heap)", g_pqc->size(), g_pqc->bytes_per_vector()); }
    g_sq_only = p.sq_only && (g_pqc != nullptr || (std::is_same<T, uint8_t>::value ? g_sq4 != nullptr : g_sqf != nullptr));
    g_colo_free = p.colo_free;
    g_colo_mem_L = p.colo_mem_L; g_colo_ssg_off = p.colo_ssg_off;
    g_colo_pipe = p.colo_pipe; g_colo_pipe_batch = p.colo_pipe_batch;
    if (g_colo_pipe && p.colo_io != 1) throw std::runtime_error("--colo-pipe needs --colo-io uring");
    if (!p.colo_sq_results.empty()) {
        std::ifstream in(p.colo_sq_results + ".new2old", std::ios::binary); in.seekg(0, std::ios::end); g_sqres_n2o.resize(static_cast<size_t>(in.tellg()) / 4);
        in.seekg(0); in.read(reinterpret_cast<char*>(g_sqres_n2o.data()), g_sqres_n2o.size() * 4);
        if (!in) throw std::runtime_error("--colo-sq-results: cannot read " + p.colo_sq_results);
    }
    g_dump_kind = !p.dump_expanded.empty();
    if (!p.colo_exact_frontier.empty()) {
        vecs.read_all_vectors_flat(); g_exact_base = vecs.vector_data_unchecked(0); g_exact_dim = vecs.dim();
        std::ifstream in(p.colo_exact_frontier + ".new2old", std::ios::binary); in.seekg(0, std::ios::end); g_exact_n2o.resize(static_cast<size_t>(in.tellg()) / 4);
        in.seekg(0); in.read(reinterpret_cast<char*>(g_exact_n2o.data()), g_exact_n2o.size() * 4);
        if (!in || !g_exact_base) throw std::runtime_error("--colo-exact-frontier: cannot load " + p.colo_exact_frontier);
        g_logger->warn("DIAGNOSTIC: colo frontier scored with exact distances ({} ids)", g_exact_n2o.size());
    }
    g_colo_cache = p.colo_cache;
    if (g_colo && g_colo->per_sector() == 1 && g_colo_cache > 0) {
        // one record per page (high-dimensional data): a page is never needed twice by the same query, so the per-query
        g_logger->info("colo layout has 1 record per sector: per-query sector cache disabled (was {})", g_colo_cache);
        g_colo_cache = 0;
    }
    if (!p.colo_perm.empty()) {
        auto rd = [&](const std::string& f, std::vector<uint32_t>& v) { std::ifstream in(f, std::ios::binary); in.seekg(0, std::ios::end); v.resize(static_cast<size_t>(in.tellg()) / 4); in.seekg(0); in.read(reinterpret_cast<char*>(v.data()), v.size() * 4); if (!in) throw std::runtime_error("cannot read " + f); };
        rd(p.colo_perm + ".new2old", g_perm_n2o); rd(p.colo_perm + ".old2new", g_perm_o2n);
        g_logger->info("colo permutation loaded: {} ids", g_perm_n2o.size());
    }
    if (p.sq_only && !g_sq_only) throw std::runtime_error("--sq-only needs --sq4-file (uint8), --sqf-file (float) or --pqc-file");
    if (p.mode == "two_layer" && !ssg_graph && !(!p.colo_file.empty() && g_sq_only)) throw std::runtime_error("without --disk_graph, two_layer needs --colo-file and --sq-only");
    ssg_opt.stop_eps = p.stop_eps; ssg_opt.stop_k = static_cast<size_t>(p.K);
    g_io_stats = p.io_stats;
    g_trace = p.trace_found;
    if (g_trace && ssg_graph) {
        const size_t nn = ssg_graph->size();
        std::vector<std::atomic<uint32_t>> ind(nn);
        for (auto& x : ind) x.store(0, std::memory_order_relaxed);
        #pragma omp parallel for schedule(static, 65536)
        for (size_t i = 0; i < nn; ++i) {
            auto v = ssg_graph->get_neighbors_view(static_cast<GID_Type>(i));
            size_t c = v.size(); if (ssg_opt.neighbor_limit > 0) c = std::min(c, static_cast<size_t>(ssg_opt.neighbor_limit));
            for (size_t j = 0; j < c; ++j) if (v.data[j] < nn) ind[v.data[j]].fetch_add(1, std::memory_order_relaxed);
        }
        g_indeg.resize(nn); for (size_t i = 0; i < nn; ++i) g_indeg[i] = ind[i].load();
    }
    g_vec_bytes = vecs.vector_bytes();
    if (ssg_graph) {
        g_ssg_meta_bytes = ssg_graph->meta_data_size();
        g_ssg_record_bytes = sizeof(GL_Type) + static_cast<size_t>(ssg_graph->max_degree()) * sizeof(GID_Type);
    }
    // Timing window: starters + all queries. Every sidecar/layout load above (SQ, co-located file, permutation, entry) stays outside it,
    // like any index load; before 10-16 the SQ/colo loads sat inside the window and inflated KSG's wall time.
    const auto t_start = std::chrono::high_resolution_clock::now();
    const auto starters = g_has_entry ? std::vector<std::vector<GID_Type>>(queries.size()) : build_starters(p, queries, sl, nv);
    const auto t_starters = std::chrono::high_resolution_clock::now();
    std::atomic<uint64_t> tot_cm{0}, tot_ksg{0}, tot_ssg{0}, tot_ssg_pages{0}, tot_ksg_vec{0}, tot_ssg_vec{0}, tot_skip{0}, tot_cs{0}, tot_ce{0}, tot_cf{0}, tot_ch{0}, tot_vec{0}, tot_vec_pages{0}, tot_rerank{0};
    const bool rerank_by_id = p.rerank_read_order == "id" || (p.rerank_read_order == "candidate" && p.exact_read_order_id);

    struct ThreadState {
        std::vector<Neighbor> starters, c1, c2;
        VisitedSet visited_ksg, visited_ssg;
        FoundTrace tr_ksg, tr_ssg;
    };
    const int threads = omp_get_max_threads();
    std::vector<ThreadState> ts(static_cast<size_t>(threads));
    for (auto& s : ts) { s.visited_ksg.reserve(nv); s.visited_ssg.reserve(nv); }

    std::mutex dump_mu; std::vector<char> dump_buf;
    std::vector<std::vector<Neighbor>> results(nq);
    std::vector<double> latency_us(nq, 0.0);
    double sum_ksg_us = 0, sum_ssg_us = 0, sum_r1 = 0, sum_r2 = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+ : sum_ksg_us, sum_ssg_us, sum_r1, sum_r2)
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        ThreadState& s = ts[static_cast<size_t>(omp_get_thread_num())];
        s.visited_ksg.clear();
        s.visited_ssg.clear();
        const auto& q = queries[qi];
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (g_adc) g_adc->prepare(q.data());
        if (g_pqc) g_pqc->prepare(q.data());
        if (g_io_stats) io_tracker().begin();
        if (g_track_exact) exact_set().clear();
        t_colo_exact.clear();
        t_colo_kind.clear();
        if (g_trace) { s.tr_ksg.begin(gt[qi], 100); s.tr_ssg.begin(gt[qi], 100); }
        if (g_has_entry) {
            thread_local std::vector<GID_Type> st_ids;
            g_entry.starters(q.data(), static_cast<size_t>(sl), st_ids);
            materialize_starters(q, vecs, st_ids, s.starters);
        } else materialize_starters(q, vecs, starters[qi], s.starters);
        if (s.starters.empty()) throw std::runtime_error("no starters for query " + std::to_string(qi));

        std::vector<Neighbor>* final_list = &s.c1;
        if (p.mode == "starter_only") {
            s.c1.assign(s.starters.begin(), s.starters.end());
            std::sort(s.c1.begin(), s.c1.end());
            if (g_dense && p.rerank_k > 0) rerank_exact(q, vecs, p.rerank_k, rerank_by_id, s.c1);
        } else if (p.mode == "ssg_only") {
            s.c1.assign(s.starters.begin(), s.starters.end());
            graph_search(q, vecs, *ssg_graph, L2, beam, s.c1, s.visited_ssg, ssg_opt, g_trace ? &s.tr_ssg : nullptr);
            if (g_dense && p.rerank_k > 0) rerank_exact(q, vecs, p.rerank_k, rerank_by_id, s.c1);
        } else {
            s.c1.assign(s.starters.begin(), s.starters.end());
            graph_search(q, vecs, *ksg_graph, L1, beam, s.c1, s.visited_ksg, ksg_opt, g_trace ? &s.tr_ksg : nullptr);
            if (p.mode == "ksg_only") {
                if (g_dense && p.rerank_k > 0) rerank_exact(q, vecs, p.rerank_k, rerank_by_id, s.c1);
            }
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (p.mode == "two_layer") {
            sum_r1 += recall_at_k(s.c1, gt[qi], K);
            const size_t hand = std::min(s.c1.size(), static_cast<size_t>(L2));
            s.c2.assign(s.c1.begin(), s.c1.begin() + static_cast<long>(hand));
            if (g_colo && g_sq_only) graph_search_colo_sq(q, *g_colo, L2, beam, s.c2, s.visited_ssg, ssg_opt);
            else graph_search(q, vecs, *ssg_graph, L2, beam, s.c2, s.visited_ssg, ssg_opt, g_trace ? &s.tr_ssg : nullptr);
            if (!g_perm_n2o.empty()) for (auto& e : t_colo_exact) e.id = g_perm_n2o[e.id];   // back to original ids
            if (!p.dump_expanded.empty() && g_colo) {
                std::lock_guard<std::mutex> lk(dump_mu);
                const uint32_t qid = static_cast<uint32_t>(qi), cnt = static_cast<uint32_t>(t_colo_exact.size());
                dump_buf.insert(dump_buf.end(), reinterpret_cast<const char*>(&qid), reinterpret_cast<const char*>(&qid) + 4);
                dump_buf.insert(dump_buf.end(), reinterpret_cast<const char*>(&cnt), reinterpret_cast<const char*>(&cnt) + 4);
                // id | kind << 30 (kind: 0 = expansion that read a new sector, 2 = expansion without a new read, 1 = free in-sector expansion)
                for (size_t k = 0; k < t_colo_exact.size(); ++k) {
                    const uint32_t v = static_cast<uint32_t>(t_colo_exact[k].id) | (uint32_t(k < t_colo_kind.size() ? t_colo_kind[k] : 0) << 30);
                    dump_buf.insert(dump_buf.end(), reinterpret_cast<const char*>(&v), reinterpret_cast<const char*>(&v) + 4);
                }
            }
            if (g_colo) {
                // the expanded nodes' exact distances came with their pages: no rerank reads at all
                s.c2.assign(t_colo_exact.begin(), t_colo_exact.end());
                std::sort(s.c2.begin(), s.c2.end());
                if (!g_sqres_n2o.empty())
                    s.c2.erase(std::unique(s.c2.begin(), s.c2.end(), [](const Neighbor& x, const Neighbor& y) { return x.id == y.id; }), s.c2.end());
            } else if (g_dense && p.rerank_k > 0) rerank_exact(q, vecs, p.rerank_k, rerank_by_id, s.c2);
            final_list = &s.c2;
        }
        const auto t2 = std::chrono::high_resolution_clock::now();
        sum_ksg_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
        sum_ssg_us += std::chrono::duration<double, std::micro>(t2 - t1).count();
        latency_us[qi] = std::chrono::duration<double, std::micro>(t2 - t0).count();
        const float r = recall_at_k(*final_list, gt[qi], K);
        sum_r2 += r;
        if (g_trace) {
            const FoundTrace &a = s.tr_ksg, &b = s.tr_ssg;
            for (size_t i = 0; i < a.gt.size(); ++i)
                for (int sel = 0; sel < 2; ++sel) {
                    if (sel == 1 && a.rk[i] >= 10) continue;
                    g_tr_total[sel]++;
                    for (int t = 0; t < kNTH; ++t) {
                        if (a.at[i] >= 0 && a.at[i] <= kTH[t]) g_tr_ksg_cum[sel][t]++;
                        if (a.at[i] >= 0 || (b.at[i] >= 0 && b.at[i] <= kTH[t])) g_tr_all_cum[sel][t]++;
                    }
                }
            if (!g_indeg.empty())
                for (size_t i = 0; i < a.gt.size(); ++i) {
                    const int cls = a.at[i] >= 0 ? 0 : (b.at[i] < 0 ? 4 : b.at[i] <= 10 ? 1 : b.at[i] <= 50 ? 2 : b.at[i] <= 200 ? 3 : 4);
                    const uint32_t d = a.gt[i] < g_indeg.size() ? g_indeg[a.gt[i]] : 0;
                    int ib = 0; while (ib + 1 < kNIB && d >= kIndegEdges[ib + 1]) ++ib;
                    g_tr_cls_indeg[cls][ib]++;
                    g_tr_cls_rank[cls][a.rk[i] < 10 ? 0 : a.rk[i] < 50 ? 1 : 2]++;
                }
            g_tr_ksg_exp += a.nexp; g_tr_ssg_exp += b.nexp; g_tr_q++;
        }
        results[qi].assign(final_list->begin(), final_list->begin() + std::min<size_t>(K, final_list->size()));
        if (g_io_stats) {
            const IoStats& st = io_tracker().q;
            tot_ksg += st.ksg_expand; tot_ssg += st.ssg_expand; tot_ssg_pages += st.ssg_pages;
            tot_ch += t_colo_stats.cache_hits; tot_cm += t_colo_stats.mem_expand; tot_cs += t_colo_stats.sectors; tot_ce += t_colo_stats.expand; tot_cf += t_colo_stats.free_expand; t_colo_stats = ColoStats{};
            tot_vec += st.vec_reads; tot_ksg_vec += st.ksg_vec; tot_ssg_vec += st.ssg_vec; tot_skip += st.rerank_skipped; tot_vec_pages += st.vec_pages; tot_rerank += st.rerank_reads;
        }
    }
    const auto t_end = std::chrono::high_resolution_clock::now();

    const double sec = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() / 1000.0;
    const double starter_ms = std::chrono::duration<double, std::milli>(t_starters - t_start).count();
    const double mean_lat = std::accumulate(latency_us.begin(), latency_us.end(), 0.0) / nq;
    if (!p.latency_dump.empty()) {   // per-query wall time t0..t2 (starters + KSG + SSG), measured under the concurrent load
        std::ofstream lf(p.latency_dump, std::ios::binary);
        for (double v : latency_us) { const float f = static_cast<float>(v); lf.write(reinterpret_cast<const char*>(&f), sizeof(f)); }
    }
    std::sort(latency_us.begin(), latency_us.end());
    auto pct = [&](double x) { return latency_us[static_cast<size_t>(x * (nq - 1))]; };
    g_logger->info("search starters in {:.2f} ms", starter_ms);
    g_logger->info("query latency mean:{:.2f}us,p50:{:.2f}us,p95:{:.2f}us,p99:{:.2f}us", mean_lat, pct(0.50), pct(0.95), pct(0.99));
    // Log line format kept for bench/baselines/remote_scripts/ksg_log_to_csv.py.
    const double r1 = p.mode == "two_layer" ? sum_r1 / nq : sum_r2 / nq;
    g_logger->info("search {} queries,average mem graph recall:{},average mem graph time:{}us,average disk graph recall:{},average disk graph time:{}us",
                   nq, r1, (sum_ksg_us + starter_ms * 1000.0) / nq, sum_r2 / nq, sum_ssg_us / nq);
    if (p.mode != "two_layer") g_logger->info("average recall:{}", sum_r2 / nq);
    g_logger->info("Graph search completed in {:.2f} seconds.QPS : {:.2f} queries/s", sec, nq / sec);
    if (p.mem_graph_ssd && ksg_graph) g_logger->info("ksg_ssd per query: list_reads={:.1f}, ksg_pages={:.1f}", ksg_graph->ssd_reads() / double(nq), ksg_graph->ssd_pages_read() / double(nq));
    if (g_io_stats) {
        auto per = [&](const std::atomic<uint64_t>& v) { return static_cast<double>(v.load()) / nq; };
        g_logger->info("io_stats per query: ksg_expand={:.1f}, ssg_expand={:.1f}, ssg_pages={:.1f}, vec_reads={:.1f}, "
                       "rerank_reads={:.1f}, vec_pages={:.1f}, ssd_pages_total={:.1f}",
                       per(tot_ksg), per(tot_ssg), per(tot_ssg_pages), per(tot_vec), per(tot_rerank), per(tot_vec_pages),
                       per(tot_ssg_pages) + per(tot_vec_pages));
        if (g_colo && g_sq_only) g_logger->info("colo_sq per query: sectors_read={:.1f}, expansions={:.1f}, free_expansions={:.1f}, sector_cache_hits={:.1f}, mem_expansions={:.1f}", per(tot_cs), per(tot_ce), per(tot_cf), per(tot_ch), per(tot_cm));
        g_logger->info("io_stats refine reads by layer: ksg_graph={:.1f}, ssg_graph={:.1f} ; rerank reads skipped (already exact)={:.1f}", per(tot_ksg_vec), per(tot_ssg_vec), per(tot_skip));
    }
    if (g_trace && g_tr_q.load() > 0) {
        const char* nm[2] = {"top100", "top10"};
        g_logger->info("trace_found mean expansions per query: ksg_graph={:.1f}, ssg_graph={:.1f}",
                       double(g_tr_ksg_exp.load()) / g_tr_q.load(), double(g_tr_ssg_exp.load()) / g_tr_q.load());
        for (int sel = 0; sel < 2; ++sel) {
            std::string a, b;
            for (int t = 0; t < kNTH; ++t) {
                a += fmt::format(" e{}:{:.3f}", kTH[t], double(g_tr_ksg_cum[sel][t].load()) / std::max<uint64_t>(1, g_tr_total[sel].load()));
                b += fmt::format(" e{}:{:.3f}", kTH[t], double(g_tr_all_cum[sel][t].load()) / std::max<uint64_t>(1, g_tr_total[sel].load()));
            }
            g_logger->info("trace_found {} ksg_cum(by KSG expansions):{}", nm[sel], a);
            g_logger->info("trace_found {} ksg_graph+ssg_graph(by SSG expansions):{}", nm[sel], b);
        }
    }
    if (g_trace && !g_indeg.empty()) {
        const char* cn[5] = {"KSG phase", "SSG<=10", "SSG<=50", "SSG<=200", "later/never"};
        std::string hdr = "in-degree bucket lower bounds:"; for (int b = 0; b < kNIB; ++b) hdr += fmt::format(" {}", kIndegEdges[b]);
        g_logger->info("trace_found discovery class vs SSG in-degree of the true neighbour (share within class). {}", hdr);
        uint64_t all_ib[kNIB] = {0}, all_total = 0;
        for (int c = 0; c < 5; ++c) for (int b = 0; b < kNIB; ++b) { all_ib[b] += g_tr_cls_indeg[c][b]; all_total += g_tr_cls_indeg[c][b]; }
        for (int c = 0; c < 5; ++c) {
            uint64_t tot = 0, rt = 0; for (int b = 0; b < kNIB; ++b) tot += g_tr_cls_indeg[c][b]; for (int b = 0; b < 3; ++b) rt += g_tr_cls_rank[c][b];
            std::string line;
            for (int b = 0; b < kNIB; ++b) line += fmt::format(" {:.3f}", double(g_tr_cls_indeg[c][b].load()) / std::max<uint64_t>(1, tot));
            g_logger->info("  class {:<11} n={:<9} indeg:{} | rank<10 {:.3f} <50 {:.3f} >=50 {:.3f}", cn[c], tot, line,
                           double(g_tr_cls_rank[c][0].load()) / std::max<uint64_t>(1, rt), double(g_tr_cls_rank[c][1].load()) / std::max<uint64_t>(1, rt), double(g_tr_cls_rank[c][2].load()) / std::max<uint64_t>(1, rt));
        }
        std::string line; for (int b = 0; b < kNIB; ++b) line += fmt::format(" {:.3f}", double(all_ib[b]) / std::max<uint64_t>(1, all_total));
        g_logger->info("  all true neighbours  n={:<9} indeg:{}", all_total, line);
    }
    if (!p.dump_expanded.empty()) { std::ofstream df(p.dump_expanded, std::ios::binary); df.write(dump_buf.data(), static_cast<std::streamsize>(dump_buf.size())); }
    save_results(p.result_file, results);
    return 0;
}

int main(int argc, char** argv) {
    Params p;
    try {
        if (!parse(argc, argv, p)) { usage(argv[0]); return 2; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        usage(argv[0]);
        return 2;
    }
    setup_logger(p.log_file);
    try {
        if (p.search_distance == "dense_ivfpq") {
            g_dense.reset(new ksg::DenseIVFPQReader(p.dense_ivfpq_file));
            if (!p.pq_index_file.empty()) {
                struct stat st{};
                if (stat(p.pq_index_file.c_str(), &st) != 0 ||
                    g_dense->header().source_bytes != static_cast<uint64_t>(st.st_size) ||
                    g_dense->header().source_mtime != static_cast<int64_t>(st.st_mtim.tv_sec))
                    throw std::runtime_error("dense IVFPQ source fingerprint does not match --pq_index_file");
            }
            g_logger->info("Loaded dense IVFPQ: vectors={}, dim={}, code_size={}", g_dense->size(), g_dense->dim(), g_dense->code_size());
            if (p.dense_resident) {
                const double s = g_dense->make_resident(std::max(1, omp_get_max_threads()));
                g_logger->info("Dense IVFPQ resident: bytes={}, load_seconds={:.2f}", g_dense->bytes(), s);
            }
            if (p.dense_adc) {
                const auto t0 = std::chrono::steady_clock::now();
                g_adc.reset(new ksg::DenseADC(*g_dense, std::max(1, omp_get_max_threads())));
                g_adc->set_gather(p.adc_gather);
                g_logger->info("Dense ADC tables ready: beta={:.2f} GiB in {:.2f} s", g_adc->beta_bytes() / 1073741824.0,
                               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        const int rc = p.data_type == "uint8" ? run<uint8_t>(p) : run<float>(p);
        g_logger->info("Search completed.");
        return rc;
    } catch (const std::exception& e) {
        g_logger->error("fatal: {}", e.what());
        return 1;
    }
}
