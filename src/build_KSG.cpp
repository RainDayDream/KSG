// KNN graph into the first-layer KSG.  It is not a complete KSG builder: the
// resulting variable-length graph is consumed later as --mem_graph by search.
//
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <faiss/index_io.h>
#include <faiss/IndexIVFPQ.h>
#include <unordered_set>
#include <queue>
#include <numeric>
#include <limits>
#include <mutex>
#include <cmath>
#include "IO_utils.h"
#include "distance.h"
#include "types.h"
#include "utils.h"
#include "union_find.h"

struct KSGParams {
    std::string pq_index_file, vec_file, data_type, log_file, ksg_file, ksg_init_file;
    std::string component_file;
    std::string connect_policy = "min_degree";
    int p_degree = 0;
    int connect_anchor_count = 8;
    bool indisk = false;
};

void print_usage() {
    // g_logger->info("Usage: build_KSG --pq_index <pq_index_file> --vec_file <vec_file> --data_type <data_type> --p_degree <p_degree> [--indisk] [--log_file <log_file>] [--ksg_file <KSG_file>]");
    std::cout << "Usage: build_KSG --ksg_init_file <ksg_init_file> --p_degree <p_degree>  | build_KSG [--pq_index <pq_index_file>] [--vec_file <vec_file>] [--data_type <data_type>] [--indisk] [--log_file <log_file>] [--ksg_file <KSG_file>]" << std::endl;
}

bool parse_args(int argc, char** argv, KSGParams& params) {
    params.log_file = "./log/build_ksg.log";
    params.ksg_file = "./index/ksg_graph.graph";
    params.indisk = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pq_index" && i + 1 < argc) params.pq_index_file = argv[++i];
        else if (arg == "--vec_file" && i + 1 < argc) params.vec_file = argv[++i];
        else if (arg == "--data_type" && i + 1 < argc) params.data_type = argv[++i];
        else if (arg == "--p_degree" && i + 1 < argc) params.p_degree = std::stoi(argv[++i]);
        else if (arg == "--log_file" && i + 1 < argc) params.log_file = argv[++i];
        else if (arg == "--ksg_file" && i + 1 < argc) params.ksg_file = argv[++i];
        else if (arg == "--ksg_init_file" && i + 1 < argc) params.ksg_init_file = argv[++i];
        else if ((arg == "--component_file" || arg == "--component-file") && i + 1 < argc) params.component_file = argv[++i];
        else if ((arg == "--connect_policy" || arg == "--connect-policy") && i + 1 < argc) params.connect_policy = argv[++i];
        else if ((arg == "--connect_anchor_count" || arg == "--connect-anchor-count") && i + 1 < argc) params.connect_anchor_count = std::stoi(argv[++i]);
        else if (arg == "--indisk") params.indisk = true;
        else if (arg == "--help") { print_usage(); return false; }
    }
    if (params.p_degree <= 0)
        return false;
    if( params.pq_index_file.empty() && params.ksg_init_file.empty() ) {
        print_usage();
        return false;
    }
    if (params.connect_policy != "min_degree" && params.connect_policy != "nearest_component") {
        std::cerr << "--connect-policy must be min_degree or nearest_component\n";
        return false;
    }
    if (params.connect_policy == "nearest_component" && (params.vec_file.empty() || params.data_type.empty())) {
        std::cerr << "--connect-policy nearest_component requires --vec_file and --data_type\n";
        return false;
    }
    return true;
}

inline faiss::IndexIVFPQ* load_pq_index(const std::string& pq_index_file) {
    g_logger->info("Loading PQ index from {}", pq_index_file);
    faiss::Index* base = faiss::read_index(pq_index_file.c_str());
    auto* pq = dynamic_cast<faiss::IndexIVFPQ*>(base);
    if (!pq) {
        g_logger->error("Index is not IVFPQ!");
        throw std::runtime_error("Index is not IVFPQ!");
    }
    return pq;
}



template<typename VecType>
std::vector<std::vector<GID_Type>> build_ksg_neighbor_graph_indisk(
    const std::string& vec_file, faiss::IndexIVFPQ* pq_index, int p_degree) {

    auto start_time = std::chrono::steady_clock::now();
    VecMMapReader<VecType> vec_reader(vec_file);
    int n = vec_reader.num_vectors();
    g_logger->info("Building KSG neighbor graph for {} nodes (p_degree={}, indisk={true})", n, p_degree);
    std::vector<std::vector<GID_Type>> ksg_graph(n);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i) {
        if(i % (n/10) == 0) g_logger->info("[INDISK][build_KSG]Processing {}/{} nodes (indisk)", i, n);
        auto vec = vec_reader.read_vector(i);
        std::vector<float> flat_queries =  vec_to_float<VecType>(vec);
        int search_p_degree = p_degree + 1;
        std::vector<faiss::Index::idx_t> all_ids(search_p_degree);
        std::vector<float> all_dists(search_p_degree);
        pq_index->search(1, flat_queries.data(), search_p_degree, all_dists.data(), all_ids.data());

        std::vector<GID_Type> neighbors;
        for (int j = 0; j < search_p_degree; ++j) {
            GID_Type id = static_cast<GID_Type>(all_ids[j]);
            if (id != static_cast<GID_Type>(i))
                neighbors.push_back(id);
            // if ((int)neighbors.size() >= p_degree) break;
        }
        ksg_graph[i] = std::move(neighbors);
    }
    auto end_time = std::chrono::steady_clock::now();
    g_logger->info("KSG get neighbors from ivfpq in {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    return ksg_graph;
}


template<typename VecType>
std::vector<std::vector<GID_Type>> build_ksg_neighbor_graph(
    const std::string& vec_file, faiss::IndexIVFPQ* pq_index, int p_degree, bool indisk) {
    
    if(indisk) return build_ksg_neighbor_graph_indisk<VecType>(vec_file, pq_index, p_degree);
    auto start_time = std::chrono::steady_clock::now();
    VecMMapReader<VecType> vec_reader(vec_file);
    int n = vec_reader.num_vectors();
    g_logger->info("[INMEM]Building KSG neighbor graph for {} nodes (p_degree={}, indisk={false})", n, p_degree);
    std::vector<std::vector<GID_Type>> ksg_graph(n);
    auto all_vecs = vec_reader.read_all_vectors();
    auto flat_queries = flatten_to_float<VecType>(all_vecs);
    int search_p_degree = p_degree + 1;
    std::vector<faiss::Index::idx_t> all_ids(n * search_p_degree);
    std::vector<float> all_dists(n * search_p_degree);
    pq_index->search(n, flat_queries.data(), search_p_degree, all_dists.data(), all_ids.data());
    // #pragma omp parallel for schedule(dynamic)
    for (int gidx = 0; gidx < n; ++gidx) {
        std::vector<GID_Type> neighbors;
        for (int j = 0; j < search_p_degree; ++j) {
            GID_Type id = static_cast<GID_Type>(all_ids[gidx * search_p_degree + j]);
            if (id != static_cast<GID_Type>(gidx))
                neighbors.push_back(id);
            // if ((int)neighbors.size() >= p_degree) break;
        }
        ksg_graph[gidx] = std::move(neighbors);
        if(gidx % (n/10) == 0) g_logger->info("Processed {}/{} nodes (batch-search)", gidx, n);
    }
    auto end_time = std::chrono::steady_clock::now();
    g_logger->info("KSG get neighbors from ivfpq in {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
    return ksg_graph;
}



inline void make_undirected(std::vector<std::vector<GID_Type>>& graph) {
    size_t n = graph.size();
    g_logger->info("Making graph undirected (symmetrizing)...");
    for (size_t i = 0; i < n; ++i) {
        std::sort(graph[i].begin(), graph[i].end());
    }
    for (size_t i = 0; i < n; ++i) {
        for (auto nb : graph[i]) {
            if (nb < n && !std::binary_search(graph[nb].begin(), graph[nb].end(), i))
                graph[nb].push_back(i);
        }
    }
}


inline void make_undirected_op(std::vector<std::vector<GID_Type>>& graph) {
    size_t n = graph.size();
    g_logger->info("Making graph undirected (symmetrizing)...");

    // Pre-sort neighbor lists for binary search
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        std::sort(graph[i].begin(), graph[i].end());
    }

    std::vector<std::vector<GID_Type>> to_undirected(n);

    std::vector<std::mutex> add_locks(n);

    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        for (auto nb : graph[i]) {
            if (nb >= n) continue;
            if (!std::binary_search(graph[nb].begin(), graph[nb].end(), i))
            {
                std::lock_guard<std::mutex> guard(add_locks[nb]);
                to_undirected[nb].push_back(i);
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        for(auto nb : to_undirected[i]) {
            graph[i].push_back(nb);
        }
    }

    to_undirected.clear();
    add_locks.clear();


}

struct ComponentBuildResult {
    std::vector<uint32_t> component;
    std::vector<std::vector<GID_Type>> nodes;
};

ComponentBuildResult find_components(const std::vector<std::vector<GID_Type>>& graph) {
    const size_t n = graph.size();
    ComponentBuildResult result;
    result.component.assign(n, std::numeric_limits<uint32_t>::max());
    for (size_t i = 0; i < n; ++i) {
        if (result.component[i] != std::numeric_limits<uint32_t>::max()) continue;
        const uint32_t cid = static_cast<uint32_t>(result.nodes.size());
        result.nodes.emplace_back();
        std::queue<GID_Type> q;
        q.push(static_cast<GID_Type>(i));
        result.component[i] = cid;
        while (!q.empty()) {
            GID_Type u = q.front();
            q.pop();
            result.nodes.back().push_back(u);
            for (GID_Type v : graph[u]) {
                if (v >= n) continue;
                if (result.component[v] == std::numeric_limits<uint32_t>::max()) {
                    result.component[v] = cid;
                    q.push(v);
                }
            }
        }
    }
    return result;
}

void save_component_file(const std::string& path,
                         const std::vector<uint32_t>& component,
                         uint32_t component_count) {
    if (path.empty()) return;
    const std::filesystem::path out_path(path);
    if (!out_path.parent_path().empty()) std::filesystem::create_directories(out_path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write component file: " + path);
    const uint32_t n = static_cast<uint32_t>(component.size());
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&component_count), sizeof(component_count));
    out.write(reinterpret_cast<const char*>(component.data()),
              static_cast<std::streamsize>(component.size() * sizeof(uint32_t)));
    g_logger->info("Saved KSG pre-connect components to {} (nodes={}, components={})",
                   path, n, component_count);
}

template<typename T>
DisType node_distance(const VecMMapReader<T>& vectors, GID_Type a, GID_Type b) {
    return squared_l2_distance_avx512<T>(vectors.vector_data_unchecked(a),
                                         vectors.vector_data_unchecked(b),
                                         vectors.dim());
}

std::vector<GID_Type> choose_component_anchors(const std::vector<std::vector<GID_Type>>& graph,
                                               const std::vector<GID_Type>& nodes,
                                               int anchor_count) {
    std::vector<GID_Type> anchors;
    if (nodes.empty() || anchor_count <= 0) return anchors;
    auto add_unique = [&](GID_Type id) {
        if (std::find(anchors.begin(), anchors.end(), id) == anchors.end()) anchors.push_back(id);
    };
    auto min_it = std::min_element(nodes.begin(), nodes.end(), [&](GID_Type a, GID_Type b) {
        return graph[a].size() < graph[b].size();
    });
    auto max_it = std::max_element(nodes.begin(), nodes.end(), [&](GID_Type a, GID_Type b) {
        return graph[a].size() < graph[b].size();
    });
    add_unique(*min_it);
    if (static_cast<int>(anchors.size()) < anchor_count) add_unique(*max_it);
    for (int i = 0; i < anchor_count && static_cast<int>(anchors.size()) < anchor_count; ++i) {
        const size_t pos = (static_cast<uint64_t>(i) * nodes.size()) / static_cast<size_t>(anchor_count);
        add_unique(nodes[std::min(pos, nodes.size() - 1)]);
    }
    return anchors;
}

template<typename T>
void connect_graph_nearest_component(std::vector<std::vector<GID_Type>>& graph,
                                     const std::string& vec_file,
                                     int anchor_count) {
    ComponentBuildResult comps = find_components(graph);
    const uint32_t comp_count = static_cast<uint32_t>(comps.nodes.size());
    if (comp_count <= 1) {
        g_logger->info("Graph is already connected");
        return;
    }
    VecMMapReader<T> vectors(vec_file);
    std::vector<uint32_t> order(comp_count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return comps.nodes[a].size() > comps.nodes[b].size();
    });
    const uint32_t root = order[0];
    std::vector<std::vector<GID_Type>> anchors(comp_count);
    #pragma omp parallel for schedule(dynamic)
    for (uint32_t c = 0; c < comp_count; ++c) {
        anchors[c] = choose_component_anchors(graph, comps.nodes[c], anchor_count);
    }

    std::vector<DisType> bridge_distances;
    bridge_distances.reserve(comp_count - 1);
    const auto& root_anchors = anchors[root];
    for (uint32_t oi = 1; oi < comp_count; ++oi) {
        const uint32_t c = order[oi];
        GID_Type best_a = root_anchors.empty() ? comps.nodes[root][0] : root_anchors[0];
        GID_Type best_b = anchors[c].empty() ? comps.nodes[c][0] : anchors[c][0];
        DisType best = std::numeric_limits<DisType>::max();
        for (GID_Type a : root_anchors) {
            for (GID_Type b : anchors[c]) {
                const DisType d = node_distance(vectors, a, b);
                if (d < best) {
                    best = d;
                    best_a = a;
                    best_b = b;
                }
            }
        }
        if (best_a >= graph.size() || best_b >= graph.size() || best_a == best_b) continue;
        graph[best_a].push_back(best_b);
        graph[best_b].push_back(best_a);
        bridge_distances.push_back(best);
    }
    if (!bridge_distances.empty()) {
        std::sort(bridge_distances.begin(), bridge_distances.end());
        long double sum = 0.0;
        for (DisType d : bridge_distances) sum += d;
        auto pct = [&](double p) {
            const size_t idx = std::min(bridge_distances.size() - 1,
                static_cast<size_t>(std::llround(p * (bridge_distances.size() - 1))));
            return bridge_distances[idx];
        };
        size_t max_size = 0;
        for (const auto& c : comps.nodes) max_size = std::max(max_size, c.size());
        g_logger->info("nearest_component connected {} components with {} bridges, max_component={}, avg_bridge_l2={:.2f}, p50={:.2f}, p90={:.2f}, p99={:.2f}",
                       comp_count, bridge_distances.size(), max_size,
                       static_cast<double>(sum / bridge_distances.size()),
                       pct(0.50), pct(0.90), pct(0.99));
    }
}

inline void connect_graph(std::vector<std::vector<GID_Type>>& graph) {
    size_t n = graph.size();
    std::vector<int> component(n, -1);
    int comp_id = 0;
    for (size_t i = 0; i < n; ++i) if (component[i] == -1) {
        std::queue<GID_Type> q; q.push(i); component[i] = comp_id;
        while (!q.empty()) {
            GID_Type u = q.front(); q.pop();
            for (auto v : graph[u]) if (component[v] == -1) { component[v] = comp_id; q.push(v); }
        }
        comp_id++;
    }
    if (comp_id == 1) {
        g_logger->info("Graph is already connected");
        return;
    }
    g_logger->warn("Graph has {} connected components, connecting them...", comp_id);
    std::vector<std::vector<GID_Type>> comps(comp_id);
    for (size_t i = 0; i < n; ++i) comps[component[i]].push_back(i);
    for (int i = 1; i < comp_id; ++i) {
        auto min_deg = [&](const std::vector<GID_Type>& vs) {
            return *std::min_element(vs.begin(), vs.end(), [&](GID_Type a, GID_Type b){return graph[a].size()<graph[b].size();});
        };
        GID_Type a = min_deg(comps[0]), b = min_deg(comps[i]);
        graph[a].push_back(b);
        graph[b].push_back(a);
        g_logger->debug("Connected component {} to 0 via {} <-> {}", i, a, b);
    }
}



inline void connect_graph_op(std::vector<std::vector<GID_Type>>& graph) {
    size_t n = graph.size();
    std::vector<int> component(n, -1);
    int comp_id = 0;
    for (size_t i = 0; i < n; ++i) if (component[i] == -1) {
        std::queue<GID_Type> q; q.push(i); component[i] = comp_id;
        while (!q.empty()) {
            GID_Type u = q.front(); q.pop();
            for (auto v : graph[u]) if (component[v] == -1) { component[v] = comp_id; q.push(v); }
        }
        comp_id++;
    }
    if (comp_id == 1) {
        g_logger->info("Graph is already connected");
        return;
    }
    g_logger->warn("Graph has {} connected components, connecting them...", comp_id);
    std::vector<std::vector<GID_Type>> comps(comp_id);
    for (size_t i = 0; i < n; ++i) comps[component[i]].push_back(i);
    std::vector<GID_Type> min_deg_nodes(comp_id);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < comp_id; ++i) {
        if (comps[i].empty()) continue;
        
        GID_Type min_node = comps[i][0];
        size_t min_deg = graph[min_node].size();
        
        for (auto node : comps[i]) {
            size_t deg = graph[node].size();
            if (deg < min_deg) {
                min_deg = deg;
                min_node = node;
            }
        }
        min_deg_nodes[i] = min_node;
    }
    GID_Type a0 = min_deg_nodes[0];
    for (int i = 1; i < comp_id; ++i) {
        if (static_cast<size_t>(i) >= min_deg_nodes.size()) continue;
        
        GID_Type b = min_deg_nodes[i];
        
        if (a0 >= n || b >= n) continue;
        graph[a0].push_back(b);
        graph[b].push_back(a0);
        if (graph[a0].size() < graph[b].size()) a0=b;
    }
    
}

inline void connect_graph_with_op(){

}



inline void save_ksg_to_file(const std::string& ksg_file, const std::vector<std::vector<GID_Type>>& graph) {
    int max_degree = -1;
    size_t total_degree = 0;
    for(const auto& nbs : graph) total_degree +=  nbs.size();
    g_logger->info("Saving KSG to {} (nodes={}, max_degree={},total_degree={})", ksg_file, graph.size(), max_degree,total_degree);
    GraphMMapRW<GID_Type,GL_Type> writer(ksg_file, graph.size(), max_degree, GraphMMapRW<GID_Type,GL_Type>::Mode::ReadWrite, total_degree);
    writer.save_graph(graph);
    writer.flush();
}


void check_graph(const std::vector<std::vector<GID_Type>>& graph) {
    size_t n = graph.size();
    bool has_self_loop = false, has_dup = false, undirected = true;
    int min_deg = INT_MAX, max_deg = 0, total_deg = 0;

    #pragma omp parallel for schedule(dynamic) reduction(min:min_deg) reduction(max:max_deg) reduction(+:total_deg)
    for (size_t i = 0; i < n; ++i) {
        std::unordered_set<GID_Type> neighbor_set;
        for (auto nb : graph[i]) {
            if (nb == i) {has_self_loop = true;g_logger->warn("Node {} has self-loop!", i);}
            if (!neighbor_set.insert(nb).second) {has_dup = true;g_logger->warn("Node {} has duplicate neighbor {}", i, nb);}
        }
        int deg = graph[i].size();
        min_deg = std::min(min_deg, deg);
        max_deg = std::max(max_deg, deg);
        total_deg += deg;
    }
    double avg_deg = double(total_deg) / n;

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (auto nb : graph[i]) {
            if (nb < n && std::find(graph[nb].begin(), graph[nb].end(), i) == graph[nb].end()) {
                #pragma omp critical
                {
                    undirected = false;
                    g_logger->warn("Edge {}-{} is not symmetric", i, nb);
                }
            }
        }
    }
    std::vector<bool> visited(n, false);
    int conn_cnt = 0;
    for (size_t i = 0; i < n; ++i) if (!visited[i]) {
        std::queue<GID_Type> q; q.push(i); visited[i] = true;
        while (!q.empty()) {
            GID_Type u = q.front(); q.pop();
            for (auto v : graph[u]) {
                if (!visited[v]) { visited[v] = true; q.push(v); }
            }
        }
        conn_cnt++;
    }
    if (has_self_loop)  g_logger->warn("Graph has self-loops!");
    if (has_dup)        g_logger->warn("Graph has duplicate neighbors!");
    if (!undirected)    g_logger->warn("Graph is not undirected!");
    if (conn_cnt > 1)   g_logger->warn("Graph has {} connected components (not fully connected)!", conn_cnt);
    g_logger->info("Graph degree: min={}, max={}, avg={:.2f}", min_deg, max_deg, avg_deg);
}



template<typename T>
void connect_graph_by_policy(std::vector<std::vector<GID_Type>>& graph,
                             const std::string& policy,
                             const std::string& vec_file,
                             int anchor_count) {
    if (policy == "nearest_component") {
        connect_graph_nearest_component<T>(graph, vec_file, anchor_count);
    } else {
        connect_graph_op(graph);
    }
}

template<typename T>
std::vector<std::vector<GID_Type>> build_ksg_from_init(const KSGParams& params) {
    g_logger->info("Loading KSG from init file: {}", params.ksg_init_file);
    VecMMapReader<GID_Type> ksg_reader(params.ksg_init_file);
    auto ksg_graph = ksg_reader.read_all_vectors();
    for (auto& neighbors : ksg_graph) {
        if (neighbors.size() > static_cast<size_t>(params.p_degree)) neighbors.resize(static_cast<size_t>(params.p_degree));
    }
    make_undirected_op(ksg_graph);
    auto comps = find_components(ksg_graph);
    save_component_file(params.component_file, comps.component, static_cast<uint32_t>(comps.nodes.size()));
    connect_graph_by_policy<T>(ksg_graph, params.connect_policy, params.vec_file, params.connect_anchor_count);
    check_graph(ksg_graph);
    return ksg_graph;
}

void build_ksg_from_init_indisk(const std::string& ksg_init_file,
                                 const std::string& ksg_file,
                                 const std::string& component_file,
                                 const std::string& connect_policy,
                                 const std::string& vec_file,
                                 const std::string& data_type,
                                 int connect_anchor_count,
                                 int p_degree) {
    // Canonical file-input path: ksg_init_file is the already-built KNN graph.
    // This function does not inspect its builder/parameter provenance; callers
    // must bind that provenance outside the KSG binary format.
    auto total_start = std::chrono::steady_clock::now();

    // Step 1: Load KNN from init file into single flat array (1 allocation, not 1B)
    g_logger->info("[indisk] Opening KNN init file: {}", ksg_init_file);
    VecMMapReader<GID_Type> knn_reader(ksg_init_file);
    size_t n = knn_reader.num_vectors();
    const int input_degree = static_cast<int>(knn_reader.dim());
    const int actual_p_degree = std::min(input_degree, p_degree);
    g_logger->info("[indisk] KNN graph: {} nodes, input degree {}, selected degree {}",
                   n, input_degree, actual_p_degree);

    knn_reader.read_all_vectors_flat();
    size_t knn_flat_mb = (n * actual_p_degree * sizeof(GID_Type)) / (1024 * 1024);
    g_logger->info("[indisk] KNN loaded into flat buffer ({} MB)", knn_flat_mb);

    // Allocate our own mutable flat buffer, copy and sort from mmap'd data
    std::vector<GID_Type> knn_flat(n * actual_p_degree);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        const GID_Type* src = knn_reader.vector_data_unchecked(i);
        GID_Type* dst = &knn_flat[i * actual_p_degree];
        std::memcpy(dst, src, actual_p_degree * sizeof(GID_Type));
        std::sort(dst, dst + actual_p_degree);
    }
    g_logger->info("[indisk] KNN neighbor lists copied and sorted");
    // knn_reader will cleanup its mmap/flat buffer on destruction

    // Step 2: Lock-free reverse edge collection (same pattern as make_ssg_undirected_limit_op)
    g_logger->info("[indisk] Collecting missing reverse edges...");
    using Edge = std::pair<GID_Type, GID_Type>;  // (dst, src)
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<Edge>> thread_buffers(num_threads);

    #pragma omp parallel
    {
        auto& buf = thread_buffers[omp_get_thread_num()];
        buf.reserve(n * actual_p_degree / num_threads / 4);
    }

    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        auto& buf = thread_buffers[omp_get_thread_num()];
        GID_Type* my_neighbors = &knn_flat[i * actual_p_degree];
        for (int j = 0; j < actual_p_degree; ++j) {
            GID_Type nb = my_neighbors[j];
            if (nb >= n) continue;
            GID_Type* nb_neighbors = &knn_flat[nb * actual_p_degree];
            if (!std::binary_search(nb_neighbors, nb_neighbors + actual_p_degree,
                                    static_cast<GID_Type>(i))) {
                buf.emplace_back(nb, static_cast<GID_Type>(i));
            }
        }
    }

    size_t total_rev_edges = 0;
    for (auto& buf : thread_buffers) total_rev_edges += buf.size();
    g_logger->info("[indisk] Missing reverse edges to add: {}", total_rev_edges);

    std::vector<Edge> all_rev_edges;
    all_rev_edges.reserve(total_rev_edges);
    for (auto& buf : thread_buffers) {
        all_rev_edges.insert(all_rev_edges.end(), buf.begin(), buf.end());
        buf.clear();
        buf.shrink_to_fit();
    }
    thread_buffers.clear();
    std::sort(all_rev_edges.begin(), all_rev_edges.end(),
        [](const Edge& a, const Edge& b) { return a.first < b.first; });

    // Prefix sum for reverse edges by dst
    std::vector<size_t> rev_offset(n + 1, 0);
    for (const auto& e : all_rev_edges) rev_offset[e.first + 1]++;
    for (size_t i = 1; i <= n; ++i) rev_offset[i] += rev_offset[i - 1];

    // Step 3: Compute final degrees
    std::vector<GL_Type> final_degree(n, 0);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        GL_Type count = static_cast<GL_Type>(actual_p_degree);
        GID_Type* my_knn = &knn_flat[i * actual_p_degree];
        size_t r_start = rev_offset[i], r_end = rev_offset[i + 1];
        for (size_t k = r_start; k < r_end; ++k) {
            GID_Type src = all_rev_edges[k].second;
            if (!std::binary_search(my_knn, my_knn + actual_p_degree, src)) ++count;
        }
        final_degree[i] = count;
    }

    // Prefix sum for final KSG flat buffer
    std::vector<size_t> ksg_offset(n + 1, 0);
    size_t total_ksg_edges = 0;
    for (size_t i = 0; i < n; ++i) {
        ksg_offset[i] = total_ksg_edges;
        total_ksg_edges += final_degree[i];
    }
    ksg_offset[n] = total_ksg_edges;
    double avg_deg = static_cast<double>(total_ksg_edges) / n;
    g_logger->info("[indisk] Final KSG: {} nodes, {} edges, avg degree {:.2f}", n, total_ksg_edges, avg_deg);

    // Step 4: Build final KSG flat buffer
    std::vector<GID_Type> ksg_flat(total_ksg_edges);
    #pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; ++i) {
        size_t pos = ksg_offset[i];
        GID_Type* my_knn = &knn_flat[i * actual_p_degree];
        for (int j = 0; j < actual_p_degree; ++j) ksg_flat[pos++] = my_knn[j];
        size_t r_start = rev_offset[i], r_end = rev_offset[i + 1];
        for (size_t k = r_start; k < r_end; ++k) {
            GID_Type src = all_rev_edges[k].second;
            if (!std::binary_search(my_knn, my_knn + actual_p_degree, src))
                ksg_flat[pos++] = src;
        }
        std::sort(&ksg_flat[ksg_offset[i]], &ksg_flat[ksg_offset[i] + final_degree[i]]);
    }

    // Release early
    knn_flat.clear(); knn_flat.shrink_to_fit();
    all_rev_edges.clear(); all_rev_edges.shrink_to_fit();
    rev_offset.clear(); rev_offset.shrink_to_fit();

    // Step 5: Save KSG in variable-length format (sequential write, ~44GB)
    g_logger->info("[indisk] Writing KSG file...");
    {
        std::ofstream ofs(ksg_file, std::ios::binary | std::ios::trunc);
        uint32_t n32 = static_cast<uint32_t>(n);
        int32_t var_fmt = -1;
        ofs.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
        ofs.write(reinterpret_cast<const char*>(&var_fmt), sizeof(var_fmt));
        // Buffer writes in 1M-node chunks
        const size_t chunk_nodes = 1000000;
        for (size_t base = 0; base < n; base += chunk_nodes) {
            size_t end = std::min(base + chunk_nodes, n);
            for (size_t i = base; i < end; ++i) {
                uint32_t deg = final_degree[i];
                ofs.write(reinterpret_cast<const char*>(&deg), sizeof(deg));
                ofs.write(reinterpret_cast<const char*>(&ksg_flat[ksg_offset[i]]),
                          deg * sizeof(GID_Type));
            }
            if (n >= 10000000 && base % (n / 10) < chunk_nodes)
                g_logger->info("[indisk] Write progress: {}/{} nodes", base, n);
        }
        ofs.close();
    }
    auto write_end = std::chrono::steady_clock::now();
    g_logger->info("[indisk] KSG file written in {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(write_end - total_start).count());

    // Step 6: Verify connectivity with Parallel Union-Find
    g_logger->info("[indisk] Verifying connectivity with parallel Union-Find...");
    {
        ParallelUnionFind uf(n);
        #pragma omp parallel for schedule(dynamic, 1024)
        for (size_t i = 0; i < n; ++i) {
            GID_Type u = static_cast<GID_Type>(i);
            size_t start = ksg_offset[i], end = ksg_offset[i + 1];
            for (size_t k = start; k < end; ++k) {
                GID_Type v = ksg_flat[k];
                if (v < n) uf.union_sets_atomic(u, v);
            }
        }
        std::vector<int> comp_id;
        std::vector<GID_Type> rep_nodes;
        int comp_count = uf.count_components_min_degree(comp_id, rep_nodes, final_degree);
        if (!component_file.empty()) {
            std::vector<uint32_t> component_u32(comp_id.size());
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < comp_id.size(); ++i) component_u32[i] = static_cast<uint32_t>(comp_id[i]);
            save_component_file(component_file, component_u32, static_cast<uint32_t>(comp_count));
        }

        if (comp_count == 1) {
            g_logger->info("[indisk] Graph is fully connected");
        } else {
            g_logger->warn("[indisk] Graph has {} components — adding bridge edges", comp_count);
            std::vector<std::pair<GID_Type, GID_Type>> bridge_list;
            bridge_list.reserve(static_cast<size_t>(comp_count - 1) * 2);
            std::vector<size_t> comp_sizes(comp_count, 0);
            for (int cid : comp_id) if (cid >= 0 && cid < comp_count) ++comp_sizes[static_cast<size_t>(cid)];
            int root = 0;
            for (int c = 1; c < comp_count; ++c) {
                if (comp_sizes[static_cast<size_t>(c)] > comp_sizes[static_cast<size_t>(root)]) root = c;
            }
            if (connect_policy == "nearest_component" && !vec_file.empty()) {
                std::vector<std::vector<GID_Type>> comp_anchors(static_cast<size_t>(comp_count));
                std::vector<GID_Type> min_node(static_cast<size_t>(comp_count), static_cast<GID_Type>(n));
                std::vector<GID_Type> max_node(static_cast<size_t>(comp_count), static_cast<GID_Type>(n));
                std::vector<GL_Type> min_deg(static_cast<size_t>(comp_count), std::numeric_limits<GL_Type>::max());
                std::vector<GL_Type> max_deg(static_cast<size_t>(comp_count), 0);
                const int anchors_per_comp = std::max(1, connect_anchor_count);
                auto add_anchor = [&](int c, GID_Type id) {
                    auto& a = comp_anchors[static_cast<size_t>(c)];
                    if (id >= n || std::find(a.begin(), a.end(), id) != a.end()) return;
                    if (static_cast<int>(a.size()) < anchors_per_comp) a.push_back(id);
                };
                for (size_t i = 0; i < n; ++i) {
                    const int c = comp_id[i];
                    if (c < 0 || c >= comp_count) continue;
                    add_anchor(c, static_cast<GID_Type>(i));
                    if (final_degree[i] < min_deg[static_cast<size_t>(c)]) {
                        min_deg[static_cast<size_t>(c)] = final_degree[i];
                        min_node[static_cast<size_t>(c)] = static_cast<GID_Type>(i);
                    }
                    if (final_degree[i] > max_deg[static_cast<size_t>(c)]) {
                        max_deg[static_cast<size_t>(c)] = final_degree[i];
                        max_node[static_cast<size_t>(c)] = static_cast<GID_Type>(i);
                    }
                }
                for (int c = 0; c < comp_count; ++c) {
                    add_anchor(c, rep_nodes[static_cast<size_t>(c)]);
                    add_anchor(c, min_node[static_cast<size_t>(c)]);
                    add_anchor(c, max_node[static_cast<size_t>(c)]);
                }
                auto add_nearest = [&](auto tag) {
                    using VecT = decltype(tag);
                    VecMMapReader<VecT> vectors(vec_file);
                    std::vector<DisType> distances;
                    distances.reserve(comp_count - 1);
                    const auto& root_anchors = comp_anchors[static_cast<size_t>(root)];
                    for (int c = 0; c < comp_count; ++c) {
                        if (c == root) continue;
                        const auto& target_anchors = comp_anchors[static_cast<size_t>(c)];
                        if (root_anchors.empty() || target_anchors.empty()) continue;
                        GID_Type best_a = root_anchors[0];
                        GID_Type best_b = target_anchors[0];
                        DisType best = std::numeric_limits<DisType>::max();
                        for (GID_Type a : root_anchors) {
                            for (GID_Type b : target_anchors) {
                                if (a >= n || b >= n || a == b) continue;
                                const DisType d = node_distance(vectors, a, b);
                                if (d < best) {
                                    best = d;
                                    best_a = a;
                                    best_b = b;
                                }
                            }
                        }
                        if (best == std::numeric_limits<DisType>::max()) continue;
                        distances.push_back(best);
                        bridge_list.emplace_back(best_a, best_b);
                        bridge_list.emplace_back(best_b, best_a);
                    }
                    if (!distances.empty()) {
                        std::sort(distances.begin(), distances.end());
                        long double sum = 0.0;
                        for (DisType d : distances) sum += d;
                        g_logger->info("[indisk] nearest_component bridges root={}, anchors_per_comp={}, avg_l2={:.2f}, p50={:.2f}, p90={:.2f}",
                            root, anchors_per_comp, static_cast<double>(sum / distances.size()),
                            distances[distances.size() / 2],
                            distances[std::min(distances.size() - 1, distances.size() * 9 / 10)]);
                    }
                };
                if (data_type == "float") add_nearest(float{});
                else add_nearest(uint8_t{});
            } else {
                for (int c = 0; c < comp_count; ++c) {
                    if (c == root) continue;
                    GID_Type a = rep_nodes[static_cast<size_t>(root)], b = rep_nodes[static_cast<size_t>(c)];
                    if (a >= n || b >= n) continue;
                    bridge_list.emplace_back(a, b);
                    bridge_list.emplace_back(b, a);
                }
            }
            std::sort(bridge_list.begin(), bridge_list.end());

            // Rebuild the flat graph so each bridge is physically inserted into
            // its owner's list. Recomputing offsets without this rebuild corrupts
            // every list after the first bridge endpoint.
            std::vector<size_t> old_offset = std::move(ksg_offset);
            std::vector<GID_Type> repaired_flat(ksg_flat.size() + bridge_list.size());
            ksg_offset.resize(n + 1);
            size_t cur = 0, bridge_pos = 0;
            for (size_t i = 0; i < n; ++i) {
                ksg_offset[i] = cur;
                const size_t old_begin = old_offset[i];
                const size_t old_end = old_offset[i + 1];
                std::copy(ksg_flat.begin() + old_begin, ksg_flat.begin() + old_end,
                          repaired_flat.begin() + cur);
                cur += old_end - old_begin;
                while (bridge_pos < bridge_list.size() && bridge_list[bridge_pos].first == i) {
                    repaired_flat[cur++] = bridge_list[bridge_pos++].second;
                }
                final_degree[i] = static_cast<GL_Type>(cur - ksg_offset[i]);
                std::sort(repaired_flat.begin() + ksg_offset[i], repaired_flat.begin() + cur);
            }
            ksg_offset[n] = cur;
            total_ksg_edges = cur;
            repaired_flat.resize(cur);
            ksg_flat.swap(repaired_flat);

            // Re-write KSG file with bridge edges
            g_logger->info("[indisk] Re-writing KSG with {} bridge edges...", bridge_list.size());
            {
                std::ofstream ofs(ksg_file, std::ios::binary | std::ios::trunc);
                uint32_t n32 = static_cast<uint32_t>(n);
                int32_t var_fmt = -1;
                ofs.write(reinterpret_cast<const char*>(&n32), sizeof(n32));
                ofs.write(reinterpret_cast<const char*>(&var_fmt), sizeof(var_fmt));
                const size_t chunk_nodes = 1000000;
                for (size_t base = 0; base < n; base += chunk_nodes) {
                    size_t end = std::min(base + chunk_nodes, n);
                    for (size_t i = base; i < end; ++i) {
                        uint32_t deg = final_degree[i];
                        ofs.write(reinterpret_cast<const char*>(&deg), sizeof(deg));
                        ofs.write(reinterpret_cast<const char*>(&ksg_flat[ksg_offset[i]]),
                                  deg * sizeof(GID_Type));
                    }
                }
                ofs.close();
            }
        }
    }

    auto total_end = std::chrono::steady_clock::now();
    g_logger->info("[indisk] KSG built in {} ms",
        std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count());
    g_logger->info("[indisk] KSG saved to {}", ksg_file);
}





// inline void make_undirected_with_limit(std::vector<std::vector<GID_Type>>& graph, int max_degree) {
//     size_t n = graph.size();
//     g_logger->info("Making graph undirected with degree limit {}...", max_degree);
//     for( size_t i = 0; i < n; ++i) {
//         for (auto nb_it = graph[i].begin(); nb_it != graph[i].end(); ) {
//             GID_Type nb = *nb_it;
//             if (nb < n && std::find(graph[nb].begin(), graph[nb].end(), i) == graph[nb].end()) {
//                 if (graph[nb].size() < size_t(max_degree)) {
//                     graph[nb].push_back(i);
//                     ++nb_it;
//                 } else {
//                     nb_it = graph[i].erase(nb_it);
//                 }
//             } else {
//                 ++nb_it;
//             }
//         }
//     }

// }



inline void make_undirected_with_limit(std::vector<std::vector<GID_Type>>& graph, int max_degree) {
    size_t n = graph.size();
    g_logger->info("Making graph undirected with degree limit {}...", max_degree);


    for( size_t i = 0; i < n; ++i) {
        if (graph[i].size() > (size_t)max_degree) {
            graph[i].resize(max_degree);
        }

        for (auto nb_it = graph[i].begin(); nb_it != graph[i].end(); ) {
            GID_Type nb = *nb_it;
            if (nb >= n) {
                 nb_it = graph[i].erase(nb_it);
                 continue;
            }

            if (std::find(graph[nb].begin(), graph[nb].end(), i) == graph[nb].end()) {
                if (graph[nb].size() < size_t(max_degree)) {
                    graph[nb].push_back(i);
                    ++nb_it;
                } else {
                    nb_it = graph[i].erase(nb_it);
                }
            } else {
                ++nb_it;
            }
        }
    }
}

std::vector<std::vector<GID_Type>> build_ksg_with_limit(const std::string& ksg_init_file, int max_degree) {
    g_logger->info("Loading KSG from init file with degree limit {}: {}", max_degree, ksg_init_file);
    VecMMapReader<GID_Type> ksg_reader(ksg_init_file);
    auto ksg_graph = ksg_reader.read_all_vectors();
    make_undirected_with_limit(ksg_graph,max_degree);
    connect_graph_op(ksg_graph);
    check_graph(ksg_graph);
    return ksg_graph;
}

template<typename T>
inline std::vector<T> compute_center(const std::vector<GID_Type>& nodes) {
    // #TODO
    return std::vector<T>();
}

template<typename T>
inline GID_Type find_closest_node(const std::vector<T>& center, const std::vector<GID_Type>& nodes) {
    // #TODO
    return GID_Type(0);
}

// #TODO
template<typename T>
std::vector<std::vector<GID_Type>> build_sub_ksg(std::vector<std::vector<GID_Type>>& graph, const std::string vec_file){
    std::vector<bool> visited(graph.size(), false);
    std::vector<std::vector<GID_Type>> components;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!visited[i]) {
            std::vector<GID_Type> component;
            std::queue<GID_Type> q;
            q.push(i);
            visited[i] = true;
            while (!q.empty()) {
                GID_Type node = q.front();
                q.pop();
                component.push_back(node);
                for (auto neighbor : graph[node]) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
            components.push_back(component);
        }
    }
    int num_components = components.size();
    g_logger->info("Found {} connected components in KSG", num_components);
    std::vector<GID_Type> centers;
    centers.reserve(num_components);
    VecMMapReader<T> vec_reader(vec_file);
    int dim = vec_reader.dim();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_components; ++i) {
        if (components[i].empty()) continue;

        std::vector<float> centroid(dim, 0.0f);
        for (GID_Type node_id : components[i]) {
            auto vec = vec_reader.read_vector(node_id);
            for(int d = 0; d < dim; ++d) {
                centroid[d] += static_cast<float>(vec[d]);
            }
        }
        for(int d = 0; d < dim; ++d) {
            centroid[d] /= components[i].size();
        }

        GID_Type best_node = components[i][0];
        float min_dist = std::numeric_limits<float>::max();

        for (GID_Type node_id : components[i]) {
            auto vec = vec_reader.read_vector(node_id);
            auto float_vec = vec_to_float(vec);
            float dist = L2Sqr(centroid.data(), float_vec.data(), &dim);
            if (dist < min_dist) {
                min_dist = dist;
                best_node = node_id;
            }
        }
        centers[i] = best_node;
        g_logger->debug("Component {} center is node {}", i, best_node);
    }

    // #TODO
    g_logger->info("Building sub-KSG (clique) from {} centers...", num_components);
    std::vector<std::vector<GID_Type>> sub_ksg(num_components);
    // for (int i = 0; i < num_components; ++i) {
    //     sub_ksg[i].reserve(num_components - 1);
    //     for (int j = 0; j < num_components; ++j) {
    //         if (i == j) continue;
    //         sub_ksg[i].push_back(j);
    //     }
    // }

    return sub_ksg;
}

int main(int argc, char** argv) {

    KSGParams params;
    if (!parse_args(argc, argv, params)) {
        print_usage();
        return 1;
    }
    setup_logger(params.log_file);
    // g_logger->info("Building KSG with parameters: pq_index_file={}, vec_file={}, data_type={}, p_degree={}, indisk={}, log_file={}, ksg_file={}",
    //     params.pq_index_file, params.vec_file, params.data_type, params.p_degree, params.indisk, params.log_file, params.ksg_file);
    auto start = std::chrono::steady_clock::now();
    if(std::filesystem::exists(params.ksg_init_file)) {
        if (params.indisk) {
            g_logger->info("[indisk] Building KSG from init file with optimized indisk path");
            build_ksg_from_init_indisk(params.ksg_init_file, params.ksg_file,
                                       params.component_file, params.connect_policy,
                                       params.vec_file, params.data_type,
                                       params.connect_anchor_count, params.p_degree);
            auto end = std::chrono::steady_clock::now();
            g_logger->info("[indisk] KSG built from init file in {} ms",
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        } else {
            g_logger->info("Loading KSG from init file: {}", params.ksg_init_file);
            std::vector<std::vector<GID_Type>> ksg_graph;
            if (params.data_type == "float") ksg_graph = build_ksg_from_init<float>(params);
            else ksg_graph = build_ksg_from_init<uint8_t>(params);
            save_ksg_to_file(params.ksg_file, ksg_graph);
            auto end = std::chrono::steady_clock::now();
            g_logger->info("KSG graph built from init file in {} ms",
                std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
            g_logger->info("KSG graph saved to {}", params.ksg_file);
        }
        return 0;
    }
    if(!std::filesystem::exists(params.pq_index_file) || !std::filesystem::exists(params.vec_file)) {
        g_logger->error("pq_index_file or vec_file does not exist: {}, {}", params.pq_index_file, params.vec_file);
        return 1;
    }
    faiss::IndexIVFPQ* pq_index = load_pq_index(params.pq_index_file);
    std::vector<std::vector<GID_Type>> ksg_graph;
    if (params.data_type == "float")
        ksg_graph = build_ksg_neighbor_graph<float>(params.vec_file, pq_index, params.p_degree, params.indisk);
    else if (params.data_type == "uint8")
        ksg_graph = build_ksg_neighbor_graph<uint8_t>(params.vec_file, pq_index, params.p_degree, params.indisk);
    else {
        g_logger->error("Unsupported data_type: only float/uint8 supported");
        throw std::runtime_error("Unsupported data_type: only float/uint8 supported");
    }
    g_logger->info("KSG graph built with {} nodes", ksg_graph.size());
    make_undirected(ksg_graph);
    {
        auto comps = find_components(ksg_graph);
        save_component_file(params.component_file, comps.component, static_cast<uint32_t>(comps.nodes.size()));
    }
    if (params.data_type == "float")
        connect_graph_by_policy<float>(ksg_graph, params.connect_policy, params.vec_file, params.connect_anchor_count);
    else
        connect_graph_by_policy<uint8_t>(ksg_graph, params.connect_policy, params.vec_file, params.connect_anchor_count);
    check_graph(ksg_graph);
    auto end = std::chrono::steady_clock::now();
    g_logger->info("KSG graph built in {} ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    save_ksg_to_file(params.ksg_file, ksg_graph);
    g_logger->info("KSG graph saved to {}", params.ksg_file);
    delete pq_index;
    return 0;
}
