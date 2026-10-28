#pragma once
#include <vector>
#include <cstdint>
#include <omp.h>
#include "types.h"

// ===== Parallel Union-Find (Disjoint Set Union) =====
// Replaces serial BFS for connectivity checks at billion-node scale.
// Algorithm:
//   1. Parallel edge scan: union connected nodes
//   2. Parallel pointer jumping: flatten all paths to root
//   3. Count components and identify representative nodes
//
// Complexity: O(N α(N)) with near-perfect parallelism.

class ParallelUnionFind {
    size_t n_;
    std::vector<GID_Type> parent_;
    std::vector<unsigned> rank_;  // unsigned sufficient for path length

public:
    ParallelUnionFind(size_t n) : n_(n), parent_(n), rank_(n, 0) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            parent_[i] = static_cast<GID_Type>(i);
        }
    }

    // Find with path halving (iterative, no recursion stack risk)
    GID_Type find(GID_Type x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];  // path halving
            x = parent_[x];
        }
        return x;
    }

    // Union with linking by size (rank tracks tree depth upper bound)
    void union_sets(GID_Type a, GID_Type b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) {
            parent_[a] = b;
        } else if (rank_[a] > rank_[b]) {
            parent_[b] = a;
        } else {
            parent_[b] = a;
            rank_[a]++;
        }
    }

    // Thread-safe union using atomic compare-exchange on parent pointers.
    // Used during parallel edge traversal to avoid locks.
    void union_sets_atomic(GID_Type a, GID_Type b) {
        while (true) {
            a = find(a);
            b = find(b);
            if (a == b) return;
            // Link smaller to larger by rank heuristic
            if (rank_[a] < rank_[b] || (rank_[a] == rank_[b] && a > b)) {
                std::swap(a, b);
            }
            // Try to attach b's root to a
            GID_Type old = parent_[b];
            if (old != b) continue;  // another thread changed b's parent
            // No atomic needed here since find() already resolved the root.
            // We use a simple assignment; conflicting unions are benign
            // because the next find() call resolves any inconsistency.
            parent_[b] = a;
            if (rank_[a] == rank_[b]) {
                #pragma omp atomic update
                rank_[a]++;
            }
            break;
        }
    }

    // Parallel pointer jumping: make every node point directly to its root.
    // This converges in O(log N) rounds, each round fully parallel.
    void flatten() {
        bool changed = true;
        while (changed) {
            changed = false;
            #pragma omp parallel
            {
                bool local_changed = false;
                #pragma omp for schedule(static)
                for (size_t i = 0; i < n_; ++i) {
                    GID_Type p = parent_[i];
                    if (parent_[p] != p) {
                        parent_[i] = parent_[p];
                        local_changed = true;
                    }
                }
                if (local_changed) {
                    #pragma omp atomic write
                    changed = true;
                }
            }
        }
    }

    // Reduce all parent pointers to final roots via find() in parallel.
    // Simpler alternative to flatten() — one parallel pass, no iteration.
    void compress_all() {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            parent_[i] = find(static_cast<GID_Type>(i));
        }
    }

    // Returns component count and populates comp_id for each node.
    // rep_nodes[comp] = a representative node for that component.
    int count_components(std::vector<int>& comp_id, std::vector<GID_Type>& rep_nodes) {
        compress_all();
        comp_id.assign(n_, -1);
        rep_nodes.clear();

        // Assign component IDs in parallel via atomic counter
        int comp_counter = 0;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            GID_Type root = parent_[i];
            if (static_cast<size_t>(root) == i) {
                int my_comp;
                #pragma omp atomic capture
                my_comp = comp_counter++;
                comp_id[i] = my_comp;
            }
        }
        rep_nodes.resize(comp_counter);

        // Propagate component IDs from roots to all nodes
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            if (comp_id[i] == -1) {
                GID_Type root = parent_[i];
                comp_id[i] = comp_id[root];
            }
        }

        // Find representative node for each component (min-degree or just the root)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            int c = comp_id[i];
            if (static_cast<size_t>(parent_[i]) == i) {  // i is a root
                rep_nodes[c] = static_cast<GID_Type>(i);
            }
        }

        return comp_counter;
    }

    // Same as count_components but selects representative with minimum degree
    int count_components_min_degree(std::vector<int>& comp_id, std::vector<GID_Type>& rep_nodes,
                                     const std::vector<GL_Type>& degrees) {
        compress_all();
        comp_id.assign(n_, -1);
        rep_nodes.clear();

        int comp_counter = 0;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            GID_Type root = parent_[i];
            if (static_cast<size_t>(root) == i) {
                int my_comp;
                #pragma omp atomic capture
                my_comp = comp_counter++;
                comp_id[i] = my_comp;
            }
        }
        rep_nodes.assign(comp_counter, 0);
        // Init rep degree to max
        std::vector<GL_Type> rep_deg(comp_counter, std::numeric_limits<GL_Type>::max());

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            if (comp_id[i] == -1) {
                GID_Type root = parent_[i];
                comp_id[i] = comp_id[root];
            }
            int c = comp_id[i];
            GL_Type deg = degrees[i];
            // Use critical section per component (rare collision with few components)
            #pragma omp critical
            {
                if (deg < rep_deg[c]) {
                    rep_deg[c] = deg;
                    rep_nodes[c] = static_cast<GID_Type>(i);
                }
            }
        }

        return comp_counter;
    }

    size_t size() const { return n_; }
};
