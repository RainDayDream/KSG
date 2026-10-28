# KSG: memory-centred graph search with an SSD supplement graph

KSG is an approximate nearest-neighbour (ANN) index for one machine with an NVMe SSD, for datasets from millions up to a billion vectors.

* **KSG (Key Sparse Graph), in memory.** A sparse graph over **all** points: approximate 5-NN lists plus reverse edges, about 9 edges per node.
  It is kept in memory together with a 5-bit affine scalar-quantised (SQ) copy of every vector (80 B/vector at d = 128). The search navigates
  and expands mostly here, with no I/O.
* **SSG (Supplement Search Graph), on SSD.** A dense graph (R = 96, search-based refinement rounds with robust prune until the graph has
  converged, then an in-degree boost).
  Each node's full-precision vector and neighbour list are stored in one record, and records are packed 7 per 4 KB page by graph
  connectivity. Reading one page gives exact distances (for the final ranking) plus extra neighbours that let the in-memory search escape
  local minima.
* **Search.**
  1. Entry: an online k-means entry index (16384 centroids) picks starting points close to the query.
  2. Best-first search over the KSG using SQ distances.
  3. Then the search alternates two phases until no unread candidate is left in the top-L:
     * an in-memory KSG phase over the best `8L` candidates;
     * pipelined SSG page reads: 32 reads in flight via io_uring with O_DIRECT, and every page is processed as soon as it arrives.
  4. The answer is the top-K by exact distance. `L` is the only recall knob.

Tools: `build_KSG` builds the KSG (loaded with `--mem_graph`); `build_ssg_search` / `build_ssg_shard` build the SSG, which the search reads from the co-located page layout (`--colo-file`).

## Requirements

* Linux x86-64 with AVX-512 (F, BW, VNNI), kernel ≥ 5.6 (io_uring), an NVMe SSD for the index.
* GCC ≥ 9, CMake ≥ 3.16, OpenMP, [faiss](https://github.com/facebookresearch/faiss) (C++ library; Python package for index building,
  GPU build optional), OpenBLAS, spdlog, fmt, Python 3 with numpy.

```bash
cmake -S . -B build && cmake --build build -j      # binaries in bin/
```

## Data format

All files use the common `.bin` layout: `[u32 n][u32 d]` followed by `n*d` values.
* `base.bin` / `query.bin`: `uint8` (d must be 128; zero-pad other dimensions with `scripts/pad_u8.py`, which leaves L2 distances unchanged)
  or `float32` (any d).
* `groundtruth.bin`: `[u32 nq][u32 k]` followed by `nq*k` `uint32` ids of the exact nearest neighbours.

## Quick start (1M vectors, a few minutes)

```bash
scripts/build_index.sh --base sift1m/base.bin --dtype float --out idx_sift1m            # add --knn-cpu without a GPU
scripts/search.sh --index idx_sift1m --query sift1m/query.bin --gt sift1m/groundtruth.bin --dtype float --L 200 --out run_L200
scripts/recall.py sift1m/groundtruth.bin run_L200
scripts/sweep.sh  --index idx_sift1m --query sift1m/query.bin --gt sift1m/groundtruth.bin --dtype float --out sweep_sift1m
```

Expected (fresh build, 24 threads, warm run): SIFT1M L200 Recall@100 0.9996, 7013 QPS, 129 pages/query; the uint8 copy
USIFT1M L200 0.9996, 7102 QPS, 129 pages/query. The build takes about 5 minutes per dataset. Absolute QPS depends on the SSD.

## Building an index

`scripts/build_index.sh` runs the full pipeline. Each step is resumable and logged with `/usr/bin/time -v`:

| step | tool | default parameters |
|---|---|---|
| approximate 5-NN graph | `tools/gpu_knn.py` (faiss IVFPQ, GPU or `--knn-cpu`) | nlist 8192, PQ16, nprobe 32 |
| KSG | `bin/build_KSG` | 5 neighbours + reverse edges + connectivity repair |
| SSG | `bin/build_ssg_search` | random R=32 start, 2 rounds, `--R 96 --Lc 128 --C 512`, α = 1.2, back-links |
| SSG convergence | `bin/build_ssg_search` | 1 more round from the result (`--extra-rounds 1`), back-links with up to 96 reverse candidates |
| in-degree boost | `bin/boost_indegree` | T 80, donor 88, src 48, protect 16, 2-hop sources (8), 3 passes; `--boost-batch B` parallelises it |
| page layout | `bin/pack_sectors` + `bin/build_colocated` | connectivity-greedy packing; records `vector + count + 96 ids + original id` (uint8: 520 B, 7/page; float vectors are stored as fp16) |
| SQ sidecar | `bin/build_sq4` (uint8, affine SQ5, 80 B) / `bin/build_sqf` (float, 8 or 4 bit) | per-dimension range [min, 99.9th percentile] |
| entry index | `bin/build_entry` | 16384 centroids × 4 representatives (1024 for n ≤ 2M) |
| renumbering + check | `bin/permute_ksg`, `tools/verify_layout_any.py` | KSG and entry index in layout ids; 2000 random records verified |

Output directory: `ksg.graph`, `entry.bin`, `codes.sq5` or `codes.sqf`, `ssg.colo`, and `layout.{new2old,old2new}`. Only these files are
needed for search; `--cleanup` removes the intermediate graphs.

Disk usage: about 0.6 KB per vector (≈ 58 GB of SSD layout at 100M uint8). In-memory sidecars: about 120 B per vector (≈ 12 GB at 100M).
The in-memory SSG construction needs ≈ 1.2–1.8 KB per vector of RAM, so use the sharded build beyond a few hundred million vectors.

**Billion scale.** `scripts/build_index_1b.sh` builds the SSG in 20 k-means shards. Each shard is extended by its KSG one-hop halo, and
only its primary nodes' lists are written. The in-degree boost then runs per shard, followed by the same layout steps. The paper's
USIFT1B index took ≈ 12 h on 48–72 threads with a peak of ≈ 450 GB RAM. It started every shard from the previous-generation SSG; the
script starts from a random graph instead (`--old random`, two refinement rounds).

**Why the extra round.** Refinement must converge. On DEEP100M and SPACEV100M, a graph after two rounds from a random start
reads about 2.3–2.4× more SSD pages per query at Recall@100 0.99 than after three. The in-degree boost alone does not close that gap: on SPACEV100M it
cut the share of nodes with in-degree below 48 from 45.6% to 0.5% without changing recall or pages. The extra round costs about
1–2 hours at 100M on 44 threads.

## Searching

`scripts/search.sh` runs the paper's configuration. The full command line is written to `RUN_DIR/command.txt`.

| option | meaning |
|---|---|
| `--L L` | recall knob: candidate list L, in-memory range 8L (`-L1 L -L2 L --colo-mem-L 8L`) |
| `--eps e` | early-stop variant: `-L1 100 -L2 3200 --colo-mem-L <L> --ssg-stop-eps e` (stop when the best unread candidate is (1+e)² farther than the K-th result) |
| `--K`, `--threads` | results per query (default 100), OpenMP threads (default 48; one query per thread) |

Fixed parameters:

| flag | value | meaning |
|---|---|---|
| `-B` | 32 | SSG reads in flight per query |
| `--colo-pipe` | on | pipelined reads |
| `--colo-free` | 6 | exact-distance expansion of the other records on a page that was read |
| `--colo-cache` | 1024 | per-query page cache, in pages |
| `--disk-neighbor-limit` | 96 | |
| `--sq-only` | on | SQ distances only for unread nodes |
| `--colo-sq-results` | on | unread candidates fill the result list |

The timing window covers the online entry search and all queries; index loading is excluded. The log reports the QPS and the mean
and p50/p95/p99 latency (`search.log`), plus SSD pages per query (`sectors_read`). `latency.f32` holds the per-query latency in µs
(float32, in query order).

`scripts/sweep.sh` runs a list of `L` (or `--eps-list`) twice per point and reports the second (warm) run. It writes `summary.tsv` with
the standard **Recall@K = |result[:K] ∩ GT[:K]| / K**, QPS, latency percentiles, pages per query and peak RSS. Note that the recall
printed by `ksg_search` itself divides by min(K, number returned); use `scripts/recall.py` for the standard definition.

**Determinism.** With `--colo-pipe` (default) pages are processed in I/O completion order, so a few queries can return a slightly different
result list from run to run (e.g. 10 of 10,000 queries on USIFT1M, recall within ±0.00001). Without `--colo-pipe` the search is
deterministic: repeated runs produce byte-identical `result.bin`.

Ablations (append with `--extra`):

| ablation | flags |
|---|---|
| ignore SSG neighbour lists, exact distances only | `--colo-ssg-off` |
| in-memory KSG only, no SSD | `--mode ksg_only` (result ids are layout ids: `scripts/recall.py --new2old idx/layout.new2old`) |
| no pipelining | omit `--colo-pipe` (edit `scripts/search.sh`) |
| beam width | `-B b` |

## Memory budget (low-memory variants, uint8)

`scripts/build_lowmem.sh --index DIR` builds two sidecars:
* PQ codes in layout order (`lowmem_pq32.pqc`, `lowmem_pq16.pqc`);
* a compressed KSG (sorted delta-varint lists, two-level offsets), either capped to each node's 5 nearest neighbours (`ksg_k5.ksgc`) or
  complete (`ksg_full.ksgc`).

`scripts/search.sh --lowmem pq32|pq16 [--ksgc file]` searches with them. All their memory is heap memory, so a cgroup limit applies to
it directly.

USIFT100M peak RSS:

| configuration | peak RSS |
|---|---|
| SQ5 (default) | 12.9 GB |
| full compressed KSG + PQ32 | 6.9 GB |
| k5 KSG + PQ32 | 5.6 GB |
| k5 KSG + PQ16 | 4.0 GB |

Recall at equal L drops with PQ codes (see the paper).

## Reproducing the paper

Machine: 2 × Intel Xeon Platinum 8457C (48 cores each), 503 GB RAM, Samsung PM9A3 7.68 TB NVMe (MZQL27T6HBLA), Linux 6.8, 48 search threads, O_DIRECT. Datasets:

| dataset | type | source |
|---|---|---|
| USIFT1M / 10M / 100M / 1B | uint8, d = 128 | BIGANN SIFT1B and its prefixes |
| SIFT1M | float, d = 128 | — |
| GIST1M | float, d = 960 | — |
| DEEP100M | float, d = 96 | first 100M of DEEP1B |
| SPACEV100M | int8 + 128, d = 100 | first 100M of SPACEV1B, zero-padded to 128 |

Exact top-100 ground truth for every dataset.

```bash
scripts/build_index.sh    --base <ds>/base.bin --dtype uint8 --out idx_<ds> --threads 48 --cleanup   # 1M-100M
scripts/build_index_1b.sh --base usift1b/base.bin --out idx_1b --threads 48 --cleanup                # 1B
scripts/sweep.sh --index idx_<ds> --query <ds>/query.bin --gt <ds>/groundtruth.bin --dtype uint8 --out sweep_<ds> \
                 --Ls "100 120 150 200 300 400 600 800 1200"
scripts/sweep.sh ... --eps-list "0.01 0.015 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09"                  # early-stop curve
```

Reference points measured with this code (warm run, standard Recall@100, 48 threads):

| dataset | L | Recall@100 | QPS | pages/query | RSS |
|---|---:|---:|---:|---:|---:|
| USIFT100M | 200 | 0.988 | 6040 | 161 | 12.9 GB |
| USIFT100M | 400 | 0.996 | 3260 | 301 | |
| DEEP100M (SQF 8-bit) | 200 | 0.987 | 6333 | 154 | 13.8 GB |
| DEEP100M (SQF 8-bit) | 400 | 0.995 | 3453 | 288 | |
| DEEP100M (SQF 8-bit) | eps 0.02 | 0.9916 | 5605 | 146 | |
| SPACEV100M | 200 | 0.981 | 7570 | 117 | 12.1 GB |
| SPACEV100M | 400 | 0.9905 | 4390 | 221 | |

Absolute QPS depends on the SSD's random-read IOPS. Pages per query and recall should match closely. DEEP100M and SPACEV100M were
built with the converged SSG described above; their row reports `eval_new100m.sh` sweeps (2 runs, warm), not the same-window
comparison with the baselines.

### Comparison with the baselines

Same-window results at Recall@100 ≥ 0.99 (48 threads, median of warm runs):

| dataset | KSG QPS | best baseline QPS | ratio |
|---|---:|---:|---:|
| DEEP100M | 5587 | Starling 3746 | 1.49× |
| SPACEV100M | 4347 | OctopusANN 2569 | 1.69× |

## Repository layout

```
src/ksg_search.cpp      search (all modes and ablations)        src/build_KSG.cpp   KSG builder
include/                SSD layout reader, io_uring wrapper, SQ/PQ/compressed-KSG readers, distance kernels
tools/                  SSG builders (in-memory and sharded), in-degree boost, layout/sidecar builders, k-NN graph, PQ, checks
scripts/                build_index.sh, build_index_1b.sh, build_lowmem.sh, search.sh, sweep.sh, recall.py, pad_u8.py
```

## License

MIT (see `LICENSE`).
