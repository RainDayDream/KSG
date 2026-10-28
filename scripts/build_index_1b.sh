#!/usr/bin/env bash
# Billion-scale KSG build (uint8, d = 128): the SSG is built per shard so that each shard fits in memory.
#   KSG as in build_index.sh -> shard_split (P k-means shards + KSG one-hop halo) -> per shard: search-based SSG build, only the lists
#   of the shard's primary nodes are written into the global SSG -> global in-degree -> per-shard in-degree boost -> layout as in
#   build_index.sh. Memory per shard ~ members x (1172 + 4 x threads) bytes (~100M members -> ~130 GB at 48 threads).
#
# Usage: scripts/build_index_1b.sh --base base.bin --out DIR [--threads N] [--shards 20] [--knn-cpu] [--cleanup]
# The paper's USIFT1B index used the previous-generation SSG as each shard's initial graph with one refinement round (--Lc 64);
# this script starts every shard from a random graph and refines it for two rounds (as validated up to 100M).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); BIN=$HERE/bin; PY=${PYTHON:-python3}
BASE=; OUT=; T=$(nproc); P=20; KNN_CPU=; CLEAN=
while [ $# -gt 0 ]; do case $1 in
  --base) BASE=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --threads) T=$2; shift 2 ;; --shards) P=$2; shift 2 ;;
  --knn-cpu) KNN_CPU=1; shift ;; --cleanup) CLEAN=1; shift ;; *) echo "unknown option $1"; exit 2 ;; esac; done
[ -n "$BASE" ] && [ -n "$OUT" ] || { sed -n 2,9p "$0"; exit 2; }
BASE=$(readlink -f "$BASE"); mkdir -p "$OUT/shards"; cd "$OUT"; export OMP_NUM_THREADS=$T
read N D < <($PY -c "import numpy as np,sys; h=np.fromfile(sys.argv[1],dtype=np.uint32,count=2); print(h[0],h[1])" "$BASE")
[ "$D" = 128 ] || { echo "uint8 d=128 only"; exit 2; }
echo "base=$BASE n=$N shards=$P threads=$T" | tee build.params
step() { local n=$1; shift; [ -f $n.done ] && return 0; echo "$(date '+%F %T') $n"; local s=$(date +%s)
  if /usr/bin/time -v "$@" > $n.log 2> $n.time; then touch $n.done; echo "  done in $(( $(date +%s) - s )) s"
  else echo "  $n FAILED"; tail -n 5 $n.log $n.time; exit 1; fi; }

step knn env ${KNN_CPU:+KNN_CPU=1} $PY $HERE/tools/gpu_knn.py "$BASE" uint8 knn5.graph 5 65536 16 32 4000000
step ksg $BIN/build_KSG --ksg_init_file knn5.graph --p_degree 5 --ksg_file ksg_raw.graph --log_file ksg_raw.log
step split $BIN/shard_split "$BASE" ksg_raw.graph $P shards
for k in $(seq 0 $((P - 1))); do
  step shard$k $BIN/build_ssg_shard --base "$BASE" --old random --ids shards/shard$k.ids --prim shards/shard$k.prim --out ssg.ssg \
    --R 96 --Lc 64 --C 512 --rounds 2
done
step indeg $BIN/indegree_count ssg.ssg ssg.indeg
for k in $(seq 0 $((P - 1))); do
  step boost$k $BIN/boost_shard --base "$BASE" --graph ssg.ssg --indeg ssg.indeg --ids shards/shard$k.ids --prim shards/shard$k.prim
done
step pack    $BIN/pack_sectors ssg.ssg 7 layout 7 2
step colo    $BIN/build_colocated "$BASE" ssg.ssg ssg.colo layout
step codes   $BIN/build_sq4 "$BASE" codes.sq5 1 layout.new2old 999
step entry   $BIN/build_entry "$BASE" entry_raw.bin 16384 2000000 4 8
step permute $BIN/permute_ksg ksg_raw.graph layout ksg.graph entry_raw.bin entry.bin
step verify  $PY $HERE/tools/verify_layout_any.py "$BASE" ssg.ssg ssg.colo layout 2000
if [ -n "$CLEAN" ]; then rm -f ssg.ssg ssg.indeg; echo "removed ssg.ssg (not needed for search)"; fi
echo "INDEX READY: $OUT"; du -sh "$OUT"
