#!/usr/bin/env bash
# Build a KSG index (in-memory KSG + on-SSD SSG layout) for one dataset on one machine, current best configuration.
# Every step writes <step>.log and is skipped when <step>.done exists (re-run to resume).
#
# Usage: scripts/build_index.sh --base base.bin --dtype uint8|float --out DIR [options]
#   --threads N        OpenMP threads (default: all cores)
#   --knn-cpu          compute the approximate k-NN graph with faiss on the CPU (default: GPU 0)
#   --R 96 --Lc 128 --C 512 --rounds 2     SSG construction (degree, search list, candidate pool, refinement rounds from the random start)
#   --extra-rounds 1   further refinement rounds with up to 96 reverse candidates per node (converges the graph; 0 = off)
#   --boost-batch B    in-degree boost: gather the source lists of B nodes in parallel (default 0 = sequential, as in the paper)
#   --sq-bits 8|4      float data only: bits per dimension of the in-memory scalar quantiser (default 8)
#   --entry-c C        k-means centroids of the entry index (default 1024 for n <= 2M, else 16384)
#   --cleanup          delete the random initial graph and the raw SSG after the layout is verified (they are not needed to search)
# uint8 data must have d = 128 (pad other dimensions with zeros: scripts/pad_u8.py); float data may have any dimension.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); BIN=$HERE/bin; PY=${PYTHON:-python3}
BASE=; DT=; OUT=; T=$(nproc); KNN_CPU=; R=96; LC=128; CC=512; ROUNDS=2; XR=1; BB=0; SQB=8; EC=; CLEAN=
while [ $# -gt 0 ]; do case $1 in
  --base) BASE=$2; shift 2 ;; --dtype) DT=$2; shift 2 ;; --out) OUT=$2; shift 2 ;; --threads) T=$2; shift 2 ;;
  --knn-cpu) KNN_CPU=1; shift ;; --R) R=$2; shift 2 ;; --Lc) LC=$2; shift 2 ;; --C) CC=$2; shift 2 ;; --rounds) ROUNDS=$2; shift 2 ;;
  --extra-rounds) XR=$2; shift 2 ;; --boost-batch) BB=$2; shift 2 ;;
  --sq-bits) SQB=$2; shift 2 ;; --entry-c) EC=$2; shift 2 ;; --cleanup) CLEAN=1; shift ;;
  *) echo "unknown option $1"; exit 2 ;; esac; done
[ -n "$BASE" ] && [ -n "$OUT" ] && { [ "$DT" = uint8 ] || [ "$DT" = float ]; } || { sed -n 2,15p "$0"; exit 2; }
BASE=$(readlink -f "$BASE"); mkdir -p "$OUT"; cd "$OUT"
read N D < <($PY -c "import numpy as np,sys; h=np.fromfile(sys.argv[1],dtype=np.uint32,count=2); print(h[0],h[1])" "$BASE")
if [ $DT = uint8 ] && [ "$D" != 128 ]; then echo "uint8 data must have d=128 (got $D): pad with scripts/pad_u8.py"; exit 2; fi
[ -z "$EC" ] && { [ "$N" -le 2000000 ] && EC=1024 || EC=16384; }
VB=$([ $DT = uint8 ] && echo $D || echo $((2 * D)))            # vector bytes in an SSD record (float -> fp16)
PER=$((4096 / (VB + 8 + 4 * R)))                                  # records per 4 KB page
[ $PER -ge 1 ] || { echo "record of $((VB + 8 + 4 * R)) B does not fit a 4 KB page: lower --R"; exit 2; }
export OMP_NUM_THREADS=$T
echo "base=$BASE n=$N d=$D dtype=$DT threads=$T R=$R Lc=$LC C=$CC rounds=$ROUNDS extra_rounds=$XR per_page=$PER entry_c=$EC" | tee build.params
step() { local n=$1; shift; [ -f $n.done ] && return 0; echo "$(date '+%F %T') $n"; local s=$(date +%s)
  if /usr/bin/time -v "$@" > $n.log 2> $n.time; then touch $n.done; echo "  done in $(( $(date +%s) - s )) s"
  else echo "  $n FAILED"; tail -n 5 $n.log $n.time; exit 1; fi; }

# 1. approximate 5-NN graph (faiss IVFPQ)  ->  2. KSG: keep 5 neighbours, add reverse edges, repair connectivity
step knn env ${KNN_CPU:+KNN_CPU=1} $PY $HERE/tools/gpu_knn.py "$BASE" $DT knn5.graph 5 8192 16 32 2000000
step ksg $BIN/build_KSG --ksg_init_file knn5.graph --p_degree 5 --ksg_file ksg_raw.graph --log_file ksg_raw.log
# 3. SSG: random R=32 start graph, then search-based refinement rounds with robust prune (alpha 1.2) to degree R
[ -f init.done ] || { $PY -c "import numpy as np,sys
n=int(sys.argv[1]); R=32; rng=np.random.default_rng(7); f=open(sys.argv[2],'wb'); f.write(np.array([n,R],dtype=np.uint32).tobytes())
for s in range(0,n,1<<22):
    m=min(1<<22,n-s); a=np.empty((m,R+1),dtype=np.uint32); a[:,0]=R; a[:,1:]=rng.integers(0,n,size=(m,R),dtype=np.uint32); f.write(a.tobytes())" $N init.ssg && touch init.done; }
step ssg $BIN/build_ssg_search --base "$BASE" --init init.ssg --out ssg_raw.ssg --R $R --Lc $LC --C $CC --rounds $ROUNDS --entries 8 --backlink 1 \
  --save_rounds 0 $([ $DT = float ] && echo --float 1)
# 3b. extra refinement round(s) from the refined graph, back-links with up to 96 reverse candidates: lets the graph converge
#     (DEEP100M / SPACEV100M: Recall@100 at L=400 0.984 -> 0.991 with 315 -> 221 pages per query)
SSG_IN=ssg_raw.ssg
if [ "$XR" -gt 0 ]; then
  step ssg_extra $BIN/build_ssg_search --base "$BASE" --init ssg_raw.ssg --out ssg_conv.ssg --R $R --Lc $LC --C $CC --rounds $XR --entries 8 \
    --backlink 1 --rev 96 --save_rounds 0 $([ $DT = float ] && echo --float 1)
  SSG_IN=ssg_conv.ssg
fi
# 4. in-degree boost: nodes with in-degree < 80 take a tail slot (never the 16 nearest) of their near out-neighbours and 2-hop
#    neighbours whose target has in-degree > 88; out-degree stays R; up to 3 passes
step boost $BIN/boost_indegree --base "$BASE" --in $SSG_IN --out ssg.ssg --T 80 --donor 88 --src 48 --protect 16 --hop2 8 --passes 3 \
  --batch $BB $([ $DT = float ] && echo --float 1)
# 5. page layout: connectivity-greedy packing (PER records per 4 KB page) -> renumbered co-located file (vector + neighbours + original id)
step pack  $BIN/pack_sectors ssg.ssg $PER layout $PER 2
step colo  $BIN/build_colocated "$BASE" ssg.ssg ssg.colo layout
# 6. in-memory sidecars in layout order: affine scalar quantiser, entry index, renumbered KSG
if [ $DT = uint8 ]; then step codes $BIN/build_sq4 "$BASE" codes.sq5 1 layout.new2old 999
else step codes $BIN/build_sqf "$BASE" codes.sqf $SQB layout.new2old 999; fi
step entry $BIN/build_entry "$BASE" entry_raw.bin $EC 2000000 4 8
step permute $BIN/permute_ksg ksg_raw.graph layout ksg.graph entry_raw.bin entry.bin
# 7. verify 2000 random records of the layout against the base vectors and the SSG
step verify $PY $HERE/tools/verify_layout_any.py "$BASE" ssg.ssg ssg.colo layout 2000
if [ -n "$CLEAN" ]; then rm -f init.ssg ssg_raw.ssg ssg_conv.ssg ssg.ssg; echo "removed init.ssg / ssg_raw.ssg / ssg_conv.ssg / ssg.ssg (not needed for search)"; fi
echo "INDEX READY: $OUT  (search: scripts/search.sh --index $OUT ...)"; du -sh "$OUT"
