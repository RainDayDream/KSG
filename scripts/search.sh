#!/usr/bin/env bash
# One KSG search run (current best configuration): K results per query, single recall knob L, 32 SSD reads in flight (io_uring, O_DIRECT).
# Usage: scripts/search.sh --index DIR --query query.bin --gt groundtruth.bin --dtype uint8|float --L L --out RUN_DIR
#                          [--base base.bin] [--K 100] [--threads 48] [--eps e] [--lowmem pq32|pq16 [--ksgc file.ksgc]] [--extra "flags"]
#   default (single knob): -L1 L -L2 L, in-memory KSG range m = 8L, unread candidates fill the result (--colo-sq-results)
#   --eps e : early-stop variant (L1 100, L2 3200, m = L, stop when the best unread candidate is (1+e)^2 farther than the K-th result)
#   --lowmem: PQ codes (from scripts/build_lowmem.sh) instead of the scalar quantiser, compressed KSG, no id map (see README)
# Writes RUN_DIR/{command.txt,search.log,result.bin,latency.f32,time.txt}; recall: scripts/recall.py.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
IDX=; Q=; GT=; DT=; L=; OUT=; BASE=; K=100; T=48; EPS=; LOW=; KSGC=; EXTRA=
while [ $# -gt 0 ]; do case $1 in
  --index) IDX=$2; shift 2 ;; --query) Q=$2; shift 2 ;; --gt) GT=$2; shift 2 ;; --dtype) DT=$2; shift 2 ;; --L) L=$2; shift 2 ;;
  --out) OUT=$2; shift 2 ;; --base) BASE=$2; shift 2 ;; --K) K=$2; shift 2 ;; --threads) T=$2; shift 2 ;; --eps) EPS=$2; shift 2 ;;
  --lowmem) LOW=$2; shift 2 ;; --ksgc) KSGC=$2; shift 2 ;; --extra) EXTRA=$2; shift 2 ;; *) echo "unknown option $1"; exit 2 ;; esac; done
[ -n "$IDX" ] && [ -n "$Q" ] && [ -n "$GT" ] && [ -n "$DT" ] && [ -n "$L" ] && [ -n "$OUT" ] || { sed -n 2,9p "$0"; exit 2; }
BASE=${BASE:-$(awk '{for(i=1;i<=NF;i++) if($i ~ /^base=/) print substr($i,6)}' $IDX/build.params)}
mkdir -p "$OUT"
if [ -n "$LOW" ]; then CODES="--pqc-file $IDX/lowmem_$LOW.pqc --mem-graph-compact ${KSGC:-$IDX/ksg_k5.ksgc}"; SQRES=
elif [ $DT = uint8 ]; then CODES="--sq4-file $IDX/codes.sq5 --sq4-resident"; SQRES="--colo-sq-results $IDX/layout"
else CODES="--sqf-file $IDX/codes.sqf --sq4-resident"; SQRES="--colo-sq-results $IDX/layout"; fi
if [ -n "$EPS" ]; then KNOB="-L1 100 -L2 3200 --colo-mem-L $L --ssg-stop-eps $EPS"; SQRES=
else KNOB="-L1 $L -L2 $L --colo-mem-L $((8 * L))"; fi
cmd=($HERE/bin/ksg_search --data_type $DT --vec "$BASE" --query "$Q" --query_groundtruth_file "$GT" -K $K $KNOB -B 32
  --mem_graph $IDX/ksg.graph --entry-file $IDX/entry.bin --colo-file $IDX/ssg.colo --colo-io uring --colo-pipe --colo-free 6 --colo-cache 1024
  --sq-only $CODES $SQRES --indisk --heap-search --prefetch 4 --mem-neighbor-limit 0 --disk-neighbor-limit 96 --exact-read-order id
  --dense-refine-push refined --dense-refine-neighbors 32 --rerank-k 0 --io-stats
  --result_file $OUT/result.bin --log_file $OUT/search.log --latency-dump $OUT/latency.f32 $EXTRA)
echo "${cmd[*]}" > $OUT/command.txt
# drop the SSD layout from the page cache so every run reads from the device (the search itself uses O_DIRECT)
python3 -c "import os,sys; fd=os.open(sys.argv[1],os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" $IDX/ssg.colo
OMP_NUM_THREADS=$T OMP_PROC_BIND=spread OMP_PLACES=cores /usr/bin/time -v "${cmd[@]}" > $OUT/stdout.txt 2> $OUT/time.txt
grep -oE "QPS : [0-9.]+|query latency mean:[0-9.]+us" $OUT/search.log | tr '\n' ' '; echo
