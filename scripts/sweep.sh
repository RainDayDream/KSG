#!/usr/bin/env bash
# Recall-QPS/latency curve: runs scripts/search.sh for every L (or eps) twice (first = cold, second = warm; the warm run is reported)
# and writes OUT/summary.tsv (standard Recall@K = |result[:K] ∩ GT[:K]| / K, QPS, mean/p50/p90/p99 latency, SSD pages per query, peak RSS).
# Usage: scripts/sweep.sh --index DIR --query q.bin --gt gt.bin --dtype uint8|float --out OUT [--Ls "100 150 200 300 400 600 800 1200"]
#                         [--eps-list "0.01 0.02 0.03 0.05 0.07 0.09"] [any other scripts/search.sh option]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
LS="100 120 150 200 300 400 600 800 1200"; EPSL=; OUT=; GT=; K=100; ARGS=()
while [ $# -gt 0 ]; do case $1 in
  --Ls) LS=$2; shift 2 ;; --eps-list) EPSL=$2; shift 2 ;; --out) OUT=$2; shift 2 ;;
  --gt) GT=$2; ARGS+=(--gt "$2"); shift 2 ;; --K) K=$2; ARGS+=(--K "$2"); shift 2 ;; *) ARGS+=("$1" "$2"); shift 2 ;; esac; done
[ -n "$OUT" ] && [ -n "$GT" ] || { sed -n 2,6p "$0"; exit 2; }
mkdir -p "$OUT"
run() { local d=$1; shift; for r in 0 1; do "$HERE/scripts/search.sh" "${ARGS[@]}" "$@" --out $d/r$r > /dev/null; done; echo "$(basename $d) done"; }
if [ -n "$EPSL" ]; then for e in $EPSL; do run $OUT/eps$e --L ${EPS_M:-1600} --eps $e; done
else for L in $LS; do run $OUT/L$L --L $L; done; fi
python3 "$HERE/scripts/recall.py" "$GT" --K $K --summary $OUT/*/r1 > $OUT/summary.tsv
cat $OUT/summary.tsv
