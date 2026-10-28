#!/usr/bin/env bash
# Low-memory sidecars for an existing uint8 index (see README "Memory budget"): PQ codes in layout order and a compressed KSG
# capped to its k nearest neighbours per node. Search with scripts/search.sh --lowmem pq32|pq16.
# Usage: scripts/build_lowmem.sh --index DIR [--k 5] [--threads N]
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); IDX=; KD=5; T=$(nproc)
while [ $# -gt 0 ]; do case $1 in --index) IDX=$2; shift 2 ;; --k) KD=$2; shift 2 ;; --threads) T=$2; shift 2 ;; *) echo "unknown $1"; exit 2 ;; esac; done
[ -n "$IDX" ] || { sed -n 2,5p "$0"; exit 2; }
BASE=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^base=/) print substr($i,6)}' $IDX/build.params); export OMP_NUM_THREADS=$T
for M in 32 16; do [ -f $IDX/lowmem_pq$M.pqc ] || python3 $HERE/tools/build_pqc.py "$BASE" uint8 $IDX/lowmem_pq$M.pqc $M $IDX/layout.new2old; done
[ -f $IDX/ksg_k$KD.ksgc ] || $HERE/bin/build_ksg_compact $IDX/ksg.graph $IDX/ksg_k$KD.ksgc $KD "$BASE" $IDX/layout.new2old
[ -f $IDX/ksg_full.ksgc ] || $HERE/bin/build_ksg_compact $IDX/ksg.graph $IDX/ksg_full.ksgc
ls -la $IDX/*.pqc $IDX/*.ksgc
