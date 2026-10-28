#!/usr/bin/env python3
# Standard Recall@K of ksg_search result files: |result[:K] ∩ GT[:K]| / K per query (missing results count as misses), averaged.
# result.bin: per query [int32 count][count ids]; groundtruth.bin: [u32 nq][u32 k] then nq*k u32 ids.
# Usage: recall.py groundtruth.bin [--K 100] [--new2old file] result.bin|RUN_DIR ...
#        recall.py groundtruth.bin --summary RUN_DIR ...   (TSV with QPS, latency percentiles, pages/query, RSS from each run dir)
import os, re, sys
import numpy as np
args = sys.argv[1:]; gtf = args.pop(0); K = 100; n2o = None; summary = False; files = []
while args:
    a = args.pop(0)
    if a == '--K': K = int(args.pop(0))
    elif a == '--new2old': n2o = np.fromfile(args.pop(0), dtype=np.uint32)
    elif a == '--summary': summary = True
    else: files.append(a)
g = np.fromfile(gtf, dtype=np.uint32); nq, gk = int(g[0]), int(g[1]); gt = g[2:2 + nq * gk].reshape(nq, gk)[:, :K]
def recall(f):
    raw = np.fromfile(f, dtype=np.int32); i = q = 0; s = 0.0
    while i < len(raw) and q < nq:
        c = int(raw[i]); ids = raw[i + 1:i + 1 + c].astype(np.int64)[:K]; i += 1 + c
        if n2o is not None: ids = n2o[ids[(ids >= 0) & (ids < len(n2o))]]
        s += len(np.intersect1d(ids, gt[q])) / K; q += 1
    return s / nq
if summary:
    print('run\trecall\tqps\tlat_mean_ms\tlat_p50_ms\tlat_p90_ms\tlat_p99_ms\tpages_per_query\tmax_rss_gib')
    for d in files:
        log = open(os.path.join(d, 'search.log')).read()
        qps = float(re.findall(r'QPS : ([0-9.]+)', log)[-1])
        lat = np.fromfile(os.path.join(d, 'latency.f32'), dtype=np.float32) / 1000.0
        pg = re.search(r'sectors_read=([0-9.]+)', log); rss = re.search(r'Maximum resident set size \(kbytes\): (\d+)', open(os.path.join(d, 'time.txt')).read())
        name = os.path.basename(os.path.dirname(os.path.abspath(d)))
        print(f'{name}\t{recall(os.path.join(d, "result.bin")):.6f}\t{qps:.1f}\t{lat.mean():.3f}\t{np.percentile(lat, 50):.3f}\t{np.percentile(lat, 90):.3f}'
              f'\t{np.percentile(lat, 99):.3f}\t{float(pg.group(1)) if pg else 0:.1f}\t{int(rss.group(1)) / 1048576 if rss else 0:.2f}')
else:
    for f in files:
        f = os.path.join(f, 'result.bin') if os.path.isdir(f) else f
        print(f'{f}\tRecall@{K}={recall(f):.6f}')
