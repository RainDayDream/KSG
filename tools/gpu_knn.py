#!/usr/bin/env python3
# Approximate k-NN graph on one GPU for the KSG input (same method as tools/build_gpu_pq_index.py: faiss IVFPQ, self removed, first k kept),
# but chunked so 100M-scale bases fit on a 48 GB GPU (that script searches the whole base in one call when RAM is large).
# Output: [u32 n][u32 k] then n*k u32 ids (build_KSG --ksg_init_file format).
# Usage: gpu_knn.py base.bin uint8|float out.graph [k=5] [nlist=8192] [m=16] [nprobe=32] [train=2000000]
# KNN_CPU=1 runs the same IVFPQ on the CPU (no GPU needed; slower).
import os, sys, time
import numpy as np
import faiss
base, dt, out = sys.argv[1], sys.argv[2], sys.argv[3]
k, nlist, m, nprobe, ntrain = [int(a) for a in (sys.argv[4:] + ['5', '8192', '16', '32', '2000000'][len(sys.argv[4:]):])]
h = np.fromfile(base, dtype=np.uint32, count=2); n, d = int(h[0]), int(h[1])
X = np.memmap(base, dtype=np.uint8 if dt == 'uint8' else np.float32, mode='r', offset=8, shape=(n, d))
f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
t0 = time.time()
cpu_idx = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, nlist, m, 8)
if os.environ.get('KNN_CPU'): idx = cpu_idx
else: res = faiss.StandardGpuResources(); idx = faiss.index_cpu_to_gpu(res, 0, cpu_idx)
rng = np.random.default_rng(1); tid = np.sort(rng.choice(n, min(ntrain, n), replace=False))
idx.train(f32(X[tid])); print(f'n={n} d={d} trained in {time.time() - t0:.0f}s', flush=True)
CH = 2_000_000
for s in range(0, n, CH): idx.add(f32(X[s:s + CH]))
print(f'added in {time.time() - t0:.0f}s', flush=True)
idx.nprobe = nprobe
with open(out, 'wb') as fo:
    fo.write(np.array([n, k], dtype=np.uint32).tobytes())
    QB = 500_000
    for s in range(0, n, QB):
        _, I = idx.search(f32(X[s:s + QB]), k + 1)
        ids = np.arange(s, s + len(I))[:, None]
        hs = (I == ids).any(1)                           # self appears at most once per row
        G = np.empty((len(I), k), dtype=np.int64)
        G[hs] = I[hs][I[hs] != ids[hs]].reshape(-1, k); G[~hs] = I[~hs, :k]
        bad = G < 0                                      # unfilled slots (tiny lists): repeat the first neighbour
        if bad.any(): G[bad] = np.broadcast_to(G[:, :1], G.shape)[bad]
        G = G.astype(np.uint32)
        fo.write(G.tobytes())
        if s % 10_000_000 == 0: print(f'searched {s + len(I)} in {time.time() - t0:.0f}s', flush=True)
print(f'done in {time.time() - t0:.0f}s', flush=True)
