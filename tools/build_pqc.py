#!/usr/bin/env python3
# PQ sidecar for the KSG search (include/pqc.h): faiss ProductQuantizer (M sub-quantisers x 256), trained on a uniform sample,
# codes written in layout order (new2old). bias = mean squared quantisation error on a held-out sample.
# Usage: build_pqc.py base.bin uint8|float out.pqc M new2old|- [train=1000000]
import sys, time
import numpy as np
import faiss
base, dt, out, M, n2of = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
ntrain = int(sys.argv[6]) if len(sys.argv) > 6 else 1_000_000
h = np.fromfile(base, dtype=np.uint32, count=2); n, d = int(h[0]), int(h[1])
X = np.memmap(base, dtype=np.uint8 if dt == 'uint8' else np.float32, mode='r', offset=8, shape=(n, d))
f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
t0 = time.time(); rng = np.random.default_rng(3)
pq = faiss.ProductQuantizer(d, M, 8); pq.train(f32(X[np.sort(rng.choice(n, ntrain, replace=False))]))
hs = f32(X[np.sort(rng.choice(n, 200_000, replace=False))]); rec = pq.decode(pq.compute_codes(hs))
bias = float(((hs - rec) ** 2).sum(1).mean()); print(f'n={n} d={d} M={M} trained in {time.time() - t0:.0f}s, mean sq. error {bias:.1f}', flush=True)
n2o = np.fromfile(n2of, dtype=np.uint32) if n2of != '-' else None
with open(out, 'wb') as f:
    f.write(np.array([n, d, M, 0x31435150], dtype=np.uint32).tobytes()); f.write(np.array([bias], dtype=np.float32).tobytes())
    f.write(faiss.vector_to_array(pq.centroids).astype(np.float32).tobytes())
    CH = 4_000_000
    for s in range(0, n, CH):
        ids = n2o[s:s + CH] if n2o is not None else np.arange(s, min(n, s + CH))
        o = np.argsort(ids); x = f32(X[ids[o]])                  # sorted reads, then back to layout order
        c = np.empty((len(ids), M), dtype=np.uint8); c[o] = pq.compute_codes(x); f.write(c.tobytes())
print(f'done in {time.time() - t0:.0f}s -> {out}', flush=True)
