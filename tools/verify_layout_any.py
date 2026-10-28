#!/usr/bin/env python3
# Spot-check a permuted co-located layout of any dtype/dim (uint8 records or fp16 records of a float base): for random layout ids i,
# the record must hold base[new2old[i]] (fp16: equal to the base rounded to fp16), the neighbours of new2old[i] in graph.ssg mapped
# through new2old, and the original id new2old[i]; old2new must invert new2old.
# Usage: verify_layout_any.py base.bin graph.ssg layout.colo perm_prefix [samples]
import sys
import numpy as np
base_p, ssg_p, colo_p, perm = sys.argv[1:5]; S = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
colo = np.memmap(colo_p, dtype=np.uint8, mode='r'); ssg = np.memmap(ssg_p, dtype=np.uint32, mode='r')
n, R, rec_len, per, dim, vb = (int(x) for x in colo[:24].view(np.uint32))
vb = vb or 1; bn, bd = (int(x) for x in np.fromfile(base_p, dtype=np.uint32, count=2))
f32 = (np.dtype(np.float32).itemsize * bd * bn + 8) == __import__('os').path.getsize(base_p)
B = np.memmap(base_p, dtype=np.float32 if f32 else np.uint8, mode='r', offset=8, shape=(bn, bd))
n2o = np.fromfile(perm + '.new2old', dtype=np.uint32); o2n = np.fromfile(perm + '.old2new', dtype=np.uint32)
gn, gR = int(ssg[0]), int(ssg[1])
assert n == bn == gn == len(n2o) and bd == dim and R == gR, (n, bn, gn, len(n2o), bd, dim, R, gR)
rng = np.random.default_rng(1); bad = 0
for i in list(rng.choice(n, S, replace=False)) + [0, n - 1]:
    i = int(i); src = int(n2o[i]); ok = int(o2n[src]) == i
    off = 4096 * (1 + i // per) + (i % per) * rec_len; vbytes = dim * vb
    rec = colo[off:off + rec_len]
    v = rec[:vbytes].view(np.float16 if vb == 2 else np.uint8)
    ok &= bool(np.array_equal(v, B[src].astype(np.float16) if vb == 2 else B[src]))
    cnt = int(rec[vbytes:vbytes + 4].view(np.uint32)[0]); ids = rec[vbytes + 4:vbytes + 4 + 4 * R].view(np.uint32)[:cnt]
    go = (src * (R + 1)) + 2; gc = int(ssg[go]); gids = ssg[go + 1:go + 1 + gc]
    ok &= cnt == gc and bool(np.array_equal(n2o[ids], gids))
    ok &= int(rec[vbytes + 4 + 4 * R:vbytes + 8 + 4 * R].view(np.uint32)[0]) == src
    bad += not ok
print(f'verify_layout_any: {S + 2} samples, {bad} mismatches, n={n} R={R} dim={dim} vbytes={vb} rec_len={rec_len} per_sector={per}')
sys.exit(1 if bad else 0)
