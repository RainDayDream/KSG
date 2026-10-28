#!/usr/bin/env python3
# Zero-pad a uint8 .bin file ([u32 n][u32 d] + n*d bytes) to d = 128 (KSG's uint8 path is specialised for 128 dimensions).
# L2 distances, and therefore the ground truth, are unchanged. Apply to both base.bin and query.bin.
# Usage: pad_u8.py in.bin out.bin
import sys
import numpy as np
h = np.fromfile(sys.argv[1], dtype=np.uint32, count=2); n, d = int(h[0]), int(h[1]); assert d <= 128, d
x = np.memmap(sys.argv[1], dtype=np.uint8, mode='r', offset=8, shape=(n, d))
y = np.memmap(sys.argv[2], dtype=np.uint8, mode='w+', offset=8, shape=(n, 128))
for s in range(0, n, 1 << 22): y[s:s + (1 << 22), :d] = x[s:s + (1 << 22)]; y[s:s + (1 << 22), d:] = 0
y.flush(); del y
with open(sys.argv[2], 'r+b') as f: f.write(np.array([n, 128], dtype=np.uint32).tobytes())
print(f'{sys.argv[2]}: {n} x 128 (padded from {d})')
