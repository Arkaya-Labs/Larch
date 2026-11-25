#!/usr/bin/env python3
"""Row-wise cosine and max abs of each config's encoder dump vs base (raw f32 [rows, d])."""
import sys, glob, os, array, math
O = sys.argv[1]
def load(p):
    a = array.array('f'); a.frombytes(open(p, 'rb').read()); return a
for f in sorted(glob.glob(os.path.join(O, '*.f32'))):
    name = os.path.basename(f)
    if name.startswith('base-'): continue
    cfg, rest = name.split('-', 1)
    b = os.path.join(O, 'base-' + rest)
    if not os.path.exists(b): continue
    x, y = load(f), load(b)
    d = 2048 if '-m15-' in name else 1024
    rows = len(y) // d
    worst, mabs = 1.0, 0.0
    for r in range(rows):
        xs, ys = x[r*d:(r+1)*d], y[r*d:(r+1)*d]
        dot = sum(p*q for p, q in zip(xs, ys)); nx = math.sqrt(sum(p*p for p in xs)); ny = math.sqrt(sum(q*q for q in ys))
        worst = min(worst, dot / (nx*ny) if nx and ny else 1.0)
        mabs = max(mabs, max(abs(p-q) for p, q in zip(xs, ys)))
    print(f"{name[:-4]:40s} rows {rows:5d}  worst cos vs f32 {worst:.6f}  max abs {mabs:.4f}")
