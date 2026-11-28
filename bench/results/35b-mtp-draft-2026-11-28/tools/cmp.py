"""Compare two tokdump outputs prompt by prompt: first index where the ids differ, and each side's top-2 logprob margin
there (top-1 minus top-2: a logit difference). Prints a summary; exit 0 always."""
import json, sys
a = {d['id']: d for d in map(json.loads, open(sys.argv[1]))}
b = {d['id']: d for d in map(json.loads, open(sys.argv[2]))}
div, margins, same = [], [], 0
def margin(d, i):
    t = d['top'][i]
    return t[1] - t[3] if len(t) >= 4 else float('nan')
for k in a:
    x, y = a[k]['ids'], b[k]['ids']
    i = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), None)
    if i is None and len(x) != len(y): i = min(len(x), len(y))
    if i is None: same += 1; continue
    ma = margin(a[k], i) if i < len(x) else float('nan'); mb = margin(b[k], i) if i < len(y) else float('nan')
    div.append((k, i, ma, mb)); margins.append(max(ma, mb))
print(f'prompts {len(a)} identical {same} diverge {len(div)} (tokens {sum(len(d["ids"]) for d in a.values())} / {sum(len(d["ids"]) for d in b.values())})')
for k, i, ma, mb in div: print(f'  {k} first diff at {i}: margin {ma:.4f} / {mb:.4f}')
if margins:
    m = sorted(margins); print(f'margin (larger side) min {m[0]:.4f} median {m[len(m)//2]:.4f} max {m[-1]:.4f}; above 0.5: {sum(x > 0.5 for x in m)}')
