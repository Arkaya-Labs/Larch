"""gain(K) = E(K) t1 / (t(K+1) + d(K) + o(K)) per task. Args: costs.json ({"t":{r:ms}, "tA1":ms, "d":{K:ms}, "o":{K:ms}})
then task=a1,a2,a3,a4 ... Prints per task gain for K=1..4 with t1 = the drafting plan's t(1) and with A's t(1)."""
import json, sys
c = json.load(open(sys.argv[1]))
t, d, o, tA1 = {int(k): v for k, v in c['t'].items()}, {int(k): v for k, v in c['d'].items()}, {int(k): v for k, v in c['o'].items()}, c['tA1']
print(f"{'task':10} " + ' '.join(f'{"K=%d" % k:>13}' for k in range(1, 5)) + '   best (vs A)')
best = []
for arg in sys.argv[2:]:
    name, rs = arg.split('='); a = [float(x) for x in rs.split(',')]
    row, top = [], (0, 0, 0)
    for K in range(1, 5):
        E, run = 1.0, 1.0
        for i in range(K): run *= a[i]; E += run
        cost = t[K + 1] + d[K] + o[K]
        g, gA = E * t[1] / cost, E * tA1 / cost
        row.append(f'{g:.3f}/{gA:.3f}')
        if gA > top[1]: top = (K, gA, g)
    best.append(top[1])
    print(f'{name:10} ' + ' '.join(f'{x:>13}' for x in row) + f'   K={top[0]} {top[1]:.3f}')
best.sort(); m = best[len(best) // 2] if len(best) % 2 else (best[len(best) // 2 - 1] + best[len(best) // 2]) / 2
print(f'median best gain vs A {m:.3f}; predicted W4@1 = 172.5 x {m:.3f} = {172.5 * m:.1f} ({(m - 1) * 100:+.1f} %)')
