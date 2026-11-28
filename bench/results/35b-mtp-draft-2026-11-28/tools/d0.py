"""D0 summaries from a run's stderr: larch_d0_round (id,tried,kept,pos,ids), larch_d0_propose (n,wall,block,head,argmax,
tail), larch_d0_observe (k,ns). Prints a_i for i=1..K, the kept per round, and the means."""
import sys, collections
rounds, prop, obs = [], collections.defaultdict(list), collections.defaultdict(list)
for line in open(sys.argv[1], errors='replace'):
    f = line.strip().split(',')
    if f[0] == 'larch_d0_round':
        rounds.append((int(f[2]), int(f[3]), int(f[4]), [int(x) for x in f[5].split()] if len(f) > 5 and f[5] else []))
    elif f[0] == 'larch_d0_propose':
        prop[int(f[1])].append([int(x) for x in f[2:]])
    elif f[0] == 'larch_d0_observe':
        obs[int(f[1])].append(int(f[2]))
def rates(rs):
    K = max((r[0] for r in rs), default=0)
    out = []
    for i in range(1, K + 1):
        reach = sum(1 for r in rs if r[0] >= i and r[1] >= i - 1)
        kept = sum(1 for r in rs if r[0] >= i and r[1] >= i)
        out.append(kept / reach if reach else float('nan'))
    return out
first = [r for r in rounds if r[2] < 16]; rest = [r for r in rounds if r[2] >= 16]
print('rounds', len(rounds), 'a_i', ' '.join(f'{a:.3f}' for a in rates(rounds)),
      'kept/round %.3f' % (sum(r[1] for r in rounds) / max(1, len(rounds))))
print('first16 a_i', ' '.join(f'{a:.3f}' for a in rates(first)), '| rest a_i', ' '.join(f'{a:.3f}' for a in rates(rest)))
for n, v in sorted(prop.items()):
    m = [sum(c) / len(v) / 1e3 for c in zip(*v)]
    print(f'propose n={n} count={len(v)} us: wall {m[0]:.1f} block {m[1]:.1f} head {m[2]:.1f} argmax {m[3]:.1f} download {m[4]:.1f}')
for k, v in sorted(obs.items()):
    print(f'observe k={k} count={len(v)} us {sum(v) / len(v) / 1e3:.1f}')
