import json,statistics as st,sys
def load(p): return {j['id']:j for j in map(json.loads,open(p)) if 'id' in j}
a=load(sys.argv[1]); b=load(sys.argv[2])
k=[i for i in a if i in b and a[i]['image_rows']!=b[i]['image_rows']]
print('items with changed image tokens:',len(k))
if not k: sys.exit()
print('image tokens median %d -> %d' % (st.median(a[i]['image_rows'] for i in k), st.median(b[i]['image_rows'] for i in k)))
for f in ('ttft_ms','wall_ms'):
  print('%s on those: median %.0f -> %.0f, total %.1f -> %.1f s' % (f, st.median(a[i][f] for i in k), st.median(b[i][f] for i in k), sum(a[i][f] for i in k)/1000, sum(b[i][f] for i in k)/1000))
print('whole run wall %.1f -> %.1f s' % (sum(x['wall_ms'] for x in a.values())/1000, sum(x['wall_ms'] for x in b.values())/1000))
