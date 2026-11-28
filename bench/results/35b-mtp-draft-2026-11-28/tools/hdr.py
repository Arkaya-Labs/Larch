import struct, sys
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))
def s(): (n,) = rd('Q'); return f.read(n).decode('utf-8', 'replace')
SZ = {0:4,1:4,2:1,3:1,4:2,5:2,6:4,7:1,10:8,11:8,12:8}
def val(t):
    if t == 8: return s()
    if t == 9:
        (et, n) = rd('IQ')
        return [val(et) for _ in range(n)] if n < 64 else ('array', et, n, [val(et) for _ in range(n)][:0])
    fm = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t]
    return rd(fm)[0]
magic, ver, nt, nkv = rd('4sIQQ')
kv = {}
for _ in range(nkv):
    k = s(); (t,) = rd('I'); kv[k] = val(t)
# block size, type size per ggml type
BT = {0:(1,4),1:(1,2),8:(32,34),2:(32,18),12:(256,144),13:(256,176),14:(256,210),11:(256,110),10:(256,84),30:(1,2),6:(32,22),15:(256,292)}
NAMES = {0:'F32',1:'F16',2:'Q4_0',8:'Q8_0',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',30:'BF16',6:'Q5_0'}
tot = 0
for _ in range(nt):
    name = s(); (nd,) = rd('I'); dims = rd('Q' * nd); (ty, off) = rd('IQ')
    n = 1
    for d in dims: n *= d
    b, ts = BT.get(ty, (1, 0)); by = n // b * ts
    if name.startswith('blk.40.') or name in ('output.weight', 'token_embd.weight'):
        print(f'{name:40s} {NAMES.get(ty, ty):5s} {list(dims)} {by}')
        if name.startswith('blk.40.'): tot += by
print('blk.40 total', tot)
for k, v in kv.items():
    if any(x in k for x in ('nextn', 'block_count', 'embedding_length', 'expert', 'vocab_size')):
        print(k, v)
print('vocab', len(kv['tokenizer.ggml.tokens'][3]) if isinstance(kv.get('tokenizer.ggml.tokens'), tuple) else kv.get('tokenizer.ggml.tokens', '')[:0])
