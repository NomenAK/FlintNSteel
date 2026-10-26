"""Dump reference inputs and outputs for the experimental EXL3 CUDA kernels (real model, fixed seed).

Experimental: EXL3 model parity is not validated. These dumps are diagnostics, not a parity claim.
"""
import argparse
import json
import sys
from pathlib import Path
import os
import torch
from safetensors import safe_open
# The port keeps F16 activations; disable the optional upstream INT8 GEMV shortcut.
os.environ['EXL3_INT8_GEMV'] = '0'


ROOT = Path(__file__).resolve().parents[2]


def build_dir():
    """Build directory: FNS_BUILD_DIR, else ROOT/build."""
    return Path(os.environ.get('FNS_BUILD_DIR') or ROOT / 'build')


def exllamav3_dir(arg):
    """exllamav3 checkout: --exllamav3-dir, else FNS_EXLLAMAV3_DIR."""
    path = arg or os.environ.get('FNS_EXLLAMAV3_DIR')
    if not path:
        sys.exit('error: set FNS_EXLLAMAV3_DIR or pass --exllamav3-dir')
    return Path(path)


def main():
    ap = argparse.ArgumentParser(description='Dump reference inputs/outputs for the experimental EXL3 CUDA kernels. Parity is not validated.')
    ap.add_argument('--exllamav3-dir', help='exllamav3 checkout (default: FNS_EXLLAMAV3_DIR)')
    ap.add_argument('--model', required=True, help='EXL3 model directory')
    ap.add_argument('--out', help='output directory (default: <build dir>/exl3-dump)')
    ap.add_argument('--base', help='llama.cpp Q8_0 KLD base file (required with --chunk-logits)')
    ap.add_argument('--chunk-logits', action='store_true', help='compare the first chunk against logits written by fns-kld --logits-out')
    ap.add_argument('--reference-batch', type=int, choices=(0, 8), default=0, help='0: long prefill; 8: same batch geometry as FNS')
    a = ap.parse_args()
    if a.chunk_logits and not a.base:
        ap.error('--chunk-logits requires --base')
    sys.path.insert(0, str(exllamav3_dir(a.exllamav3_dir)))
    from exllamav3.ext import exllamav3_ext as ext
    root = Path(a.model)
    out = Path(a.out) if a.out else build_dir() / 'exl3-dump'
    out.mkdir(parents=True, exist_ok=True)
    if a.chunk_logits:
        compare_chunk(root, out, a.base, a.reference_batch)
        return
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    torch.manual_seed(1731)
    cases = []
    def load(key):
        with safe_open(root / index[key], framework='pt', device='cpu') as f:
            return f.get_tensor(key).cuda()
    def linear(key):
        return [load(key + '.' + n) for n in ('trellis', 'suh', 'svh')]
    def save(name, t):
        t.detach().cpu().contiguous().numpy().tofile(out / name)
    with torch.inference_mode():
        keys = ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.gate_proj',
                'model.layers.0.mlp.down_proj', 'model.layers.3.self_attn.o_proj',
                'model.layers.48.self_attn.k_proj', 'lm_head']
        for key in keys:
            b, su, sv = linear(key)
            k, n, bits = b.shape[0]*16, b.shape[1]*16, b.shape[2]//16
            for m in (1, 8):
                for fp32 in (False, True):
                    x = torch.randn((m, k), device='cuda', dtype=torch.half)*0.2
                    y = torch.empty((m, n), device='cuda', dtype=torch.float if fp32 else torch.half)
                    xh = torch.empty_like(x)
                    # Explicit fixed kernel parameters for reproducibility; no autotune.
                    ext.exl3_gemm(x, b, y, su, xh, sv, 2 if bits == 5 else -1, False, True, 82 if bits == 5 else 0)
                    stem = f'linear_{len(cases)}'
                    save(stem+'.x', x); save(stem+'.y', y)
                    cases.append(dict(kind='linear', key=key, m=m, fp32=fp32, stem=stem))
            del b, su, sv
        for layer in (3, 48):
            keys = [f'model.layers.{layer}.mlp.experts.{e}' for e in range(12)]
            weights = [[linear(key + '.' + p + '_proj') for key in keys] for p in ('gate', 'up', 'down')]
            tables = [torch.tensor([w[j].data_ptr() for w in proj],device='cuda',dtype=torch.long) for proj in weights for j in range(3)]
            for m in (1, 8):
                for gu_fp32 in (False, True):
                    slots, I, H = 48, 512, 2048
                    x = torch.randn((m,H),device='cuda',dtype=torch.half)*0.2
                    sel = (torch.arange(m*6,device='cuda').reshape(m,6)%12).long()
                    rw = torch.softmax(torch.randn((m,6),device='cuda'),-1).half()
                    hg, hu = [torch.empty((slots,H),device='cuda',dtype=torch.half) for _ in range(2)]
                    gg, gu = [torch.empty((slots,I),device='cuda',dtype=torch.float if gu_fp32 else torch.half) for _ in range(2)]
                    act = torch.empty((slots,I),device='cuda',dtype=torch.half)
                    down = torch.empty((slots,H),device='cuda',dtype=torch.float)
                    ctr = torch.zeros(slots*(I//128)+8*(H//128)+2+slots+1+slots,device='cuda',dtype=torch.int)
                    y = torch.empty((8,H),device='cuda',dtype=torch.float)
                    ext.exl3_moe_coop(x,sel,rw,-1,-1,H,*tables,None,None,None,4.,4.,4.,False,True,0,0.,True,
                                      hg,hu,gg,gu,act,down,ctr,y,None,None)
                    stem=f'moe_{len(cases)}'
                    for suffix,t in (('x',x),('y',y[:m]),('sel',sel),('rw',rw)): save(stem+'.'+suffix,t)
                    cases.append(dict(kind='moe',layer=layer,m=m,stem=stem,experts=12,gu_fp32=gu_fp32))
            del weights, tables
        for layer in (3,48):
            wt=load(f'model.layers.{layer}.mlp.gate.weight')
            wi=torch.empty((2,256,2048),device='cuda',dtype=torch.int8)
            ws=torch.empty((256,),device='cuda',dtype=torch.float)
            ext.det_quant_weight(wt,wi,ws)
            for rows in (1,8):
                x=torch.randn((rows,2048),device='cuda',dtype=torch.half)*3
                scores=torch.empty((rows,256),device='cuda',dtype=torch.half)
                sel=torch.empty((rows,6),device='cuda',dtype=torch.long)
                rw=torch.empty((rows,6),device='cuda',dtype=torch.half)
                ext.routing_ds3_nogroup(x,wt.T.contiguous(),scores,None,sel,rw,128.,wt,0,wi,ws)
                stem=f'router_{len(cases)}'
                save(stem+'.x',x);save(stem+'.y',rw);save(stem+'.sel',sel)
                cases.append(dict(kind='router',layer=layer,m=rows,stem=stem))
        (out/'manifest.json').write_text(json.dumps(cases,indent=2))
        print(f'{len(cases)} cases written to {out}')

def compare_chunk(root, out, base, batch):
    import numpy as np
    from exllamav3.model.config import Config
    from exllamav3.model.model import Model
    from exllamav3.cache import Cache
    with Path(base).open('rb') as f:
        assert f.read(8) == b'_logits_'
        ctx, vocab, chunks = np.frombuffer(f.read(12), dtype=np.int32)
        ids = np.frombuffer(f.read(int(ctx)*4), dtype=np.int32).astype(np.int64)
    model = Model.from_config(Config.from_directory(str(root)))
    cache = Cache(model, int(ctx), max_batch_size=1) if batch else None
    model.load(progressbar=True)
    got = np.memmap(out/'fns_chunk0.f32', dtype=np.float32, mode='r', shape=(int(ctx), int(vocab)))
    with torch.inference_mode():
        if batch:
            params = {'attn_mode':'flash_attn', 'cache':cache, 'batch_shape':(1,int(ctx)), 'past_len':0}
            ref_cpu = torch.empty((int(ctx),int(vocab)),dtype=torch.half)
            for p in range(0,int(ctx),batch):
                params['past_len'] = p
                ref_cpu[p:p+batch] = model.forward(torch.from_numpy(ids[p:p+batch])[None,:],params)[0,:,:int(vocab)].cpu()
            ref = ref_cpu
        else:
            ref = model.forward(torch.from_numpy(ids)[None,:], {'attn_mode':'flash_attn_nc'})[0,:,:int(vocab)]
        kl = 0.; same = 0; count = 0; scored_sum = 0.; scored_n = 0; scored_max = 0.
        for start in range(0, int(ctx), 64):
            r = ref[start:start+64].cuda().float()
            g = torch.from_numpy(np.array(got[start:start+64])).cuda()
            lp, lg = torch.log_softmax(r,-1), torch.log_softmax(g,-1)
            kl += (lp.exp()*(lp-lg)).sum(-1).sum().item()
            same += (r.argmax(-1)==g.argmax(-1)).sum().item()
            count += r.shape[0]
            lo = max(int(ctx)//2-start, 0)
            hi = min(int(ctx)-1-start, r.shape[0])
            if hi > lo:
                sk = (lp[lo:hi].exp()*(lp[lo:hi]-lg[lo:hi])).sum(-1)
                scored_sum += sk.sum().item(); scored_n += sk.numel(); scored_max = max(scored_max, sk.max().item())
    result = dict(positions=count, kld_exllamav3_fns=kl/count, top1_pct=100*same/count,
                  activation_int8=False, torch=torch.__version__)
    result['reference_batch'] = batch
    result['scored_kld'] = scored_sum/scored_n
    result['scored_max_kld'] = scored_max
    (out/'chunk0_comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
