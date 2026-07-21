#!/usr/bin/env python
"""
Population causal test: run feature-ablation causal patching over the top-N
binding-sensitive features for a TF, plus matched controls, and report what
fraction show a significant BINDING-SPECIFIC causal effect (KL_bound>KL_unbound).

Env: MODEL(nt) LAYER(14) TF  TOPN_FEAT(15) NWIN(600) K(32) EXP(16)
GPU job.
"""
import os, json, numpy as np, torch
os.environ.setdefault("HF_HOME","/projects/bhkb/sali4/genomic_sae/hf_cache")
import torch.nn as nn, random
from pyfaidx import Fasta
from model_loader_lm import load_lm_model as load_model
from scipy.stats import mannwhitneyu

ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL=os.environ.get("MODEL","nt"); LAYER=int(os.environ.get("LAYER",14))
TF=os.environ["TF"]; TOPN_FEAT=int(os.environ.get("TOPN_FEAT",15))
NWIN=int(os.environ.get("NWIN",600))
K=int(os.environ.get("K",32)); EXP=int(os.environ.get("EXP_FACTOR",16))
device="cuda" if torch.cuda.is_available() else "cpu"
path=f"{ROOT}/model_nt_500m" if MODEL=="nt" else f"{ROOT}/model_{MODEL}"
if MODEL=="dnabert2": path=f"{ROOT}/model_dnabert2"

ckpt=torch.load(f"{ROOT}/sae_{MODEL}_full_layer{LAYER}_k{K}_x{EXP}.pt",
                map_location=device, weights_only=False)
W_enc=ckpt["W_enc"].to(device); W_dec=ckpt["W_dec"].to(device)
b_pre=ckpt["b_pre"].to(device); x_mean=ckpt["x_mean"].to(device); scale=ckpt["scale"].to(device)

tok, model = load_model(MODEL, path, device)
layers=None
for name,mod in model.named_modules():
    if isinstance(mod,nn.ModuleList) and len(mod)>=4:
        if layers is None or len(mod)>len(layers): layers=mod
ABLATE={"feat":None}
def hook(m, inp, out):
    if ABLATE["feat"] is None: return out
    h=out[0] if isinstance(out,(tuple,list)) else out
    sq=(h.ndim==3); hh=h[0] if sq else h
    hn=(hh-x_mean)*scale
    fv=torch.relu((hn-b_pre)@W_enc[ABLATE["feat"]])
    hn2=hn-torch.outer(fv,W_dec[:,ABLATE["feat"]])
    hh2=(hn2/scale+x_mean).to(h.dtype)
    if sq: h=h.clone(); h[0]=hh2
    else: h=hh2
    return (h,)+tuple(out[1:]) if isinstance(out,(tuple,list)) else h
layers[LAYER-1].register_forward_hook(hook)

fa=Fasta(f"{ROOT}/hg38.fa")
rows=[l.strip().split("\t") for l in open(f"{ROOT}/binding_windows_{TF}.tsv")][1:]
by={}
for cls,c,s,e in rows: by.setdefault(cls,[]).append((c,int(s),int(e)))
random.seed(0)
for k in by: random.shuffle(by[k])
bound=by["bound"][:NWIN]; unbound=by["unbound_motif"][:NWIN]

def kl_windows(wins, feat):
    out=[]
    for c,s,e in wins:
        seq=str(fa[c][s:e]).upper()
        if not set(seq)<=set("ACGT"): continue
        enc=tok(seq,return_tensors="pt").to(device)
        with torch.no_grad():
            ABLATE["feat"]=None; base=model(**enc).logits[0]
            ABLATE["feat"]=feat; pat=model(**enc).logits[0]; ABLATE["feat"]=None
        lp=torch.log_softmax(base,-1); lq=torch.log_softmax(pat,-1)
        out.append(((lp.exp()*(lp-lq)).sum(-1).mean().item()))
    return np.array(out)

def test_feat(feat):
    kb=kl_windows(bound,feat); ku=kl_windows(unbound,feat)
    try: u,p=mannwhitneyu(kb,ku,alternative="greater"); auc=u/(len(kb)*len(ku))
    except ValueError: p,auc=1.0,0.5
    return dict(feature=int(feat),kl_bound=float(kb.mean()),kl_unbound=float(ku.mean()),
               auc=float(auc),pval=float(p))

# select top-N binding-sensitive, GC-robust features from the binding test
bt=json.load(open(f"{ROOT}/bindingtest_{MODEL}_{TF}_layer{LAYER}.json"))
cand=[r for r in bt if r["auc_bound_vs_unbound"]>0.55 and abs(r.get("gc_corr",0))<0.25]
cand.sort(key=lambda r:-r["auc_bound_vs_unbound"])
targets=[r["feature"] for r in cand[:TOPN_FEAT]]
print(f"[cfg] {TF}: testing {len(targets)} binding-sensitive GC-robust features", flush=True)

results=[]
for i,f in enumerate(targets):
    r=test_feat(f); r["type"]="target"; results.append(r)
    print(f"  target {f:5d}: KL_b={r['kl_bound']:.2e} KL_u={r['kl_unbound']:.2e} "
          f"AUC={r['auc']:.3f} p={r['pval']:.1e}", flush=True)

# matched random controls (same count)
rng=np.random.default_rng(1)
live=np.nonzero(W_dec.norm(dim=0).cpu().numpy()>0)[0]
for f in rng.choice(live,len(targets),replace=False):
    r=test_feat(int(f)); r["type"]="random"; results.append(r)
    print(f"  random {int(f):5d}: AUC={r['auc']:.3f} p={r['pval']:.1e}", flush=True)

if len(results)==0:
    print(f"[{TF}] no binding-sensitive features to test -> NEGATIVE CONTROL PASSED (0 causal)")
    json.dump([], open(f"{ROOT}/causalpop_{MODEL}_{TF}_layer{LAYER}.json","w"))
    import sys; sys.exit(0)
bonf=0.05/len(results)
sig_t=[r for r in results if r["type"]=="target" and r["pval"]<bonf and r["auc"]>0.55]
sig_r=[r for r in results if r["type"]=="random" and r["pval"]<bonf and r["auc"]>0.55]
print(f"\n[{TF}] binding-specific causal effect (Bonferroni p<{bonf:.1e}, AUC>0.55):")
print(f"  targets: {len(sig_t)}/{len(targets)}   random controls: {len(sig_r)}/{len(targets)}")
json.dump(results, open(f"{ROOT}/causalpop_{MODEL}_{TF}_layer{LAYER}.json","w"), indent=2)
print(f"[done] saved causalpop_{MODEL}_{TF}_layer{LAYER}.json", flush=True)
