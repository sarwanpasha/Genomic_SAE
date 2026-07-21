#!/usr/bin/env python
"""
Causal patching: ablate an SAE feature's contribution at layer L during NT's
forward pass, measure the change in NT's own MLM output distribution (KL),
and test whether the causal effect is BINDING-SPECIFIC (larger at bound vs
unbound TF motif sites).

Ablation: at layer L hidden state x, remove the component along the SAE
feature's decoder direction:  x' = x - f_i * dec_i      (f_i = feature activation)
Then continue the forward pass and read NT's logits.

Readout: per window, mean KL(unpatched || patched) over tokens.
Test: KL_bound vs KL_unbound (Mann-Whitney). Binding-specific causal use =>
KL larger at bound sites.

Controls: also ablate (a) a random feature, (b) a strong motif-but-not-binding
feature, and verify they do NOT show the bound>unbound differential.

Env: MODEL(nt) LAYER(14) TF FEATURE  K(32) EXP(16)  NWIN(800)
GPU job.
"""
import os, json, numpy as np, torch
os.environ.setdefault("HF_HOME","/projects/bhkb/sali4/genomic_sae/hf_cache")
import torch.nn as nn
from pyfaidx import Fasta
from model_loader import load_model
from scipy.stats import mannwhitneyu

ROOT = os.environ.get("GENOMIC_SAE_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL=os.environ.get("MODEL","nt"); LAYER=int(os.environ.get("LAYER",14))
TF=os.environ["TF"]; FEATURE=int(os.environ["FEATURE"])
K=int(os.environ.get("K",32)); EXP=int(os.environ.get("EXP_FACTOR",16))
NWIN=int(os.environ.get("NWIN",800))
device="cuda" if torch.cuda.is_available() else "cpu"
path=f"{ROOT}/model_nt_500m"

ckpt=torch.load(f"{ROOT}/sae_{MODEL}_full_layer{LAYER}_k{K}_x{EXP}.pt",
                map_location=device, weights_only=False)
W_enc=ckpt["W_enc"].to(device); W_dec=ckpt["W_dec"].to(device)
b_pre=ckpt["b_pre"].to(device); x_mean=ckpt["x_mean"].to(device); scale=ckpt["scale"].to(device)
n_feats=W_enc.shape[0]

tok, model = load_model(MODEL, path, device)
layers=None
for name,mod in model.named_modules():
    if isinstance(mod,nn.ModuleList) and len(mod)>=4:
        if layers is None or len(mod)>len(layers): layers=mod

# ablation hook factory: removes feature `feat`'s decoder direction from layer output
ABLATE={"feat":None}
def hook(m, inp, out):
    if ABLATE["feat"] is None:
        return out
    h = out[0] if isinstance(out,(tuple,list)) else out      # [1, n_tok, d]
    squeeze = (h.ndim==3)
    hh = h[0] if squeeze else h                               # [n_tok, d]
    # encode to get this feature's activation, in normalized space
    hn = (hh - x_mean)*scale
    f_val = torch.relu((hn - b_pre) @ W_enc[ABLATE["feat"]])  # [n_tok]
    # reconstruct the feature's contribution and remove it (in normalized space)
    contrib_n = torch.outer(f_val, W_dec[:, ABLATE["feat"]])  # [n_tok, d]
    hn_ablated = hn - contrib_n
    hh_new = hn_ablated/scale + x_mean                        # back to raw space
    hh_new = hh_new.to(h.dtype)
    if squeeze:
        h = h.clone(); h[0]=hh_new
    else:
        h = hh_new
    if isinstance(out,(tuple,list)):
        return (h,)+tuple(out[1:])
    return h
layers[LAYER-1].register_forward_hook(hook)

fa=Fasta(f"{ROOT}/hg38.fa")
rows=[l.strip().split("\t") for l in open(f"{ROOT}/binding_windows_{TF}.tsv")][1:]
import random; random.seed(0)
by_cls={}
for cls,c,s,e in rows: by_cls.setdefault(cls,[]).append((c,int(s),int(e)))
def sample(cls,n):
    L=by_cls.get(cls,[]); random.shuffle(L); return L[:n]
bound=sample("bound",NWIN); unbound=sample("unbound_motif",NWIN)
print(f"[cfg] {TF} feat{FEATURE}: bound={len(bound)} unbound={len(unbound)}",flush=True)

def kl_for_windows(wins, feat):
    kls=[]
    for c,s,e in wins:
        seq=str(fa[c][s:e]).upper()
        if not (set(seq)<=set("ACGT")): continue
        enc=tok(seq, return_tensors="pt").to(device)
        with torch.no_grad():
            ABLATE["feat"]=None
            base=model(**enc).logits[0]           # [n_tok, vocab]
            ABLATE["feat"]=feat
            pat=model(**enc).logits[0]
            ABLATE["feat"]=None
        lp=torch.log_softmax(base,-1); lq=torch.log_softmax(pat,-1)
        p=lp.exp()
        kl=(p*(lp-lq)).sum(-1)                     # [n_tok]
        kls.append(kl.mean().item())
    return np.array(kls)

def run_feature(feat, tag):
    kb=kl_for_windows(bound, feat); ku=kl_for_windows(unbound, feat)
    try: u,p=mannwhitneyu(kb,ku,alternative="greater"); auc=u/(len(kb)*len(ku))
    except ValueError: p,auc=1.0,0.5
    print(f"[{tag}] feat {feat}: KL_bound={kb.mean():.4f} KL_unbound={ku.mean():.4f} "
          f"ratio={kb.mean()/(ku.mean()+1e-9):.2f} AUC={auc:.3f} p={p:.2e}", flush=True)
    return dict(tag=tag, feature=feat, kl_bound=float(kb.mean()),
               kl_unbound=float(ku.mean()), auc=float(auc), pval=float(p))

results=[]
# 1) the target binding feature
results.append(run_feature(FEATURE, "TARGET"))
# 2) control: random live feature
rng=np.random.default_rng(0)
live=np.nonzero((W_dec.norm(dim=0).cpu().numpy()>0))[0]
rand_feat=int(rng.choice(live))
results.append(run_feature(rand_feat, "RANDOM_CTRL"))
# 3) control: strong motif-selective but low binding-sensitivity feature
bt=json.load(open(f"{ROOT}/bindingtest_{MODEL}_{TF}_layer{LAYER}.json"))
motif_only=[r for r in bt if r["auc_motif_vs_bg"]>0.6 and r["auc_bound_vs_unbound"]<0.52]
if motif_only:
    results.append(run_feature(int(motif_only[0]["feature"]), "MOTIF_ONLY_CTRL"))

json.dump(results, open(f"{ROOT}/causal_{MODEL}_{TF}_feat{FEATURE}.json","w"), indent=2)
print(f"\n[done] saved causal_{MODEL}_{TF}_feat{FEATURE}.json", flush=True)
